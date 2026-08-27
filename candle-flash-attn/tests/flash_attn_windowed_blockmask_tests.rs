use anyhow::Result;
use candle::{DType, Device, Tensor, D};

// Eager reference replicating modeling_deepseek_v4.py eager_attention_forward exactly:
//   attn = (Q @ K^T) * scale + additive_mask        [B,H,Sq,Skv]
//   combined = cat([attn, sink_logit[h]], dim=-1)   sink = extra logit column
//   combined -= max(combined)                       max-subtract includes sink
//   p = softmax(combined)
//   scores = p[..., :-1]                            drop sink column (probability leak)
//   out = scores @ V
//
// q: [B, Sq, H, D]; k/v: [B, Skv, Hk, D] (MQA); mask: [B, 1, Sq, Skv]; sink: [H].
fn eager_ref(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    sink: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let (b, sq, h, d) = q.dims4()?;
    let (_, skv, hk, _) = k.dims4()?;
    let groups = h / hk;

    // repeat_kv -> [B, Skv, H, D]
    let k = k
        .unsqueeze(2)?
        .expand((b, skv, hk, groups, d))?
        .reshape((b, skv, h, d))?;
    let v = v
        .unsqueeze(2)?
        .expand((b, skv, hk, groups, d))?
        .reshape((b, skv, h, d))?;

    // attn = (Q @ K^T) * scale  -> [B, H, Sq, Skv]
    let qp = q.permute((0, 2, 1, 3))?;
    let kp = k.permute((0, 2, 1, 3))?;
    let attn = (qp.matmul(&kp.transpose(2, 3)?)? * scale as f64)?;
    let attn = (attn + mask)?;

    // sink column appended on the KV axis
    let sinkv = sink.reshape((1, h, 1, 1))?.expand((b, h, sq, 1))?;
    let combined = Tensor::cat(&[&attn, &sinkv], 3)?; // [B, H, Sq, Skv+1]

    let maxv = combined.max(D::Minus1)?;
    let combined = (combined - maxv)?;
    let probs = candle_nn::ops::softmax(&combined, D::Minus1)?;
    let scores = probs.narrow(3, 0, skv)?; // drop sink column

    let vp = v.permute((0, 2, 1, 3))?; // [B, H, Skv, D]
    let out = scores.matmul(&vp)?; // [B, H, Sq, D]
    let out = out.permute((0, 2, 1, 3))?; // [B, Sq, H, D]
    Ok(out.to_dtype(in_dtype)?)
}

// Build an additive mask [B, 1, Sq, Skv] (0 / -inf) for a single batch, as a flat f32 vec.
//   local:  causal sliding-window over the local KV prefix (window w).
//   compressed: per-query causal block_bias over the compressed suffix (compress_rate).
fn build_hetero_mask(
    sq: usize,
    local_len: usize,
    compressed_len: usize,
    window: usize,
    compress_rate: usize,
) -> Vec<f32> {
    let skv = local_len + compressed_len;
    let mut mask = Vec::with_capacity(sq * skv);
    for q in 0..sq {
        for j in 0..skv {
            let visible = if j < local_len {
                // sliding causal over local prefix: q-w < j <= q
                j <= q && j > q.saturating_sub(window)
            } else {
                // compressed suffix: entry c = j - local_len visible iff causal for query q
                let c = j - local_len;
                c < (q + 1) / compress_rate
            };
            mask.push(if visible { 0.0f32 } else { f32::NEG_INFINITY });
        }
    }
    mask
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?;
    let b = b.to_dtype(DType::F32)?;
    Ok(a.sub(&b)?.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?)
}

fn run_case(
    device: &Device,
    b: usize,
    sq: usize,
    skv: usize,
    h: usize,
    hk: usize,
    d: usize,
    scale: f32,
    mask: Vec<f32>,
    sink: Vec<f32>,
) -> Result<()> {
    let total_q = b * sq * h * d;
    let total_kv = b * skv * hk * d;
    let q = Tensor::arange(0u32, total_q as u32, device)?
        .to_dtype(DType::BF16)?
        .reshape((b, sq, h, d))?;
    let q = (&q / 30.)?;
    let k = (&Tensor::arange(0u32, total_kv as u32, device)?
        .to_dtype(DType::BF16)?
        .reshape((b, skv, hk, d))?
        / 40.)?;
    let v = (&Tensor::arange(0u32, total_kv as u32, device)?
        .to_dtype(DType::BF16)?
        .reshape((b, skv, hk, d))?
        / 50.)?;

    let mask_t = Tensor::from_vec(mask, (b, 1, sq, skv), device)?;
    let sink_t = Tensor::from_vec(sink, (h,), device)?;

    let expected = eager_ref(&q, &k, &v, &mask_t, &sink_t, scale)?;
    let got =
        candle_flash_attn::flash_attn_windowed_blockmask(&q, &k, &v, &mask_t, &sink_t, scale)?;

    assert_eq!(got.dims(), &[b, sq, h, d]);
    let diff = max_abs_diff(&expected, &got)?;
    // bf16 tolerance (matches the other bf16 flash tests in this suite)
    assert!(
        diff < 0.125,
        "kernel deviates from eager reference by {diff} (>0.125)"
    );
    Ok(())
}

#[test]
fn windowed_blockmask_sliding_only() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let b = 1;
    let sq = 16;
    let local_len = 16;
    let compressed_len = 0;
    let h = 8;
    let hk = 1;
    let d = 128;
    let scale = 1.0 / (d as f32).sqrt();
    let mask = build_hetero_mask(sq, local_len, compressed_len, /*window=*/ 8, 4);
    let sink = vec![0.0f32; h];
    run_case(
        &device,
        b,
        sq,
        local_len + compressed_len,
        h,
        hk,
        d,
        scale,
        mask,
        sink,
    )
}

#[test]
fn windowed_blockmask_local_compressed_block_bias() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let b = 1;
    let sq = 8;
    let local_len = 10;
    let compressed_len = 4;
    let h = 8;
    let hk = 1;
    let d = 128;
    let scale = 1.0 / (d as f32).sqrt();
    let mask = build_hetero_mask(sq, local_len, compressed_len, /*window=*/ 4, 4);
    let sink = vec![0.0f32; h];
    run_case(
        &device,
        b,
        sq,
        local_len + compressed_len,
        h,
        hk,
        d,
        scale,
        mask,
        sink,
    )
}

// Exercise a non-trivial per-head sink so the max-subtract / drop-sink path is covered.
#[test]
fn windowed_blockmask_sink_logits() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let b = 1;
    let sq = 8;
    let local_len = 12;
    let compressed_len = 3;
    let h = 4;
    let hk = 1;
    let d = 128;
    let scale = 1.0 / (d as f32).sqrt();
    let mask = build_hetero_mask(sq, local_len, compressed_len, /*window=*/ 6, 4);
    let sink = (0..h).map(|i| i as f32 * 0.5).collect();
    run_case(
        &device,
        b,
        sq,
        local_len + compressed_len,
        h,
        hk,
        d,
        scale,
        mask,
        sink,
    )
}
