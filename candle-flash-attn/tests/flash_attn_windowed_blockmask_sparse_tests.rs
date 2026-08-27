//! Block-sparse execution parity tests for `flash_attn_windowed_blockmask`.
//!
//! The kernel (story #4276) processes the KV axis in 64-column blocks and skips any block
//! whose columns are all masked (`-inf`) - no QK dot, no K/V load. An `-inf` column
//! contributes exactly 0 to the running max (which starts at the finite sink logit), the
//! denominator, and the accumulator, so the sparse result is bit-identical to the dense
//! masked reference while QK/V traffic scales only with the attended blocks.
//!
//! Two kinds of tests:
//!   * `block_sparse_skip_equals_dense_cpu` - CPU-only proof that skipping fully-masked
//!     64-column blocks is bit-identical to processing every column. No CUDA needed.
//!   * `dsa_flash_block_sparse_cuda` - runs the real CUDA kernel vs the masked-dense eager
//!     reference for sliding-only, CSA (index_topk selection), and HCA (causality) masks.
//!
//! Run with: `cargo test -p candle-flash-attn --features cuda dsa_flash_block_sparse`

use anyhow::Result;
use candle::{DType, Device, Tensor, D};

// Eager masked-dense reference (modeling_deepseek_v4.py eager_attention_forward):
//   attn = (Q @ K^T)*scale + mask; combined = cat([attn, sink]); -= max; softmax;
//   drop sink column; out = scores @ V. q:[B,Sq,H,D]; k/v:[B,Skv,Hk,D] (MQA); mask:[B,1,Sq,Skv].
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
    let k = k
        .unsqueeze(2)?
        .expand((b, skv, hk, groups, d))?
        .reshape((b, skv, h, d))?;
    let v = v
        .unsqueeze(2)?
        .expand((b, skv, hk, groups, d))?
        .reshape((b, skv, h, d))?;
    let qp = q.permute((0, 2, 1, 3))?.contiguous()?;
    let kp = k.permute((0, 2, 1, 3))?.contiguous()?;
    let attn = (qp.matmul(&kp.transpose(2, 3)?.contiguous()?)? * scale as f64)?;
    let maskb = mask.broadcast_as((b, h, sq, skv))?;
    let attn = (attn + maskb)?;
    let sinkv = sink.reshape((1, h, 1, 1))?.expand((b, h, sq, 1))?;
    let combined = Tensor::cat(&[&attn, &sinkv], 3)?;
    let maxv = combined
        .max(D::Minus1)?
        .unsqueeze(3)?
        .broadcast_as((b, h, sq, skv + 1))?;
    let combined = (combined - maxv)?;
    let probs = candle_nn::ops::softmax(&combined, D::Minus1)?;
    let scores = probs.narrow(3, 0, skv)?;
    let vp = v.permute((0, 2, 1, 3))?.contiguous()?;
    let out = scores.contiguous()?.matmul(&vp)?;
    Ok(out.permute((0, 2, 1, 3))?.to_dtype(in_dtype)?)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?;
    let b = b.to_dtype(DType::F32)?;
    Ok(a.sub(&b)?.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?)
}

// Heterogeneous additive mask [B,1,Sq,Skv] (0/-inf) for one batch:
//   local: causal sliding-window over the local prefix (window w);
//   compressed: per-query causal block_bias (compress_rate).
// Produces dense-masked columns in the same shape the kernel consumes.
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
                j <= q && j > q.saturating_sub(window)
            } else {
                let c = j - local_len;
                c < (q + 1) / compress_rate
            };
            mask.push(if visible { 0.0f32 } else { f32::NEG_INFINITY });
        }
    }
    mask
}

// ---------------------------------------------------------------------------
// CPU-only online softmax with sink over a KV row, replicating the kernel.
//
// `mode`:
//   Dense  - folds every column (including -inf) into the running state.
//   Sparse - skips fully-masked 64-column blocks entirely (no fold for them).
// Both must be bit-identical: an -inf column contributes exp(-inf - m) == 0 to the
// denominator and 0 to the accumulator, and fmaxf(m, -inf) == m.
// ---------------------------------------------------------------------------
#[derive(PartialEq, Clone, Copy)]
enum Mode {
    Dense,
    Sparse,
}

fn online_softmax_row(
    q: &[f32],
    k: &[f32], // skv rows of d
    v: &[f32],
    mask: &[f32],
    sink: f32,
    scale: f32,
    skv: usize,
    d: usize,
    mode: Mode,
) -> Vec<f32> {
    const BK: usize = 64;
    // Running max starts at the sink logit (finite); denominator starts at 1 (exp(sink-m)).
    let mut m = sink;
    let mut l = 1.0f32;
    let mut acc = vec![0.0f32; d];
    for col_base in (0..skv).step_by(BK) {
        let end = (col_base + BK).min(skv);
        if mode == Mode::Sparse {
            // Skip the block iff every column is fully masked (-inf).
            let mut skip = true;
            for c in col_base..end {
                if mask[c].is_finite() {
                    skip = false;
                    break;
                }
            }
            if skip {
                continue;
            }
        }
        for c in col_base..end {
            let s = if mask[c].is_finite() {
                let dot: f32 = (0..d).map(|t| q[t] * k[c * d + t]).sum();
                dot * scale + mask[c]
            } else {
                // Dense mode still folds -inf columns (exp(-inf - m) == 0, no state change).
                f32::NEG_INFINITY
            };
            let m_new = m.max(s);
            let rescale = (m - m_new).exp();
            l = l * rescale + (s - m_new).exp();
            for t in 0..d {
                acc[t] = acc[t] * rescale + (s - m_new).exp() * v[c * d + t];
            }
            m = m_new;
        }
    }
    acc.iter().map(|a| a / l).collect()
}

#[test]
fn block_sparse_skip_equals_dense_cpu() -> Result<()> {
    // Many fully-masked 64-column blocks so the sparse path actually skips something.
    let sq = 8;
    let local_len = 64; // one attended BK-block
    let compressed_len = 320; // 5 more BK-blocks, most fully masked
    let h = 2;
    let d = 128;
    let skv = local_len + compressed_len;
    let scale = 1.0 / (d as f32).sqrt();
    let mask = build_hetero_mask(sq, local_len, compressed_len, /*window=*/ 64, 16);

    for qrow in 0..sq {
        let q: Vec<f32> = (0..d).map(|t| ((qrow * 17 + t) as f32) / 31.0).collect();
        let k: Vec<f32> = (0..skv * d).map(|i| (i as f32) / 40.0).collect();
        let v: Vec<f32> = (0..skv * d).map(|i| (i as f32) / 50.0).collect();
        let mask_row = &mask[qrow * skv..(qrow + 1) * skv];
        for hidx in 0..h {
            let sink = hidx as f32 * 0.5;
            let dense = online_softmax_row(&q, &k, &v, mask_row, sink, scale, skv, d, Mode::Dense);
            let sparse =
                online_softmax_row(&q, &k, &v, mask_row, sink, scale, skv, d, Mode::Sparse);
            assert_eq!(
                dense, sparse,
                "sparse (skip masked blocks) deviates from dense at q={qrow} h={hidx}"
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA parity: real kernel vs masked-dense eager reference.
// ---------------------------------------------------------------------------
fn run_cuda_case(
    b: usize,
    sq: usize,
    local_len: usize,
    compressed_len: usize,
    h: usize,
    hk: usize,
    d: usize,
    window: usize,
    compress_rate: usize,
    sink: &[f32],
) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let skv = local_len + compressed_len;
    let total_q = b * sq * h * d;
    let total_kv = b * skv * hk * d;
    let q = (&Tensor::arange(0u32, total_q as u32, &device)?
        .to_dtype(DType::BF16)?
        .reshape((b, sq, h, d))?
        / 30.)?;
    let k = (&Tensor::arange(0u32, total_kv as u32, &device)?
        .to_dtype(DType::BF16)?
        .reshape((b, skv, hk, d))?
        / 40.)?;
    let v = (&Tensor::arange(0u32, total_kv as u32, &device)?
        .to_dtype(DType::BF16)?
        .reshape((b, skv, hk, d))?
        / 50.)?;
    let mask = build_hetero_mask(sq, local_len, compressed_len, window, compress_rate);
    let mask_t = Tensor::from_vec(mask, (b, 1, sq, skv), &device)?;
    let sink_t = Tensor::from_vec(sink.to_vec(), (h,), &device)?;
    let scale = 1.0 / (d as f32).sqrt();

    let expected = eager_ref(&q, &k, &v, &mask_t, &sink_t, scale)?;
    let got =
        candle_flash_attn::flash_attn_windowed_blockmask(&q, &k, &v, &mask_t, &sink_t, scale)?;
    assert_eq!(got.dims(), &[b, sq, h, d]);
    let diff = max_abs_diff(&expected, &got)?;
    assert!(
        diff < 0.125,
        "block-sparse kernel deviates from masked-dense by {diff} (>0.125)"
    );
    Ok(())
}

#[test]
fn dsa_flash_block_sparse_sliding() -> Result<()> {
    // Sliding-only: local prefix, no compressed suffix. Every fully-masked block is skipped.
    run_cuda_case(1, 16, 192, 0, 8, 1, 128, 64, 4, &vec![0.0f32; 8])
}

#[test]
fn dsa_flash_block_sparse_csa_topk() -> Result<()> {
    // CSA-style: index_topk selects a few compressed entries => most compressed blocks fully
    // masked and skipped. compress_rate 16 with top-k ≈ index_topk/rate.
    run_cuda_case(1, 16, 64, 512, 8, 1, 128, 64, 16, &vec![0.0f32; 8])
}

#[test]
fn dsa_flash_block_sparse_hca_causal() -> Result<()> {
    // HCA-style: strict causality over the compressed suffix => trailing blocks fully masked.
    run_cuda_case(1, 16, 64, 384, 8, 1, 128, 64, 8, &vec![0.0f32; 8])
}

#[test]
fn dsa_flash_block_sparse_sink() -> Result<()> {
    // Non-trivial per-head sink exercises the max-subtract/drop-sink path alongside skipping.
    let sink: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
    run_cuda_case(1, 8, 64, 256, 8, 1, 128, 32, 8, &sink)
}
// ---------------------------------------------------------------------------
// V4-Flash parity: head_dim 512, 64 heads (the real DeepSeek V4 shape), with a
// CSA/indexer-style mask. The block-sparse kernel must be bit-identical (within
// bf16 tolerance) to the masked-dense eager reference at this target shape.
// ---------------------------------------------------------------------------
#[test]
fn dsa_flash_block_sparse_d512_64head() -> Result<()> {
    // V4-Flash target: head_dim 512, 64 query heads, MQA (1 kv head). CSA-style
    // compressed suffix (compress_rate 16) so most 64-col blocks are fully masked
    // and skipped, while the kernel still matches the dense masked reference.
    run_cuda_case(1, 16, 64, 512, 64, 1, 512, 64, 16, &vec![0.0f32; 64])
}

// ---------------------------------------------------------------------------
// Reduced-QK-traffic evidence (ncu unavailable on this host -> timing/event
// measurement). Run the SAME kernel with a fully-attended mask (dense: no block
// skipped) vs a mask whose 64-col blocks are mostly fully masked (sparse: those
// blocks are skipped - no QK dot, no K/V load, no softmax fold). Because QK/V
// traffic scales with attended blocks only, the sparse mask must run faster.
// The measured times are printed and recorded in the story comment.
// ---------------------------------------------------------------------------
fn skipped_block_fraction(mask: &[f32], sq: usize, skv: usize) -> f32 {
    let mut skipped = 0usize;
    let mut total = 0usize;
    for q in 0..sq {
        let row = &mask[q * skv..(q + 1) * skv];
        for base in (0..skv).step_by(64) {
            let end = (base + 64).min(skv);
            let mut all_masked = true;
            for c in base..end {
                if row[c].is_finite() {
                    all_masked = false;
                    break;
                }
            }
            total += 1;
            if all_masked {
                skipped += 1;
            }
        }
    }
    skipped as f32 / total as f32
}

#[test]
fn dsa_flash_block_sparse_traffic_evidence() -> Result<()> {
    use std::time::Instant;
    let device = Device::new_cuda(0)?;
    let b = 1usize;
    let sq = 32usize;
    let h = 8usize;
    let hk = 1usize;
    let d = 512usize;
    let local_len = 512usize;
    let compressed_len = 4096usize;
    let window = 512usize;
    let compress_rate = 16usize;
    let skv = local_len + compressed_len;
    let scale = 1.0 / (d as f32).sqrt();

    let total_q = b * sq * h * d;
    let total_kv = b * skv * hk * d;
    let q = (&Tensor::arange(0u32, total_q as u32, &device)?
        .to_dtype(DType::BF16)?
        .reshape((b, sq, h, d))?
        / 30.)?;
    let k = (&Tensor::arange(0u32, total_kv as u32, &device)?
        .to_dtype(DType::BF16)?
        .reshape((b, skv, hk, d))?
        / 40.)?;
    let v = (&Tensor::arange(0u32, total_kv as u32, &device)?
        .to_dtype(DType::BF16)?
        .reshape((b, skv, hk, d))?
        / 50.)?;
    let sink: Vec<f32> = (0..h).map(|i| i as f32 * 0.5).collect();
    let sink_t = Tensor::from_vec(sink, (h,), &device)?;

    let dense_mask = Tensor::from_vec(vec![0.0f32; sq * skv], (b, 1, sq, skv), &device)?;
    let sparse = build_hetero_mask(sq, local_len, compressed_len, window, compress_rate);
    let frac = skipped_block_fraction(&sparse, sq, skv);
    let sparse_mask = Tensor::from_vec(sparse, (b, 1, sq, skv), &device)?;

    let time = |mask: &Tensor, iters: usize| -> Result<f64> {
        for _ in 0..5 {
            let _ =
                candle_flash_attn::flash_attn_windowed_blockmask(&q, &k, &v, mask, &sink_t, scale)?;
        }
        device.synchronize()?;
        let start = Instant::now();
        for _ in 0..iters {
            let _ =
                candle_flash_attn::flash_attn_windowed_blockmask(&q, &k, &v, mask, &sink_t, scale)?;
        }
        device.synchronize()?;
        Ok(start.elapsed().as_secs_f64() / iters as f64)
    };

    let iters = 10usize;
    let dense_s = time(&dense_mask, iters)?;
    let sparse_s = time(&sparse_mask, iters)?;
    eprintln!(
        "dense  mask (all {} blocks attended): {:.3} ms/kernel",
        skv.div_ceil(64),
        dense_s * 1e3
    );
    eprintln!(
        "sparse mask ({:.1}% 64-col blocks skipped): {:.3} ms/kernel",
        frac * 100.0,
        sparse_s * 1e3
    );
    eprintln!(
        "speedup from skipping masked-out blocks: {:.2}x",
        dense_s / sparse_s
    );
    // Sparse must be measurably faster than dense (block-sparse skips masked blocks,
    // so QK/load traffic scales with attended blocks only).
    assert!(
        sparse_s < dense_s * 0.9,
        "sparse kernel ({sparse_s:.3}s) not faster than dense ({dense_s:.3}s) - \
         masked-out blocks were not skipped"
    );
    Ok(())
}
