//! Varlen + paged parity tests for `flash_attn_varlen_paged_windowed_blockmask`.
//!
//! This is the batched continuous-batching decode entry point: a batch of
//! sequences of differing lengths whose K/V lives in a physical paged cache
//! (`[num_blocks, page_block_size, Hk, D]`) addressed through a per-sequence
//! `block_table`, with a materialized per-query additive mask and per-head sink
//! logits. The kernel fuses QK^T*scale + mask + sink + max-subtract + softmax +
//! drop-sink + @V and skips fully-masked 64-column blocks.
//!
//! Each test builds a small varlen batch, scatters per-sequence K/V into pages,
//! runs the kernel on CUDA and compares every query row to an eager per-row
//! masked reference (mirroring `eager_attention_forward`).
//!
//! Run with: `cargo test -p candle-flash-attn varlen_paged_blockmask`

use anyhow::Result;
use candle::{DType, Device, Tensor, D};

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let d = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?;
    let v = d.to_vec1::<f32>()?;
    Ok(v.into_iter().fold(0.0f32, f32::max))
}

// Eager masked reference for one query row (batch `b`, local query row `qr`).
// q_row [H, D]; seq_k/seq_v [kv_len, Hk, D] (MQA broadcast to H); mask_row
// [kv_len]; sink [H]. Returns [H, D].
fn eager_row(
    q_row: &Tensor,    // [H, D]
    seq_k: &Tensor,    // [kv_len, Hk, D]
    seq_v: &Tensor,    // [kv_len, Hk, D]
    mask_row: &Tensor, // [kv_len]
    sink: &Tensor,     // [H]
    scale: f32,
) -> Result<Tensor> {
    let h = q_row.dim(0)?;
    let kv_len = seq_k.dim(0)?;
    let d = seq_k.dim(2)?;
    // k/v per query head: [kv_len, H, D] -> [H, kv_len, D]
    let k = seq_k.broadcast_as((kv_len, h, d))?.transpose(0, 1)?; // [H, kv_len, D]
    let v = seq_v.broadcast_as((kv_len, h, d))?.transpose(0, 1)?; // [H, kv_len, D]
                                                                  // att[h, j] = sum_d q_row[h,d] * k[h,j,d]
    let att = q_row
        .unsqueeze(1)? // [H, 1, D]
        .matmul(&k.transpose(1, 2)?)? // [H, D, kv_len] -> [H, 1, kv_len]
        .squeeze(1)?
        .affine(scale as f64, 0.0)?; // [H, kv_len]
    let mask = mask_row.unsqueeze(0)?.broadcast_as((h, kv_len))?;
    let att = att.broadcast_add(&mask)?;
    let sinks = sink.reshape((h, 1))?;
    let att = Tensor::cat(&[att, sinks], 1)?.contiguous()?; // [H, kv_len+1]
    let att = candle_nn::ops::softmax_last_dim(&att)?;
    let att = att.narrow(D::Minus1, 0, kv_len)?; // drop sink
                                                 // out[h, d] = sum_j att[h,j] * v[h,j,d]
    let out = att.unsqueeze(1)?.matmul(&v)?.squeeze(1)?; // [H, 1, kv_len] @ [H, kv_len, D]
    Ok(out)
}

// Build a causal + sliding mask row over [kv_len]: 0 for the last `window`
// columns (the local sliding window), -inf elsewhere (compressed entries are
// selected separately; here we simply attend to the whole window).
fn mask_row(kv_len: usize, window: usize) -> Vec<f32> {
    let mut m = vec![f32::NEG_INFINITY; kv_len];
    for j in 0..kv_len {
        if kv_len - j <= window {
            m[j] = 0.0;
        }
    }
    m
}

// Runs one varlen/paged case: `b` sequences with per-seq kv_lens, head dim `d`,
// `h` query heads, `hk` kv heads (MQA), page_block_size. Returns kernel output
// [total_q, h, d] and the eager reference [total_q, h, d] (for the test to
// compare).
fn run_case(
    b: usize,
    kv_lens: &[usize],
    h: usize,
    hk: usize,
    d: usize,
    window: usize,
    page_block_size: usize,
) -> Result<(Tensor, Tensor)> {
    let dev = Device::new_cuda(0)?;
    let max_kv = kv_lens.iter().copied().max().unwrap_or(0);

    // Per-sequence contiguous K==V, then scatter into pages.
    let mut k_pages: Vec<Tensor> = Vec::new();
    let mut block_table: Vec<Vec<i32>> = vec![Vec::new(); b];
    let mut max_blocks = 0usize;
    let mut seqs_kv: Vec<Tensor> = Vec::new();
    for bi in 0..b {
        let kv_len = kv_lens[bi];
        let kv = det_tensor(&[kv_len, hk, d], (10 + bi) as f32)
            .to_device(&dev)?
            .to_dtype(DType::BF16)?;
        seqs_kv.push(kv.clone());
        let n_blocks = kv_len.div_ceil(page_block_size);
        max_blocks = max_blocks.max(n_blocks);
        let mut row = Vec::with_capacity(n_blocks);
        for blk in 0..n_blocks {
            let phys = k_pages.len() as i32;
            row.push(phys);
            let start = blk * page_block_size;
            let end = (start + page_block_size).min(kv_len);
            let slice = kv.narrow(0, start, end - start)?; // [len, Hk, D]
                                                           // Pad to page_block_size rows with zeros.
            let pad = Tensor::zeros((page_block_size - (end - start), hk, d), kv.dtype(), &dev)?;
            let page = Tensor::cat(&[slice, pad], 0)?.contiguous()?; // [pbs, Hk, D]
            k_pages.push(page);
        }
        block_table[bi] = row;
    }
    let _num_blocks = k_pages.len();
    let k_paged = Tensor::stack(&k_pages, 0)?.contiguous()?; // [num_blocks, pbs, Hk, D]

    // Query rows: one per sequence (decode Sq=1), total_q == b.
    let total_q = b;
    let mut q_rows: Vec<Tensor> = Vec::new();
    for r in 0..total_q {
        q_rows.push(
            det_tensor(&[h, d], (50 + r) as f32)
                .to_device(&dev)?
                .to_dtype(DType::BF16)?,
        );
    }
    let q = Tensor::stack(&q_rows, 0)?.contiguous()?; // [total_q, h, d]

    // Mask [total_q, max_kv].
    let mut mask_vec = vec![f32::NEG_INFINITY; total_q * max_kv];
    for r in 0..total_q {
        let kv_len = kv_lens[r];
        let m = mask_row(kv_len, window);
        for j in 0..kv_len {
            mask_vec[r * max_kv + j] = m[j];
        }
    }
    let mask = Tensor::from_vec(mask_vec.clone(), (total_q, max_kv), &dev)?;
    let sink = det_tensor(&[h], 99.0)
        .to_device(&dev)?
        .to_dtype(DType::F32)?;

    // seqlens_q: one query per seq -> [0,1,2,...,b]; seqlens_k cumulative kv.
    let mut seqlens_q = vec![0i32];
    for r in 0..b {
        seqlens_q.push(seqlens_q[r] + 1);
    }
    let mut seqlens_k = vec![0i32];
    for bi in 0..b {
        seqlens_k.push(seqlens_k[bi] + kv_lens[bi] as i32);
    }
    let seqlens_q = Tensor::from_vec(seqlens_q, (b + 1,), &dev)?;
    let seqlens_k = Tensor::from_vec(seqlens_k, (b + 1,), &dev)?;
    let bt = Tensor::from_vec(
        block_table
            .iter()
            .flat_map(|r| {
                let mut v = r.clone();
                v.resize(max_blocks, -1);
                v
            })
            .collect::<Vec<i32>>(),
        (b, max_blocks),
        &dev,
    )?;

    let scale = (d as f32).powf(-0.5);
    let out = candle_flash_attn::flash_attn_varlen_paged_windowed_blockmask(
        &q,
        &k_paged,
        &k_paged,
        &seqlens_q,
        &seqlens_k,
        &bt,
        &mask,
        &sink,
        page_block_size,
        scale,
    )?; // [total_q, h, d]

    // Eager reference per row.
    let mut ref_rows: Vec<Tensor> = Vec::new();
    for r in 0..total_q {
        let q_row = q.narrow(0, r, 1)?.squeeze(0)?.contiguous()?; // [h, d]
        let kv_len = kv_lens[r];
        let seq_k = seqs_kv[r].to_dtype(DType::F32)?;
        let seq_v = seqs_kv[r].to_dtype(DType::F32)?;
        let m_row = Tensor::from_vec(
            mask_vec[r * max_kv..r * max_kv + kv_len].to_vec(),
            (kv_len,),
            &dev,
        )?;
        let r_row = eager_row(
            &q_row.to_dtype(DType::F32)?,
            &seq_k,
            &seq_v,
            &m_row,
            &sink,
            scale,
        )?;
        ref_rows.push(r_row);
    }
    let ref_out = Tensor::stack(&ref_rows, 0)?.contiguous()?; // [total_q, h, d]
    Ok((out, ref_out))
}

/// Deterministic weight tensor: `w[i] = sin((i + base) * 0.13) * 0.5`.
fn det_tensor(shape: &[usize], base: f32) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + base) * 0.13).sin() * 0.5)
        .collect();
    Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
}

// ---------------------------------------------------------------------------
// Cases.
// ---------------------------------------------------------------------------

#[test]
fn varlen_paged_decode_parity() -> Result<()> {
    let kv_lens = vec![80usize, 200, 150]; // different lengths, >1 page each
    let (out, ref_out) = run_case(3, &kv_lens, 8, 1, 128, 32, 64)?;
    let diff = max_abs_diff(&out, &ref_out)?;
    assert!(
        diff < 2e-2,
        "varlen/paged decode parity failed: max_abs_diff={diff}"
    );
    Ok(())
}

#[test]
fn varlen_paged_decode_parity_head_dim_512() -> Result<()> {
    // V4-Flash head shape (512) over a small varlen batch.
    let kv_lens = vec![130usize, 300, 260];
    let (out, ref_out) = run_case(3, &kv_lens, 64, 1, 512, 64, 128)?;
    let diff = max_abs_diff(&out, &ref_out)?;
    assert!(
        diff < 2e-2,
        "varlen/paged decode parity (d512) failed: max_abs_diff={diff}"
    );
    Ok(())
}
