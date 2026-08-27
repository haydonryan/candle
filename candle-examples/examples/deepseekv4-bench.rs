//! DeepSeek V4 flash-vs-eager attention benchmark (prefill + decode).
//!
//! Benchmarks `DeepseekV4Attention` on a Compressed-Sparse-Attention (CSA)
//! layer (the case where the custom DSA flash kernel replaces eager
//! block-sparse attention) at two shapes: head_dim 128 / 8 heads
//! (tiny-model shape) and head_dim 512 / 64 heads (V4-Flash shape). It measures
//! prefill (single large forward) and decode (per-step Sq=1 through the
//! incremental compressed-KV cache) wall time for eager vs flash, the
//! flash/eager speedup, and peak GPU memory (sampled via nvidia-smi).
//!
//! Build/run with the flash feature on a CUDA GPU:
//!
//! ```text
//! cargo run --release --features cuda,flash-attn --example deepseekv4-bench
//! ```

use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::deepseek_v4::{DeepseekV4Attention, DeepseekV4Config, LayerType};
use std::collections::HashMap;
use std::time::Instant;

fn det_tensor(shape: &[usize], base: f32, dev: &Device) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n)
        .map(|i| ((i as f32 + base) * 0.13).sin() * 0.5)
        .collect();
    Tensor::from_vec(v, shape, dev).unwrap()
}

/// Peak GPU memory used (sampled via nvidia-smi); the GPU is otherwise idle so
/// the delta vs idle (~2 MiB) approximates the benchmark's allocation.
fn gpu_mem_mb() -> u64 {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout)
            .trim()
            .split('\n')
            .next()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0),
        Err(_) => 0,
    }
}

/// Deterministic CSA attention weights under `DeepseekV4Attention::new`'s
/// namespace (`compressor.*` included), matching the shapes the model expects.
fn csa_weights(cfg: &DeepseekV4Config, dev: &Device) -> HashMap<String, Tensor> {
    let h = cfg.num_attention_heads;
    let d = cfg.head_dim;
    let hd = cfg.head_dim;
    let ihd = cfg.index_head_dim;
    let rate = cfg.compress_rates.compressed_sparse_attention;
    let mut m = HashMap::new();
    m.insert(
        "q_a_proj.weight".into(),
        det_tensor(&[cfg.q_lora_rank, cfg.hidden_size], 1.0, dev),
    );
    m.insert(
        "q_a_norm.weight".into(),
        det_tensor(&[cfg.q_lora_rank], 2.0, dev),
    );
    m.insert(
        "q_b_proj.weight".into(),
        det_tensor(&[h * d, cfg.q_lora_rank], 3.0, dev),
    );
    m.insert(
        "kv_proj.weight".into(),
        det_tensor(&[d, cfg.hidden_size], 4.0, dev),
    );
    m.insert("kv_norm.weight".into(), det_tensor(&[d], 5.0, dev));
    m.insert(
        "o_a_proj.weight".into(),
        det_tensor(
            &[cfg.o_groups * cfg.o_lora_rank, h * d / cfg.o_groups],
            6.0,
            dev,
        ),
    );
    m.insert(
        "o_b_proj.weight".into(),
        det_tensor(&[cfg.hidden_size, cfg.o_groups * cfg.o_lora_rank], 7.0, dev),
    );
    m.insert("sinks".into(), det_tensor(&[h], 8.0, dev));
    m.insert(
        "compressor.kv_proj.weight".into(),
        det_tensor(&[2 * hd, cfg.hidden_size], 21.0, dev),
    );
    m.insert(
        "compressor.gate_proj.weight".into(),
        det_tensor(&[2 * hd, cfg.hidden_size], 22.0, dev),
    );
    m.insert(
        "compressor.position_bias".into(),
        det_tensor(&[rate, 2 * hd], 23.0, dev),
    );
    m.insert(
        "compressor.kv_norm.weight".into(),
        det_tensor(&[hd], 24.0, dev),
    );
    m.insert(
        "compressor.indexer.kv_proj.weight".into(),
        det_tensor(&[2 * ihd, cfg.hidden_size], 31.0, dev),
    );
    m.insert(
        "compressor.indexer.gate_proj.weight".into(),
        det_tensor(&[2 * ihd, cfg.hidden_size], 32.0, dev),
    );
    m.insert(
        "compressor.indexer.position_bias".into(),
        det_tensor(&[rate, 2 * ihd], 33.0, dev),
    );
    m.insert(
        "compressor.indexer.kv_norm.weight".into(),
        det_tensor(&[ihd], 34.0, dev),
    );
    m.insert(
        "compressor.indexer.q_b_proj.weight".into(),
        det_tensor(&[cfg.index_n_heads * ihd, cfg.q_lora_rank], 35.0, dev),
    );
    m.insert(
        "compressor.indexer.scorer.weights_proj.weight".into(),
        det_tensor(&[cfg.index_n_heads, cfg.hidden_size], 36.0, dev),
    );
    m
}

/// Load the tiny config (or a V4-Flash variant), CSA-only, hash layers off.
fn csa_config(head_dim: usize, num_heads: usize) -> DeepseekV4Config {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../sample_models/deepseek-v4-tiny/config.json");
    let raw = std::fs::read_to_string(&path).expect("tiny config.json");
    let mut cfg: DeepseekV4Config = serde_json::from_str(&raw).unwrap();
    cfg.num_attention_heads = num_heads;
    cfg.head_dim = head_dim;
    cfg.q_lora_rank = head_dim;
    cfg.o_lora_rank = head_dim;
    cfg.num_hash_layers = 0;
    cfg.layer_types = vec![LayerType::CompressedSparseAttention];
    cfg.compress_ratios = vec![cfg.compress_rates.compressed_sparse_attention];
    cfg.num_hidden_layers = 1;
    cfg
}

fn build_pair(cfg: &DeepseekV4Config, dev: &Device) -> (DeepseekV4Attention, DeepseekV4Attention) {
    let weights = csa_weights(cfg, dev);
    let bf16: HashMap<String, Tensor> = weights
        .into_iter()
        .map(|(k, v)| (k, v.to_dtype(DType::BF16).unwrap()))
        .collect();
    let eager = DeepseekV4Attention::new(
        cfg,
        0,
        false,
        VarBuilder::from_tensors(bf16.clone(), DType::BF16, dev),
    )
    .unwrap();
    let flash = DeepseekV4Attention::new(
        cfg,
        0,
        true,
        VarBuilder::from_tensors(bf16, DType::BF16, dev),
    )
    .unwrap();
    (eager, flash)
}

fn time_prefill(dev: &Device, cfg: &DeepseekV4Config, label: &str, seq_len: usize, iters: usize) {
    let (mut eager, mut flash) = build_pair(cfg, dev);
    let x = det_tensor(&[1, seq_len, cfg.hidden_size], 51.0, dev)
        .to_dtype(DType::BF16)
        .unwrap();

    // Warm-up.
    let _ = eager.forward(&x, 0, None).unwrap();
    let _ = flash.forward(&x, 0, None).unwrap();
    dev.synchronize().unwrap();

    let t_eager = {
        let start = Instant::now();
        for _ in 0..iters {
            let _ = eager.forward(&x, 0, None).unwrap();
        }
        dev.synchronize().unwrap();
        start.elapsed().as_secs_f64() / iters as f64
    };
    let t_flash = {
        let start = Instant::now();
        for _ in 0..iters {
            let _ = flash.forward(&x, 0, None).unwrap();
        }
        dev.synchronize().unwrap();
        start.elapsed().as_secs_f64() / iters as f64
    };
    let mem = gpu_mem_mb();
    println!(
        "[{label}] prefill seq_len={seq_len}: eager {:.3} ms  flash {:.3} ms  speedup {:.2}x  gpu_mem ~{mem} MiB",
        t_eager * 1e3,
        t_flash * 1e3,
        t_eager / t_flash
    );
}

fn time_decode(dev: &Device, cfg: &DeepseekV4Config, label: &str, n_steps: usize) {
    let (mut eager, mut flash) = build_pair(cfg, dev);

    // Warm-up one step each.
    let _ = eager
        .forward(
            &det_tensor(&[1, 1, cfg.hidden_size], 61.0, dev)
                .to_dtype(DType::BF16)
                .unwrap(),
            0,
            None,
        )
        .unwrap();
    let _ = flash
        .forward(
            &det_tensor(&[1, 1, cfg.hidden_size], 61.0, dev)
                .to_dtype(DType::BF16)
                .unwrap(),
            0,
            None,
        )
        .unwrap();
    dev.synchronize().unwrap();

    let t_eager = {
        eager.clear_kv_cache();
        let start = Instant::now();
        for step in 0..n_steps {
            let x = det_tensor(&[1, 1, cfg.hidden_size], (61 + step) as f32, dev)
                .to_dtype(DType::BF16)
                .unwrap();
            let _ = eager.forward(&x, step, None).unwrap();
        }
        dev.synchronize().unwrap();
        start.elapsed().as_secs_f64() / n_steps as f64
    };
    let t_flash = {
        flash.clear_kv_cache();
        let start = Instant::now();
        for step in 0..n_steps {
            let x = det_tensor(&[1, 1, cfg.hidden_size], (61 + step) as f32, dev)
                .to_dtype(DType::BF16)
                .unwrap();
            let _ = flash.forward(&x, step, None).unwrap();
        }
        dev.synchronize().unwrap();
        start.elapsed().as_secs_f64() / n_steps as f64
    };
    let mem = gpu_mem_mb();
    println!(
        "[{label}] decode {n_steps} steps: eager {:.3} ms/step  flash {:.3} ms/step  speedup {:.2}x  gpu_mem ~{mem} MiB",
        t_eager * 1e3,
        t_flash * 1e3,
        t_eager / t_flash
    );
}

fn main() -> candle::Result<()> {
    let dev = Device::new_cuda(0)?;
    println!("GPU: {dev:?}");

    // head_dim 128, 8 heads (tiny-model shape).
    let cfg128 = csa_config(128, 8);
    println!("== head_dim 128, 8 heads (tiny shape) ==");
    time_prefill(&dev, &cfg128, "hd128", 512, 5);
    time_decode(&dev, &cfg128, "hd128", 64);

    // head_dim 512, 64 heads (V4-Flash shape).
    let cfg512 = csa_config(512, 64);
    println!("== head_dim 512, 64 heads (V4-Flash shape) ==");
    time_prefill(&dev, &cfg512, "hd512", 512, 5);
    time_decode(&dev, &cfg512, "hd512", 64);
    Ok(())
}
