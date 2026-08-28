#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use candle::{DType, IndexOp, Tensor};
use candle_transformers::models::deepseek_v4::quantized::load_quantized_for_causal_lm;
use candle_transformers::models::deepseek_v4::DeepseekV4Config;
use clap::Parser;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Directory containing config.json, tokenizer.json and the model shards.
    #[arg(long)]
    model_dir: PathBuf,

    /// Use the flash DSA decode path (requires the `flash-attn` feature + CUDA GPU).
    #[arg(long)]
    use_flash_attn: bool,

    /// The prompt to complete (prefill only; logits are dumped).
    #[arg(long, default_value = "The capital of France is")]
    prompt: String,

    /// Resident dequantized-weight cache budget in bytes (offload target).
    #[arg(long, default_value_t = 80_000_000_000)]
    offload_budget_bytes: usize,

    /// fp8/fp4 dequant block.
    #[arg(long, default_value_t = 128)]
    block: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let dir = args.model_dir;
    let config: DeepseekV4Config = {
        let config_file = dir.join("config.json");
        serde_json::from_slice(&std::fs::read(config_file)?)?
    };
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(E::msg)?;

    let device = candle_examples::device(false)?; // GPU (cuda) unless forced cpu by candle_examples
    println!("device: {device:?}");

    // Build the 46 shard paths from the snapshot dir.
    let mut shards: Vec<PathBuf> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .map(|f| {
                    let s = f.to_string_lossy();
                    s.starts_with("model-") && s.ends_with(".safetensors")
                })
                .unwrap_or(false)
        })
        .collect();
    shards.sort();
    println!("found {} shards", shards.len());
    if shards.is_empty() {
        anyhow::bail!("no model shards in {dir:?}");
    }

    println!(
        "config: layers={} hidden={} heads={} head_dim={} vocab={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.head_dim,
        config.vocab_size
    );
    println!(
        "layer_types(len={}): {:?}",
        config.effective_layer_types().len(),
        config.effective_layer_types()[..8.min(config.effective_layer_types().len())].to_vec()
    );

    let block = (args.block, args.block);
    // Real deepseek-ai V4-Flash: flat-name schema (embed.weight / attn.* / ffn.* /
    // hc_*) with `.scale` fp8 scales, per-expert FP4 w1/w2/w3, and MTP layers that
    // candle does not model. `load_quantized_for_causal_lm(.., real_schema = true)`
    // performs the name remap + FP4 expert assembly.
    let scale_suffix = ".scale";
    let fp4_prefixes: Vec<String> = vec![
        "ffn.experts.".into(),
    ];

    let t0 = std::time::Instant::now();
    let mut model = unsafe {
        load_quantized_for_causal_lm(
            &config,
            args.use_flash_attn,
            &shards,
            &device,
            DType::BF16,
            block,
            scale_suffix,
            &fp4_prefixes,
            args.offload_budget_bytes,
            true,
        )?
    };
    let load_time = t0.elapsed();
    println!("LOAD OK in {:?}", load_time);

    let tokens = tokenizer
        .encode(&*args.prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    println!(
        "prompt tokens ({}): {:?}",
        tokens.len(),
        &tokens[..tokens.len().min(16)]
    );
    let prompt = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;

    let t1 = std::time::Instant::now();
    let logits = model.forward(&prompt, 0)?;
    let fwd_time = t1.elapsed();

    let flat = logits.flatten_all()?;
    let f = flat.to_vec1::<f32>()?;
    let finite = f.iter().all(|x| x.is_finite());
    let n_nan = f.iter().filter(|x| x.is_nan()).count();
    let n_inf = f.iter().filter(|x| x.is_infinite()).count();
    let max = f.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = f.iter().cloned().fold(f32::INFINITY, f32::min);
    println!(
        "FORWARD OK in {:?}; logits shape {:?}, len {}, finite={}, nan={}, inf={}, max={:.3}, min={:.3}",
        fwd_time,
        logits.shape(),
        f.len(),
        finite,
        n_nan,
        n_inf,
        max,
        min
    );

    // Top-5 argmax at the last token as a sanity signal.
    let last = logits.i((0, tokens.len() - 1, ..))?.to_vec1::<f32>()?;
    let mut idx: Vec<usize> = (0..last.len()).collect();
    idx.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
    let toks: Vec<_> = idx[..5].to_vec();
    println!("top-5 tokens at last position: {:?}", toks);
    for t in toks {
        if let Ok(s) = tokenizer.decode(&[t as u32], false) {
            println!("  token {t} -> {s:?}");
        }
    }

    Ok(())
}
