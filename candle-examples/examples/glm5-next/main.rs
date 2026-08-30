#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::glm5_next::model::Glm5NextForCausalLM;
use candle_transformers::models::glm5_next::Glm5NextTextConfig;

use candle::{DType, Tensor, D};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::io::Write;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Override the compute dtype (default: bf16 on GPU, f32 on CPU).
    #[arg(long, value_parser = ["f32", "bf16"])]
    dtype: Option<String>,

    /// Directory containing config.json, tokenizer.json and the (sharded)
    /// safetensors weights.
    #[arg(long, default_value = "sample_models/glm-5.3-flash-tiny")]
    model_dir: PathBuf,

    /// The raw prompt to complete. The checkpoint chat template is not applied;
    /// the prompt is tokenized verbatim (documented raw-prompt path).
    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 64)]
    sample_len: usize,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

/// Load the text model config from a model directory.
///
/// The official `zai-org/GLM-5.3-Flash` `config.json` nests the text-only
/// config under `text_config` (the full checkpoint is multimodal). When that
/// key is absent the whole file is treated as the text config, so a reduced
/// single-model local fixture also works.
fn load_text_config(dir: &Path) -> Result<Glm5NextTextConfig> {
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let text = value.get("text_config").unwrap_or(&value);
    serde_json::from_value(text.clone()).map_err(E::msg)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let dir = args.model_dir;
    let config = load_text_config(&dir)?;
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(E::msg)?;

    let device = candle_examples::device(args.cpu)?;
    // Load the sharded BF16 safetensors via the index, or a single file when
    // the checkpoint is not sharded.
    let filenames = if dir.join("model.safetensors.index.json").exists() {
        candle_examples::hub_load_local_safetensors(&dir, "model.safetensors.index.json")?
    } else {
        vec![dir.join("model.safetensors")]
    };
    let dtype = match args.dtype.as_deref() {
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        _ => {
            if device.is_cpu() {
                DType::F32
            } else {
                DType::BF16
            }
        }
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let mut model = Glm5NextForCausalLM::new(&config, vb)?;

    // Sampling: ArgMax (temp <= 0) or TopK/TopP/TopKThenTopP/All.
    let temperature = args.temperature.unwrap_or(0.);
    let sampling = if temperature <= 0. {
        Sampling::ArgMax
    } else {
        match (args.top_k, args.top_p) {
            (None, None) => Sampling::All { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, sampling);

    let mut tokenizer = TokenOutputStream::new(tokenizer);
    let mut all = tokenizer
        .tokenizer()
        .encode(&*args.prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    for &t in all.iter() {
        if let Some(t) = tokenizer.next_token(t)? {
            print!("{t}");
        }
    }
    std::io::stdout().flush()?;

    let prompt = Tensor::new(&all[..], &device)?.unsqueeze(0)?;
    let repeat_penalty = args.repeat_penalty;
    let repeat_last_n = args.repeat_last_n;
    let eos = config.eos_token_id.clone().unwrap_or_default();

    // Autoregressive decode over the incremental compressed-KV cache: the first
    // forward prefills the prompt, then one token per step. The GLM-5.3 KDA/DSA
    // caches are stateful and drive decode internally.
    model.clear_kv_cache();
    let mut generated = Vec::with_capacity(args.sample_len);
    let mut pos = 0usize;
    let mut next = prompt;
    for _ in 0..args.sample_len {
        let logits = model.forward(&next, pos)?; // [1, S, vocab]
        let last = logits
            .narrow(D::Minus2, logits.dim(D::Minus2)? - 1, 1)?
            .squeeze(0)?
            .squeeze(0)?; // [vocab]
        let logits = if repeat_penalty == 1. {
            last
        } else {
            let start_at = all.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &last,
                repeat_penalty,
                &all[start_at..],
            )?
        };
        let tok = logits_processor.sample(&logits)?;
        all.push(tok);
        generated.push(tok);
        if eos.contains(&(tok as usize)) {
            break;
        }
        pos += next.dim(D::Minus1)?;
        next = Tensor::new(&[tok], &device)?.unsqueeze(0)?;
    }
    for &t in generated.iter() {
        if let Some(t) = tokenizer.next_token(t)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    std::io::stdout().flush()?;
    println!();
    Ok(())
}
