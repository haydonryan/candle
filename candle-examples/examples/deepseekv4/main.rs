#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle_transformers::models::deepseek_v4::{DeepseekV4Config, DeepseekV4ForCausalLM};

use candle::{DType, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Use the flash DSA decode path (requires the `flash-attn` feature and a CUDA GPU).
    #[arg(long)]
    use_flash_attn: bool,

    /// Directory containing config.json, tokenizer.json and the model weights.
    #[arg(long, default_value = "sample_models/deepseek-v4-tiny")]
    model_dir: PathBuf,

    /// The prompt to complete.
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

fn main() -> Result<()> {
    let args = Args::parse();

    let dir = args.model_dir;
    let config: DeepseekV4Config = {
        let config_file = dir.join("config.json");
        serde_json::from_slice(&std::fs::read(config_file)?)?
    };
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(E::msg)?;

    let device = candle_examples::device(args.cpu)?;
    let filenames = if dir.join("model.safetensors.index.json").exists() {
        candle_examples::hub_load_local_safetensors(&dir, "model.safetensors.index.json")?
    } else {
        vec![dir.join("model.safetensors")]
    };
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let mut model = DeepseekV4ForCausalLM::new(&config, args.use_flash_attn, vb)?;

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
    let eos = config.eos_token_id;

    let generated = model.generate(&prompt, args.sample_len, eos, |logits| {
        let logits = if repeat_penalty == 1. {
            logits.clone()
        } else {
            let start_at = all.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                logits,
                repeat_penalty,
                &all[start_at..],
            )?
        };
        let tok = logits_processor.sample(&logits)?;
        all.push(tok);
        Ok(tok)
    })?;

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
