# DeepSeek V4

DeepSeek V4 is a DeepSeek-MoE-style model with sliding-window MLA plus
Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA)
compressors. This example runs prefill + incremental KV-cache decode with
sampling (`ArgMax`/`TopK`/`TopP`/`TopKThenTopP`) and repeat penalty.

## Running the example

Load a local model directory containing `config.json`, `tokenizer.json` and
the safetensors weights:

```bash
$ cargo run --example deepseekv4 --release -- \
    --model-dir sample_models/deepseek-v4-tiny \
    --prompt "The capital of France is" --sample-len 64

$ cargo run --example deepseekv4 --release -- \
    --model-dir /path/to/model --prompt "Once upon a time" \
    --temperature 0.8 --top-p 0.9 --top-k 40 --seed 42 --sample-len 128
```

The flash DSA decode path (`--use-flash-attn`) requires building with the
`flash-attn` feature on a CUDA GPU; without it the eager decode path is used.
