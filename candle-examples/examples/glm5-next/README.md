# GLM-5.3-Flash

GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) is a 320B-parameter hybrid
sparse/linear-attention causal LM. This example loads a local BF16 checkpoint
through the `glm5_next` model API and performs cached greedy/sampled
generation with repeat penalty.

## Running the example

Point `--model-dir` at a local directory containing `config.json`,
`tokenizer.json` and the sharded safetensors weights
(`model.safetensors.index.json` plus the shard files, or a single
`model.safetensors` for an unsharded checkpoint):

```bash
$ cargo run --example glm5-next --release -- \
    --model-dir /path/to/GLM-5.3-Flash \
    --prompt "The capital of France is" --sample-len 64

$ cargo run --example glm5-next --release -- \
    --model-dir /path/to/GLM-5.3-Flash \
    --prompt "def fib(n):" --temperature 0.8 --top-p 0.9 --top-k 40 \
    --seed 42 --sample-len 128
```

## Prompt handling

The example uses a **documented raw-prompt path**: the prompt is tokenized
verbatim and no chat template is applied. GLM-5.3-Flash is an instruct model
tuned for its chat template, so for representative output wrap the prompt in
the checkpoint's chat roles (e.g. `<|user|>\n...<|assistant|>`) yourself.

## Options

The example accepts the usual device/dtype/seed/prompt options used by nearby
Candle examples: `--cpu`, `--dtype f32|bf16` (default bf16 on GPU, f32 on CPU),
`--prompt`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--sample-len`,
`--repeat-penalty`, `--repeat-last-n`. With `--temperature 0` (the default) it
performs greedy (ArgMax) decoding; otherwise it samples.

The model directory's `config.json` is read as the official multimodal
checkpoint format, which nests the text config under `text_config`; a reduced
single-model fixture (text config at the top level) is also accepted.

## CI smoke test

`cargo test -p candle-transformers glm5_next` runs a reduced local-fixture
smoke test (`generate_one_token_smoke`) that prefills a short prompt and
samples one next token through the greedy decode step, proving one generation
step without requiring the full 320B checkpoint.
