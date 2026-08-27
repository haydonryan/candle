#!/usr/bin/env python3
"""DeepSeek V4 numerical parity harness: candle (eager + flash) vs transformers.

Compares the candle `deepseekv4` example's logits against the HuggingFace
`transformers` reference on the tiny sample model for a fixed prompt+seed,
within a documented bf16 tolerance. The torch reference is computed by
`transformers` (eager or its `flash_attention_2` path); the candle side is
produced by the `deepseekv4` example in its deterministic `--dump-logits`
prefill mode (eager without `--use-flash-attn`, flash with it).

The comparison is *skippable*: if `transformers`/`torch` are not importable or
the tiny model directory has no weights + tokenizer, the harness prints a
SKIP notice with run instructions and exits 0 so it can be wired into a
skippable cargo test. It does not fabricate numbers it cannot produce.

The candle side needs the weights in candle's fused-expert layout (HF stores
routed experts per-expert as `w1/w2/w3` and the LM head as `head.weight`;
candle expects `mlp.experts.gate_up_proj`/`down_proj` and `lm_head.weight`).
`convert_weights.py` performs that rewrite; the harness applies it
automatically to a scratch copy so the transformers side keeps reading the
original HF checkpoint.

Usage (from the repo root, with a CUDA-enabled candle build):

    # eager candle vs eager transformers
    python3 candle-examples/examples/deepseekv4/parity_harness.py \
        --model-dir sample_models/deepseek-v4-tiny --mode eager

    # flash candle (--use-flash-attn) vs transformers (flash_attention_2 on GPU
    # torch, else the eager reference)
    python3 candle-examples/examples/deepseekv4/parity_harness.py \
        --model-dir sample_models/deepseek-v4-tiny --mode flash

    # both eager and flash
    python3 candle-examples/examples/deepseekv4/parity_harness.py \
        --model-dir sample_models/deepseek-v4-tiny --mode both
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

from safetensors import safe_open


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default="sample_models/deepseek-v4-tiny")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mode",
        choices=["eager", "flash", "both"],
        default="both",
        help="Which candle/transformers path(s) to compare.",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Max absolute logit difference accepted (bf16 round-trip + flash "
        "reassociation). Default 0.1.",
    )
    p.add_argument(
        "--candle-bin",
        default=None,
        help="Path to a prebuilt `deepseekv4` example binary. If omitted, "
        "cargo run --example deepseekv4 is used (requires a candle build).",
    )
    p.add_argument(
        "--cargo-args",
        default="--release --features cuda,flash-attn",
        help="Extra cargo args for the candle build, e.g. "
        "'--release --features cuda,flash-attn'.",
    )
    return p.parse_args()


def log(msg):
    print(f"[parity] {msg}", flush=True)


def skip(reason, instructions):
    log(f"SKIP: {reason}")
    log("To run the candle-vs-transformers comparison, install `transformers` "
        "+ GPU `torch` and provide the tiny model weights + tokenizer under the "
        "model dir.")
    log(instructions)
    sys.exit(0)


def check_environment(args):
    """Skip cleanly when torch/transformers or the tiny checkpoint are missing."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        skip(
            f"torch/transformers not importable ({e})",
            "Install: pip install 'transformers[torch]' (GPU build of torch "
            "recommended for the flash path).",
        )

    cfg = os.path.join(args.model_dir, "config.json")
    tok = os.path.join(args.model_dir, "tokenizer.json")
    weights = os.path.join(args.model_dir, "model.safetensors")
    index = os.path.join(args.model_dir, "model.safetensors.index.json")
    if not os.path.isfile(cfg):
        skip(f"model dir has no config.json: {args.model_dir}", "")
    if not os.path.isfile(tok):
        skip(
            "model dir has no tokenizer.json (required for a tokenized prompt)",
            f"Place tokenizer.json under {args.model_dir} (e.g. from the HF "
            "DeepSeek-V4 tokenizer).",
        )
    if not (os.path.isfile(weights) or os.path.isfile(index)):
        skip(
            "model dir has no safetensors weights",
            f"Place the HF DeepSeek-V4 safetensors under {args.model_dir}, or "
            "download via: huggingface-cli download <hf-id> "
            f"--local-dir {args.model_dir}",
        )


def torch_ref_logits(args, mode, out_json):
    """Compute transformers reference logits `[1, S, vocab]` (flattened f32)
    for the fixed prompt and dump to `out_json`."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    # `flash_attention_2` requires a CUDA torch build; on CPU-only torch we fall
    # back to the eager attention implementation (still the authoritative
    # reference) so the candle flash DSA kernel is compared against it.
    want_flash = mode == "flash" and torch.cuda.is_available()
    if mode == "flash" and not torch.cuda.is_available():
        log("transformers flash_attention_2 needs a CUDA torch build; using the "
            "eager reference for the candle-flash comparison")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        config=config,
        attn_implementation="flash_attention_2" if want_flash else "eager",
    )
    # Force every parameter (including small norms/bias tensors that load as
    # fp32) to bf16 so the forward pass is dtype-consistent and comparable to
    # the candle bf16 path.
    model.to(torch.bfloat16)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=ids)  # logits [1, S, vocab]
    logits = out.logits.detach().float().cpu().reshape(-1).tolist()
    with open(out_json, "w") as f:
        json.dump(logits, f)
    log(f"transformers {mode}: wrote {len(logits)} reference logits -> {out_json}")
    return len(logits)


def prepare_candle_dir(args, tmp):
    """Build a candle-loadable model dir in `tmp`: config + tokenizer + the
    safetensors converted to candle's fused-expert layout (HF stores routed
    experts per-expert as w1/w2/w3 and the LM head as `head.weight`, while
    candle expects `gate_up_proj`/`down_proj` and `lm_head.weight`). The
    transformers side keeps reading the original HF checkpoint."""
    candle_dir = os.path.join(tmp, "candle_model")
    os.makedirs(candle_dir, exist_ok=True)
    for fn in ("config.json", "tokenizer.json"):
        shutil.copy(os.path.join(args.model_dir, fn), os.path.join(candle_dir, fn))
    src_weights = os.path.join(args.model_dir, "model.safetensors")
    out = os.path.join(candle_dir, "model.safetensors")
    if os.path.isfile(src_weights):
        with safe_open(src_weights, framework="pt", device="cpu") as f:
            already_fused = any("gate_up_proj" in k for k in f.keys())
        if already_fused:
            shutil.copy(src_weights, out)
        else:
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "convert_weights.py"),
                    src_weights,
                    "--out",
                    out,
                ],
                check=True,
            )
    return candle_dir


def candle_logits(args, use_flash, out_json, tmp):
    """Run the deepseekv4 example prefill dump for eager/flash and return the
    logit count. Uses a converted candle-loadable model dir so the fused-expert
    layout matches what the candle model expects."""
    candle_dir = prepare_candle_dir(args, tmp)
    extra = ["--use-flash-attn"] if use_flash else []
    if args.candle_bin:
        cmd = [args.candle_bin, "--model-dir", candle_dir, "--prompt", args.prompt]
    else:
        cargo_args = args.cargo_args.split() if args.cargo_args else []
        cmd = [
            "cargo",
            "run",
            "--example",
            "deepseekv4",
            *cargo_args,
            "--",
            "--model-dir",
            candle_dir,
            "--prompt",
            args.prompt,
        ]
    cmd += extra + ["--dump-logits", out_json]
    log("running candle %s: %s" % ("flash" if use_flash else "eager", " ".join(cmd)))
    subprocess.run(cmd, check=True)
    with open(out_json) as f:
        return len(json.load(f))


def compare(candle_json, ref_json, mode, tolerance):
    with open(candle_json) as f:
        candle = json.load(f)
    with open(ref_json) as f:
        ref = json.load(f)
    if len(candle) != len(ref):
        log(
            f"FAIL({mode}): logit count mismatch candle={len(candle)} "
            f"ref={len(ref)}"
        )
        return False
    max_diff = 0.0
    n_exceed = 0
    for a, b in zip(candle, ref):
        d = abs(a - b)
        if d > max_diff:
            max_diff = d
        if d > tolerance:
            n_exceed += 1
    ok = n_exceed == 0
    status = "PASS" if ok else "FAIL"
    log(
        f"{status}({mode}): max_abs_diff={max_diff:.5f} tolerance={tolerance} "
        f"exceeding={n_exceed}/{len(candle)}"
    )
    return ok


def main():
    args = parse_args()
    check_environment(args)

    modes = ["eager", "flash"] if args.mode == "both" else [args.mode]
    all_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        # torch reference (recompute per mode so the eager vs flash reference
        # reflects the requested attention impl).
        for mode in modes:
            ref_json = os.path.join(tmp, f"ref_{mode}.json")
            n_ref = torch_ref_logits(args, mode, ref_json)
            candle_json = os.path.join(tmp, f"candle_{mode}.json")
            use_flash = mode == "flash"
            n_candle = candle_logits(args, use_flash, candle_json, tmp)
            if n_ref != n_candle:
                log(f"FAIL({mode}): shape mismatch ref={n_ref} candle={n_candle}")
                all_ok = False
                continue
            if not compare(candle_json, ref_json, mode, args.tolerance):
                all_ok = False
    log("parity harness " + ("PASSED" if all_ok else "FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
