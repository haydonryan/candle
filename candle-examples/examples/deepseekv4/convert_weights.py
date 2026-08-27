#!/usr/bin/env python3
"""Convert a HuggingFace DeepSeek-V4 checkpoint to the candle fused-expert format.

The HF `transformers` checkpoint stores routed experts as per-expert tensors
(`mlp.experts.<i>.w1.weight` gate / `.w2.weight` down / `.w3.weight` up) and the
LM head as `head.weight`, whereas the candle `deepseek_v4` model expects the
fused `mlp.experts.gate_up_proj` / `mlp.experts.down_proj` layout and
`lm_head.weight`. This script rewrites those keys in place so
`candle-examples/examples/deepseekv4` (eager + flash) can load the tiny sample
checkpoint for the transformers-parity harness.

Only the expert + lm_head key layouts differ; every other tensor (attention,
norms, DSA compressors/indexer, hyper-connections, MoE shared experts, hash
`tid2eid`, `e_score_correction_bias`) is passed through unchanged.

Usage:
    python3 candle-examples/examples/deepseekv4/convert_weights.py \
        sample_models/deepseek-v4-tiny/model.safetensors
"""

import argparse
import os
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("safetensors", help="path to the HF DeepSeek-V4 safetensors")
    p.add_argument(
        "--out",
        default=None,
        help="output path (default: overwrite the input file)",
    )
    args = p.parse_args()

    src = args.safetensors
    out = args.out or src
    tensors = {}
    with safe_open(src, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # Collect per-expert routed weights: mlp.experts.<e>.w{1,2,3}.weight
    expert_w1, expert_w2, expert_w3 = {}, {}, {}
    for key in list(tensors.keys()):
        parts = key.split(".")
        # model.layers.N.mlp.experts.<e>.w<k>.weight
        if "experts" in parts and len(parts) >= 6 and parts[-1] == "weight":
            if parts[-2].startswith("w") and parts[-2][1:].isdigit():
                e_idx = int(parts[-3])
                w_idx = int(parts[-2][1:])
                bucket = {1: expert_w1, 2: expert_w2, 3: expert_w3}[w_idx]
                prefix = ".".join(parts[:-3])  # .../mlp.experts
                bucket.setdefault(prefix, {})[e_idx] = tensors.pop(key)

    for prefix, w_by_e in expert_w1.items():
        # Reconstruct the parent safetensor name for gate_up_proj / down_proj.
        # gate_up_proj[e] = cat([w1 (gate), w3 (up)], dim=0) -> [2*inter, hidden]
        e = max(w_by_e.keys()) + 1
        w1 = torch.stack([w_by_e[i] for i in range(e)], dim=0)  # [E, inter, hidden]
        w3 = torch.stack([expert_w3[prefix][i] for i in range(e)], dim=0)
        gate_up = torch.cat([w1, w3], dim=1)  # [E, 2*inter, hidden]
        w2 = torch.stack([expert_w2[prefix][i] for i in range(e)], dim=0)  # [E, hidden, inter]
        tensors[f"{prefix}.gate_up_proj"] = gate_up
        tensors[f"{prefix}.down_proj"] = w2

    # Rename the LM head.
    if "head.weight" in tensors:
        tensors["lm_head.weight"] = tensors.pop("head.weight")

    # Atomically write the converted checkpoint.
    tmp = out + ".tmp"
    save_file(tensors, tmp)
    os.replace(tmp, out)
    print(f"converted {src} -> {out} ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
