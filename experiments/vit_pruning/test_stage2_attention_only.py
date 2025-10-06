#!/usr/bin/env python3
"""
Smoke test to verify Stage-2 now removes ONLY attention submodules (keeps FFN/MLP intact).
- Builds a tiny ViT from config (no downloads).
- Records per-block attention and MLP parameter counts.
- Runs prune_vit_attention_blocks with a fixed num_to_prune.
- Verifies:
  * encoder.layer length is unchanged
  * attention params for pruned indices drop to zero
  * MLP params remain unchanged for all blocks
  * forward pass still works
"""

import sys
from pathlib import Path

# Ensure project root (with src/) on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification

from src.vit_pruning import prune_vit_attention_blocks, _get_encoder, _count_attention_params_per_block


def count_mlp_params_per_block(model: nn.Module):
    enc = _get_encoder(model)
    counts = []
    for layer in enc.layer:
        inter = layer.intermediate.dense
        out = layer.output.dense
        cnt = sum(p.numel() for p in inter.parameters()) + sum(p.numel() for p in out.parameters())
        counts.append(cnt)
    return counts


def main():
    torch.manual_seed(0)

    # Small ViT from config (no pretrained weights required)
    cfg = ViTConfig(
        image_size=32,
        patch_size=16,
        num_channels=3,
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        num_hidden_layers=4,
        num_labels=10,
    )
    model = ViTForImageClassification(cfg)

    # Baseline counts
    enc = _get_encoder(model)
    B = len(enc.layer)
    attn_before = _count_attention_params_per_block(model)
    mlp_before = count_mlp_params_per_block(model)

    print("Blocks:", B)
    print("Attention params per block (before):", attn_before)
    print("MLP params per block (before):", mlp_before)

    # Prune attention in exactly 2 blocks using heuristic selection (fast)
    res = prune_vit_attention_blocks(
        model,
        sparsity=0.5,
        dataloader=None,
        device="cpu",
        importance_mode="heuristic",
        show_progress=False,
        num_to_prune=2,
    )
    pruned_indices = sorted(res["pruned_indices"])
    print("Pruned attention indices:", pruned_indices)

    # After counts
    enc_after = _get_encoder(model)
    B_after = len(enc_after.layer)
    attn_after = _count_attention_params_per_block(model)
    mlp_after = count_mlp_params_per_block(model)

    print("Blocks after:", B_after)
    print("Attention params per block (after):", attn_after)
    print("MLP params per block (after):", mlp_after)

    # Assertions
    assert B_after == B, "Encoder depth changed; entire blocks were pruned instead of attention-only."
    for i in range(B):
        if i in pruned_indices:
            assert attn_after[i] == 0, f"Attention params for pruned block {i} should be 0."
        else:
            assert attn_after[i] == attn_before[i], f"Attention params for kept block {i} should be unchanged."

        assert mlp_after[i] == mlp_before[i], f"MLP params for block {i} should be unchanged."

    # Forward pass still works
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 32, 32)
        out = model(pixel_values=x)
        assert out.logits.shape[-1] == cfg.num_labels, "Forward logits shape mismatch."

    print("All checks passed: Stage-2 attention-only pruning works as expected.")


if __name__ == "__main__":
    main()
