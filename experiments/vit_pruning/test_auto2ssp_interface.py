#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
import timm


# Make pruning_srp-main importable (contains mask_conjunction.py with Auto2SSPInterface)
ROOT = Path(__file__).resolve().parents[2]
PRUNING_DIR = ROOT / "pruning_srp-main"
if str(PRUNING_DIR) not in sys.path:
    sys.path.insert(0, str(PRUNING_DIR))

from mask_conjunction import Auto2SSPInterface  # noqa: E402


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = pick_device()
    print(f"[TEST] Using device: {device}")

    # Tiny timm ViT (no downloads when pretrained=False)
    model = timm.create_model("vit_tiny_patch16_224", num_classes=10, pretrained=False)
    model.to(device)
    model.eval()

    # Instantiate interface: no dataloader -> uses heuristic for attention and L1 for FFN width
    iface = Auto2SSPInterface(
        model=model,
        pruning_dataloader=None,
        device=device,
        importance_mode="heuristic",
        batch_limit=3,
        min_remaining=128,
    )
    att_imp, mlp_imp = iface.fit()

    # Report shapes
    print("[TEST] att_importance:", type(att_imp), "shape:", tuple(att_imp.shape) if isinstance(att_imp, torch.Tensor) else None)
    print("[TEST] mlp_importance blocks:", len(mlp_imp) if isinstance(mlp_imp, list) else None)
    if isinstance(mlp_imp, list) and len(mlp_imp) > 0:
        print("[TEST] mlp_importance[0] shape:", tuple(mlp_imp[0].shape))

    # Basic sanity checks against interface spec:
    assert isinstance(att_imp, torch.Tensor), "att_importance must be 1D tensor of length = #blocks"
    assert att_imp.dim() == 1 and att_imp.numel() > 0, "att_importance must be 1D non-empty"
    assert isinstance(mlp_imp, list) and len(mlp_imp) == att_imp.numel(), "mlp_importance must be list per block"
    assert all(isinstance(t, torch.Tensor) and t.dim() == 1 and t.numel() > 0 for t in mlp_imp), "each mlp_importance[b] must be 1D tensor [d_int]"

    print("[TEST] Auto2SSPInterface fit() shapes OK.")


if __name__ == "__main__":
    main()
