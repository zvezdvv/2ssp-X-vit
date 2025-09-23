import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
import copy
import json
import time
import os
from dataclasses import dataclass

__all__ = [
    "prune_vit_mlp_width",
    "evaluate_top1",
    "prune_vit_attention_blocks",
    "plan_2ssp_allocation",
    "count_total_params",
    "count_block_params",
    "compute_actual_sparsity",
    "save_cifar_adapter",
    "load_cifar_adapter",
    "save_report",
]

# =========================
# Helpers for model anatomy
# =========================

@torch.no_grad()
def _get_encoder(vit_model):
    """Get encoder module from different ViT model variants.

    Supports huggingface ViTModel / ViTForImageClassification structures.
    """
    if hasattr(vit_model, "vit"):
        # ViTForImageClassification has a vit attribute
        base = vit_model.vit
    elif hasattr(vit_model, "base_model"):
        # Some variants have base_model
        base = vit_model.base_model
    else:
        # Otherwise, assume model is already encoder-like
        base = vit_model

    # Get encoder from base
    encoder = base.encoder if hasattr(base, "encoder") else base
    return encoder

@torch.no_grad()
def _gather_mlp_pairs(vit_model) -> List[Tuple[nn.Linear, nn.Linear]]:
    """Return (intermediate_dense, output_dense) linear layer pairs for each ViT encoder block.

    Supports huggingface ViTModel / ViTForImageClassification structures.
    """
    mlp_pairs = []
    encoder = _get_encoder(vit_model)
    for layer in encoder.layer:  # type: ignore[attr-defined]
        inter_dense = layer.intermediate.dense
        out_dense = layer.output.dense
        mlp_pairs.append((inter_dense, out_dense))
    return mlp_pairs

@torch.no_grad()
def _get_hidden_and_inter_sizes(vit_model) -> Tuple[int, List[int]]:
    """Get hidden size and list of intermediate sizes for MLPs across blocks."""
    pairs = _gather_mlp_pairs(vit_model)
    hidden = pairs[0][0].weight.size(1) if pairs else getattr(vit_model.config, "hidden_size", None)
    inter_sizes = [inter.weight.size(0) for inter, _ in pairs]
    return hidden, inter_sizes

# =========================
# Parameter accounting
# =========================

@torch.no_grad()
def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def count_block_params(model: nn.Module) -> List[int]:
    """Return per-block parameter counts for encoder.layer[i]."""
    encoder = _get_encoder(model)
    counts = []
    for layer in encoder.layer:  # type: ignore[attr-defined]
        counts.append(sum(p.numel() for p in layer.parameters()))
    return counts

@torch.no_grad()
def compute_actual_sparsity(before_params: int, after_params: int) -> float:
    """Return achieved global sparsity fraction given parameter counts."""
    if before_params <= 0:
        return 0.0
    return (before_params - after_params) / before_params

# =========================
# Stage-1: Width pruning
# =========================

@torch.no_grad()
def prune_vit_mlp_width(
    vit_model,
    sparsity: Optional[float] = None,
    strategy: str = "l1",
    min_remaining: int = 256,
    n_to_prune_per_block: Optional[List[int]] = None,
):
    """Width pruning of MLP intermediate dimension across all ViT blocks.

    Either provide `sparsity` (fraction per block) OR `n_to_prune_per_block` (absolute, per block).
    If both provided, `n_to_prune_per_block` takes precedence.

    Args:
        vit_model: HuggingFace ViT (ViTModel or ViTForImageClassification)
        sparsity: fraction (0-1) of intermediate neurons to remove (uniform across blocks).
        strategy: currently only 'l1'
        min_remaining: lower bound for remaining neurons after pruning each block.
        n_to_prune_per_block: list with number of neurons to remove per block (len == #blocks).
    """
    mlp_pairs = _gather_mlp_pairs(vit_model)

    if n_to_prune_per_block is not None:
        if len(n_to_prune_per_block) != len(mlp_pairs):
            raise ValueError("n_to_prune_per_block length must match number of blocks")
    else:
        if sparsity is None:
            raise ValueError("Provide either sparsity or n_to_prune_per_block")
        if not (0.0 <= sparsity < 1.0):
            raise AssertionError("sparsity must be in [0,1)")

    for block_idx, (inter_dense, out_dense) in enumerate(mlp_pairs):
        W_int: torch.Tensor = inter_dense.weight  # [intermediate, hidden]
        B_int: torch.Tensor = inter_dense.bias    # [intermediate]
        W_out: torch.Tensor = out_dense.weight    # [hidden, intermediate]

        if strategy == "l1":
            importance = W_int.abs().sum(dim=1)
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        n_channels = W_int.size(0)
        if n_to_prune_per_block is not None:
            n_prune = int(n_to_prune_per_block[block_idx])
        else:
            n_prune = int(n_channels * sparsity)

        # respect min_remaining
        if n_channels - n_prune < min_remaining:
            n_prune = max(0, n_channels - min_remaining)
        if n_prune <= 0:
            continue

        keep_idx = torch.argsort(importance, descending=True)[: n_channels - n_prune]
        keep_idx, _ = torch.sort(keep_idx)

        new_W_int = W_int[keep_idx].clone()
        new_B_int = B_int[keep_idx].clone() if B_int is not None else None
        new_W_out = W_out[:, keep_idx].clone()

        hidden = W_int.size(1)
        new_intermediate = new_W_int.size(0)

        inter_dense.weight = nn.Parameter(new_W_int)
        if new_B_int is not None:
            inter_dense.bias = nn.Parameter(new_B_int)
        inter_dense.out_features = new_intermediate
        inter_dense.in_features = hidden

        out_dense.weight = nn.Parameter(new_W_out)
        out_dense.in_features = new_intermediate

    return vit_model

# =========================
# Evaluation
# =========================

@torch.no_grad()
def evaluate_top1(model, dataloader, device: str = "cuda", max_batches: int | None = None, progress: bool = False):
    """Compute top-1 accuracy.

    Args:
        model: classification model with .logits output
        dataloader: iterable yielding dict with pixel_values, labels
        device: target device
        max_batches: limit number of batches (for quick estimation)
        progress: show tqdm progress bar
    """
    model.eval()
    correct = 0
    total = 0
    autocast_device = device if (str(device).startswith("cuda") or str(device).startswith("mps")) else "cpu"
    iterator = dataloader
    if progress:
        try:
            from tqdm.auto import tqdm  # lazy import
            iterator = tqdm(dataloader, total=(max_batches if max_batches is not None else None), desc="eval")
        except Exception:
            pass
    for i, batch in enumerate(iterator):
        if max_batches is not None and i >= max_batches:
            break
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=autocast_device, enabled=True):
            out = model(pixel_values=pixel_values)
            logits = out.logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(1, total)

# =========================
# Stage-2: Depth pruning
# =========================

@torch.no_grad()
def prune_vit_attention_blocks(
    vit_model,
    sparsity: float,
    dataloader=None,
    device: str = "cuda",
    batch_limit: int = 5,
    metric_fn=None,
    importance_mode: str = "copy",
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Remove entire transformer blocks (Depth Pruning, Stage-2 of 2SSP).

    Note: Although initially drafted as "attention" block removal, this function removes whole
    encoder.layer blocks to ensure structural consistency and speedup.

    Args:
        vit_model: HuggingFace ViT model (ViTModel or ViTForImageClassification)
        sparsity: Fraction (0-1) of blocks to remove
        dataloader: Optional dataloader for metric-based selection
        device: Device for evaluation
        batch_limit: Max batches to process for evaluation
        metric_fn: Custom metric function to evaluate block importance (unused, reserved)

    Returns:
        Dict with pruned model and pruning information
    """
    assert 0.0 <= sparsity < 1.0, "sparsity must be in [0,1)"

    # Create a copy of the model for calibration
    vit_model.eval()
    model_copy = copy.deepcopy(vit_model)
    model_copy.eval()

    # Get encoder
    encoder = _get_encoder(vit_model)
    encoder_copy = _get_encoder(model_copy)

    num_blocks = len(encoder.layer)
    num_to_prune = max(0, min(num_blocks - 1, int(round(num_blocks * sparsity))))

    if num_to_prune == 0:
        print("No encoder blocks to prune based on sparsity.")
        return {"model": vit_model, "pruned_indices": [], "original_metrics": None, "final_metrics": None}

    # If no dataloader provided, use simple position heuristic
    # If fast heuristic requested or no dataloader, use heuristic regardless
    if dataloader is None or (isinstance(importance_mode, str) and importance_mode.lower() == "heuristic"):
        print("Using heuristic for depth pruning importance (position-based).")
        # Heuristic: middle blocks tend to be less critical
        importance_scores = [(i if i < num_blocks / 2 else num_blocks - i) for i in range(num_blocks)]
        to_prune = sorted(range(num_blocks), key=lambda i: importance_scores[i])[:num_to_prune]
        original_metrics = None
        final_metrics = None
    else:
        print(f"Evaluating {num_blocks} blocks by impact on accuracy (copy-remove)...")
        original_metrics = evaluate_top1(vit_model, dataloader, device, max_batches=batch_limit, progress=True)
        print(f"Baseline accuracy: {original_metrics:.4f}")

        impact_scores = []

        # Evaluate impact by actually removing one block at a time on a copy,
        # to avoid forward-signature/shape mismatches when patching.
        try:
            from tqdm.auto import tqdm as _tqdm
        except Exception:
            _tqdm = None
        iterable = range(num_blocks)
        if show_progress and _tqdm is not None:
            iterable = _tqdm(range(num_blocks), desc="Depth eval (copy-remove)", leave=False)

        for block_idx in iterable:
            try:
                model_copy = copy.deepcopy(vit_model)
                model_copy.eval()
                enc_copy = _get_encoder(model_copy)

                keep_indices = [i for i in range(num_blocks) if i != block_idx]
                new_layers = nn.ModuleList([copy.deepcopy(enc_copy.layer[i]) for i in keep_indices])
                enc_copy.layer = new_layers
                if hasattr(model_copy, "config"):
                    model_copy.config.num_hidden_layers = len(new_layers)

                score = evaluate_top1(model_copy, dataloader, device, max_batches=batch_limit, progress=False)
                impact = max(0.0, original_metrics - score)
                impact_scores.append(impact)
                if show_progress:
                    print(f"[Depth] Block {block_idx} impact: {impact:.4f}", flush=True)
            except Exception as e:
                print(f"Error evaluating block {block_idx}: {e}")
                impact_scores.append(0.0)
            finally:
                try:
                    del model_copy, enc_copy
                except Exception:
                    pass

        # select lowest impact blocks
        to_prune = sorted(range(num_blocks), key=lambda i: impact_scores[i])[:num_to_prune]
        print(f"Selected blocks to prune: {to_prune}")

    # Perform actual pruning on original model
    keep_indices = [i for i in range(num_blocks) if i not in to_prune]
    new_layers = nn.ModuleList([copy.deepcopy(encoder.layer[i]) for i in keep_indices])
    encoder.layer = new_layers

    # Update config if available
    if hasattr(vit_model, "config"):
        vit_model.config.num_hidden_layers = len(new_layers)

    if dataloader is not None:
        final_metrics = evaluate_top1(vit_model, dataloader, device, max_batches=batch_limit, progress=True)
        print(f"Final accuracy after pruning: {final_metrics:.4f}")
        if original_metrics is not None:
            print(f"Accuracy change: {final_metrics - original_metrics:.4f}")
    else:
        final_metrics = None

    return {
        "model": vit_model,
        "pruned_indices": sorted(list(to_prune)),
        "original_metrics": original_metrics,
        "final_metrics": final_metrics,
    }

# ==================================
# Auto-allocation for single sparsity
# ==================================

@dataclass
class TwoSSPPlan:
    target_sparsity: float
    num_blocks_total: int
    blocks_to_prune: int
    per_block_neurons_to_prune: int
    stage2_fraction: float
    estimated_total_removed_params: int
    est_error_params: int

@torch.no_grad()
def _estimate_width_removal_per_block(hidden: int, t_remove: int) -> int:
    """Estimate parameter reduction per block when removing t intermediate neurons.

    Removing one neuron removes one row from inter_dense: hidden weights + 1 bias,
    and one column from out_dense: hidden weights. Total approx = t * (2*hidden + 1).
    """
    if t_remove <= 0:
        return 0
    return t_remove * (2 * hidden + 1)

@torch.no_grad()
def plan_2ssp_allocation(
    vit_model,
    target_sparsity: float,
    min_remaining: int = 256,
) -> TwoSSPPlan:
    """Plan Stage-1 (width) and Stage-2 (depth) amounts given a single TARGET_SPARSITY.

    Strategy:
    - Try K in [0..B-1] removed blocks (depth).
    - For each K, compute remaining budget in params to reach target.
    - Convert remaining budget to per-block neurons to prune uniformly (width).
    - Respect min_remaining and per-block available neurons.
    - Pick K,t minimizing absolute error in removed params.

    Returns:
        TwoSSPPlan with chosen K and t and stage2 fraction.
    """
    assert 0.0 < target_sparsity < 1.0, "target_sparsity must be in (0,1)"

    total_params = count_total_params(vit_model)
    block_params = count_block_params(vit_model)
    B = len(block_params)
    P_target = int(round(total_params * target_sparsity))

    hidden, inter_sizes = _get_hidden_and_inter_sizes(vit_model)
    if hidden is None or len(inter_sizes) != B:
        raise RuntimeError("Unable to determine hidden/intermediate sizes for planning.")

    # maximum neurons removable per block respecting min_remaining
    max_removable_per_block = [max(0, inter - min_remaining) for inter in inter_sizes]
    t_max_uniform = min(max_removable_per_block) if max_removable_per_block else 0

    best = None

    P_block_mean = sum(block_params) / max(1, B)

    for K in range(0, max(0, B - 1) + 1):
        # Removed by depth:
        P_removed_depth = int(round(K * P_block_mean))

        # Remaining to remove by width:
        P_remaining = max(0, P_target - P_removed_depth)

        # Convert to uniform t:
        denom = B * (2 * hidden + 1)
        t = int(round(P_remaining / denom)) if denom > 0 else 0
        t = max(0, min(t, t_max_uniform))

        # Estimate total removed:
        P_removed_width = _estimate_width_removal_per_block(hidden, t) * B
        P_removed_total = P_removed_depth + P_removed_width

        err = abs(P_target - P_removed_total)

        cand = (err, K, t, P_removed_total)
        if (best is None) or (cand < best):
            best = cand

        # small local tweaks around t for improved fit
        for dt in (-1, 1, 2, -2):
            tt = max(0, min(t + dt, t_max_uniform))
            P_removed_width_tt = _estimate_width_removal_per_block(hidden, tt) * B
            P_removed_total_tt = P_removed_depth + P_removed_width_tt
            err_tt = abs(P_target - P_removed_total_tt)
            cand_tt = (err_tt, K, tt, P_removed_total_tt)
            if cand_tt < best:
                best = cand_tt

    if best is None:
        # fallback (no pruning)
        return TwoSSPPlan(
            target_sparsity=target_sparsity,
            num_blocks_total=B,
            blocks_to_prune=0,
            per_block_neurons_to_prune=0,
            stage2_fraction=0.0,
            estimated_total_removed_params=0,
            est_error_params=P_target,
        )

    err, K_best, t_best, P_removed_est = best
    return TwoSSPPlan(
        target_sparsity=target_sparsity,
        num_blocks_total=B,
        blocks_to_prune=K_best,
        per_block_neurons_to_prune=t_best,
        stage2_fraction=(K_best / B) if B > 0 else 0.0,
        estimated_total_removed_params=P_removed_est,
        est_error_params=int(err),
    )

# =========================
# Artifacts and Reporting
# =========================

@torch.no_grad()
def save_cifar_adapter(model: nn.Module, out_dir: str, filename: str = "adapter.pt", extra: Optional[Dict[str, Any]] = None) -> str:
    """Save classifier/adapter module state_dict and metadata.

    Args:
        model: ViTForImageClassification with classifier or adapter in model.classifier
        out_dir: directory to write file
        filename: file name
        extra: optional metadata to store along adapter

    Returns:
        Full path to saved file.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    payload = {
        "state_dict": model.classifier.state_dict(),
        "classifier_type": model.classifier.__class__.__name__,
        "num_labels": getattr(model.config, "num_labels", None),
        "hidden_size": getattr(model.config, "hidden_size", None),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path

@torch.no_grad()
def load_cifar_adapter(path: str, model: nn.Module) -> nn.Module:
    """Load a previously saved CIFAR adapter/head (from save_cifar_adapter) into a model.

    This reconstructs either a Linear classifier or a Sequential(adapter) based on saved payload,
    sets model.classifier accordingly, and updates model.config.num_labels.

    Args:
        path: path to adapter.pt saved by save_cifar_adapter
        model: ViTForImageClassification instance to modify in-place

    Returns:
        The same model instance with classifier replaced and weights loaded.
    """
    payload = torch.load(path, map_location="cpu")
    state_dict: Dict[str, torch.Tensor] = payload.get("state_dict", {})
    classifier_type: str = payload.get("classifier_type", "Linear")
    num_labels: Optional[int] = payload.get("num_labels", None)
    saved_hidden: Optional[int] = payload.get("hidden_size", None)

    # Try to infer shapes from state_dict if metadata missing
    inferred_hidden = None
    inferred_bottleneck = None
    inferred_out = None

    if "weight" in state_dict:  # Linear classifier case
        w = state_dict["weight"]
        inferred_out, inferred_hidden = int(w.shape[0]), int(w.shape[1])
    else:
        # Sequential adapter case: expect keys like "0.weight" and "2.weight"
        if "0.weight" in state_dict and "2.weight" in state_dict:
            w0 = state_dict["0.weight"]  # [bottleneck, hidden]
            w2 = state_dict["2.weight"]  # [out, bottleneck]
            inferred_bottleneck = int(w0.shape[0])
            inferred_hidden = int(w0.shape[1])
            inferred_out = int(w2.shape[0])

    hidden = saved_hidden or getattr(model.config, "hidden_size", None) or inferred_hidden
    if hidden is None:
        raise RuntimeError("Cannot determine hidden size for adapter loading.")
    if num_labels is None and inferred_out is not None:
        num_labels = inferred_out

    if classifier_type == "Linear" or ("weight" in state_dict and "bias" in state_dict):
        # Simple linear head
        if num_labels is None:
            raise RuntimeError("num_labels is None for Linear classifier.")
        head = nn.Linear(hidden, num_labels)
        head.load_state_dict(state_dict)
        model.classifier = head
        model.config.num_labels = num_labels
    else:
        # Sequential adapter (Linear -> GELU -> Linear)
        if inferred_bottleneck is None or num_labels is None:
            # best-effort: try to read from extra or raise
            extra = payload.get("extra", {})
            # Try to detect bottleneck from state_dict keys if not found yet
            if inferred_bottleneck is None and "0.weight" in state_dict:
                inferred_bottleneck = int(state_dict["0.weight"].shape[0])
            if inferred_out is None and "2.weight" in state_dict:
                inferred_out = int(state_dict["2.weight"].shape[0])
                num_labels = num_labels or inferred_out
            if inferred_bottleneck is None or num_labels is None:
                raise RuntimeError("Cannot reconstruct adapter architecture from payload/state_dict.")
        bottleneck = inferred_bottleneck
        adapter = nn.Sequential(
            nn.Linear(hidden, bottleneck, bias=False),
            nn.GELU(),
            nn.Linear(bottleneck, num_labels, bias=True),
        )
        adapter.load_state_dict(state_dict)
        model.classifier = adapter
        model.config.num_labels = num_labels

    return model

def _to_serializable(obj):
    try:
        json.dumps(obj)
        return obj
    except Exception:
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _to_serializable(v) for k, v in obj.items()}
        return str(obj)

def save_report(report: Dict[str, Any], out_dir: str, run_id: Optional[str] = None) -> Dict[str, str]:
    """Save JSON and Markdown consolidated report."""
    os.makedirs(out_dir, exist_ok=True)
    if run_id is None:
        run_id = time.strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(out_dir, f"report-{run_id}.json")
    md_path = os.path.join(out_dir, f"report-{run_id}.md")

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(report), f, indent=2, ensure_ascii=False)

    # Markdown summary
    lines = []
    lines.append(f"# 2SSP ViT Pruning Report ({run_id})")
    lines.append("")
    if "config" in report:
        lines.append("## Config")
        for k, v in report["config"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    if "metrics" in report:
        m = report["metrics"]
        lines.append("## Parameters reduction")
        lines.append(f"- Stage-1 (Width): {m.get('params_before_stage1_millions')}M -> {m.get('params_after_stage1_millions')}M ({m.get('stage1_reduction_percent')}%)")
        lines.append(f"- Stage-2 (Depth): {m.get('params_after_stage1_millions')}M -> {m.get('params_after_stage2_millions')}M ({m.get('stage2_reduction_percent')}%)")
        lines.append(f"- Final result: {m.get('params_before_stage1_millions')}M -> {m.get('params_after_stage2_millions')}M ({m.get('total_reduction_percent')}%)")
        lines.append("")
        lines.append("## Latency")
        lines.append(f"- Baseline: {m.get('latency_baseline_ms')} ms")
        lines.append(f"- Stage-1 (Width): {m.get('latency_stage1_ms')} ms ({m.get('latency_stage1_change_percent')}%)")
        lines.append(f"- Stage-2 (Depth): {m.get('latency_stage2_ms')} ms ({m.get('latency_stage2_change_percent')}%)")
        lines.append(f"- Final change: {m.get('latency_total_change_percent')}%")
        lines.append("")
        lines.append("## Accuracy")
        lines.append(f"- Baseline: {m.get('acc_baseline')}")
        lines.append(f"- Stage-1 (Width): {m.get('acc_stage1')} (drop: {m.get('acc_drop_stage1_percent')}%)")
        lines.append(f"- Stage-2 (Depth): {m.get('acc_stage2')} (drop: {m.get('acc_drop_stage2_percent')}%)")
        lines.append(f"- Final change: {m.get('acc_total_drop_percent')}%")
        lines.append("")
    if "plan" in report:
        p = report["plan"]
        lines.append("## Auto-allocation plan")
        lines.append(f"- Target sparsity: {p.get('target_sparsity')}")
        lines.append(f"- Blocks total: {p.get('num_blocks_total')}")
        lines.append(f"- Blocks to prune (Stage-2): {p.get('blocks_to_prune')} ({p.get('stage2_fraction'):.4f})")
        lines.append(f"- Per-block neurons to prune (Stage-1): {p.get('per_block_neurons_to_prune')}")
        lines.append(f"- Estimated total removed params: {p.get('estimated_total_removed_params')}")
        lines.append(f"- Estimation error (params): {p.get('est_error_params')}")
        lines.append("")
    if "artifacts" in report:
        lines.append("## Artifacts")
        for k, v in report["artifacts"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"json": json_path, "md": md_path}
