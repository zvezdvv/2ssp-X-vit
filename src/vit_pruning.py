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
def _compute_ffn_activation_importance(
    vit_model,
    dataloader,
    device: str = "cuda",
    batch_limit: Optional[int] = None,
    progress: bool = False,
) -> List[torch.Tensor]:
    """Compute per-neuron importance for FFN intermediate activations using average L2 norm over tokens,
    averaged across calibration samples.

    Returns:
        List of length B (num blocks), each a 1D tensor [d_int] with importance scores s_j.
    """
    vit_model.eval()
    encoder = _get_encoder(vit_model)
    B = len(encoder.layer)

    sums: List[Optional[torch.Tensor]] = [None for _ in range(B)]
    counts = 0

    def make_hook(b: int):
        def hook(module, _inp, out):
            # out: [batch, seq_len, d_int]
            with torch.no_grad():
                val = out
                if isinstance(val, (tuple, list)):
                    val = val[0]
                # L2 over tokens -> [batch, d_int], then sum over batch -> [d_int]
                per_sample = torch.linalg.vector_norm(val, ord=2, dim=1)  # [B, d_int]
                acc = per_sample.sum(dim=0)  # [d_int]
                acc_cpu = acc.detach().to("cpu")
                if sums[b] is None:
                    sums[b] = acc_cpu
                else:
                    sums[b] += acc_cpu
        return hook

    handles = []
    try:
        for b in range(B):
            handles.append(encoder.layer[b].intermediate.register_forward_hook(make_hook(b)))

        iterator = dataloader
        if progress:
            try:
                from tqdm.auto import tqdm as _tqdm  # lazy
                iterator = _tqdm(dataloader, total=(batch_limit if batch_limit is not None else None), desc="S1 activations", leave=False)
            except Exception:
                pass

        autocast_dev = "cuda" if str(device).startswith("cuda") else ("mps" if str(device).startswith("mps") else "cpu")
        for i, batch in enumerate(iterator):
            if batch_limit is not None and i >= batch_limit:
                break
            px = batch["pixel_values"].to(device, non_blocking=True)
            with torch.autocast(device_type=autocast_dev, enabled=True):
                _ = vit_model(pixel_values=px)
            counts += px.size(0)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    imps: List[torch.Tensor] = []
    for b in range(B):
        d_int = encoder.layer[b].intermediate.dense.out_features
        if sums[b] is None:
            imps.append(torch.zeros(d_int))
        else:
            imps.append(sums[b] / max(1, counts))
    return imps

@torch.no_grad()
def prune_vit_mlp_width(
    vit_model,
    sparsity: Optional[float] = None,
    strategy: str = "l1",
    min_remaining: int = 256,
    n_to_prune_per_block: Optional[List[int]] = None,
    dataloader=None,
    device: str = "cuda",
    batch_limit: Optional[int] = None,
    progress: bool = False,
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

    # Activation-based importance (paper S1): average L2 of activations over tokens and calibration samples
    importance_blocks: Optional[List[torch.Tensor]] = None
    if strategy == "act_l2" and dataloader is not None:
        print("[S1-LOG] Using activation-based importance (avg L2 over tokens, averaged across calibration samples)")
        importance_blocks = _compute_ffn_activation_importance(
            vit_model, dataloader, device=device, batch_limit=batch_limit, progress=progress
        )

    for block_idx, (inter_dense, out_dense) in enumerate(mlp_pairs):
        W_int: torch.Tensor = inter_dense.weight  # [intermediate, hidden]
        B_int: torch.Tensor = inter_dense.bias    # [intermediate]
        W_out: torch.Tensor = out_dense.weight    # [hidden, intermediate]
        n_channels = W_int.size(0)

        if strategy == "act_l2" and importance_blocks is not None:
            importance = importance_blocks[block_idx].to(W_int.device)
            if importance.numel() != n_channels:
                raise RuntimeError("act_l2 importance size mismatch with intermediate width")
        elif strategy == "l1":
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
        print(f"[S1-LOG] block={block_idx}, inter={n_channels}, n_prune={n_prune}, strategy={strategy}")
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
    num_to_prune: Optional[int] = None,
) -> Dict[str, Any]:
    """Prune attention submodules only in selected blocks (Stage-2 of 2SSP).

    This function selects encoder blocks by importance and replaces each selected block's
    attention module (e.g., ViTAttention with its Q/K/V and output projection) with a
    parameter-free bypass. The bypass outputs zeros so that the residual addition leaves
    the hidden states unchanged for the attention part, while the block's MLP (FFN) remains
    intact and functional.

    Args:
        vit_model: HuggingFace ViT model (ViTModel or ViTForImageClassification)
        sparsity: Fraction (0-1) of blocks whose attention submodule will be removed
        dataloader: Optional dataloader for metric-based selection
        device: Device for evaluation
        batch_limit: Max batches to process for evaluation
        metric_fn: Custom metric function to evaluate importance (unused, reserved)
        importance_mode: "copy" to evaluate impact via copy-edit, "heuristic" for position heuristic
        show_progress: Show progress during evaluation
        num_to_prune: If provided, prune attention in exactly this many blocks

    Returns:
        Dict with pruned model and pruning information
    """
    assert 0.0 <= sparsity < 1.0, "sparsity must be in [0,1)"

    class AttentionBypass(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, hidden_states, head_mask=None, output_attentions: bool = False, *args, **kwargs):
            zeros = torch.zeros_like(hidden_states)
            if output_attentions:
                return (zeros, None)
            return (zeros,)

    vit_model.eval()
    encoder = _get_encoder(vit_model)
    num_blocks = len(encoder.layer)

    # Determine how many blocks to modify
    if num_to_prune is None:
        num_to_prune = int(round(num_blocks * sparsity))
    # Keep at least one block intact to avoid degenerate encoders in edge-cases
    num_to_prune = max(0, min(num_blocks - 1, int(num_to_prune)))

    if num_to_prune == 0:
        print("No attention submodules to prune (num_to_prune=0).")
        return {"model": vit_model, "pruned_indices": [], "original_metrics": None, "final_metrics": None}

    # Importance selection
    if dataloader is None or (isinstance(importance_mode, str) and importance_mode.lower() == "heuristic"):
        print("Using heuristic for attention pruning importance (position-based).")
        importance_scores = [(i if i < num_blocks / 2 else num_blocks - i) for i in range(num_blocks)]
        to_prune = sorted(range(num_blocks), key=lambda i: importance_scores[i])[:num_to_prune]
        original_metrics = None
        final_metrics = None
    else:
        print(f"Evaluating {num_blocks} blocks by impact of removing attention (copy-replace)...")
        original_metrics = evaluate_top1(vit_model, dataloader, device, max_batches=batch_limit, progress=True)
        print(f"Baseline accuracy: {original_metrics:.4f}")

        impact_scores: List[float] = []
        try:
            from tqdm.auto import tqdm as _tqdm
        except Exception:
            _tqdm = None
        iterable = range(num_blocks)
        if show_progress and _tqdm is not None:
            iterable = _tqdm(range(num_blocks), desc="Attn eval (copy-replace)", leave=False)

        for block_idx in iterable:
            try:
                model_copy = copy.deepcopy(vit_model)
                model_copy.eval()
                enc_copy = _get_encoder(model_copy)
                if hasattr(enc_copy.layer[block_idx], "attention"):
                    enc_copy.layer[block_idx].attention = AttentionBypass()

                score = evaluate_top1(model_copy, dataloader, device, max_batches=batch_limit, progress=False)
                impact = max(0.0, original_metrics - score)
                impact_scores.append(impact)
                if show_progress:
                    print(f"[Attn] Block {block_idx} impact: {impact:.4f}", flush=True)
            except Exception as e:
                print(f"Error evaluating attention removal for block {block_idx}: {e}")
                impact_scores.append(0.0)

        to_prune = sorted(range(num_blocks), key=lambda i: impact_scores[i])[:num_to_prune]
        print(f"Selected blocks to remove attention: {to_prune}")

    # Apply pruning on the original model: replace attention with bypass
    for idx in to_prune:
        if hasattr(encoder.layer[idx], "attention"):
            encoder.layer[idx].attention = AttentionBypass()

    # Optional evaluation after pruning
    if dataloader is not None:
        final_metrics = evaluate_top1(vit_model, dataloader, device, max_batches=batch_limit, progress=True)
        print(f"Final accuracy after attention pruning: {final_metrics:.4f}")
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

@torch.no_grad()
def _count_attention_params_per_block(vit_model) -> List[int]:
    """Return per-block parameter counts for the attention submodule only (encoder.layer[i].attention)."""
    encoder = _get_encoder(vit_model)
    counts: List[int] = []
    for layer in encoder.layer:  # type: ignore[attr-defined]
        attn = getattr(layer, "attention", None)
        if attn is None:
            counts.append(0)
        else:
            counts.append(sum(p.numel() for p in attn.parameters()))
    return counts

@torch.no_grad()
def _count_ffn_params_per_block(vit_model) -> List[int]:
    """Return per-block parameter counts for the FFN submodules (intermediate.dense + output.dense)."""
    encoder = _get_encoder(vit_model)
    counts: List[int] = []
    for layer in encoder.layer:  # type: ignore[attr-defined]
        inter = layer.intermediate.dense
        out = layer.output.dense
        cnt = sum(p.numel() for p in inter.parameters()) + sum(p.numel() for p in out.parameters())
        counts.append(cnt)
    return counts

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
    forced_blocks: Optional[int] = None,
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

    # Debug: constants used by planner
    denom_const = B * (2 * hidden + 1)
    print(f"[PLAN-LOG] B={B}, target_sparsity={target_sparsity}, P_target={P_target}")
    print(f"[PLAN-LOG] hidden={hidden}, inter_sizes={inter_sizes}, min_remaining={min_remaining}")
    print(f"[PLAN-LOG] total_params={total_params}, block_params={block_params}")
    print(f"[PLAN-LOG] t_max_uniform={t_max_uniform}, denom=B*(2*hidden+1)={denom_const}")

    # Prefer using some attention pruning when multiple allocations achieve similar total removal.
    # Tolerance: treat solutions within ~2% of the target removed params as equivalent, then prefer larger K.
    tol = max(1, int(0.02 * P_target))
    best = None

    attn_param_counts = _count_attention_params_per_block(vit_model)
    P_attn_mean = sum(attn_param_counts) / max(1, B)

    # Compute FFN params per block to apply the paper's allocation formula:
    # N_attn = round( B * s^(|W_FFN| / (alpha * |W_Attn|)) )
    ffn_param_counts = _count_ffn_params_per_block(vit_model)
    W_FFN = sum(ffn_param_counts) / max(1, B)   # avg FFN params per block
    W_Attn = P_attn_mean                         # avg Attention params per block
    alpha = 1.5

    # Debug: inputs to formula
    print(f"[PLAN-LOG] attn_params_per_block={attn_param_counts}")
    print(f"[PLAN-LOG] ffn_params_per_block={ffn_param_counts}")
    print(f"[PLAN-LOG] mean_params_per_block: W_FFN_avg={int(W_FFN)}, W_Attn_avg={int(W_Attn)}, alpha={alpha}")
    if W_Attn > 0:
        exponent = W_FFN / (alpha * W_Attn)
    else:
        exponent = float("inf")
    print(f"[PLAN-LOG] exponent = W_FFN/(alpha*W_Attn) = {exponent if exponent != float('inf') else 'inf'}")

    if forced_blocks is not None:
        # Respect exact user override
        K_values = [max(0, min(B - 1, int(forced_blocks)))]
        print(f"[PLAN-LOG] forced_blocks provided: K_values={K_values}")
    else:
        if W_Attn > 0:
            K_formula = int(round(B * (target_sparsity ** exponent)))
        else:
            K_formula = 0
        K_formula = max(0, min(B - 1, K_formula))
        # Evaluate a small neighborhood around the formula-based K to best-fit the global target
        neighborhood = sorted(set([K_formula + d for d in (-2, -1, 0, 1, 2)]))
        K_values = [k for k in neighborhood if 0 <= k <= B - 1]
        print(f"[PLAN-LOG] K_formula={K_formula}, K_candidates={K_values}")

    for K in K_values:
        # Removed by depth (approximate with mean attention params per block):
        P_removed_depth = int(round(K * P_attn_mean))

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
        if best is None:
            best = cand
        else:
            best_err, best_K, _, _ = best
            # Prefer strictly smaller error; if errors are within tolerance, prefer larger K (more attention pruning)
            if (err < best_err - tol) or (abs(err - best_err) <= tol and K > best_K):
                best = cand

        # small local tweaks around t for improved fit
        for dt in (-1, 1, 2, -2):
            tt = max(0, min(t + dt, t_max_uniform))
            P_removed_width_tt = _estimate_width_removal_per_block(hidden, tt) * B
            P_removed_total_tt = P_removed_depth + P_removed_width_tt
            err_tt = abs(P_target - P_removed_total_tt)
            cand_tt = (err_tt, K, tt, P_removed_total_tt)
            if best is None:
                best = cand_tt
            else:
                best_err, best_K, _, _ = best
                if (err_tt < best_err - tol) or (abs(err_tt - best_err) <= tol and K > best_K):
                    best = cand_tt

    # If the best allocation ends up with K=0 but the target budget is substantial
    # relative to the average attention params per block, prefer a non-zero K alternative
    # within tolerance to avoid degenerate "all-width" solutions.
    if best is not None and forced_blocks is None:
        best_err, best_K, best_t, best_total = best
        if best_K == 0 and P_attn_mean > 0:
            if P_target >= 0.5 * P_attn_mean:
                # Explore K around the rough ratio of target to per-block attention params
                K_guess = max(1, int(round(P_target / max(1, P_attn_mean))))
                K_cand_max = min(B - 1, K_guess + 2)
                best_alt = None
                for K_alt in range(1, K_cand_max + 1):
                    P_removed_depth_alt = int(round(K_alt * P_attn_mean))
                    P_remaining_alt = max(0, P_target - P_removed_depth_alt)
                    t_alt = int(round(P_remaining_alt / denom)) if denom > 0 else 0
                    t_alt = max(0, min(t_alt, t_max_uniform))
                    P_removed_width_alt = _estimate_width_removal_per_block(hidden, t_alt) * B
                    P_total_alt = P_removed_depth_alt + P_removed_width_alt
                    err_alt = abs(P_target - P_total_alt)
                    cand_alt = (err_alt, K_alt, t_alt, P_total_alt)
                    if best_alt is None:
                        best_alt = cand_alt
                    else:
                        alt_err, alt_K, _, _ = best_alt
                        if (err_alt < alt_err - tol) or (abs(err_alt - alt_err) <= tol and K_alt > alt_K):
                            best_alt = cand_alt
                if best_alt is not None:
                    alt_err, alt_K, _, _ = best_alt
                    # Switch to non-zero K if it is within tolerance of the original best error
                    # or improves it. This enforces some attention pruning when reasonable.
                    if (alt_err < best_err - tol) or (abs(alt_err - best_err) <= tol):
                        best = best_alt

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

    # Debug: chosen allocation and budget breakdown
    P_removed_depth_chosen = int(round(K_best * P_attn_mean))
    P_removed_width_chosen = _estimate_width_removal_per_block(hidden, t_best) * B
    stage2_fraction_chosen = (K_best / B) if B > 0 else 0.0
    print(f"[PLAN-LOG] chosen: K={K_best}, t={t_best}, stage2_fraction={stage2_fraction_chosen:.6f}")
    print(f"[PLAN-LOG] removal_depth(attn)={P_removed_depth_chosen}, removal_width(ffn)={P_removed_width_chosen}, total={P_removed_est}, target={P_target}, err={int(err)}")

    return TwoSSPPlan(
        target_sparsity=target_sparsity,
        num_blocks_total=B,
        blocks_to_prune=K_best,
        per_block_neurons_to_prune=t_best,
        stage2_fraction=stage2_fraction_chosen,
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
