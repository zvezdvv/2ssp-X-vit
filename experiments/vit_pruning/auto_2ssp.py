#!/usr/bin/env python3

"""python3 experiments/vit_pruning/auto_2ssp.py --target 0.1 --load-cifar --cifar-train-pct 0.4 --cifar-test-pct 0.2 --load-adapter experiments/vit_pruning/artifacts/20250923-213804/adapter.pt --freeze-backbone --depth-importance copy --eval-batches 5 --force-depth-blocks 1"""
"""python3 experiments/vit_pruning/auto_2ssp.py --stage s1 --s1-sparsity 0.15 --load-cifar --cifar-train-pct 0.4 --cifar-test-pct 0.2 --load-adapter experiments/vit_pruning/artifacts/20250923-213804/adapter.pt --eval-batches 5"""
import argparse
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from transformers import AutoImageProcessor, ViTForImageClassification

# Local imports
import sys
ROOT = Path(__file__).resolve().parents[2]  # project root (contains src/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vit_pruning import (
    plan_2ssp_allocation,
    prune_vit_mlp_width,
    prune_vit_attention_blocks,
    evaluate_top1,
    count_total_params,
    compute_actual_sparsity,
    save_cifar_adapter,
    load_cifar_adapter,
    save_report,
    _get_encoder,
    _get_hidden_and_inter_sizes,
)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def measure_latency(model: nn.Module, device: str, warmup: int = 3, iters: int = 10) -> float:
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224, device=device)
        for _ in range(warmup):
            _ = model(pixel_values=dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = model(pixel_values=dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - start) / iters  # seconds / image


def load_cifar10(processor, device: str, train_pct: float = 0.25, test_pct: float = 0.25, num_workers: Optional[int] = None):
    # Lazy imports
    from datasets import load_dataset
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from torch.utils.data import DataLoader

    if num_workers is None:
        num_workers = 2 if device != "cpu" else 0

    train_split = f"train[:{int(train_pct * 100)}%]"
    test_split = f"test[:{int(test_pct * 100)}%]"

    train_ds = load_dataset("cifar10", split=train_split)
    test_ds = load_dataset("cifar10", split=test_split)

    normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    train_tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    def preprocess(example, train=True):
        img = example["img"]
        img = train_tf(img) if train else test_tf(img)
        return {"pixel_values": img, "labels": example["label"]}

    train_ds = train_ds.map(lambda e: preprocess(e, True))
    test_ds = test_ds.map(lambda e: preprocess(e, False))
    train_ds.set_format(type="torch", columns=["pixel_values", "labels"])
    test_ds.set_format(type="torch", columns=["pixel_values", "labels"])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))
    return train_loader, test_loader


def maybe_finetune_head(model: nn.Module, train_loader, device: str, freeze_backbone: bool, epochs: int = 1, lr: float = 5e-5):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        print("[INFO] No trainable parameters; skipping fine-tune.")
        return

    print(f"[INFO] Fine-tuning head for {epochs} epoch(s) with {sum(p.numel() for p in trainable)/1e6:.2f}M trainable params")
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    autocast_dev = "cuda" if device == "cuda" else ("mps" if device == "mps" else "cpu")

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=autocast_dev, enabled=True):
                out = model(pixel_values=pixel_values)
                loss = criterion(out.logits, labels)
            if device == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

    print("[INFO] Fine-tuning complete.")


def save_pruned_model_and_processor(model, processor, out_root: Path, run_id: str) -> str:
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save model and processor in HF format (does NOT touch HF cache)
    model.save_pretrained(out_dir.as_posix())
    try:
        processor.save_pretrained(out_dir.as_posix())
    except Exception:
        pass
    return out_dir.as_posix()


def run(args):
    device = pick_device()
    print(f"[INFO] Using device: {device}")

    model_name = args.model
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = ViTForImageClassification.from_pretrained(model_name)

    hidden = model.config.hidden_size
    # Configure classifier / adapter
    if getattr(args, "load_adapter", None):
        model = load_cifar_adapter(args.load_adapter, model)
        print(f"[INFO] Loaded adapter from: {args.load_adapter} (num_labels={getattr(model.config,'num_labels', None)}, type={model.classifier.__class__.__name__})")
    else:
        if args.use_adapter:
            original_out = model.classifier.out_features
            bottleneck = max(hidden // args.adapter_reduction, 32)
            model.classifier = nn.Sequential(
                nn.Linear(hidden, bottleneck, bias=False),
                nn.GELU(),
                nn.Linear(bottleneck, original_out, bias=True),
            )
            print(f"[INFO] Using adapter head with bottleneck={bottleneck}")
        elif args.replace_classifier:
            model.classifier = nn.Linear(hidden, 10)
            model.config.num_labels = 10
            print("[INFO] Replaced classifier for 10 classes")

    if args.freeze_backbone:
        for p in model.vit.parameters():
            p.requires_grad = False
        print("[INFO] Backbone frozen; training head-only")

    model.to(device)

    # Data
    if args.load_cifar:
        train_loader, test_loader = load_cifar10(processor, device, train_pct=args.cifar_train_pct, test_pct=args.cifar_test_pct)
    else:
        train_loader = test_loader = None

    # Optional fine-tune
    if args.do_finetune and train_loader is not None:
        maybe_finetune_head(model, train_loader, device, args.freeze_backbone, epochs=args.ft_epochs, lr=args.ft_lr)

    # Baseline metrics
    params_before = count_total_params(model)
    latency_baseline = measure_latency(model, device, warmup=3, iters=10)
    if test_loader is not None:
        acc_baseline = evaluate_top1(model, test_loader, device=device, max_batches=args.eval_batches, progress=True)
    else:
        acc_baseline = None

    # Plan allocation (only for stage='both'; stage-only runs do their own sizing)
    if args.stage == "both":
        plan = plan_2ssp_allocation(
            model,
            args.target,
            min_remaining=args.min_remaining,
            forced_blocks=args.force_depth_blocks,
        )
        print(f"[PLAN] target={plan.target_sparsity:.3f}, blocks_to_prune={plan.blocks_to_prune}, per_block_neurons_to_prune={plan.per_block_neurons_to_prune}")
    else:
        plan = None

    # Stage-1: width pruning
    B = len(_get_encoder(model).layer)
    ffn_masks = None
    ffn_indices = None

    if args.stage in ("both", "s1"):
        if args.stage == "both":
            n_to_prune_per_block = [plan.per_block_neurons_to_prune] * B
        else:
            # Stage-1-only: interpret --s1-sparsity as fraction of FFN params per block.
            # Since |FFN_params| = d_int * (2*hidden + 1), removing t neurons removes t*(2*hidden+1).
            # Fraction removed = t / d_int => t = round(s1_sparsity * d_int)
            hidden, inter_sizes = _get_hidden_and_inter_sizes(model)
            if args.s1_sparsity is None:
                raise ValueError("When --stage s1, you must provide --s1-sparsity (fraction of FFN params per block to remove).")
            n_to_prune_per_block = []
            for inter in inter_sizes:
                t = int(round(args.s1_sparsity * inter))
                # respect min_remaining
                t = max(0, min(t, max(0, inter - args.min_remaining)))
                n_to_prune_per_block.append(t)
            print(f"[S1] Using per-component sparsity: s1_sparsity={args.s1_sparsity}, n_to_prune_per_block[0]={n_to_prune_per_block[0]}")

        # Use calibration-driven importance (avg L2 of FFN activations over tokens/samples)
        cal_loader = train_loader if train_loader is not None else test_loader
        s1_res = prune_vit_mlp_width(
            model,
            n_to_prune_per_block=n_to_prune_per_block,
            min_remaining=args.min_remaining,
            strategy="act_l2",
            dataloader=cal_loader,
            device=device,
            batch_limit=args.eval_batches,
            progress=True,
            collect_masks=True,
        )
        if isinstance(s1_res, dict):
            model = s1_res["model"]
            ffn_indices = s1_res.get("ffn_pruned_indices", None)
            ffn_masks = s1_res.get("ffn_prune_masks", None)
        else:
            model = s1_res
    else:
        # Skip Stage-1 entirely for stage='s2'
        pass

    params_after_stage1 = count_total_params(model) if args.stage != "s2" else params_before
    latency_stage1 = measure_latency(model, device, warmup=3, iters=10)
    if test_loader is not None:
        acc_stage1 = evaluate_top1(model, test_loader, device=device, max_batches=args.eval_batches, progress=False)
    else:
        acc_stage1 = None

    # Stage-2: depth pruning
    pruned_indices = []
    if args.stage in ("both", "s2"):
        if args.stage == "both":
            depth_fraction = (plan.blocks_to_prune / max(1, B))
            num_to_prune = args.force_depth_blocks
            print(f"[INFO] Depth importance mode: {args.depth_importance}")
            if args.force_depth_blocks is not None:
                print(f"[INFO] Forcing depth pruning to remove exactly {args.force_depth_blocks} block(s). Width (Stage-1) adjusted to meet target.")
        else:
            # Stage-2-only: interpret --s2-sparsity as fraction of attention params, i.e. fraction of blocks
            if args.s2_sparsity is None:
                raise ValueError("When --stage s2, you must provide --s2-sparsity (fraction of Attention params / blocks to remove).")
            k = int(round(B * args.s2_sparsity))
            k = max(0, min(B - 1, k))
            num_to_prune = k
            depth_fraction = (k / max(1, B))
            print(f"[S2] Using per-component sparsity: s2_sparsity={args.s2_sparsity} -> K={k}/{B} blocks")
        res = prune_vit_attention_blocks(
            model,
            sparsity=depth_fraction,
            dataloader=test_loader if test_loader is not None else None,
            device=device,
            batch_limit=args.eval_batches,
            importance_mode=args.depth_importance,
            show_progress=True,
            num_to_prune=num_to_prune,
        )
        model = res["model"]
        pruned_indices = res["pruned_indices"]
    else:
        # Skip Stage-2 for stage='s1'
        pass

    params_after_stage2 = count_total_params(model)
    latency_stage2 = measure_latency(model, device, warmup=3, iters=10)
    if test_loader is not None:
        acc_stage2 = evaluate_top1(model, test_loader, device=device, max_batches=args.eval_batches, progress=False)
    else:
        acc_stage2 = None

    # Achieved sparsities
    s1 = compute_actual_sparsity(params_before, params_after_stage1)
    s2_local = compute_actual_sparsity(params_after_stage1, params_after_stage2)
    s_total = compute_actual_sparsity(params_before, params_after_stage2)

    # Optional: persist pruned model (user-controlled)
    pruned_model_dir = None
    if args.save_pruned_model:
        pruned_model_dir = save_pruned_model_and_processor(
            model,
            processor,
            Path(args.pruned_output_dir),
            run_id,
        )

    # Save artifacts and report
    run_id = time.strftime("%Y%m%d-%H%M%S")
    reports_dir = Path(__file__).resolve().parent / "reports"
    artifacts_dir = Path(__file__).resolve().parent / "artifacts" / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Persist masks/indices as requested
    if ffn_masks is not None:
        try:
            hidden, inter_sizes = _get_hidden_and_inter_sizes(model)
        except Exception:
            inter_sizes = None
        ffn_masks_path = artifacts_dir / "ffn_prune_masks.json"
        with open(ffn_masks_path, "w", encoding="utf-8") as f:
            json.dump({
                "format_version": 1,
                "stage": "s1",
                "strategy": "act_l2",
                "min_remaining": args.min_remaining,
                "s1_sparsity": args.s1_sparsity if getattr(args, "s1_sparsity", None) is not None else None,
                "block_inter_sizes": inter_sizes,
                "masks": ffn_masks,            # list of lists, 1=prune, 0=keep
                "indices": ffn_indices,        # list of lists of pruned neuron indices
            }, f, indent=2)
        print(f"[ARTIFACT] FFN prune masks saved to: {ffn_masks_path}")

    attn_indices_path = None
    if pruned_indices:
        attn_indices_path = artifacts_dir / "attention_pruned_indices.json"
        with open(attn_indices_path, "w", encoding="utf-8") as f:
            json.dump({
                "format_version": 1,
                "stage": "s2",
                "indices": pruned_indices,     # list of block indices with attention removed
            }, f, indent=2)
        print(f"[ARTIFACT] Attention pruned indices saved to: {attn_indices_path}")

    artifacts: Dict[str, Any] = {
        "pruned_block_indices": pruned_indices,
    }
    if ffn_masks is not None:
        artifacts["ffn_prune_masks_path"] = str(ffn_masks_path)
    if attn_indices_path is not None:
        artifacts["attn_pruned_indices_path"] = str(attn_indices_path)
    if pruned_model_dir is not None:
        artifacts["pruned_model_dir"] = pruned_model_dir

    # Build plan section only for stage='both'
    plan_section = None
    if plan is not None:
        plan_section = {
            "target_sparsity": plan.target_sparsity,
            "num_blocks_total": plan.num_blocks_total,
            "blocks_to_prune": plan.blocks_to_prune,
            "per_block_neurons_to_prune": plan.per_block_neurons_to_prune,
            "stage2_fraction": plan.stage2_fraction,
            "estimated_total_removed_params": plan.estimated_total_removed_params,
            "est_error_params": plan.est_error_params,
        }

    if args.save_adapter:
        adapter_path = save_cifar_adapter(
            model,
            out_dir=str(artifacts_dir),
            filename="adapter.pt",
            extra={
                "model_name": model_name,
                "target_sparsity": args.target,
                "use_adapter": args.use_adapter,
                "replace_classifier": args.replace_classifier,
            },
        )
        artifacts["adapter_path"] = adapter_path

    report = {
        "config": {
            "model": model_name,
            "target_sparsity": args.target,
            "stage": args.stage,
            "s1_sparsity": args.s1_sparsity,
            "s2_sparsity": args.s2_sparsity,
            "freeze_backbone": args.freeze_backbone,
            "replace_classifier": args.replace_classifier,
            "use_adapter": args.use_adapter,
            "adapter_reduction": args.adapter_reduction if args.use_adapter else None,
            "eval_batches": args.eval_batches,
            "min_remaining": args.min_remaining,
            "cifar_load": args.load_cifar,
        },
        "metrics": {
            "params_before_stage1": params_before,
            "params_after_stage1": params_after_stage1,
            "params_after_stage2": params_after_stage2,
            "params_before_stage1_millions": round(params_before / 1e6, 2),
            "params_after_stage1_millions": round(params_after_stage1 / 1e6, 2),
            "params_after_stage2_millions": round(params_after_stage2 / 1e6, 2),
            "stage1_reduction_percent": round(s1 * 100, 1),
            "stage2_reduction_percent": round(s2_local * 100, 1),
            "total_reduction_percent": round(s_total * 100, 1),
            "latency_baseline_ms": round(latency_baseline * 1000, 2),
            "latency_stage1_ms": round(latency_stage1 * 1000, 2),
            "latency_stage2_ms": round(latency_stage2 * 1000, 2),
            "latency_stage1_change_percent": round((latency_stage1 / max(1e-12, latency_baseline) - 1) * 100, 1),
            "latency_stage2_change_percent": round((latency_stage2 / max(1e-12, latency_stage1) - 1) * 100, 1),
            "latency_total_change_percent": round((latency_stage2 / max(1e-12, latency_baseline) - 1) * 100, 1),
            "acc_baseline": round(acc_baseline, 4) if acc_baseline is not None else None,
            "acc_stage1": round(acc_stage1, 4) if acc_stage1 is not None else None,
            "acc_stage2": round(acc_stage2, 4) if acc_stage2 is not None else None,
            "acc_drop_stage1_percent": round(((acc_baseline - acc_stage1) / max(1e-12, acc_baseline)) * 100, 2) if (acc_baseline is not None and acc_stage1 is not None) else None,
            "acc_drop_stage2_percent": round(((acc_stage1 - acc_stage2) / max(1e-12, acc_stage1)) * 100, 2) if (acc_stage1 is not None and acc_stage2 is not None) else None,
            "acc_total_drop_percent": round(((acc_baseline - acc_stage2) / max(1e-12, acc_baseline)) * 100, 2) if (acc_baseline is not None and acc_stage2 is not None) else None,
        },
        "artifacts": artifacts,
    }
    if plan_section is not None:
        report["plan"] = plan_section

    saved = save_report(report, out_dir=str(reports_dir), run_id=run_id)
    print("[SUMMARY]")
    print(json.dumps(report["metrics"], indent=2))
    print(f"[INFO] Report saved to: {saved['json']} and {saved['md']}")
    if "adapter_path" in artifacts:
        print(f"[INFO] Adapter saved to: {artifacts['adapter_path']}")
    if "pruned_model_dir" in artifacts:
        print(f"[INFO] Pruned model saved to: {artifacts['pruned_model_dir']}")
    else:
        print("[INFO] Pruned model was NOT saved (default). Pass --save-pruned-model to persist.")


def build_argparser():
    p = argparse.ArgumentParser(description="Auto 2SSP for ViT with single TARGET sparsity.")
    p.add_argument("--model", type=str, default="google/vit-base-patch16-224", help="HF model id")
    p.add_argument("--target", type=float, required=False, help="Global target sparsity (0..1). Used when --stage both (default).")
    p.add_argument("--stage", type=str, default="both", choices=["both", "s1", "s2"], help="Which stage to run: both (default), s1 (FFN width only), s2 (Attention depth only).")
    p.add_argument("--s1-sparsity", type=float, default=None, help="Component-wise sparsity for FFN (fraction of FFN params per block). Used when --stage s1.")
    p.add_argument("--s2-sparsity", type=float, default=None, help="Component-wise sparsity for Attention (fraction of attention params / blocks). Used when --stage s2.")
    p.add_argument("--min-remaining", type=int, default=512, help="Min remaining intermediate size per block after width pruning")
    p.add_argument("--load-cifar", action="store_true", help="Load CIFAR-10 for quick accuracy evaluation")
    p.add_argument("--cifar-train-pct", type=float, default=0.25)
    p.add_argument("--cifar-test-pct", type=float, default=0.25)
    p.add_argument("--do-finetune", action="store_true", help="Lightly fine-tune the head/adapter")
    p.add_argument("--ft-epochs", type=int, default=1)
    p.add_argument("--ft-lr", type=float, default=5e-5)
    p.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone and train only head/adapter")
    p.add_argument("--replace-classifier", action="store_true", help="Replace classifier with 10-class head")
    p.add_argument("--use-adapter", action="store_true", help="Use adapter bottleneck instead of replacing head")
    p.add_argument("--adapter-reduction", type=int, default=4)
    p.add_argument("--save-adapter", action="store_true", help="Save adapter/classifier state dict")
    p.add_argument("--eval-batches", type=int, default=5, help="Max batches to use for quick evaluation")
    p.add_argument("--load-adapter", type=str, default=None, help="Path to saved adapter.pt to load into model classifier")
    p.add_argument(
        "--depth-importance",
        type=str,
        default="copy",
        choices=["copy", "heuristic"],
        help="Depth importance mode: 'copy' (accurate, slower; shows per-block progress) or 'heuristic' (fast, no eval).",
    )
    p.add_argument(
        "--force-depth-blocks",
        type=int,
        default=None,
        help="Force Stage-2 to remove exactly this number of encoder blocks. Stage-1 width pruning will be adjusted to hit the global target.",
    )
    # Control whether pruned model is persisted or discarded
    p.add_argument("--save-pruned-model", action="store_true", help="Persist pruned model to --pruned-output-dir (default: do not save)")
    p.add_argument("--pruned-output-dir", type=str, default=str((Path(__file__).resolve().parent / "pruned_models")), help="Directory to save pruned model (subfolder by run_id)")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
