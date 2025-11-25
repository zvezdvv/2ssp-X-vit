#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python3 experiments/vit_pruning/apply_mask_prune.py --mask manual-experiments/mask.json --eval-on test --calib-per-class 2 --eval-batches 5
"""
Apply a binary (0/1) mask to prune ViT FFN neurons block-wise (Stage-1 width pruning) while measuring latency and accuracy. Based on experiments/vit_pruning/auto_2ssp.py.

Details:
- Mask: JSON containing leaf dict structures whose keys are "i:j" (i = MLP block index 0..11, j = neuron index) and values 0/1 (1 = prune, 0 = keep). If the format differs, the code searches any leaf dicts with such keys anywhere in the JSON tree.
- For each MLP block the exact number of neurons marked with 1 in that block's mask is pruned.
- Pruning uses src.vit_pruning.prune_vit_mlp_width with:
  - n_to_prune_per_block = [count of ones for block i]
  - precomputed_importance = vectors of length d_int: mask=0 -> importance +1, mask=1 -> -1.
    The function then keeps the highest importance neurons (descending keep_idx) and removes the marked ones.
- CIFAR-100: data loaders identical to auto_2ssp.py
- Model: SRP checkpoint B/16 with top10_idx=8 (resolution 224x224) as in auto_2ssp.py:
    timm -> weight transfer into HF ViTForImageClassification

Examples:
  1) Apply mask and measure metrics (CIFAR-100, 5 batches quick evaluation):
     python3 experiments/vit_pruning/apply_mask_prune.py --mask manual-experiments/mask.json --eval-batches 5

  2) Only evaluate without pruning (baseline metrics):
     python3 experiments/vit_pruning/apply_mask_prune.py --mask manual-experiments/mask.json --eval-batches 5 --dry-run

  3) Custom dataset fractions:
     python3 experiments/vit_pruning/apply_mask_prune.py --mask manual-experiments/mask.json --cifar-train-pct 0.25 --cifar-test-pct 0.25 --eval-batches 10 --calib_per_class 1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ViTForImageClassification
import timm

# Project root
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local libs
from src.vit_pruning import (
    prune_vit_mlp_width,
    evaluate_top1,
    count_total_params,
    compute_actual_sparsity,
    save_report,
    _get_encoder,
    _get_hidden_and_inter_sizes,
)

# SRP utilities (as in auto_2ssp.py)
PRUNING_DIR = ROOT / "pruning_srp-main"
if str(PRUNING_DIR) not in sys.path:
    sys.path.insert(0, str(PRUNING_DIR))
from process_models import load_model_timm as srp_load_model_timm  # type: ignore

KEY_RE = re.compile(r"^(\d+):(\d+)$")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def measure_latency(model: nn.Module, device: str, warmup: int = 3, iters: int = 10, img_size: int = 224) -> float:
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def _fwd(m, x):
        try:
            return m(pixel_values=x)
        except TypeError:
            try:
                return m(x)
            except Exception:
                return m(x=x) if hasattr(m, "forward") else m(x)

    with torch.no_grad():
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        for _ in range(warmup):
            _ = _fwd(model, dummy)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = _fwd(model, dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - start) / iters


def load_cifar(processor, device: str, dataset: str = "cifar100", train_pct: float = 0.25, test_pct: float = 0.25, calib_per_class: int = 0, num_workers: Optional[int] = None, img_size: int = 224):
    # Lazy imports
    from datasets import load_dataset
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from torch.utils.data import DataLoader

    if num_workers is None:
        num_workers = 2 if device != "cpu" else 0

    ds_name = dataset.lower()
    assert ds_name in ("cifar10", "cifar100"), f"Unsupported dataset: {dataset}"
    num_classes = 10 if ds_name == "cifar10" else 100

    train_split = f"train[:{int(train_pct * 100)}%]" if train_pct is not None else "train"
    test_split = f"test[:{int(test_pct * 100)}%]" if test_pct is not None else "test"

    train_raw = load_dataset(ds_name, split=train_split)
    test_raw = load_dataset(ds_name, split=test_split)

    normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    tf_test = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    def extract_label(example):
        if "label" in example:
            return int(example["label"])
        if "fine_label" in example:
            return int(example["fine_label"])
        return int(example.get("labels", 0))

    def preprocess(example, train=True):
        img = example["img"]
        img = tf_train(img) if train else tf_test(img)
        return {"pixel_values": img, "labels": extract_label(example)}

    train_ds = train_raw.map(lambda e: preprocess(e, True))
    test_ds = test_raw.map(lambda e: preprocess(e, False))
    # Ограничение числа примеров на класс если задано (>0)
    if calib_per_class and calib_per_class > 0:
        # Собираем индексы с максимумом calib_per_class на класс
        label_column = "labels"
        counts = {}
        kept_idx = []
        labels = train_ds[label_column]
        for idx, lbl in enumerate(labels):
            c = counts.get(int(lbl), 0)
            if c < calib_per_class:
                kept_idx.append(idx)
                counts[int(lbl)] = c + 1
        if kept_idx:
            train_ds = train_ds.select(kept_idx)
            print(f"[INFO] calib_per_class={calib_per_class}: train subset size={len(train_ds)}")
    train_ds.set_format(type="torch", columns=["pixel_values", "labels"])
    test_ds.set_format(type="torch", columns=["pixel_values", "labels"])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))
    return train_loader, test_loader


def timm2transformers(tf_model, timm_model):
    # Перенос весов (адаптировано из auto_2ssp.py)
    tf_model.vit.embeddings.cls_token = timm_model.cls_token
    tf_model.vit.embeddings.position_embeddings = timm_model.pos_embed
    tf_model.vit.embeddings.patch_embeddings.projection = timm_model.patch_embed.proj

    sd = {}
    for m1, m2 in zip(tf_model.vit.encoder.layer, timm_model.blocks):
        sd["weight"], sd["bias"] = m2.attn.qkv.weight[:768], m2.attn.qkv.bias[:768]
        m1.attention.attention.query.load_state_dict(sd)
        sd["weight"], sd["bias"] = m2.attn.qkv.weight[768:768 * 2], m2.attn.qkv.bias[768:768 * 2]
        m1.attention.attention.key.load_state_dict(sd)
        sd["weight"], sd["bias"] = m2.attn.qkv.weight[768 * 2:768 * 3], m2.attn.qkv.bias[768 * 2:768 * 3]
        m1.attention.attention.value.load_state_dict(sd)
        sd["weight"], sd["bias"] = m2.attn.proj.weight, m2.attn.proj.bias
        m1.attention.output.dense.load_state_dict(sd)
        sd["weight"], sd["bias"] = m2.mlp.fc1.weight, m2.mlp.fc1.bias
        m1.intermediate.dense.load_state_dict(sd)
        sd["weight"], sd["bias"] = m2.mlp.fc2.weight, m2.mlp.fc2.bias
        m1.output.dense.load_state_dict(sd)

        m1.layernorm_before = m2.norm1
        m1.layernorm_after = m2.norm2

    tf_model.vit.layernorm = timm_model.norm
    sd["weight"], sd["bias"] = timm_model.head.weight, timm_model.head.bias
    tf_model.classifier.load_state_dict(sd)
    return tf_model


def looks_like_leaf_ij_dict(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict) or not d:
        return False
    for k, v in d.items():
        if not (isinstance(k, str) and KEY_RE.match(k)):
            return False
        if not isinstance(v, (int, float)):
            return False
    return True


def find_leaf_ij_dicts(obj: Any, path: List[str] | None = None, out: List[Tuple[Tuple[str, ...], Dict[str, float]]] | None = None):
    if path is None:
        path = []
    if out is None:
        out = []
    if isinstance(obj, dict):
        if looks_like_leaf_ij_dict(obj):
            leaf = {k: float(v) for k, v in obj.items()}
            out.append((tuple(path), leaf))
            return out
        for k, v in obj.items():
            find_leaf_ij_dicts(v, path + [str(k)], out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_leaf_ij_dicts(v, path + [f"[{i}]"], out)
    return out


def load_mask(path: Path) -> Dict[int, Dict[int, int]]:
    """
    Загружает маску и конвертирует в словарь вида:
        block_idx -> { neuron_idx -> bit(0/1) }
    Если в JSON несколько листов с ij-ключами, они объединяются.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    leaves = find_leaf_ij_dicts(data)
    if not leaves:
        raise RuntimeError(f"Mask file has no ij-leaf dicts: {path}")
    blocks: Dict[int, Dict[int, int]] = {}
    for _pth, leaf in leaves:
        for k, v in leaf.items():
            m = KEY_RE.match(k)
            if not m:
                continue
            i = int(m.group(1))
            j = int(m.group(2))
            bit = int(round(float(v)))
            blocks.setdefault(i, {})[j] = 1 if bit != 0 else 0
    return blocks


def build_importance_and_counts(blocks_mask: Dict[int, Dict[int, int]], inter_sizes: List[int]) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Возвращает:
      - precomputed_importance: List[Tensor[d_int]] со значениями (+1 для keep, -1 для prune)
      - n_to_prune_per_block: число единиц в маске по каждому блоку
    Любые отсутствующие индексы считаются 0 (keep).
    """
    B = len(inter_sizes)
    imp: List[torch.Tensor] = []
    n_prune: List[int] = []
    for i in range(B):
        d_int = inter_sizes[i]
        vec = torch.ones(d_int, dtype=torch.float32)  # default keep (+1)
        bm = blocks_mask.get(i, {})
        cnt = 0
        for j in range(d_int):
            if bm.get(j, 0) == 1:
                vec[j] = -1.0
                cnt += 1
        imp.append(vec)
        n_prune.append(cnt)
    return imp, n_prune


def load_model_cifar100_with_srp(device: str) -> Tuple[nn.Module, Any, int]:
    """
    Загружает HF ViTForImageClassification и переносит в него веса timm из SRP (B/16, top10_idx=8, res=224).
    Возвращает (model, processor, input_res).
    """
    model_name = "google/vit-base-patch16-224"
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    model = ViTForImageClassification.from_pretrained(model_name)

    # Переносим вес timm (SRP) -> HF
    ds_name_srp = "cifar100"
    num_classes = 100
    model.classifier = torch.nn.Linear(768, num_classes, device=device)
    timm_model = srp_load_model_timm("B/16", ds_name_srp, top10_idx=8, verbose=True)
    model = timm2transformers(model, timm_model)

    input_res = 224
    return model, processor, input_res


def run(args):
    device = pick_device()
    print(f"[INFO] Using device: {device}")

    # Load model (SRP B/16 top10_idx=8 @224) and CIFAR loaders
    model, processor, input_res = load_model_cifar100_with_srp(device)
    model.to(device)

    # Data
    train_loader, test_loader = load_cifar(
        processor,
        device,
        dataset="cifar100",
        train_pct=args.cifar_train_pct,
        test_pct=args.cifar_test_pct,
        img_size=input_res,
        calib_per_class=args.calib_per_class
    )

    # Baseline metrics
    params_before = count_total_params(model)
    latency_baseline = measure_latency(model, device, warmup=3, iters=10, img_size=input_res)
    eval_loader = test_loader if args.eval_on == "test" else train_loader
    acc_baseline = evaluate_top1(model, eval_loader, device=device, max_batches=args.eval_batches, progress=True)
    print(f"[BASE] params={params_before}, latency={latency_baseline*1000:.2f} ms, acc={acc_baseline:.4f}")

    if args.dry_run:
        print("[DRY] Skipping pruning; baseline measured only.")
        metrics = {
            "params_before_stage1": params_before,
            "params_after_stage1": params_before,
            "params_before_stage1_millions": round(params_before / 1e6, 2),
            "params_after_stage1_millions": round(params_before / 1e6, 2),
            "stage1_reduction_percent": 0.0,
            "latency_baseline_ms": round(latency_baseline * 1000, 2),
            "latency_stage1_ms": round(latency_baseline * 1000, 2),
            "latency_stage1_change_percent": 0.0,
            "acc_baseline": round(acc_baseline, 4),
            "acc_stage1": round(acc_baseline, 4),
            "acc_drop_stage1_percent": 0.0,
        }
        report = {
            "config": {
                "mode": "dry-run",
                "mask_path": args.mask,
                "dataset": "cifar100",
                "eval_batches": args.eval_batches,
                "min_remaining": args.min_remaining,
            },
            "metrics": metrics,
        }
        saved = save_report(report, out_dir=str((Path(__file__).resolve().parent / "reports")))
        print(f"[INFO] Report saved to: {saved['json']} and {saved['md']}")
        return

    # Load mask
    mask_path = Path(args.mask)
    blocks_mask = load_mask(mask_path)

    # Build importance and prune counts
    hidden, inter_sizes = _get_hidden_and_inter_sizes(model)
    if hidden is None or len(inter_sizes) == 0:
        raise RuntimeError("Cannot obtain intermediate sizes from model.")
    precomp_imp, n_to_prune = build_importance_and_counts(blocks_mask, inter_sizes)

    # Sanity: min_remaining per block
    for i, (d_int, k) in enumerate(zip(inter_sizes, n_to_prune)):
        if d_int - k < args.min_remaining:
            adj = max(0, d_int - args.min_remaining)
            if k > adj:
                print(f"[WARN] Block {i}: requested prune {k} exceeds min_remaining constraint ({args.min_remaining}). Adjusting to {adj}.")
                n_to_prune[i] = adj

    # Apply pruning according to mask
    s1_res = prune_vit_mlp_width(
        model,
        n_to_prune_per_block=n_to_prune,
        min_remaining=args.min_remaining,
        strategy="l1",  # ignored when precomputed_importance provided
        dataloader=None,
        device=device,
        batch_limit=args.eval_batches,
        progress=False,
        collect_masks=True,
        precomputed_importance=precomp_imp,
    )
    if isinstance(s1_res, dict):
        model = s1_res["model"]
        ffn_indices = s1_res.get("ffn_pruned_indices", None)
        ffn_masks = s1_res.get("ffn_prune_masks", None)
    else:
        model = s1_res
        ffn_indices, ffn_masks = None, None

    # Post-prune metrics
    params_after = count_total_params(model)
    latency_after = measure_latency(model, device, warmup=3, iters=10, img_size=input_res)
    acc_after = evaluate_top1(model, eval_loader, device=device, max_batches=args.eval_batches, progress=True)

    s1 = compute_actual_sparsity(params_before, params_after)

    # Report
    artifacts: Dict[str, Any] = {
        "mask_path": str(mask_path),
        "n_to_prune_per_block": n_to_prune,
        "inter_sizes": inter_sizes,
    }
    if ffn_masks is not None:
        artifacts["ffn_prune_masks"] = ffn_masks
    if ffn_indices is not None:
        artifacts["ffn_pruned_indices"] = ffn_indices

    report = {
        "config": {
            "mode": "apply-mask",
            "mask_path": str(mask_path),
            "dataset": "cifar100",
            "eval_batches": args.eval_batches,
            "eval_on": args.eval_on,
            "calib_per_class": args.calib_per_class,
            "min_remaining": args.min_remaining,
            "model": "ViT B/16 (SRP timm -> HF), top10_idx=8, res=224",
        },
        "metrics": {
            "params_before_stage1": params_before,
            "params_after_stage1": params_after,
            "params_before_stage1_millions": round(params_before / 1e6, 2),
            "params_after_stage1_millions": round(params_after / 1e6, 2),
            "stage1_reduction_percent": round(s1 * 100, 1),
            "latency_baseline_ms": round(latency_baseline * 1000, 2),
            "latency_stage1_ms": round(latency_after * 1000, 2),
            "latency_stage1_change_percent": round((latency_after / max(1e-12, latency_baseline) - 1) * 100, 1),
            "acc_baseline": round(acc_baseline, 4),
            "acc_stage1": round(acc_after, 4),
            "acc_drop_stage1_percent": round(((acc_baseline - acc_after) / max(1e-12, acc_baseline)) * 100, 2),
        },
        "artifacts": artifacts,
    }
    saved = save_report(report, out_dir=str((Path(__file__).resolve().parent / "reports")))
    print("[SUMMARY]")
    print(json.dumps(report["metrics"], indent=2))
    print(f"[INFO] Report saved to: {saved['json']} and {saved['md']}")


def build_argparser():
    p = argparse.ArgumentParser(description="Apply binary FFN pruning mask (equal-per-block) to ViT B/16 SRP and evaluate metrics.")
    p.add_argument("--mask", type=str, required=True, help="Путь к JSON маске (0/1) с ключами 'i:j'")
    p.add_argument("--min-remaining", type=int, default=512, help="Мин. оставшаяся ширина FFN на блок после урезания (как в auto_2ssp.py)")
    p.add_argument("--cifar-train-pct", type=float, default=0.25)
    p.add_argument("--cifar-test-pct", type=float, default=0.25)
    p.add_argument("--eval-batches", type=int, default=5, help="Число батчей для быстрой оценки accuracy")
    p.add_argument("--eval-on", type=str, default="test", choices=["test", "train"], help="Which split to evaluate accuracy on: 'test' or 'train' (default: test)")
    p.add_argument("--dry-run", action="store_true", help="Не выполнять прунинг, только измерить baseline метрики")
    p.add_argument("--calib-per-class", type=int, default=0, help="Макс. число тренировочных изображений на класс (0 = без ограничения)")
    p.add_argument("--calib_per_class", type=int, dest="calib_per_class", help="Алиас для --calib-per-class")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
