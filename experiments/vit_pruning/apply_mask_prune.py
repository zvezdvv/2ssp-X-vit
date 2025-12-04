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
import random
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, ViTForImageClassification
import timm
from torchvision.transforms import v2
from torchvision.datasets import CIFAR100 as C100
from torch.utils.data import DataLoader, random_split

# Project root
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local libs
from src.vit_pruning import (
    prune_vit_mlp_width,
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


def set_seed(seed: int) -> None:
    """
    Устанавливает seed для всех основных генераторов случайности и включает детерминизм в PyTorch.
    """
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    # Детерминизм ядра (может замедлить, но стабилизирует)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    # На новых версиях PyTorch можно жёстко требовать детерминизм
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def seed_worker_init(worker_id: int):
    """
    Инициализирует RNG в воркере DataLoader детерминированно на основе torch.initial_seed().
    """
    worker_seed = torch.initial_seed() % 2**32
    try:
        np.random.seed(worker_seed)
    except Exception:
        pass
    try:
        random.seed(worker_seed)
    except Exception:
        pass
    try:
        torch.manual_seed(worker_seed)
    except Exception:
        pass


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


@torch.no_grad()
def evaluate_top1_simple(model, dataloader, device: str = "cuda", max_batches: Optional[int] = None, progress: bool = False):
    """
    Локальная версия top-1 accuracy без использования src.vit_pruning.evaluate_top1.
    Поддерживает как dict-батчи {"pixel_values","labels"}, так и tuple (img, label).
    """
    model.eval()
    correct = 0
    total = 0
    autocast_device = "cuda" if str(device).startswith("cuda") else ("mps" if str(device).startswith("mps") else "cpu")
    iterator = dataloader
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(dataloader, total=(max_batches if max_batches is not None else None), desc="eval")
        except Exception:
            pass
    for i, batch in enumerate(iterator):
        if max_batches is not None and i >= max_batches:
            break
        if isinstance(batch, dict):
            x = batch.get("pixel_values", batch.get("input", None))
            y = batch.get("labels", batch.get("label", None))
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise TypeError("Unsupported batch type for evaluate_top1_simple")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=autocast_device, enabled=True):
            try:
                out = model(pixel_values=x)
            except TypeError:
                try:
                    out = model(x)
                except Exception:
                    out = model(x=x) if hasattr(model, "forward") else model(x)
            if isinstance(out, torch.Tensor):
                logits = out
            elif hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                logits = out[0]
            else:
                raise RuntimeError("Model forward output is not a tensor or does not contain logits")
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(1, total)


@torch.no_grad()
def snp_test_model(model, dataloader, device: str, max_batches: Optional[int] = None, progress: bool = False):
    """
    Точная валидация как в manual-experiments/snp.ipynb::test_model:
    Softmax -> argmax, аккумулирование correct и деление на размер датасета.
    Если ограничение по батчам задано, делим на число реально обработанных примеров,
    чтобы не занижать метрику (в ноутбуке ограничений не было).
    """
    sm = torch.nn.Softmax(dim=1)
    correct = 0
    seen = 0
    iterator = dataloader
    if progress:
        try:
            from tqdm.auto import tqdm as _tqdm
            iterator = _tqdm(dataloader, total=(max_batches if max_batches is not None else None), desc="eval")
        except Exception:
            pass
    for i, (features, labels) in enumerate(iterator):
        if max_batches is not None and i >= max_batches:
            break
        features = features.to(device)
        labels = labels.to(device)
        seen += int(labels.size(0))
        try:
            out = model(features)
        except Exception:
            out = model(pixel_values=features)
        if isinstance(out, torch.Tensor):
            logits = out
        elif hasattr(out, "logits"):
            logits = out.logits
        elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            logits = out[0]
        else:
            raise RuntimeError("Unexpected model output type for snp_test_model")
        clf = sm(logits).argmax(1)
        correct += (clf == labels).sum()
    denom = seen if (max_batches is not None) else len(dataloader.dataset)
    acc = correct / max(1, denom)
    return float(acc)


def load_cifar(processor, device: str, dataset: str = "cifar100", train_pct: float = 0.25, test_pct: float = 0.25, calib_per_class: int = 0, num_workers: Optional[int] = None, img_size: int = 224, seed: Optional[int] = None, calib_select: str = "first"):
    # Lazy imports
    from datasets import load_dataset
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from torch.utils.data import DataLoader

    if num_workers is None:
        num_workers = 2 if device != "cpu" else 0

    # Seed all RNGs (optional) before mapping/transforms to make Random* ops deterministic
    if seed is not None:
        try:
            from datasets import set_seed as hf_set_seed  # type: ignore
            hf_set_seed(seed)
        except Exception:
            pass
        set_seed(seed)

    # DataLoader RNG for deterministic shuffle + per-worker seeding
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
        seed_worker = seed_worker_init
    else:
        gen = None
        seed_worker = None

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
        labels = train_ds["labels"]
        kept_idx: List[int] = []
        if calib_select.lower() == "random":
            # Детерминированная случайная выборка по классам с учетом seed
            rng = np.random.default_rng(seed if seed is not None else 0)
            np_labels = np.array([int(x) for x in labels], dtype=np.int64)
            for cls in range(num_classes):
                cls_idx = np.flatnonzero(np_labels == cls)
                if cls_idx.size == 0:
                    continue
                take = min(calib_per_class, int(cls_idx.size))
                if take > 0:
                    sel = rng.choice(cls_idx, size=take, replace=False)
                    kept_idx.extend(sel.tolist())
            kept_idx = sorted(kept_idx)
            mode_info = "random"
        else:
            # "first": берем первые calib_per_class на класс в исходном порядке
            counts: Dict[int, int] = {}
            for idx, lbl in enumerate(labels):
                y = int(lbl)
                c = counts.get(y, 0)
                if c < calib_per_class:
                    kept_idx.append(idx)
                    counts[y] = c + 1
            mode_info = "first"
        if kept_idx:
            train_ds = train_ds.select(kept_idx)
            print(f"[INFO] calib_per_class={calib_per_class} ({mode_info}): train subset size={len(train_ds)}")
    train_ds.set_format(type="torch", columns=["pixel_values", "labels"])
    test_ds.set_format(type="torch", columns=["pixel_values", "labels"])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"), generator=gen, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))
    return train_loader, test_loader


def seed_gen(seed):
    return torch.Generator().manual_seed(seed)

def load_dataset(batch_size: int, subset_size: float = 1., seed=42):
    tr = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[.5]*3, std=[.5]*3),
        v2.Resize(224),
    ])
    dataset = C100('input/', train=False, transform=tr, download=True)
    if subset_size < 1.:
        n = int(round(len(dataset) * float(subset_size)))
        n = max(1, min(n, len(dataset)))
        dataset, _ = random_split(dataset, [n, len(dataset) - n], generator=seed_gen(seed))
    dl = DataLoader(dataset, batch_size=batch_size, generator=seed_gen(seed), shuffle=False, pin_memory=True)
    return dl




def _tv_make_transform(img_size: int, processor=None):
    """Build torchvision v2 transform; fixed mean/std = 0.5 as requested."""
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    try:
        # torchvision v2
        from torchvision.transforms import v2 as V2  # type: ignore
        return V2.Compose([
            V2.ToImage(),
            V2.ToDtype(torch.float32, scale=True),
            V2.Normalize(mean=mean, std=std),
            V2.Resize((img_size, img_size)),
        ])
    except Exception:
        # Fallback to torchvision v1
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def load_cifar_torchvision(
    processor,
    device: str,
    dataset: str = "cifar100",
    split: str = "test",
    root: str = "input",
    batch_size: int = 64,
    subset_size: float = 1.0,
    seed: Optional[int] = None,
    img_size: int = 224,
):
    """
    Простой загрузчик через torchvision.datasets (как в примере пользователя), детерминированный через generator и worker_init_fn.
    - dataset: 'cifar10' | 'cifar100'
    - split: 'train' | 'test'
    - subset_size: доля [0..1] примеров из выбранного split (детерминированно по seed)
    """
    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import CIFAR10, CIFAR100

    ds_name = (dataset or "cifar100").lower()
    DS = CIFAR100 if ds_name == "cifar100" else CIFAR10

    tr = _tv_make_transform(img_size, processor=processor)
    ds = DS(root, train=(split == "train"), transform=tr, download=True)

    if 0.0 < subset_size < 1.0:
        n = int(round(len(ds) * float(subset_size)))
        n = max(1, min(n, len(ds)))
        ds, _ = random_split(ds, [n, len(ds) - n], generator=seed_gen(seed))

    # Обертка в словарный формат под evaluate_top1
    ds = TVDictWrapper(ds)

    num_workers = 2 if device != "cpu" else 0
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,  # как в примере; порядок фиксируется generator'ом
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        generator=seed_gen(seed),
        worker_init_fn=seed_worker_init,
    )
    return dl


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

    # Optional global seeding for deterministic shuffle/aug
    if getattr(args, "seed", None) is not None:
        set_seed(int(args.seed))

    # Load model (SRP B/16 top10_idx=8 @224) and CIFAR loaders
    model, processor, input_res = load_model_cifar100_with_srp(device)
    model.to(device)

    # Data
    if getattr(args, "use_tv_loader", False):
        eval_loader_local = load_dataset(
            batch_size=getattr(args, "tv_test_batch", 64),
            subset_size=getattr(args, "tv_test_subset", 1.0),
            seed=getattr(args, "seed", 42),
        )
        train_loader = eval_loader_local
        test_loader = eval_loader_local
    else:
        train_loader, test_loader = load_cifar(
            processor,
            device,
            dataset="cifar100",
            train_pct=args.cifar_train_pct,
            test_pct=args.cifar_test_pct,
            img_size=input_res,
            calib_per_class=args.calib_per_class,
            seed=getattr(args, "seed", None),
            calib_select=getattr(args, "calib_select", "first"),
        )

    # Baseline metrics
    params_before = count_total_params(model)
    latency_baseline = measure_latency(model, device, warmup=3, iters=10, img_size=input_res)
    eval_loader = test_loader if args.eval_on == "test" else train_loader
    chosen_eval = snp_test_model if getattr(args, "snp_validation", False) else evaluate_top1_simple
    acc_baseline = chosen_eval(model, eval_loader, device=device, max_batches=args.eval_batches, progress=True)
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
    acc_after = chosen_eval(model, eval_loader, device=device, max_batches=args.eval_batches, progress=True)

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
    p.add_argument("--seed", type=int, default=None, help="Глобальный seed для детерминированного шафла/аугментаций")
    p.add_argument("--calib-select", type=str, default="first", choices=["first", "random"], help="Как выбирать подвыборку на класс при calib-per-class: 'first' (первые) или 'random' (детерминированно при --seed)")

    # Альтернативная простая загрузка через torchvision (как в примере)
    p.add_argument("--use-tv-loader", action="store_true", help="Использовать простой torchvision-загрузчик CIFAR вместо HuggingFace datasets")
    p.add_argument("--tv-root", type=str, default="input", help="Корень для torchvision CIFAR (по умолчанию: input)")
    p.add_argument("--tv-train-subset", type=float, default=1.0, help="Доля train набора [0..1] для torchvision loader")
    p.add_argument("--tv-test-subset", type=float, default=1.0, help="Доля test набора [0..1] для torchvision loader")
    p.add_argument("--tv-train-batch", type=int, default=32, help="Размер батча для train (torchvision loader)")
    p.add_argument("--tv-test-batch", type=int, default=64, help="Размер батча для test (torchvision loader)")
    p.add_argument("--snp-validation", action="store_true", help="Использовать точную схему валидации как в manual-experiments/snp.ipynb (Softmax+argmax; acc = correct/len(dataset))")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
