#!/usr/bin/env python3

"""python3 experiments/vit_pruning/auto_2ssp.py --target 0.1 --load-cifar --cifar-train-pct 0.4 --cifar-test-pct 0.2 --load-adapter experiments/vit_pruning/artifacts/20250923-213804/adapter.pt --freeze-backbone --depth-importance copy --eval-batches 5 --force-depth-blocks 1"""
"""python3 experiments/vit_pruning/auto_2ssp.py --stage s1 --s1-sparsity 0.15 --load-cifar --cifar-train-pct 0.4 --cifar-test-pct 0.2 --load-adapter experiments/vit_pruning/artifacts/20250923-213804/adapter.pt --eval-batches 5"""
"""first run (no adapter yet): python3 experiments/vit_pruning/auto_2ssp.py --stage s1 --s1-sparsity 0.0 --load-cifar --dataset cifar100 --cifar-train-pct 0.4 --cifar-test-pct 0.2 --calib-per-class 2 --replace-classifier --freeze-backbone --do-finetune --ft-epochs 2 --ft-lr 5e-5 --save-adapter --eval-batches 5"""
"""CIFAR100: python3 experiments/vit_pruning/auto_2ssp.py --stage s1 --s1-sparsity 0.15 --load-cifar --dataset cifar100 --cifar-train-pct 0.4 --cifar-test-pct 0.2 --calib-per-class 2 --load-adapter experiments/vit_pruning/artifacts/20251024-215318/adapter.pt --eval-batches 5"""
"""CIFAR100: python3 experiments/vit_pruning/auto_2ssp.py --stage s2 --s2-sparsity 0.05 --load-cifar --dataset cifar100 --cifar-train-pct 0.4 --cifar-test-pct 0.2 --calib-per-class 2 --load-adapter experiments/vit_pruning/artifacts/20251024-215318/adapter.pt --eval-batches 5"""

import argparse
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from transformers import AutoImageProcessor, ViTForImageClassification
import timm
import re
import csv
from types import SimpleNamespace
import urllib.request
import urllib.error
import subprocess
import ssl

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


def load_cifar(processor, device: str, dataset: str = "cifar10", train_pct: float = 0.25, test_pct: float = 0.25, calib_per_class: int = 2, num_workers: Optional[int] = None, img_size: int = 224):
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

    # Base splits for optional fine-tune and test evaluation
    train_split = f"train[:{int(train_pct * 100)}%]" if train_pct is not None else "train"
    test_split = f"test[:{int(test_pct * 100)}%]" if test_pct is not None else "test"

    train_raw = load_dataset(ds_name, split=train_split)
    test_raw = load_dataset(ds_name, split=test_split)
    full_train_raw = load_dataset(ds_name, split="train")  # for calibration coverage

    normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
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
        img = train_tf(img) if train else test_tf(img)
        return {"pixel_values": img, "labels": extract_label(example)}

    # Map transforms for train/test
    train_ds = train_raw.map(lambda e: preprocess(e, True))
    test_ds = test_raw.map(lambda e: preprocess(e, False))
    train_ds.set_format(type="torch", columns=["pixel_values", "labels"])
    test_ds.set_format(type="torch", columns=["pixel_values", "labels"])

    # Build calibration subset with at least calib_per_class samples per class
    calib_indices = []
    counts = [0] * num_classes
    for i in range(len(full_train_raw)):
        y = extract_label(full_train_raw[i])
        if 0 <= y < num_classes and counts[y] < calib_per_class:
            calib_indices.append(i)
            counts[y] += 1
            if all(c >= calib_per_class for c in counts):
                break
    # Fallback to ensure at least one sample per class (shouldn't be needed for CIFAR)
    if any(c == 0 for c in counts):
        for i in range(len(full_train_raw)):
            y = extract_label(full_train_raw[i])
            if counts[y] == 0:
                calib_indices.append(i)
                counts[y] = 1
            if all(c >= 1 for c in counts):
                break

    calib_raw_subset = full_train_raw.select(calib_indices)
    calib_ds = calib_raw_subset.map(lambda e: preprocess(e, True))
    calib_ds.set_format(type="torch", columns=["pixel_values", "labels"])

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda"))
    cal_loader = DataLoader(calib_ds, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda"))

    return train_loader, test_loader, cal_loader


def maybe_finetune_head(model: nn.Module, train_loader, device: str, freeze_backbone: bool, epochs: int = 1, lr: float = 5e-5):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        print("[INFO] No trainable parameters; skipping fine-tune.")
        return

    total_params_m = sum(p.numel() for p in trainable) / 1e6
    print(f"[INFO] Fine-tuning head for {epochs} epoch(s) with {total_params_m:.2f}M trainable params")

    optimizer = torch.optim.AdamW(trainable, lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Use new torch.amp.GradScaler API for CUDA only; disable on MPS/CPU
    try:
        scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    except Exception:
        from torch.cuda.amp import GradScaler as _CudaGradScaler
        scaler = _CudaGradScaler(enabled=(device == "cuda"))

    autocast_dev = "cuda" if device == "cuda" else ("mps" if device == "mps" else "cpu")

    # Progress bar over batches with running loss on all backends (including MPS)
    try:
        from tqdm.auto import tqdm as _tqdm
    except Exception:
        _tqdm = None

    for epoch in range(epochs):
        model.train()
        running = 0.0
        nsteps = 0
        iterable = train_loader
        if _tqdm is not None:
            iterable = _tqdm(train_loader, desc=f"Finetune epoch {epoch+1}/{epochs}", leave=True)

        for batch in iterable:
            optimizer.zero_grad(set_to_none=True)
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type=autocast_dev, enabled=True):
                out = model(pixel_values=pixel_values)
                loss = criterion(out.logits, labels)

            if device == "cuda" and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_val = float(loss.detach().cpu())
            running += loss_val
            nsteps += 1
            if _tqdm is not None:
                iterable.set_postfix(loss=f"{loss_val:.4f}", avg=f"{(running/max(1,nsteps)):.4f}")

        print(f"[INFO] Epoch {epoch+1}/{epochs} done. Mean loss: {(running/max(1,nsteps)):.4f}")

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


def _select_srp_checkpoint(index_csv_path: str, model_type: str, dataset_name: str) -> tuple[str, int]:
    rows = []
    try:
        with open(index_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception as e:
        raise FileNotFoundError(f"Failed to read SRP index CSV at {index_csv_path}: {e}")

    cand = [r for r in rows if r.get("name") == model_type and r.get("adapt_ds") == dataset_name and r.get("adapt_filename")]
    if not cand:
        raise RuntimeError(f"No SRP checkpoint found in {index_csv_path} for name='{model_type}', adapt_ds='{dataset_name}'")

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return float("-inf")

    cand.sort(key=lambda r: _to_float(r.get("adapt_final_val", "-inf")), reverse=True)
    checkpoint = cand[0]["adapt_filename"]
    # infer resolution
    res = 224
    m = re.search(r"res[_\-]?(\d+)", checkpoint)
    if m:
        try:
            res = int(m.group(1))
        except Exception:
            pass
    return checkpoint, res


def _load_srp_model(model_type: str, dataset_name: str, models_dir: str, index_csv_path: Optional[str] = None, checkpoint_npz: Optional[str] = None, verbose: bool = False):
    timm_modelnames = {
        "Ti/16-224": "vit_tiny_patch16_224",
        "Ti/16-384": "vit_tiny_patch16_384",
        "S/16-224": "vit_small_patch16_224",
        "S/16-384": "vit_small_patch16_384",
        "B/16-224": "vit_base_patch16_224",
        "B/16-384": "vit_base_patch16_384",
    }

    if checkpoint_npz:
        checkpoint = os.path.splitext(os.path.basename(checkpoint_npz))[0]
        res = 224
        m = re.search(r"res[_\-]?(\d+)", checkpoint)
        if m:
            try:
                res = int(m.group(1))
            except Exception:
                pass
    else:
        if index_csv_path is None:
            index_csv_path = os.path.join(models_dir, "index.csv")
        checkpoint, res = _select_srp_checkpoint(index_csv_path, model_type, dataset_name)
        checkpoint_npz = os.path.join(models_dir, f"{checkpoint}.npz")

    key = f"{model_type}-{res}"
    if key not in timm_modelnames:
        key = f"{model_type}-224"
    timm_name = timm_modelnames[key]

    num_classes = 100 if dataset_name.lower() == "cifar100" else 37
    model = timm.create_model(timm_name, num_classes=num_classes)

    if not os.path.isfile(checkpoint_npz):
        # Try to download from public GCS; multiple fallbacks to avoid local CA issues
        os.makedirs(models_dir, exist_ok=True)
        src_url = f"https://storage.googleapis.com/vit_models/augreg/{checkpoint}.npz"
        print(f"[SRP] Checkpoint not found locally. Attempting download: {src_url}")
        download_ok = False
        err_msgs = []
        # 1) urllib with default SSL
        try:
            urllib.request.urlretrieve(src_url, checkpoint_npz)
            download_ok = True
            print(f"[SRP] Downloaded to: {checkpoint_npz} (urllib)")
        except Exception as e1:
            err_msgs.append(f"urllib default SSL: {e1}")
        # 2) curl -L (often has proper CA bundle on macOS)
        if not download_ok:
            try:
                subprocess.run(["curl", "-L", src_url, "-o", checkpoint_npz], check=True)
                download_ok = True
                print(f"[SRP] Downloaded to: {checkpoint_npz} (curl)")
            except Exception as e2:
                err_msgs.append(f"curl: {e2}")
        # 3) urllib with unverified SSL context (last resort)
        if not download_ok:
            try:
                ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(src_url, context=ctx) as r, open(checkpoint_npz, "wb") as f:
                    f.write(r.read())
                download_ok = True
                print(f"[SRP] Downloaded to: {checkpoint_npz} (urllib, unverified SSL)")
            except Exception as e3:
                err_msgs.append(f"urllib unverified SSL: {e3}")
        if not download_ok:
            raise FileNotFoundError(f"SRP checkpoint not found and download failed: {checkpoint_npz}. Tried methods: {' | '.join(err_msgs)}")
    timm.models.load_checkpoint(model, checkpoint_npz)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cfg = getattr(model, "default_cfg", {}) or {}
    mean = cfg.get("mean", [0.5, 0.5, 0.5])
    std = cfg.get("std", [0.5, 0.5, 0.5])
    input_size = cfg.get("input_size", (3, res, res))
    res = int(input_size[-1]) if isinstance(input_size, (list, tuple)) else res

    meta = {
        "timm_name": timm_name,
        "checkpoint": checkpoint,
        "res": res,
        "mean": mean,
        "std": std,
        "num_classes": num_classes,
        "checkpoint_path": checkpoint_npz,
    }
    if verbose:
        print(f"[SRP] Loaded {timm_name} with checkpoint {checkpoint} ({checkpoint_npz}), res={res}, num_classes={num_classes}")
    return model, meta
def timm2transformers(tf_model, timm_model):
    tf_model.vit.embeddings.cls_token = timm_model.cls_token
    tf_model.vit.embeddings.position_embeddings = timm_model.pos_embed
    tf_model.vit.embeddings.patch_embeddings.projection = timm_model.patch_embed.proj
    for i in range(12):
        tf_model.vit.encoder.layer[i].attention.attention.query.weight = torch.nn.Parameter(timm_model.blocks[i].attn.qkv.weight[:768])
        tf_model.vit.encoder.layer[i].attention.attention.key.weight = torch.nn.Parameter(timm_model.blocks[i].attn.qkv.weight[768:768*2])
        tf_model.vit.encoder.layer[i].attention.attention.value.weight = torch.nn.Parameter(timm_model.blocks[i].attn.qkv.weight[768*2:768*3])
        tf_model.vit.encoder.layer[i].attention.attention.query.bias = torch.nn.Parameter(timm_model.blocks[i].attn.qkv.bias[:768])
        tf_model.vit.encoder.layer[i].attention.attention.key.bias = torch.nn.Parameter(timm_model.blocks[i].attn.qkv.bias[768:768*2])
        tf_model.vit.encoder.layer[i].attention.attention.value.bias = torch.nn.Parameter(timm_model.blocks[i].attn.qkv.bias[768*2:768*3])
        tf_model.vit.encoder.layer[i].attention.output.dense = timm_model.blocks[i].attn.proj
        tf_model.vit.encoder.layer[i].intermediate.dense = timm_model.blocks[i].mlp.fc1
        tf_model.vit.encoder.layer[i].output.dense = timm_model.blocks[i].mlp.fc2
        tf_model.vit.encoder.layer[i].layernorm_before = timm_model.blocks[i].norm1
        tf_model.vit.encoder.layer[i].layernorm_after = timm_model.blocks[i].norm2
    tf_model.vit.layernorm = timm_model.norm
    tf_model.classifier = timm_model.head

    return tf_model
def load_model_timm(model_type, dataset_name, verbose=False):
    """ 
    model   types: B/16, S/16 or Ti/16
    dataset names: cifar100 or oxford-iiit-pet
    """
    import pandas as pd
    index = pd.read_csv('experiments/vit_pruning/models/index.csv')
    pretrains = set(
        index.query('ds=="i21k"').groupby('name').apply(
        lambda df: df.sort_values('final_val').iloc[-1], 
        include_groups=False).filename
    )
    finetunes = index.loc[index.filename.apply(lambda name: name in pretrains)]
    checkpoint = (
        finetunes.query(f'name=="{model_type}" and adapt_ds=="{dataset_name}"')
        .sort_values('adapt_final_val').iloc[-1].adapt_filename
    ) # Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--cifar100-steps_10k-lr_0.003-res_224
    if verbose: print(f"Loaded checkpoint: {checkpoint}")
    
    timm_modelnames = {
        'Ti/16-224': 'vit_tiny_patch16_224',
        'Ti/16-384': 'vit_tiny_patch16_384',
        'S/16-224': 'vit_small_patch16_224',
        'S/16-384': 'vit_small_patch16_384',
        'B/16-224': 'vit_base_patch16_224',
        'B/16-384': 'vit_base_patch16_384'
    }
    num_classes = 100 if dataset_name == 'cifar100' else 37
    res = int(checkpoint.split('_')[-1])
    model = timm.create_model(timm_modelnames[f'{model_type}-{res}'], num_classes=num_classes)
    
    # downloading a checkpoint automatically
    # may show an error, but still downloads the checkpoint
    from tensorflow.io import gfile # type: ignore
    if not gfile.exists(f'experiments/vit_pruning/models/{checkpoint}.npz'):     
        gfile.copy(f'gs://vit_models/augreg/{checkpoint}.npz', f'experiments/vit_pruning/models/{checkpoint}.npz')
    timm.models.load_checkpoint(model, f'experiments/vit_pruning/models/{checkpoint}.npz')

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else
        "cpu"
    ))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def run(args):
    device = pick_device()
    print(f"[INFO] Using device: {device}")

    run_id = time.strftime("%Y%m%d-%H%M%S")

    # Decide model source: HF or SRP timm checkpoint
    input_res = 224
    if getattr(args, "use_srp_checkpoint", False):
        model_name = args.model
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        model = ViTForImageClassification.from_pretrained(model_name)
        model = timm2transformers(model, load_model_timm('B/16', 'cifar100'))
        input_res = 224
        # models_dir = args.srp_models_dir
        # index_csv = args.srp_index_csv
        # srp_ds = getattr(args, "srp_dataset", "cifar100")
        # model, srp_meta = _load_srp_model(
        #     model_type=args.srp_model_type,
        #     dataset_name=srp_ds,
        #     models_dir=models_dir,
        #     index_csv_path=index_csv if args.srp_checkpoint_npz is None else None,
        #     checkpoint_npz=args.srp_checkpoint_npz,
        #     verbose=True,
        # )
        # input_res = int(args.srp_res) if args.srp_res is not None else int(srp_meta["res"])
        # processor = SimpleNamespace(image_mean=srp_meta["mean"], image_std=srp_meta["std"])
        # model_name = f"timm/{srp_meta['timm_name']}@{input_res} (SRP:{srp_meta['checkpoint']})"
        # Disable head changes and finetuning for SRP models by default
        args.use_adapter = False
        args.replace_classifier = False
        args.freeze_backbone = False
        args.do_finetune = False
    else:
        model_name = args.model
        processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        model = ViTForImageClassification.from_pretrained(model_name)
        input_res = 224

    # Determine dataset-derived number of labels if CIFAR is requested
    ds_name = None
    expected_num_labels = None
    if args.load_cifar:
        ds_name = (getattr(args, "dataset", "cifar10") or "cifar10").lower()
        expected_num_labels = 10 if ds_name == "cifar10" else 100

    if not getattr(args, "use_srp_checkpoint", False):
        hidden = model.config.hidden_size
        # Configure classifier / adapter (HF model path)
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
                out_dim = expected_num_labels if expected_num_labels is not None else getattr(model.classifier, "out_features", 10)
                model.classifier = nn.Linear(hidden, out_dim)
                model.config.num_labels = out_dim
                print(f"[INFO] Replaced classifier for {out_dim} classes")
        if args.freeze_backbone:
            for p in model.vit.parameters():
                p.requires_grad = False
            print("[INFO] Backbone frozen; training head-only")
    else:
        print("[INFO] Using SRP timm checkpoint; skipping head/adapter changes and backbone freezing.")

    model.to(device)

    # Data
    if args.load_cifar:
        train_loader, test_loader, cal_loader = load_cifar(
            processor,
            device,
            dataset=ds_name or (getattr(args, "dataset", "cifar10") or "cifar10"),
            train_pct=args.cifar_train_pct,
            test_pct=args.cifar_test_pct,
            calib_per_class=getattr(args, "calib_per_class", 2),
            img_size=input_res,
        )
    else:
        train_loader = test_loader = cal_loader = None

    # Optional fine-tune
    if args.do_finetune and train_loader is not None:
        maybe_finetune_head(model, train_loader, device, args.freeze_backbone, epochs=args.ft_epochs, lr=args.ft_lr)

    # Baseline metrics
    params_before = count_total_params(model)
    latency_baseline = measure_latency(model, device, warmup=3, iters=10, img_size=input_res)
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
    enc = _get_encoder(model)
    if hasattr(enc, "layer"):
        B = len(enc.layer)
    elif hasattr(enc, "blocks"):
        B = len(enc.blocks)
    else:
        raise AttributeError("Unsupported ViT model structure: expected encoder.layer or blocks")
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
        cal_loader = cal_loader if cal_loader is not None else (train_loader if train_loader is not None else test_loader)
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
    latency_stage1 = measure_latency(model, device, warmup=3, iters=10, img_size=input_res)
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
    latency_stage2 = measure_latency(model, device, warmup=3, iters=10, img_size=input_res)
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
        if getattr(args, "use_srp_checkpoint", False):
            # Save timm state dict and SRP meta
            pruned_dir = Path(args.pruned_output_dir) / run_id
            pruned_dir.mkdir(parents=True, exist_ok=True)
            sd_path = pruned_dir / "timm_model.pth"
            torch.save(model.state_dict(), sd_path.as_posix())
            meta = {}
            try:
                meta = srp_meta  # available when SRP branch was taken
            except NameError:
                meta = {}
            with open(pruned_dir / "srp_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            pruned_model_dir = pruned_dir.as_posix()
        else:
            pruned_model_dir = save_pruned_model_and_processor(
                model,
                processor,
                Path(args.pruned_output_dir),
                run_id,
            )

    # Save artifacts and report
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

    if args.save_adapter and not getattr(args, "use_srp_checkpoint", False):
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
            "dataset": ds_name,
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
    p.add_argument("--load-cifar", action="store_true", help="Load CIFAR-10/100 for quick accuracy evaluation")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"], help="When --load-cifar is set, choose dataset variant.")
    p.add_argument("--calib-per-class", type=int, default=2, help="Min samples per class for Stage-1 calibration dataloader")
    p.add_argument("--cifar-train-pct", type=float, default=0.25)
    p.add_argument("--cifar-test-pct", type=float, default=0.25)
    p.add_argument("--do-finetune", action="store_true", help="Lightly fine-tune the head/adapter")
    p.add_argument("--ft-epochs", type=int, default=1)
    p.add_argument("--ft-lr", type=float, default=5e-5)
    p.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone and train only head/adapter")
    p.add_argument("--replace-classifier", action="store_true", help="Replace classifier head to match dataset classes (10 for CIFAR-10, 100 for CIFAR-100)")
    p.add_argument("--use-adapter", action="store_true", help="Use adapter bottleneck instead of replacing head")
    p.add_argument("--adapter-reduction", type=int, default=4)
    p.add_argument("--save-adapter", action="store_true", help="Save adapter/classifier state dict")
    p.add_argument("--eval-batches", type=int, default=5, help="Max batches to use for quick evaluation")
    p.add_argument("--load-adapter", type=str, default=None, help="Path to saved adapter.pt to load into model classifier")
    # SRP/timm checkpoint options
    p.add_argument("--use-srp-checkpoint", action="store_true", help="Load SRP timm checkpoint via models/index.csv instead of HF model")
    p.add_argument("--srp-model-type", type=str, default="B/16", help="SRP model type: Ti/16, S/16, or B/16")
    p.add_argument("--srp-dataset", type=str, default="cifar100", choices=["cifar100", "oxford-iiit-pet"], help="Dataset used for SRP fine-tuning to pick checkpoint from index.csv")
    p.add_argument("--srp-index-csv", type=str, default=str(ROOT / "pruning_srp-main" / "models" / "index.csv"), help="Path to SRP models/index.csv")
    p.add_argument("--srp-models-dir", type=str, default=str(ROOT / "pruning_srp-main" / "models"), help="Directory containing SRP .npz checkpoints")
    p.add_argument("--srp-checkpoint-npz", type=str, default=None, help="Direct path to SRP .npz checkpoint to load (bypass index.csv)")
    p.add_argument("--srp-res", type=int, default=None, help="Force input resolution (e.g. 224 or 384) for SRP model")
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
