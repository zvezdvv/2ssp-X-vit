#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python3 manual-experiments/aggregate_and_mask-summation.py --pattern "manual-experiments/normalized/has-scores.json" --pattern "manual-experiments/normalized/2ssp_vit_b16_ffn_importances.json" --prune 20
"""
Aggregate (element-wise sum) values from provided normalized JSON files
and build a binary pruning mask (0/1) for MLP neurons based on the sums.

Key ideas:
- Expected leaf dictionaries have the form {"i:j": number}, where:
  i = MLP block index (0..11), j = neuron index within the block.
- Aggregation: for the same keys "i:j" values are summed across files.
  Missing keys in a file are treated as 0 for the aggregation.
- JSON structure is reconstructed only for leaves with "i:j" keys.
  Non-leaf parts (metadata, etc.) do not affect sums.
- Mask: user provides percent/fraction of parameters to prune (e.g., 20 or 0.2).
  For each MLP block, K smallest aggregated values are marked with 1 (prune), others are 0 (keep).
  K is the same for all 12 blocks (equal number of pruned neurons per block).
  By default, K is computed as rounding(percent * block_size) and then unified to
  common K = min(K_i) over all blocks so that no block exceeds its capacity.
  It is also possible to set K directly via --per-block-k.

Outputs (by default):
  - Sums: manual-experiments/aggregated_sums.json
  - Mask: manual-experiments/mask.json

Note on output ordering for visualization:
- The resulting mask preserves a stable natural ordering of keys by (block_index, neuron_index),
  i.e. sorted numerically by i then j, so keys don't appear visually shuffled.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

KEY_RE = re.compile(r"^(\d+):(\d+)$")


def is_number(x: Any) -> bool:
    """True for int/float but not bool (bool is a subclass of int)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def looks_like_leaf_ij_dict(d: Dict[str, Any]) -> bool:
    """Return True if dict looks like a leaf with keys 'i:j' and numeric values."""
    if not isinstance(d, dict) or not d:
        return False
    for k, v in d.items():
        if not (isinstance(k, str) and KEY_RE.match(k)):
            return False
        if not is_number(v):
            return False
    return True


PathTuple = Tuple[str, ...]


def find_leaf_ij_dicts(obj: Any, path: List[str] | None = None, out: List[Tuple[PathTuple, Dict[str, float]]] | None = None):
    """
    Find all leaf dicts where keys are 'i:j' and values are numbers.
    Returns list of tuples (path_tuple, dict[str,float]).
    """
    if path is None:
        path = []
    if out is None:
        out = []

    if isinstance(obj, dict):
        if looks_like_leaf_ij_dict(obj):
            # ensure float
            leaf = {k: float(v) for k, v in obj.items()}
            out.append((tuple(path), leaf))
            return out
        # otherwise descend
        for k, v in obj.items():
            find_leaf_ij_dicts(v, path + [str(k)], out)
    elif isinstance(obj, list):
        # Support lists; path element becomes "[i]"
        for i, v in enumerate(obj):
            find_leaf_ij_dicts(v, path + [f"[{i}]"], out)
    return out


def collect_files(default_glob_dir: Path, patterns: List[str], files: List[str]) -> List[Path]:
    collected: List[Path] = []

    # explicit files
    for p in files:
        path = Path(p)
        if path.exists() and path.suffix.lower() == ".json":
            collected.append(path)

    # glob patterns
    for pat in patterns:
        for p in Path(".").glob(pat):
            if p.exists() and p.suffix.lower() == ".json":
                collected.append(p)

    # default — all *.json in normalized
    if not collected:
        default = sorted((default_glob_dir).glob("*.json"))
        collected.extend(default)

    # unique by realpath preserving order
    seen = set()
    unique: List[Path] = []
    for p in collected:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json_atomic(data: Any, out_path: Path, compact: bool = True) -> None:
    """Atomic JSON write with optional compact separators."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        if compact:
            json.dump(data, f, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        else:
            json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
    os.replace(tmp_path, out_path)


def aggregate_leaves(files: List[Path]) -> Dict[PathTuple, Dict[str, float]]:
    """
    Sum all leaf ij-dicts by path.
    Returns mapping: path_tuple -> {"i:j": sum}.
    """
    sums: Dict[PathTuple, Dict[str, float]] = {}
    for src in files:
        try:
            data = load_json(src)
        except Exception as e:
            print(f"[warn] skip {src}: {e}")
            continue
        leaves = find_leaf_ij_dicts(data)
        if not leaves:
            print(f"[info] no leaf ij-dicts in {src}")
        for path_tuple, leaf in leaves:
            target = sums.setdefault(path_tuple, {})
            for k, v in leaf.items():
                target[k] = target.get(k, 0.0) + float(v)
    return sums


def reconstruct_from_leaves(leaves: Dict[PathTuple, Dict[str, float]]) -> Dict[str, Any]:
    """
    Rebuild a tree-shaped JSON containing only ij-leaves (no other wrappers).
    """
    root: Dict[str, Any] = {}
    for path, leaf in leaves.items():
        cur = root
        for key in path:
            if key.startswith("[") and key.endswith("]"):
                # list case: keep as string key (rare)
                key = key
            cur = cur.setdefault(key, {})
        cur.update(leaf)
    return root


def parse_fraction(p: float) -> float:
    """
    Convert percent or fraction to [0,1].
    If p > 1, treat as percentage (p/100).
    """
    if p < 0:
        return 0.0
    return p / 100.0 if p > 1.0 else p


def rounding_fn(name: str):
    if name == "floor":
        return math.floor
    if name == "ceil":
        return math.ceil
    return lambda x: int(round(x))


def build_block_groups(leaf: Dict[str, float]) -> Dict[int, List[Tuple[str, float]]]:
    """
    Group ("i:j", value) pairs by i (block index).
    """
    groups: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
    for k, v in leaf.items():
        m = KEY_RE.match(k)
        if not m:
            continue
        i = int(m.group(1))
        groups[i].append((k, float(v)))
    return groups


def make_mask_for_leaf(
    leaf: Dict[str, float],
    prune_fraction: float,
    rounding: str = "round",
    per_block_k: int | None = None,
) -> Dict[str, int]:
    """
    Build a binary mask for a single ij-leaf:
      - group by blocks i
      - compute K_i = round(prune_fraction * N_i)
      - unify to common K = min_i K_i (same K for all blocks)
      - in each block pick K smallest values -> 1 (prune), others -> 0 (keep)

    To improve readability, the resulting mask preserves a stable key order
    sorted numerically by (block_index, neuron_index), i.e. by 'i' then 'j'.
    If --per-block-k is provided, it takes precedence over prune_fraction.
    """
    groups = build_block_groups(leaf)
    if not groups:
        # preserve natural ordering of keys when returning zeros
        mask: Dict[str, int] = {}
        keys_sorted = sorted(
            leaf.keys(),
            key=lambda kk: (int(kk.split(":")[0]), int(kk.split(":")[1])) if KEY_RE.match(kk) else (1 << 30, 1 << 30),
        )
        for kk in keys_sorted:
            mask[kk] = 0
        return mask

    # Warn if block count differs from 12 (non-fatal)
    unique_blocks = sorted(groups.keys())
    if len(unique_blocks) != 12:
        print(f"[warn] leaf has {len(unique_blocks)} block(s), expected 12. Proceeding anyway: {unique_blocks}")

    # Compute common K across blocks
    if per_block_k is None:
        rfun = rounding_fn(rounding)
        k_candidates = []
        for i, items in groups.items():
            n_i = len(items)
            k_i = max(0, min(n_i, rfun(prune_fraction * n_i)))
            k_candidates.append(k_i)
        common_k = min(k_candidates) if k_candidates else 0
    else:
        common_k = max(0, per_block_k)

    # Determine pruned set per block based on smallest sums
    pruned_keys_set = set()
    for i, items in groups.items():
        items_sorted_by_val = sorted(items, key=lambda kv: kv[1])  # ascending by aggregated sum
        k_i = min(common_k, len(items_sorted_by_val))
        pruned_keys_set |= {k for (k, _) in items_sorted_by_val[:k_i]}

    # Build mask in stable numeric key order (block, neuron)
    mask: Dict[str, int] = {}
    keys_sorted = sorted(
        leaf.keys(),
        key=lambda kk: (int(kk.split(":")[0]), int(kk.split(":")[1])) if KEY_RE.match(kk) else (1 << 30, 1 << 30),
    )
    for kk in keys_sorted:
        mask[kk] = 1 if kk in pruned_keys_set else 0
    return mask


def apply_masks_to_leaves(aggregated_leaves: Dict[PathTuple, Dict[str, float]], masks_by_path: Dict[PathTuple, Dict[str, int]]) -> Dict[str, Any]:
    """
    Assemble a tree of masks from per-leaf masks.
    """
    root: Dict[str, Any] = {}
    for path, m in masks_by_path.items():
        cur = root
        for key in path:
            cur = cur.setdefault(key, {})
        cur.update(m)  # insertion order of m (already stable) is preserved
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate normalized JSON scores and build equal-per-block pruning masks for 12 MLP blocks.")
    parser.add_argument("files", nargs="*", help="Input JSON files to aggregate (optional; you can use --pattern instead)")
    parser.add_argument("--pattern", action="append", default=[], help="Glob pattern(s), e.g. 'manual-experiments/normalized/*.json'")
    parser.add_argument("--aggregated", type=str, default=None, help="Path to precomputed sums (if you want to build the mask without recomputing sums)")
    parser.add_argument("--aggregate-out", type=str, default="manual-experiments/aggregated_sums.json", help="Where to save aggregated sums (default: manual-experiments/aggregated_sums.json)")
    parser.add_argument("--mask-out", type=str, default="manual-experiments/mask.json", help="Where to save the mask (default: manual-experiments/mask.json)")
    parser.add_argument("--prune", type=float, default=None, help="Percent or fraction (0..1) of neurons to prune. E.g., 20 or 0.2")
    parser.add_argument("--rounding", type=str, choices=["floor", "round", "ceil"], default="round", help="Rounding mode when computing K from percent")
    parser.add_argument("--per-block-k", type=int, default=None, help="Set exact K neurons per block (overrides --prune)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files, only print statistics")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    default_norm_dir = script_dir / "normalized"

    # Prepare aggregated sums
    aggregated_data: Dict[str, Any] | None = None
    aggregated_leaves: Dict[PathTuple, Dict[str, float]] | None = None

    if args.aggregated:
        # use precomputed sums
        agg_path = Path(args.aggregated)
        if not agg_path.exists():
            print(f"[error] aggregated file not found: {agg_path}")
            return
        aggregated_data = load_json(agg_path)
        # extract ij-leaves (in case the file was edited manually)
        aggregated_leaves = {p: leaf for p, leaf in find_leaf_ij_dicts(aggregated_data)}
        print(f"[info] loaded aggregated from: {agg_path} (leaf groups: {len(aggregated_leaves)})")
    else:
        # collect inputs and compute sums
        inputs = collect_files(default_norm_dir, args.pattern, args.files)
        if not inputs:
            print("[error] no input JSON files to aggregate.")
            return
        print(f"[info] aggregating {len(inputs)} file(s)...")
        aggregated_leaves = aggregate_leaves(inputs)
        print(f"[info] found {len(aggregated_leaves)} leaf group(s) with ij-keys.")
        aggregated_data = reconstruct_from_leaves(aggregated_leaves)

        if args.dry_run:
            # Simple stats
            total_keys = sum(len(d) for d in aggregated_leaves.values())
            blocks = set()
            for leaf in aggregated_leaves.values():
                blocks |= set(int(KEY_RE.match(k).group(1)) for k in leaf if KEY_RE.match(k))
            print(f"[dry] aggregated leaf keys total: {total_keys}; unique blocks seen: {sorted(blocks)}")
        else:
            out_path = Path(args.aggregate_out)
            dump_json_atomic(aggregated_data, out_path, compact=True)
            print(f"[ok] aggregated sums saved to: {out_path}")

    # Build mask if requested
    if (args.prune is not None) or (args.per_block_k is not None):
        if aggregated_leaves is None:
            aggregated_leaves = {p: leaf for p, leaf in find_leaf_ij_dicts(aggregated_data or {})}
        if not aggregated_leaves:
            print("[error] no ij-leaf groups found in aggregated data; cannot build mask.")
            return

        prune_fraction = 0.0 if args.per_block_k is not None and args.prune is None else parse_fraction(args.prune or 0.0)

        masks_by_path: Dict[PathTuple, Dict[str, int]] = {}
        per_leaf_stats: List[str] = []
        for path, leaf in aggregated_leaves.items():
            mask = make_mask_for_leaf(
                leaf,
                prune_fraction=prune_fraction,
                rounding=args.rounding,
                per_block_k=args.per_block_k,
            )
            masks_by_path[path] = mask
            # Per-leaf stats
            groups = build_block_groups(leaf)
            n_blocks = len(groups)
            n_total = sum(len(v) for v in groups.values())
            any_block = next(iter(groups.keys()))
            k_block = sum(v for k, v in mask.items() if k.startswith(f"{any_block}:"))
            # more accurate per-block K via explicit count
            k_block = sum(1 for k in groups[any_block] if mask[k[0]] == 1)
            per_leaf_stats.append(f"path={'/'.join(path) or '<root>'} blocks={n_blocks} total={n_total} K_per_block~{k_block}")

        mask_tree = apply_masks_to_leaves(aggregated_leaves, masks_by_path)

        if args.dry_run:
            print("[dry] mask would be saved to:", args.mask_out)
            for s in per_leaf_stats:
                print("[dry]", s)
        else:
            mask_out = Path(args.mask_out)
            dump_json_atomic(mask_tree, mask_out, compact=True)
            print(f"[ok] mask saved to: {mask_out}")
            for s in per_leaf_stats:
                print("[info]", s)


if __name__ == "__main__":
    main()
