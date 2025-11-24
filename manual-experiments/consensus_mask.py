#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Example:
#   python3 manual-experiments/consensus_mask.py --pattern "manual-experiments/normalized/has-scores.json" --pattern "manual-experiments/normalized/2ssp_vit_b16_ffn_importances.json" --prune 20
"""
Build a binary pruning mask (0/1) by consensus across multiple methods (files), without summation.

Concept:
- Input: multiple normalized JSON files with leaves shaped like {"i:j": score}, where
  i = MLP block index (0..11), j = neuron index within that block.
- Each file independently proposes pruning the bottom-k neurons per block (smallest values).
  We then take the intersection (consensus) of these proposed sets across all files per block.
- To reach a user-specified pruning fraction per block, we increase the internal selection fraction t
  for each file until the size of the intersection for every block reaches the common K_per_block.
- Output format matches manual-experiments/aggregate_and_mask-summation.py: leaves of {"i:j": 0/1},
  where 1 means "prune", 0 means "keep".

Notes:
- Equal number of pruned neurons in each of the 12 blocks is ensured by choosing a common
  K_per_block = min(round(p * N_i)) across all blocks, where N_i is the number of keys present
  in all files for block i (i.e., intersection of keys).
- Consensus uses only keys ("i:j") that appear in all files for a given block.
- If intersection cannot reach the target even at t=1.0 (e.g., due to key mismatches),
  that block will end up with fewer 1s than requested.

Output key ordering (for readability):
- Resulting mask dictionaries preserve a stable natural ordering by (block_index=i, neuron_index=j),
  i.e., numerically sorted by i then j, so keys are not visually shuffled.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

KEY_RE = re.compile(r"^(\d+):(\d+)$")

PathTuple = Tuple[str, ...]


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def looks_like_leaf_ij_dict(d: Dict[str, Any]) -> bool:
    """True if dict looks like a leaf with keys 'i:j' and numeric values."""
    if not isinstance(d, dict) or not d:
        return False
    for k, v in d.items():
        if not (isinstance(k, str) and KEY_RE.match(k)):
            return False
        if not is_number(v):
            return False
    return True


def find_leaf_ij_dicts(obj: Any, path: List[str] | None = None, out: List[Tuple[PathTuple, Dict[str, float]]] | None = None):
    """Collect all leaves with 'i:j' keys and numeric values anywhere in the JSON tree."""
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


def collect_files(default_dir: Path, patterns: List[str], files: List[str]) -> List[Path]:
    """Collect explicit files and glob patterns, defaulting to all *.json in default_dir."""
    collected: List[Path] = []
    for p in files:
        path = Path(p)
        if path.exists() and path.suffix.lower() == ".json":
            collected.append(path)
    for pat in patterns:
        for p in Path(".").glob(pat):
            if p.exists() and p.suffix.lower() == ".json":
                collected.append(p)
    if not collected:
        collected = sorted(default_dir.glob("*.json"))
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
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        if compact:
            json.dump(data, f, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        else:
            json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
    os.replace(tmp, out_path)


def parse_fraction(p: float) -> float:
    """Convert percent or fraction to [0,1]. If p > 1 treat as percentage (p/100)."""
    if p < 0:
        return 0.0
    return p / 100.0 if p > 1.0 else p


def rounding_fn(name: str):
    if name == "floor":
        return math.floor
    if name == "ceil":
        return math.ceil
    return lambda x: int(round(x))


def group_by_path(files: List[Path]) -> Dict[PathTuple, List[Dict[str, float]]]:
    """
    Build: path_tuple -> list of ij-leaves (one per file if present in that file).
    """
    bag: Dict[PathTuple, List[Dict[str, float]]] = {}
    for src in files:
        try:
            data = load_json(src)
        except Exception as e:
            print(f"[warn] skip {src}: {e}")
            continue
        leaves = find_leaf_ij_dicts(data)
        if not leaves:
            print(f"[info] {src}: no ij-leaves")
            continue
        for pth, leaf in leaves:
            bag.setdefault(pth, []).append(leaf)
    return bag


def split_by_block(leaf: Dict[str, float]) -> Dict[int, Dict[str, float]]:
    """Split a leaf {"i:j": val} into blocks i -> {"i:j": val}."""
    blocks: Dict[int, Dict[str, float]] = {}
    for k, v in leaf.items():
        m = KEY_RE.match(k)
        if not m:
            continue
        i = int(m.group(1))
        blocks.setdefault(i, {})[k] = float(v)
    return blocks


def key_to_tuple(k: str) -> Tuple[int, int]:
    """Convert 'i:j' -> (i, j) for natural numeric ordering."""
    m = KEY_RE.match(k)
    if not m:
        return (1 << 30, 1 << 30)
    return (int(m.group(1)), int(m.group(2)))


def consensus_for_path(
    leaves_for_files: List[Dict[str, float]],
    prune_fraction: float,
    rounding: str = "round",
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Build a per-path mask by consensus:
    - For each block i, compute the intersection of bottom-k key sets across all files.
    - Increase internal selection fraction t from prune_fraction up to 1.0 until
      |intersection_i| >= K_common for every block i, where
      K_common = min_i round(prune_fraction * N_i) and
      N_i = number of keys present in all files for block i.
    - If intersection_i exceeds K_common, pick exactly K_common keys with smallest mean value across files.
    Mask keys are inserted in stable numeric order by (i, j).
    """
    rfun = rounding_fn(rounding)

    # file_blocks[file_idx][block_i] = {"i:j": val, ...}
    per_file_blocks: List[Dict[int, Dict[str, float]]] = [split_by_block(leaf) for leaf in leaves_for_files]

    # All blocks seen in at least one file
    all_blocks = sorted(set().union(*[set(b.keys()) for b in per_file_blocks])) if per_file_blocks else []

    # Keys common to all files per block (use natural numeric order by j)
    keys_common: Dict[int, List[str]] = {}
    for i in all_blocks:
        sets = []
        for fb in per_file_blocks:
            keys_i = set(fb.get(i, {}).keys())
            sets.append(keys_i)
        common_set = set.intersection(*sets) if sets else set()
        keys_common[i] = sorted(common_set, key=key_to_tuple)

    # Compute targets
    N_per_block = {i: len(keys_common[i]) for i in all_blocks}
    if not N_per_block:
        return {}

    K_targets = {i: max(0, min(N_per_block[i], rfun(prune_fraction * N_per_block[i]))) for i in all_blocks}
    K_common = min(K_targets.values()) if K_targets else 0

    if verbose:
        print(f"[consensus] blocks={len(all_blocks)}; N_per_block[0]={N_per_block.get(all_blocks[0], 0) if all_blocks else 0}; K_target_common={K_common}")

    # If K_common == 0, return all zeros in stable order
    if K_common <= 0:
        mask: Dict[str, int] = {}
        for i in all_blocks:
            for key in sorted(keys_common[i], key=key_to_tuple):
                mask[key] = 0
        return mask

    # Compute intersection for a given t in [0,1]
    def intersection_for_t(t: float) -> Dict[int, List[str]]:
        inter: Dict[int, List[str]] = {}
        for i in all_blocks:
            keys_i = keys_common[i]
            n = len(keys_i)
            if n == 0:
                inter[i] = []
                continue
            k = max(0, min(n, rfun(t * n)))
            if k == 0:
                inter[i] = []
                continue
            # bottom-k for each file by values (restricted to keys_i)
            bottom_sets = []
            for fb in per_file_blocks:
                vals = fb.get(i, {})
                sorted_keys = sorted(keys_i, key=lambda kk: (vals.get(kk, float("inf")), key_to_tuple(kk)))
                bottom_sets.append(set(sorted_keys[:k]))
            inter_i = set.intersection(*bottom_sets) if bottom_sets else set()
            # keep stable numeric order
            inter[i] = sorted(inter_i, key=key_to_tuple)
        return inter

    # Increase t until min_i |intersection_i| >= K_common or t reaches 1.0
    t_low = max(0.0, prune_fraction)
    t = t_low
    inter = intersection_for_t(t)
    min_inter = min((len(v) for v in inter.values()), default=0)

    iters = 0
    while min_inter < K_common and t < 1.0 and iters < 100:
        t = min(1.0, t * 1.2 if t > 0 else 0.02)  # multiplicative growth for quicker convergence
        inter = intersection_for_t(t)
        min_inter = min((len(v) for v in inter.values()), default=0)
        iters += 1

    if verbose:
        print(f"[consensus] t_final={t:.4f}, min_intersection={min_inter}, K_common={K_common}, iters={iters}")

    # Build final mask in stable order
    mask: Dict[str, int] = {}
    for i in all_blocks:
        keys_i = keys_common[i]
        # initialize zeros in numeric order
        for key in sorted(keys_i, key=key_to_tuple):
            mask[key] = 0

        inter_keys = inter.get(i, [])
        if not inter_keys:
            continue

        if len(inter_keys) <= K_common:
            for key in inter_keys:
                mask[key] = 1
        else:
            # choose exactly K_common with smallest mean value across files (tie-breaker: key order)
            means: List[Tuple[str, float]] = []
            for key in inter_keys:
                vals = []
                for fb in per_file_blocks:
                    v = fb.get(i, {}).get(key, None)
                    vals.append(float(v) if v is not None else float("inf"))
                means.append((key, sum(vals) / max(1, len(vals))))
            means_sorted = sorted(means, key=lambda kv: (kv[1], key_to_tuple(kv[0])))
            chosen = {k for k, _ in means_sorted[:K_common]}
            for key in sorted(keys_i, key=key_to_tuple):
                if key in chosen:
                    mask[key] = 1
    return mask


def reconstruct_mask_tree(path_to_mask: Dict[PathTuple, Dict[str, int]]) -> Dict[str, Any]:
    """Assemble a tree of masks from per-path masks. Insertion order is preserved."""
    root: Dict[str, Any] = {}
    for path, leaf_mask in path_to_mask.items():
        cur = root
        for key in path:
            cur = cur.setdefault(key, {})
        # leaf_mask insertion order already sorted; update preserves it in Python 3.7+
        cur.update(leaf_mask)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consensus-based pruning mask (equal-per-block) from multiple normalized JSON files.")
    parser.add_argument("files", nargs="*", help="Input JSON files with normalized values (you can list multiple)")
    parser.add_argument("--pattern", action="append", default=[], help="Glob pattern(s), e.g. 'manual-experiments/normalized/*.json'")
    parser.add_argument("--prune", type=float, required=True, help="Percent or fraction (0..1) of neurons per block to prune (target equal K across blocks)")
    parser.add_argument("--rounding", type=str, choices=["floor", "round", "ceil"], default="round", help="Rounding for K computation")
    parser.add_argument("--mask-out", type=str, default="manual-experiments/mask_consensus.json", help="Output path for the consensus mask")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    default_norm_dir = script_dir / "normalized"

    inputs = collect_files(default_norm_dir, args.pattern, args.files)
    if not inputs:
        print("[error] no input JSON files")
        return
    print(f"[info] using {len(inputs)} file(s)")

    # Group by path and keep only paths present in at least two files (consensus has meaning then)
    paths_bag = group_by_path(inputs)
    common_paths = {pth: leaves for pth, leaves in paths_bag.items() if len(leaves) >= 2}
    if not common_paths:
        print("[error] no common paths with >=2 files having ij-leaves")
        return

    prune_fraction = parse_fraction(args.prune)
    result_masks: Dict[PathTuple, Dict[str, int]] = {}
    total_ones = 0
    for pth, leaves in common_paths.items():
        mask_leaf = consensus_for_path(leaves, prune_fraction=prune_fraction, rounding=args.rounding, verbose=True)
        result_masks[pth] = mask_leaf
        total_ones += sum(mask_leaf.values())

    mask_tree = reconstruct_mask_tree(result_masks)

    if args.dry_run:
        print("[dry] consensus mask would be saved to:", args.mask_out)
        print(f"[dry] total ones (global) = {total_ones}")
        first_path = next(iter(result_masks.keys()))
        k_sample = sum(result_masks[first_path].values())
        print(f"[dry] sample path={'/'.join(first_path) or '<root>'} ones={k_sample}")
    else:
        out = Path(args.mask_out)
        dump_json_atomic(mask_tree, out, compact=True)
        print(f"[ok] consensus mask saved to: {out}")
        print(f"[ok] total ones (global) = {total_ones}")


if __name__ == "__main__":
    main()
