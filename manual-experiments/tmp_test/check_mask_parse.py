#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
import re
from typing import Any, Dict, List, Tuple

KEY_RE = re.compile(r"^(\d+):(\d+)$")


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def looks_like_leaf_ij_dict(d: Dict[str, Any]) -> bool:
    if not isinstance(d, dict) or not d:
        return False
    for k, v in d.items():
        if not (isinstance(k, str) and KEY_RE.match(k)):
            return False
        if not is_number(v):
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


def parse_fraction(p: float) -> float:
    if p < 0:
        return 0.0
    return p / 100.0 if p > 1.0 else p


def main():
    if len(sys.argv) < 3:
        print("Usage: check_mask_parse.py MASK_PATH MAX_BLOCK_FRAC(0..1 or percent)")
        sys.exit(2)
    mask_path = sys.argv[1]
    try:
        max_block_frac = float(sys.argv[2])
    except Exception:
        print("MAX_BLOCK_FRAC must be a number")
        sys.exit(2)

    max_block_frac = parse_fraction(max_block_frac)
    # clamp
    max_block_frac = max(0.0, min(1.0, max_block_frac))

    with open(mask_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    leaves = find_leaf_ij_dicts(data)
    if not leaves:
        print("[FAIL] No ij-leaf dicts found in mask")
        sys.exit(1)

    # Aggregate across all leaves
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

    print(f"[INFO] Found {len(blocks)} blocks in mask")
    total_N = 0
    total_K = 0
    total_caps = 0
    ok = True
    for i in sorted(blocks.keys()):
        bm = blocks[i]
        N_i = len(bm)
        K_i = sum(1 for b in bm.values() if b == 1)
        cap_i = int(round(max_block_frac * N_i))
        total_N += N_i
        total_K += K_i
        total_caps += cap_i
        cond = K_i <= cap_i
        state = "OK" if cond else "EXCEEDS_CAP"
        print(f"[BLOCK {i}] N={N_i} ones(K)={K_i} cap={cap_i} status={state}")
        if not cond:
            ok = False

    print(f"[TOTAL] N={total_N} K={total_K} sum_caps={total_caps} K<=sum_caps? {'YES' if total_K <= total_caps else 'NO'}")

    if ok and total_K <= total_caps:
        print("[PASS] Mask format and per-block cap constraints look valid for apply_mask_prune loader.")
        sys.exit(0)
    else:
        print("[FAIL] Mask violates per-block cap constraints.")
        sys.exit(1)


if __name__ == "__main__":
    main()
