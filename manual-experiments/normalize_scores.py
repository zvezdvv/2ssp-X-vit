#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#python3 manual-experiments/normalize_scores.py
"""
Normalize numeric importance scores in JSON files to [0, 1] without changing the file structure.
- All numeric values are transformed as: v_norm = (v - min_val) / (max_val - min_val)
- If all values are equal (max_val == min_val), normalized values become 0.0
- Non-numeric values (including booleans) are preserved as-is
- Nested structures (dicts/lists) are preserved exactly in shape and key order
- By default, normalized files are written to manual-experiments/normalized/<same-name>.json
- Use --inplace to overwrite originals (a .bak is created unless --no-backup)

Examples:
  - Normalize all JSONs in manual-experiments to a separate folder:
      python manual-experiments/normalize_scores.py

  - Overwrite in place (with backup .bak):
      python manual-experiments/normalize_scores.py --inplace

  - Specific files or patterns:
      python manual-experiments/normalize_scores.py manual-experiments/snp_scores.json
      python manual-experiments/normalize_scores.py --pattern "manual-experiments/*.json"

  - Dry run (just show min/max and targets, no writes):
      python manual-experiments/normalize_scores.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Tuple


def is_number(x: Any) -> bool:
    """True for int/float but not bool (bool is a subclass of int in Python)."""
    return (isinstance(x, (int, float)) and not isinstance(x, bool))


def scan_min_max_raw(obj: Any) -> Tuple[float | None, float | None]:
    """Traverse nested JSON-like structure and compute (min_val, max_val) across all numbers (raw values)."""
    min_val = math.inf
    max_val = -math.inf

    stack: list[Any] = [obj]
    while stack:
        cur = stack.pop()
        if is_number(cur):
            v = float(cur)
            if v < min_val:
                min_val = v
            if v > max_val:
                max_val = v
        elif isinstance(cur, list):
            stack.extend(cur)
        elif isinstance(cur, dict):
            stack.extend(cur.values())
        # else: ignore None, str, bool, etc.

    if min_val is math.inf:
        return None, None
    return float(min_val), float(max_val)


def normalize_value(v: float, min_val: float, max_val: float) -> float:
    """Raw min-max normalize to [0,1]."""
    if max_val == min_val:
        return 0.0
    return (float(v) - min_val) / (max_val - min_val)


def normalize_structure(obj: Any, min_val: float, max_val: float) -> Any:
    """Return a new structure with all numeric values normalized, preserving shapes/keys."""
    if is_number(obj):
        return normalize_value(obj, min_val, max_val)
    if isinstance(obj, list):
        return [normalize_structure(x, min_val, max_val) for x in obj]
    if isinstance(obj, dict):
        # dict order preserved by default in Python 3.7+
        return {k: normalize_structure(v, min_val, max_val) for k, v in obj.items()}
    return obj  # str, bool, None, etc.


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json_atomic(data: Any, out_path: Path) -> None:
    """Write JSON compactly and atomically to out_path."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        # compact separators keep size similar to typical machine-generated JSON
        json.dump(data, f, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    os.replace(tmp_path, out_path)


def process_file(src: Path, dst: Path, dry_run: bool = False) -> None:
    data = load_json(src)
    min_val, max_val = scan_min_max_raw(data)
    if min_val is None:
        print(f"[skip] {src} — no numeric values found")
        return

    if dry_run:
        print(f"[dry]  {src}  min_val={min_val:.6g}, max_val={max_val:.6g}  => {dst}")
        return

    normalized = normalize_structure(data, min_val, max_val)
    dump_json_atomic(normalized, dst)
    print(f"[ok]   {src}  ->  {dst}  (min_val={min_val:.6g}, max_val={max_val:.6g})")


def collect_files(script_dir: Path, patterns: list[str], files: list[str]) -> list[Path]:
    collected: list[Path] = []

    # explicit files first
    for p in files:
        path = Path(p)
        if path.exists() and path.suffix.lower() == ".json":
            collected.append(path)

    # then glob patterns (relative to current working directory)
    for pat in patterns:
        for p in Path(".").glob(pat):
            if p.exists() and p.suffix.lower() == ".json":
                collected.append(p)

    # default: all *.json in script directory if none provided
    if not collected:
        collected = sorted(script_dir.glob("*.json"))

    # uniq while preserving order
    seen = set()
    unique: list[Path] = []
    for p in collected:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize numeric importance scores in JSON files to [0,1] using raw min-max scaling.")
    parser.add_argument("files", nargs="*", help="Explicit JSON files to process")
    parser.add_argument("--pattern", action="append", default=[], help="Glob pattern(s), e.g. 'manual-experiments/*.json'")
    parser.add_argument("--inplace", action="store_true", help="Overwrite original files in place")
    parser.add_argument("--no-backup", action="store_true", help="Do not create .bak backups when using --inplace")
    parser.add_argument("--output-dir", type=str, default=None, help="Dir for outputs (default: manual-experiments/normalized)")
    parser.add_argument("--dry-run", action="store_true", help="Only show min/max and targets without writing")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    targets = collect_files(script_dir, args.pattern, args.files)

    # Filter only jsons and existing paths (defensive)
    targets = [p for p in targets if p.exists() and p.suffix.lower() == ".json"]

    if not targets:
        print("No JSON files found to process.")
        return

    if args.inplace:
        for src in targets:
            if not args.no_backup:
                bak = src.with_suffix(src.suffix + ".bak")
                if not bak.exists():
                    try:
                        shutil.copy2(src, bak)
                    except Exception as e:
                        print(f"[warn] Could not create backup for {src}: {e}")
            dst = src
            process_file(src, dst, dry_run=args.dry_run)
    else:
        out_dir = Path(args.output_dir) if args.output_dir else (script_dir / "normalized")
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in targets:
            dst = out_dir / src.name
            process_file(src, dst, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
