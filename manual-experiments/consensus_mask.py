#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python3 manual-experiments/consensus_mask.py --pattern "manual-experiments/normalized/has-scores.json" --pattern "manual-experiments/normalized/2ssp_vit_b16_ffn_importances.json" --prune 20
"""
Строит бинарную маску (0/1) по консенсусу нескольких методов (файлов) без суммирования:
- Вход: несколько нормализованных JSON-файлов с листами вида {"i:j": score}, где i — MLP-блок (0..11), j — индекс нейрона.
- Идея: каждый файл по-своему "предлагает" удалить K наименьших нейронов в каждом блоке (по своим значениям).
  Затем берём пересечение (consensus) предложений всех файлов по каждому блоку.
- Чтобы добиться пользовательской доли/процента удаления, мы увеличиваем внутреннюю долю отбора в каждом файле,
  пока размер пересечения по каждому блоку не достигнет целевого равного K_per_block.
- На выходе: маска того же формата, что и у aggregate_and_mask.py: словарь листов "i:j" -> 0/1, где 1 означает "удалить".

Примеры:
  - Построить консенсусную маску на 20%:
      python3 manual-experiments/consensus_mask.py --pattern "manual-experiments/normalized/*.json" --prune 20

  - Несколько файлов явно:
      python3 manual-experiments/consensus_mask.py file1.json file2.json --prune 0.2

  - Статистика без записи:
      python3 manual-experiments/consensus_mask.py --pattern "manual-experiments/normalized/*.json" --prune 15 --dry-run

Замечания:
- Равное число урезаний в каждом из 12 блоков обеспечивается выбором K_per_block = min(round(p*N_i)) по всем блокам.
- Консенсус считается только по тем ключам "i:j", которые присутствуют во всех файлах (по каждому блоку отдельно).
- Если пересечение невозможно достичь даже при внутреннем t=1.0 (например, из-за несовпадения ключей), блок получит меньше 1-иц.
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
    if not isinstance(d, dict) or not d:
        return False
    for k, v in d.items():
        if not (isinstance(k, str) and KEY_RE.match(k)):
            return False
        if not is_number(v):
            return False
    return True


def find_leaf_ij_dicts(obj: Any, path: List[str] | None = None, out: List[Tuple[PathTuple, Dict[str, float]]] | None = None):
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        if compact:
            json.dump(data, f, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        else:
            json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
    os.replace(tmp, out_path)


def parse_fraction(p: float) -> float:
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
    Возвращает: path_tuple -> список листов (по одному на файл, если у файла такой лист есть).
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
    """
    Разбивает лист {"i:j": val} по блокам i -> { "i:j": val }.
    """
    blocks: Dict[int, Dict[str, float]] = {}
    for k, v in leaf.items():
        m = KEY_RE.match(k)
        if not m:
            continue
        i = int(m.group(1))
        blocks.setdefault(i, {})[k] = float(v)
    return blocks


def consensus_for_path(
    leaves_for_files: List[Dict[str, float]],
    prune_fraction: float,
    rounding: str = "round",
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Для одного path_tuple строит маску по консенсусу:
    - Для каждого блока i найдём пересечение bottom_k множеств ключей по всем файлам.
    - Внутреннюю долю t увеличиваем от prune_fraction до 1.0, пока |intersection| по каждому блоку
      не достигнет K_common = min_i round(prune_fraction * N_i), где N_i — количество ключей,
      присутствующих во всех файлах для блока i.
    - Если intersection по блоку превышает K_common, берём K_common с наименьшим средним значением по файлам.
    """
    rfun = rounding_fn(rounding)

    # Разложим каждый файл на блоки: file_blocks[file_index][block_i] = {"i:j": val, ...}
    per_file_blocks: List[Dict[int, Dict[str, float]]] = [split_by_block(leaf) for leaf in leaves_for_files]

    # Определим множество блоков, которые есть хотя бы у одного файла
    all_blocks = sorted(set().union(*[set(b.keys()) for b in per_file_blocks])) if per_file_blocks else []

    # Для консенсуса используем только ключи, присутствующие во всех файлах (по каждому блоку отдельно)
    # keys_common[i] = set("i:j") присутствующих во всех файлах
    keys_common: Dict[int, List[str]] = {}
    for i in all_blocks:
        sets = []
        for fb in per_file_blocks:
            keys_i = set(fb.get(i, {}).keys())
            sets.append(keys_i)
        if not sets:
            common_set = set()
        else:
            common_set = set.intersection(*sets) if sets else set()
        keys_common[i] = sorted(common_set)

    # Рассчитаем K_target_i и общий K_common для равенства на всех блоках
    N_per_block = {i: len(keys_common[i]) for i in all_blocks}
    if not N_per_block:
        return {}

    K_targets = {i: max(0, min(N_per_block[i], rfun(prune_fraction * N_per_block[i]))) for i in all_blocks}
    K_common = min(K_targets.values()) if K_targets else 0

    if verbose:
        print(f"[consensus] blocks={len(all_blocks)}; N_per_block[0]={N_per_block.get(all_blocks[0], 0) if all_blocks else 0}; K_target_common={K_common}")

    # Если K_common == 0, просто маска из нулей
    if K_common <= 0:
        mask = {}
        for i in all_blocks:
            for key in keys_common[i]:
                mask[key] = 0
        return mask

    # Функция, возвращающая пересечение bottom_k множеств для t в [0,1]
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
            # bottom_k для каждого файла по значениям
            bottom_sets = []
            for fb in per_file_blocks:
                vals = fb.get(i, {})
                # сортируем только по keys_i, чтобы все файлы имели одинаковое множество для сравнения
                sorted_keys = sorted(keys_i, key=lambda kk: vals.get(kk, float("inf")))
                bottom_sets.append(set(sorted_keys[:k]))
            inter_i = set.intersection(*bottom_sets) if bottom_sets else set()
            inter[i] = sorted(inter_i)
        return inter

    # Увеличиваем t пока min_i |intersection_i| >= K_common или пока t < 1.0
    t_low = max(0.0, prune_fraction)
    t = t_low
    inter = intersection_for_t(t)
    min_inter = min((len(v) for v in inter.values()), default=0)

    iters = 0
    while min_inter < K_common and t < 1.0 and iters < 100:
        # Увеличиваем t мультипликативно (быстрее сходится, чем +step)
        t = min(1.0, t * 1.2 if t > 0 else 0.02)
        inter = intersection_for_t(t)
        min_inter = min((len(v) for v in inter.values()), default=0)
        iters += 1

    if verbose:
        print(f"[consensus] t_final={t:.4f}, min_intersection={min_inter}, K_common={K_common}, iters={iters}")

    # Построим итоговую маску: если intersection_i >= K_common, берём K_common по наименьшему среднему
    mask: Dict[str, int] = {}
    for i in all_blocks:
        keys_i = keys_common[i]
        # инициализация нулями
        for key in keys_i:
            mask[key] = 0

        inter_keys = inter.get(i, [])
        if not inter_keys:
            continue

        if len(inter_keys) <= K_common:
            for key in inter_keys:
                mask[key] = 1
        else:
            # выбрать K_common с наименьшим средним значением по файлам
            means: List[Tuple[str, float]] = []
            for key in inter_keys:
                vals = []
                for fb in per_file_blocks:
                    v = fb.get(i, {}).get(key, None)
                    if v is None:
                        # теоретически не должно случиться (мы отфильтровали keys_common), но защитимся
                        v = float("inf")
                    vals.append(float(v))
                means.append((key, sum(vals) / max(1, len(vals))))
            means_sorted = sorted(means, key=lambda kv: kv[1])
            chosen = {k for k, _ in means_sorted[:K_common]}
            for key in keys_i:
                if key in chosen:
                    mask[key] = 1
    return mask


def reconstruct_mask_tree(path_to_mask: Dict[PathTuple, Dict[str, int]]) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    for path, leaf_mask in path_to_mask.items():
        cur = root
        for key in path:
            cur = cur.setdefault(key, {})
        cur.update(leaf_mask)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consensus-based pruning mask (equal-per-block) from multiple normalized JSON files.")
    parser.add_argument("files", nargs="*", help="Входные JSON-файлы с нормализованными значениями (можно перечислить несколько)")
    parser.add_argument("--pattern", action="append", default=[], help="Глоб-шаблон(ы), например 'manual-experiments/normalized/*.json'")
    parser.add_argument("--prune", type=float, required=True, help="Процент или доля (0..1) нейронов на блок к удалению (целевой равный K)")
    parser.add_argument("--rounding", type=str, choices=["floor", "round", "ceil"], default="round", help="Округление при расчёте K")
    parser.add_argument("--mask-out", type=str, default="manual-experiments/mask_consensus.json", help="Куда сохранить маску (по умолчанию manual-experiments/mask_consensus.json)")
    parser.add_argument("--dry-run", action="store_true", help="Только показать статистику без записи")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    default_norm_dir = script_dir / "normalized"

    inputs = collect_files(default_norm_dir, args.pattern, args.files)
    if not inputs:
        print("[error] no input JSON files")
        return
    print(f"[info] using {len(inputs)} file(s)")

    # Сгруппируем по path и возьмем только те пути, что присутствуют хотя бы в двух файлах (консенсус имеет смысл)
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
        # Простейшая сводка по первому path
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
