#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python3 manual-experiments/aggregate_and_mask-summation.py --pattern "manual-experiments/normalized/has-scores.json" --pattern "manual-experiments/normalized/2ssp_vit_b16_ffn_importances.json" --prune 20
"""
Скрипт агрегирует (поэлементно суммирует) значения из указанных нормализованных JSON-файлов
и по суммам строит бинарную маску (0/1) для урезания MLP-нейронов.

Основные принципы:
- Ожидаются листовые словари вида {"i:j": number}, где i = индекс MLP-блока (0..11), j = индекс нейрона в блоке.
- Агрегирование: по одинаковым ключам "i:j" суммы складываются. Если ключ отсутствует в каком-то файле, он трактуется как 0.
- Структура JSON восстанавливается только для листов с ключами "i:j". Не-листовые части (метаданные и пр.) не влияют на сумму.
- Маска: пользователь задаёт процент параметров для урезания (например, 20 или 0.2).
  Для каждого MLP-блока выбираются K наименьших суммарных значений и помечаются 1 (урезать), остальные 0 (сохранить).
  K одинаков для всех 12 блоков (равное количество урезаний из каждого блока).
  По умолчанию K вычисляется как округление процента от размера блока и затем
  приводится к общему K = min(K_i) по всем блокам, чтобы не превысить ни один блок (равенство количества).
  Можно задать K напрямую флагом --per-block-k.

Примеры:
  1) Только агрегировать несколько файлов (шаблон по умолчанию возьмёт все *.json из manual-experiments/normalized):
     python3 manual-experiments/aggregate_and_mask-summation.py --pattern "manual-experiments/normalized/*.json"

  2) Агрегировать и сразу построить маску на 20%:
     python3 manual-experiments/aggregate_and_mask-summation.py --pattern "manual-experiments/normalized/*.json" --prune 20

  3) Построить маску из уже готовых сумм:
     python3 manual-experiments/aggregate_and_mask-summation.py --aggregated manual-experiments/aggregated_sums.json --prune 0.2

  4) Задать ровно K нейронов на блок (например, 128), игнорируя процент:
     python3 manual-experiments/aggregate_and_mask-summation.py --pattern "manual-experiments/normalized/*.json" --per-block-k 128

Выходы по умолчанию:
  - Суммы: manual-experiments/aggregated_sums.json
  - Маска: manual-experiments/mask.json
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


PathTuple = Tuple[str, ...]


def find_leaf_ij_dicts(obj: Any, path: List[str] | None = None, out: List[Tuple[PathTuple, Dict[str, float]]] | None = None):
    """
    Находит все словари-листы, где ключи "i:j" и значения числа.
    Возвращает список кортежей (path_tuple, dict[str,float]).
    """
    if path is None:
        path = []
    if out is None:
        out = []

    if isinstance(obj, dict):
        if looks_like_leaf_ij_dict(obj):
            # гарантируем float
            leaf = {k: float(v) for k, v in obj.items()}
            out.append((tuple(path), leaf))
            return out
        # иначе углубляемся
        for k, v in obj.items():
            find_leaf_ij_dicts(v, path + [str(k)], out)
    elif isinstance(obj, list):
        # Списки поддерживаем, но ключом пути станет строка "[i]"
        for i, v in enumerate(obj):
            find_leaf_ij_dicts(v, path + [f"[{i}]"], out)
    return out


def collect_files(default_glob_dir: Path, patterns: List[str], files: List[str]) -> List[Path]:
    collected: List[Path] = []

    # явные файлы
    for p in files:
        path = Path(p)
        if path.exists() and path.suffix.lower() == ".json":
            collected.append(path)

    # шаблоны
    for pat in patterns:
        for p in Path(".").glob(pat):
            if p.exists() and p.suffix.lower() == ".json":
                collected.append(p)

    # по умолчанию — все *.json в normalized
    if not collected:
        default = sorted((default_glob_dir).glob("*.json"))
        collected.extend(default)

    # уникализируем по realpath
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
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        if compact:
            json.dump(data, f, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        else:
            json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
    os.replace(tmp_path, out_path)


def aggregate_leaves(files: List[Path]) -> Dict[PathTuple, Dict[str, float]]:
    """
    Суммирует все листовые ij-словари по пути.
    Возвращает словарь: path_tuple -> {"i:j": sum}.
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
    Собирает древовидный JSON только из листов ij-словари (без прочей обвязки).
    """
    root: Dict[str, Any] = {}
    for path, leaf in leaves.items():
        cur = root
        for key in path:
            if key.startswith("[") and key.endswith("]"):
                # для списка создадим словарь с ключом как строка индекса (редкий кейс)
                key = key
            cur = cur.setdefault(key, {})
        cur.update(leaf)
    return root


def parse_fraction(p: float) -> float:
    """
    Превращает пользовательский ввод процента или доли в [0,1].
    Если p > 1, трактуем как проценты (p/100).
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
    Группирует пары ("i:j", value) по i (индекс блока).
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
    Строит бинарную маску для одного ij-листа:
      - находит группы по блокам i
      - считает K_i = round(prune_fraction * N_i)
      - приводит к общему K = min_i K_i (равное K для всех блоков)
      - в каждом блоке выбирает K наименьших значений -> 1, остальные -> 0
    Если задано per_block_k — он используется напрямую как K (с проверкой на длину блока).
    """
    groups = build_block_groups(leaf)
    if not groups:
        return {k: 0 for k in leaf.keys()}

    # Проверим ровно 12 блоков (необязательно, но предупредим)
    unique_blocks = sorted(groups.keys())
    if len(unique_blocks) != 12:
        print(f"[warn] leaf has {len(unique_blocks)} block(s), expected 12. Proceeding anyway: {unique_blocks}")

    # Считаем K
    if per_block_k is None:
        rfun = rounding_fn(rounding)
        k_candidates = []
        for i, items in groups.items():
            n_i = len(items)
            k_i = max(0, min(n_i, rfun(prune_fraction * n_i)))
            k_candidates.append(k_i)
        common_k = min(k_candidates) if k_candidates else 0
    else:
        # фиксированное K
        common_k = max(0, per_block_k)

    # Строим маску
    mask: Dict[str, int] = {}
    for i, items in groups.items():
        # сортируем по возрастанию сумм
        items_sorted = sorted(items, key=lambda kv: kv[1])
        k_i = min(common_k, len(items_sorted))
        pruned_keys = {k for (k, _) in items_sorted[:k_i]}
        for k, _v in items_sorted:
            mask[k] = 1 if k in pruned_keys else 0
    return mask


def apply_masks_to_leaves(aggregated_leaves: Dict[PathTuple, Dict[str, float]], masks_by_path: Dict[PathTuple, Dict[str, int]]) -> Dict[str, Any]:
    """
    Собирает дерево масок из масок по листам.
    """
    root: Dict[str, Any] = {}
    for path, m in masks_by_path.items():
        cur = root
        for key in path:
            cur = cur.setdefault(key, {})
        cur.update(m)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate normalized JSON scores and build equal-per-block pruning masks for 12 MLP blocks.")
    parser.add_argument("files", nargs="*", help="JSON-файлы для суммирования (можно не указывать, использовать --pattern)")
    parser.add_argument("--pattern", action="append", default=[], help="Глоб-шаблон(ы), например 'manual-experiments/normalized/*.json'")
    parser.add_argument("--aggregated", type=str, default=None, help="Путь к уже посчитанным суммам (если хотим строить маску без пересчёта сумм)")
    parser.add_argument("--aggregate-out", type=str, default="manual-experiments/aggregated_sums.json", help="Куда сохранить суммы (по умолчанию manual-experiments/aggregated_sums.json)")
    parser.add_argument("--mask-out", type=str, default="manual-experiments/mask.json", help="Куда сохранить маску (по умолчанию manual-experiments/mask.json)")
    parser.add_argument("--prune", type=float, default=None, help="Процент или доля (0..1) параметров для урезания. Например, 20 или 0.2")
    parser.add_argument("--rounding", type=str, choices=["floor", "round", "ceil"], default="round", help="Округление при расчёте K из процента")
    parser.add_argument("--per-block-k", type=int, default=None, help="Прямо задать K нейронов на блок (перебивает --prune)")
    parser.add_argument("--dry-run", action="store_true", help="Не записывать файлы, только показать статистику")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    default_norm_dir = script_dir / "normalized"

    # Подготовим агрегированные суммы
    aggregated_data: Dict[str, Any] | None = None
    aggregated_leaves: Dict[PathTuple, Dict[str, float]] | None = None

    if args.aggregated:
        # использовать готовые суммы
        agg_path = Path(args.aggregated)
        if not agg_path.exists():
            print(f"[error] aggregated file not found: {agg_path}")
            return
        aggregated_data = load_json(agg_path)
        # выделим листья из агрегированного файла (на случай если он руками правился)
        aggregated_leaves = {p: leaf for p, leaf in find_leaf_ij_dicts(aggregated_data)}
        print(f"[info] loaded aggregated from: {agg_path} (leaf groups: {len(aggregated_leaves)})")
    else:
        # соберём входы и посчитаем суммы
        inputs = collect_files(default_norm_dir, args.pattern, args.files)
        if not inputs:
            print("[error] no input JSON files to aggregate.")
            return
        print(f"[info] aggregating {len(inputs)} file(s)...")
        aggregated_leaves = aggregate_leaves(inputs)
        print(f"[info] found {len(aggregated_leaves)} leaf group(s) with ij-keys.")
        aggregated_data = reconstruct_from_leaves(aggregated_leaves)

        if args.dry_run:
            # Печать краткой статистики
            total_keys = sum(len(d) for d in aggregated_leaves.values())
            blocks = set()
            for leaf in aggregated_leaves.values():
                blocks |= set(int(KEY_RE.match(k).group(1)) for k in leaf if KEY_RE.match(k))
            print(f"[dry] aggregated leaf keys total: {total_keys}; unique blocks seen: {sorted(blocks)}")
        else:
            out_path = Path(args.aggregate_out)
            dump_json_atomic(aggregated_data, out_path, compact=True)
            print(f"[ok] aggregated sums saved to: {out_path}")

    # Генерация маски при необходимости
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
            # статистика по листу
            groups = build_block_groups(leaf)
            n_blocks = len(groups)
            n_total = sum(len(v) for v in groups.values())
            # все блоки должны получить одинаковый K; берём любой и проверим
            any_block = next(iter(groups.keys()))
            k_block = sum(v for k, v in mask.items() if k.startswith(f"{any_block}:"))
            # Но это может дать не точный K, лучше пересчитать по блоку явно
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
