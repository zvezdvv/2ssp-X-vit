#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python3 manual-experiments/run_consensus_grid.py --sizes 2,3,4 --prune-levels 5,10,15,20,25,30,35,40,45,50,55,60,65,70
"""
Оркестратор последовательного запуска для consensus-подхода:
1) manual-experiments/consensus_mask.py (построение консенсусной маски)
2) experiments/vit_pruning/apply_mask_prune.py (применение маски, измерение latency/accuracy и печать [SUMMARY])

Требования:
- Перебрать все комбинации из 4 файлов: только пары, тройки и четвёрка (одиночки НЕ запускать).
- Для каждой комбинации прогнать --prune в {5,10,15,...,70}.
- Запуск строго последовательно (никакого параллелизма).
- Сохранять итоговые метрики (особенно: методы (=комбинация), sparsity (=prune), latency, accuracy).
- Формат results.csv такой же, как раньше (summation), но НЕ затирать старый файл.
  Для этого пишем в отдельную папку:
    final-results-for-presentation/consensus/results.csv
- Выходная маска consensus_mask.py: manual-experiments/mask_consensus.json
  Поэтому в apply_mask_prune.py передаём --mask manual-experiments/mask_consensus.json

Запуск (полный):
  python3 manual-experiments/run_consensus_grid.py

Запуск (мини-прогон для проверки: только пары и prune=5,10, ограничить 1 первую комбинацию):
  python3 manual-experiments/run_consensus_grid.py --sizes 2 --prune-levels 5,10 --first-n-combos 1

Опции CLI:
  --sizes "2,3,4"         — какие размеры сочетаний брать (по умолчанию 2,3,4)
  --prune-levels "5,10"   — перечисление уровней sparsity (если не задано — 5..70 шаг 5)
  --first-n-combos N      — ограничить число первых комбинаций (после сортировки) для быстрого теста
  --no-resume             — не пропускать уже завершённые (ok) прогоны

CSV-колонки:
  methods, prune,
  params_before_stage1, params_after_stage1,
  params_before_stage1_millions, params_after_stage1_millions,
  stage1_reduction_percent,
  latency_baseline_ms, latency_stage1_ms, latency_stage1_change_percent,
  acc_baseline, acc_stage1, acc_drop_stage1_percent,
  status
"""

from __future__ import annotations

import argparse
import itertools
import json
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

# Константы путей
ROOT = Path(__file__).resolve().parents[1]
AGG_SCRIPT = ROOT / "manual-experiments" / "consensus_mask.py"
APPLY_SCRIPT = ROOT / "experiments" / "vit_pruning" / "apply_mask_prune.py"
MASK_PATH = ROOT / "manual-experiments" / "mask_consensus.json"

# Входные нормализованные JSON-файлы (ровно те 4, о которых просил пользователь)
SCORE_FILES: List[Path] = [
    ROOT / "manual-experiments" / "normalized" / "2ssp_vit_b16_ffn_importances.json",
    ROOT / "manual-experiments" / "normalized" / "has-scores.json",
    ROOT / "manual-experiments" / "normalized" / "pablos-method.json",
    ROOT / "manual-experiments" / "normalized" / "snp_scores.json",
]

# Диапазон sparsity (в процентах) по умолчанию
DEFAULT_PRUNE_LEVELS: List[int] = list(range(5, 71, 5))

# Куда писать сводные результаты и логи (без временных меток)
OUT_DIR = ROOT / "final-results-for-presentation" / "consensus"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = OUT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "results.csv"

SUMMARY_MARK_RE = re.compile(r"\[SUMMARY\]\s*\n(\{.*?\})", re.S)


def check_prerequisites() -> None:
    if not AGG_SCRIPT.exists():
        raise FileNotFoundError(f"Consensus script not found: {AGG_SCRIPT}")
    if not APPLY_SCRIPT.exists():
        raise FileNotFoundError(f"Apply script not found: {APPLY_SCRIPT}")
    missing = [str(p) for p in SCORE_FILES if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input normalized score files:\n  " + "\n  ".join(missing))


def stem(p: Path) -> str:
    return p.stem


def sanitize_token(token: str) -> str:
    """
    Сделать токен безопасным для имен файлов (оставляем буквы, цифры, _-+).
    """
    return re.sub(r"[^A-Za-z0-9_\-\+]+", "_", token)


def combo_key(files: Sequence[Path]) -> str:
    """
    Ключ комбинации: соединение stem'ов отсортированного набора через '+'.
    """
    stems = sorted(stem(p) for p in files)
    return "+".join(stems)


def ensure_csv_header(path: Path) -> None:
    if path.exists():
        return
    header = [
        "methods",
        "prune",
        "params_before_stage1",
        "params_after_stage1",
        "params_before_stage1_millions",
        "params_after_stage1_millions",
        "stage1_reduction_percent",
        "latency_baseline_ms",
        "latency_stage1_ms",
        "latency_stage1_change_percent",
        "acc_baseline",
        "acc_stage1",
        "acc_drop_stage1_percent",
        "status",
    ]
    path.write_text(",".join(header) + "\n", encoding="utf-8")


def load_completed_ok(path: Path) -> Set[Tuple[str, int]]:
    """
    Считывает CSV и возвращает множество пар (methods, prune) со статусом ok,
    чтобы избегать повторных прогонов (resume).
    """
    done: Set[Tuple[str, int]] = set()
    if not path.exists():
        return done
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    methods = (row.get("methods", "") or "").strip()
                    prune_s = (row.get("prune", "") or "").strip()
                    status = (row.get("status", "") or "").strip()
                    if not methods or not prune_s:
                        continue
                    prune = int(prune_s)
                    if status == "ok":
                        done.add((methods, prune))
                except Exception:
                    continue
    except Exception:
        return done
    return done


def append_csv_row(path: Path, row: Dict[str, object]) -> None:
    # Пишем в CSV в стабильном порядке колонок
    cols = [
        "methods",
        "prune",
        "params_before_stage1",
        "params_after_stage1",
        "params_before_stage1_millions",
        "params_after_stage1_millions",
        "stage1_reduction_percent",
        "latency_baseline_ms",
        "latency_stage1_ms",
        "latency_stage1_change_percent",
        "acc_baseline",
        "acc_stage1",
        "acc_drop_stage1_percent",
        "status",
    ]
    values: List[str] = []
    for c in cols:
        v = row.get(c, "")
        if isinstance(v, float):
            values.append(f"{v}")
        else:
            values.append(str(v))
    with path.open("a", encoding="utf-8") as f:
        f.write(",".join(values) + "\n")


def extract_summary(stdout_text: str) -> Optional[Dict[str, object]]:
    """
    Извлечь JSON-объект с метриками после маркера [SUMMARY].
    """
    m = SUMMARY_MARK_RE.search(stdout_text)
    if not m:
        return None
    text = m.group(1)
    try:
        return json.loads(text)
    except Exception:
        return None


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """
    Запустить команду и дождаться завершения. Возвращает (returncode, stdout, stderr).
    Никакого параллелизма — строго последовательный запуск.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def build_aggregate_cmd(files: Sequence[Path], prune: int) -> List[str]:
    cmd = [sys.executable, str(AGG_SCRIPT)]
    # Передаём файлы позиционными аргументами (а не --pattern), чтобы избежать glob с абсолютными путями
    for p in files:
        try:
            rel = p.relative_to(ROOT)
        except Exception:
            rel = p
        cmd.append(str(rel))
    cmd += ["--prune", str(prune)]
    return cmd


def build_apply_cmd() -> List[str]:
    # Для consensus-пайплайна используем маску manual-experiments/mask_consensus.json
    return [
        sys.executable,
        str(APPLY_SCRIPT),
        "--mask",
        str(MASK_PATH),
        "--eval-on",
        "test",
        "--calib-per-class",
        "2",
        "--eval-batches",
        "5",
    ]


def iterate_combinations(files: Sequence[Path], sizes: Set[int]) -> Iterable[Tuple[Path, ...]]:
    # Все сочетания выбранных размерностей (2,3,4), порядок не важен
    files_sorted = sorted(files, key=lambda p: p.stem)
    for r in sorted(sizes):
        for comb in itertools.combinations(files_sorted, r):
            yield comb


def parse_sizes(arg: Optional[str]) -> Set[int]:
    if not arg or arg.strip() == "":
        return {2, 3, 4}
    parts = [p.strip() for p in arg.split(",")]
    vals: Set[int] = set()
    for p in parts:
        if not p:
            continue
        try:
            v = int(p)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid size value: {p}")
        if v not in (2, 3, 4):
            raise argparse.ArgumentTypeError(f"Size must be in 2..4 for consensus, got: {v}")
        vals.add(v)
    if not vals:
        vals = {2, 3, 4}
    return vals


def parse_prune_levels(arg: Optional[str]) -> List[int]:
    if not arg or arg.strip() == "":
        return list(DEFAULT_PRUNE_LEVELS)
    parts = [p.strip() for p in arg.split(",")]
    vals: List[int] = []
    for p in parts:
        if not p:
            continue
        try:
            v = int(p)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid prune level: {p}")
        if v < 0:
            raise argparse.ArgumentTypeError(f"Prune level must be non-negative percent, got: {v}")
        vals.append(v)
    if not vals:
        vals = list(DEFAULT_PRUNE_LEVELS)
    return vals


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Grid-раннер для consensus-маски и оценки метрик.")
    ap.add_argument("--sizes", type=str, default=None, help="Какие размерности сочетаний использовать, например '2,3' или '2,3,4' (по умолчанию 2,3,4).")
    ap.add_argument("--prune-levels", type=str, default=None, help="Список уровней sparsity через запятую, например '5,10,15'. По умолчанию 5..70 шаг 5.")
    ap.add_argument("--first-n-combos", type=int, default=0, help="Ограничить число первых комбинаций (после сортировки) для быстрого теста. 0 = без ограничения.")
    ap.add_argument("--no-resume", action="store_true", help="Не пропускать уже успешные (status=ok) записи из CSV; по умолчанию пропускать (resume).")
    return ap


def main() -> None:
    # Проверки наличия скриптов и входов
    check_prerequisites()

    # CLI
    ap = build_argparser()
    args = ap.parse_args()
    sizes: Set[int] = parse_sizes(args.sizes)
    prune_levels: List[int] = parse_prune_levels(args.prune_levels)
    first_n: int = int(args.first_n_combos or 0)

    # CSV заголовок
    ensure_csv_header(CSV_PATH)
    # Режим resume: пропускаем уже успешные (status=ok) комбинации/уровни из CSV,
    # если не указан флаг --no-resume
    completed_ok: Set[Tuple[str, int]] = load_completed_ok(CSV_PATH) if not getattr(args, "no_resume", False) else set()

    # Формируем список комбинаций и сортируем по ключу
    combos = list(iterate_combinations(SCORE_FILES, sizes))
    combos = sorted(combos, key=lambda c: combo_key(c))
    if first_n > 0:
        combos = combos[:first_n]

    total_runs = 0
    for files_combo in combos:
        ckey = combo_key(files_combo)
        ckey_safe = sanitize_token(ckey)
        print(f"\n=== COMBO: {ckey} ===")

        for prune in prune_levels:
            # Пропуск уже успешно выполненных запусков (resume)
            if (ckey, prune) in completed_ok:
                print(f"[SKIP] already done (ok): {ckey}, prune={prune}")
                continue
            total_runs += 1
            print(f"\n--- [{total_runs}] prune={prune} ---")

            # 1) Построение консенсусной маски
            agg_cmd = build_aggregate_cmd(files_combo, prune)
            print("[RUN] ", " ".join(agg_cmd))
            rc1, out1, err1 = run_cmd(agg_cmd)
            if rc1 != 0:
                print(f"[ERROR] consensus_mask failed (rc={rc1}). Перехожу к следующему прогону.")
                # Логируем в CSV статус ошибки (без метрик)
                append_csv_row(
                    CSV_PATH,
                    {
                        "methods": ckey,
                        "prune": prune,
                        "status": f"consensus_failed_rc_{rc1}",
                    },
                )
                # Сохраним stderr/stdout для отладки
                (LOGS_DIR / f"{ckey_safe}_p{prune}.consensus.stderr.txt").write_text(err1 or "", encoding="utf-8")
                (LOGS_DIR / f"{ckey_safe}_p{prune}.consensus.stdout.txt").write_text(out1 or "", encoding="utf-8")
                continue

            # 2) Применение маски и измерения (тяжёлая часть — строго последовательно, ждём завершения)
            apply_cmd = build_apply_cmd()
            print("[RUN] ", " ".join(apply_cmd))
            rc2, out2, err2 = run_cmd(apply_cmd)

            # Пишем сырые логи второй стадии
            (LOGS_DIR / f"{ckey_safe}_p{prune}.stdout.txt").write_text(out2 or "", encoding="utf-8")
            (LOGS_DIR / f"{ckey_safe}_p{prune}.stderr.txt").write_text(err2 or "", encoding="utf-8")

            if rc2 != 0:
                print(f"[ERROR] apply_mask_prune failed (rc={rc2}).")
                append_csv_row(
                    CSV_PATH,
                    {
                        "methods": ckey,
                        "prune": prune,
                        "status": f"apply_failed_rc_{rc2}",
                    },
                )
                continue

            # Парсим [SUMMARY] { ... }
            summary = extract_summary(out2)
            if not summary:
                print("[ERROR] Не удалось найти/распарсить блок [SUMMARY] в stdout второй стадии.")
                append_csv_row(
                    CSV_PATH,
                    {
                        "methods": ckey,
                        "prune": prune,
                        "status": "summary_parse_failed",
                    },
                )
                continue

            # Формируем строку CSV с интересующими метриками
            row: Dict[str, object] = {
                "methods": ckey,
                "prune": prune,
                "params_before_stage1": summary.get("params_before_stage1", ""),
                "params_after_stage1": summary.get("params_after_stage1", ""),
                "params_before_stage1_millions": summary.get("params_before_stage1_millions", ""),
                "params_after_stage1_millions": summary.get("params_after_stage1_millions", ""),
                "stage1_reduction_percent": summary.get("stage1_reduction_percent", ""),
                "latency_baseline_ms": summary.get("latency_baseline_ms", ""),
                "latency_stage1_ms": summary.get("latency_stage1_ms", ""),
                "latency_stage1_change_percent": summary.get("latency_stage1_change_percent", ""),
                "acc_baseline": summary.get("acc_baseline", ""),
                "acc_stage1": summary.get("acc_stage1", ""),
                "acc_drop_stage1_percent": summary.get("acc_drop_stage1_percent", ""),
                "status": "ok",
            }
            append_csv_row(CSV_PATH, row)

    print("\nГотово. Результаты: ", CSV_PATH)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
