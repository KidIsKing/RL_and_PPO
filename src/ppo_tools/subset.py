"""Выбрать из каталога результатов только запуски нужных алгоритмов (и/или сред).

Зачем. Анализ (`a2c_research.analyze`) принимает несколько каталогов результатов, но если один и тот же
запуск (алгоритм, серия, значение, seed) лежит в двух каталогах как независимая копия, обе копии
исключаются как «конфликтующие дубликаты». Репозиторий сокомандника содержит и A2C, и PPO. Поэтому, чтобы
сравнить ВАШ PPO с ЕГО A2C, нужно взять у него только A2C:

    python -m ppo_tools.subset --src ../A2C_RL/results_search --dst results_a2c_only --algos a2c
    python -m a2c_research.analyze --results results_search results_a2c_only --out analysis_compare \
        --study-plan configs/main_search_v1.json

Каталог-источник не изменяется. --link создаёт жёсткие ссылки вместо копий (быстро, без места на диске).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

PATTERNS = ("*/*/*/seed_*/config.json", "*/*/*/*/seed_*/config.json")


def select_runs(src: Path, algos: set[str], envs: set[str] | None = None) -> list[Path]:
    """Папки запусков (в них лежит config.json), удовлетворяющие фильтру."""
    found = []
    for pattern in PATTERNS:
        for config_path in sorted(src.glob(pattern)):
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))["config"]
            except (OSError, ValueError, KeyError):
                continue
            if cfg.get("algo") in algos and (envs is None or cfg.get("env_id") in envs):
                found.append(config_path.parent)
    return found


def copy_runs(src: Path, dst: Path, runs: list[Path], link: bool = False) -> int:
    """Копирует папки запусков с сохранением относительной структуры. Возвращает число скопированных."""
    copier = os.link if link else shutil.copy2
    count = 0
    for run in runs:
        target = dst / run.relative_to(src)
        if target.exists():
            raise FileExistsError(f"{target} уже существует; выберите пустой каталог назначения")
        shutil.copytree(run, target, copy_function=copier)
        count += 1
    return count


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--algos", nargs="+", choices=("a2c", "ppo"), required=True)
    parser.add_argument("--envs", nargs="+")
    parser.add_argument("--link", action="store_true", help="жёсткие ссылки вместо копий")
    args = parser.parse_args(argv)
    if not args.src.is_dir():
        parser.error(f"нет каталога {args.src}")
    runs = select_runs(args.src, set(args.algos), set(args.envs) if args.envs else None)
    if not runs:
        parser.error("подходящих запусков не найдено")
    args.dst.mkdir(parents=True, exist_ok=True)
    copied = copy_runs(args.src, args.dst, runs, args.link)
    print(f"Скопировано запусков: {copied} -> {args.dst}")


if __name__ == "__main__":
    main()
