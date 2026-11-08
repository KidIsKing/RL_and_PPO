"""Выбрать из каталога результатов только запуски нужных алгоритмов (и/или сред).

Зачем. Анализ (`a2c_research.analyze`) принимает несколько каталогов результатов, но если один и тот же
запуск (алгоритм, серия, значение, seed) лежит в двух каталогах как независимая копия, обе копии
исключаются как «конфликтующие дубликаты». Репозиторий сокомандника содержит и A2C, и PPO. Поэтому, чтобы
сравнить ВАШ PPO с ЕГО A2C, нужно взять у него только A2C:

    python -m ppo_tools.subset --src ../A2CvsPPO/results_search --dst results_a2c_only --algos a2c
    python -m a2c_research.analyze --results results_search results_a2c_only --out analysis_compare \
        --study-plan configs/main_search_v1.json

Откуда взять A2C: готовые запуски лежат в ветке a2c репозитория RATARETO/A2CvsPPO (каталог results_search;
в самом A2C_RL готовых запусков нет):

    git clone --depth 1 --branch a2c --single-branch https://github.com/RATARETO/A2CvsPPO.git ../A2CvsPPO

--src можно указать и на корень клона (results_search найдётся сам). Каталог-источник не изменяется.
--link создаёт жёсткие ссылки (быстро, без места на диске); если они невозможны (другой диск),
файлы копируются.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

PATTERNS = ("*/*/*/seed_*/config.json", "*/*/*/*/seed_*/config.json")


def resolve_source(src: Path) -> Path:
    """Если указан корень репозитория, берём results_search внутри него."""
    nested = src / "results_search"
    return nested if nested.is_dir() else src


def diagnose_empty(src: Path, algos: set[str]) -> str:
    """Понятное объяснение, почему ничего не найдено."""
    total = sum(1 for _ in src.rglob("config.json"))
    if total == 0:
        return (f"в {src} нет ни одного config.json, это не каталог результатов запусков. В репозитории A2C_RL "
                "готовых запусков нет: они лежат в ветке a2c репозитория RATARETO/A2CvsPPO (папка results_search).")
    return (f"в {src} найдено файлов config.json: {total}, но ни один не относится к алгоритмам "
            f"{sorted(algos)} или не лежит в ожидаемой структуре <среда>/<алгоритм>/<серия>/<значение>/seed_N.")


def _link_or_copy(source: str, target: str) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


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
    copier = _link_or_copy if link else shutil.copy2
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
        parser.error(f"нет каталога {args.src.resolve()} (путь считается от текущей папки). "
                     "Клонируйте результаты сокомандника, команда в докстринге модуля.")
    src = resolve_source(args.src)
    runs = select_runs(src, set(args.algos), set(args.envs) if args.envs else None)
    if not runs:
        parser.error("подходящих запусков не найдено: " + diagnose_empty(src, set(args.algos)))
    args.dst.mkdir(parents=True, exist_ok=True)
    copied = copy_runs(src, args.dst, runs, args.link)
    print(f"Скопировано запусков: {copied} -> {args.dst}")


if __name__ == "__main__":
    main()
