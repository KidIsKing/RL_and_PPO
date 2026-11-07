"""Видео для уже обученных моделей.

    # одна модель (по умолчанию sampled: действия сэмплируются, как на графиках)
    python -m ppo_tools.render --run results_search/ppo/baseline/default/seed_0
    # для Acrobot и Pendulum в пути есть папка среды:
    python -m ppo_tools.render --run results_search/Acrobot-v1/ppo/baseline/default/seed_0

    # «до и после»: случайная политика и обученная бок о бок
    python -m ppo_tools.render --compare random results_search/ppo/baseline/default/seed_0

    # плохой и хороший гиперпараметр бок о бок (2–4 запуска)
    python -m ppo_tools.render --compare "<папка запуска 1>" "<папка запуска 2>" --episodes 2

Флаг --greedy включает детерминированную политику (самое вероятное действие). Она выглядит лучше
кривых обучения и может скрывать нестабильность, поэтому по умолчанию используется sampled.
По умолчанию берутся ТЕ ЖЕ начальные состояния, что в итоговой оценке (final_episodes.csv).
Результат: папка videos/ (mp4 + gif).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .video import load_run, record_panels

ROOT = Path(__file__).resolve().parents[2]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9=._,+-]+", "_", text).strip("_")[:150]


def title_of(run: str) -> str:
    if run == "random":
        return "random"
    path = Path(run)
    return f"{path.parent.parent.name}_{path.parent.name}_{path.name}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--run", help="папка одного запуска")
    parser.add_argument("--compare", nargs="+", metavar="RUN", help="2-4 запуска бок о бок (можно 'random')")
    parser.add_argument("--env", default="CartPole-v1", help="среда для 'random', если нет обученных панелей")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--greedy", action="store_true", help="детерминированная политика вместо sampled")
    parser.add_argument("--seed-start", type=int, default=None,
                        help="первый seed эпизода (по умолчанию — final_eval_seed из config.json)")
    parser.add_argument("--out", type=Path, default=ROOT / "videos")
    parser.add_argument("--formats", nargs="+", default=[".mp4", ".gif"])
    args = parser.parse_args(argv)
    if bool(args.run) == bool(args.compare):
        parser.error("укажите ровно одно из: --run <папка> или --compare <запуск> <запуск> ...")
    runs_text = [args.run] if args.run else args.compare
    if args.compare and not 2 <= len(runs_text) <= 4:
        parser.error("--compare принимает от 2 до 4 запусков")

    runs = [load_run(r) for r in runs_text]
    envs = {r.env_id for r in runs if r.env_id}
    if len(envs) > 1:
        parser.error(f"все запуски должны быть из одной среды, найдено: {sorted(envs)}")
    env_id = envs.pop() if envs else args.env
    mode = "greedy" if args.greedy else "sampled"
    for run in runs:
        if run.model is not None:
            run.label += f" | {mode}"
    name = "compare_" + "_vs_".join(title_of(r) for r in runs_text) if args.compare else \
        f"{title_of(runs_text[0])}"
    base = args.out / env_id / safe_name(f"{name}_{mode}")
    paths, results = record_panels(runs, env_id, base, episodes=args.episodes,
                                   seed_start=args.seed_start, deterministic=args.greedy,
                                   formats=tuple(args.formats))
    print("Сохранено:", *paths, sep="\n  ")
    for run, panel in zip(runs, results):
        print(f"  {run.label}: returns по эпизодам = {[round(x[0], 1) for x in panel]}")


if __name__ == "__main__":
    main()
