"""Видео для уже обученной модели (или случайной политики для сравнения «до/после»).

    python -m src.render --run results/ppo/baseline/default/seed_0
    python -m src.render --random
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .algos import ALGOS
from .config import ROOT
from .video import record_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Записать видео политики")
    parser.add_argument("--run", help="папка запуска, например results/ppo/baseline/default/seed_0")
    parser.add_argument("--random", action="store_true", help="видео случайной политики (до обучения)")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    if args.random:
        out = ROOT / "videos" / "random_policy"
        paths = record_episodes(None, args.env, out, label="RANDOM policy (no training)", n_episodes=args.episodes)
    elif args.run:
        run_dir = Path(args.run)
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
        model = ALGOS[cfg["algo"]].load(run_dir / "model", device="cpu")
        label = f"{cfg['algo'].upper()} | {cfg['name']} | seed {cfg['seed']}"
        paths = record_episodes(model, cfg["env_id"], run_dir / "policy", label=label,
                                n_episodes=args.episodes, seed=cfg["seed"])
    else:
        parser.error("укажи --run <папка> или --random")
    print("Сохранено:", *paths, sep="\n  ")


if __name__ == "__main__":
    main()
