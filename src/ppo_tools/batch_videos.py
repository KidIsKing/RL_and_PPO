"""Видео-сравнения по ВСЕМ сериям гиперпараметров одной командой (замена make_compare_videos.sh).

    python -m ppo_tools.batch_videos --results results_search --env CartPole-v1 --seed 0
    python -m ppo_tools.batch_videos --results results_search --env CartPole-v1 --groups learning_rate clip_range
    python -m ppo_tools.batch_videos --results results_search --list      # только показать, что будет записано

Для обычной серии: «минимум | значение по умолчанию | максимум» (режим triples) или четыре ступеньки
от минимума к максимуму (режим ladder). Для сетки: четыре угла (мин+мин, мин+макс, макс+мин, макс+макс).
Недостающие запуски пропускаются с сообщением. Пути к запускам строятся тем же кодом, что и при обучении
(make_config), поэтому они не могут разойтись с реальной структурой результатов.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from a2c_research.config import DEFAULTS, make_config
from a2c_research.design import GRID_AXES
from a2c_research.experiments import SWEEPS

from .render import ROOT, safe_name
from .video import load_run, record_panels


@dataclass(frozen=True)
class VideoSpec:
    group: str
    kind: str                 # triples | ladder | corners
    overrides: tuple          # по одному словарю на панель


def _same(a, b) -> bool:
    return a == b


def pick_levels(levels: list[dict], factor: str, default, mode: str) -> list[dict]:
    """Какие уровни одной обычной серии показывать. Порядок — как объявлен в SWEEPS (по возрастанию)."""
    if len(levels) < 2:
        return []
    if mode == "triples":
        default_idx = next((i for i, row in enumerate(levels) if _same(row.get(factor), default)), None)
        if len(levels) == 2:
            return list(levels)
        if default_idx is None or default_idx in (0, len(levels) - 1):
            default_idx = len(levels) // 2
        return [levels[0], levels[default_idx], levels[-1]]
    if mode == "ladder":
        if len(levels) <= 4:
            return list(levels)
        step = (len(levels) - 1) / 3
        return [levels[round(i * step)] for i in range(4)]
    raise ValueError("mode must be triples or ladder")


def plan_videos(algo: str, groups: list[str] | None = None, mode: str = "triples") -> list[VideoSpec]:
    """Чистая функция: что записывать, без обращения к диску."""
    specs = []
    for group, levels in SWEEPS[algo].items():
        if groups is not None and group not in groups:
            continue
        if group in GRID_AXES:
            first, second = GRID_AXES[group]
            xs = list(dict.fromkeys(row[first] for row in levels))
            ys = list(dict.fromkeys(row[second] for row in levels))
            corners = [{first: x, second: y} for x in (xs[0], xs[-1]) for y in (ys[0], ys[-1])]
            specs.append(VideoSpec(group, "corners", tuple(corners)))
        elif group != "baseline":
            factor = group
            picked = pick_levels(levels, factor, DEFAULTS[algo].get(factor), mode)
            if picked:
                specs.append(VideoSpec(group, mode, tuple(picked)))
    return specs


def run_dirs(spec: VideoSpec, algo: str, env_id: str, seed: int, results: Path) -> list[Path]:
    """Папки запусков панелей; индекс не влияет на путь, поэтому передаём 0."""
    return [make_config(algo, spec.group, overrides, seed, 0, env_id=env_id).run_dir(results)
            for overrides in spec.overrides]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results_search"))
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--algo", choices=sorted(DEFAULTS), default="ppo")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--mode", choices=("triples", "ladder"), default="triples")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "videos")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    specs = plan_videos(args.algo, args.groups, args.mode)
    if args.groups and not specs:
        parser.error("для этих серий нечего записывать (проверьте названия)")
    done = skipped = 0
    for spec in specs:
        dirs = run_dirs(spec, args.algo, args.env, args.seed, args.results)
        missing = [d for d in dirs if not (d / "model.zip").exists()]
        label = f"{spec.group} ({spec.kind}): " + " | ".join(d.parent.name for d in dirs)
        if missing:
            skipped += 1
            print(f"ПРОПУСК {label}\n    нет модели: {missing[0]}" +
                  (f" и ещё {len(missing) - 1}" if len(missing) > 1 else ""))
            continue
        print(("ЗАПИСАТЬ " if args.list else "ЗАПИСЬ ") + label)
        if args.list:
            continue
        runs = [load_run(d) for d in dirs]
        mode = "greedy" if args.greedy else "sampled"
        for run in runs:
            run.label += f" | {mode}"
        base = args.out / args.env / args.algo / safe_name(f"{spec.group}_{spec.kind}_seed_{args.seed}_{mode}")
        record_panels(runs, args.env, base, episodes=args.episodes, deterministic=args.greedy)
        done += 1
    print(f"\nГотово: записано {done}, пропущено {skipped}." if not args.list else "")


if __name__ == "__main__":
    main()
