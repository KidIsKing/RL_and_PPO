"""Запись видео обученных политик (CartPole, Acrobot, Pendulum).

Главное свойство: эпизод в видео — ТОТ ЖЕ, что посчитан в final_episodes.csv. Берётся тот же
оценочный seed (final_eval_seed + номер эпизода) и та же схема засева генераторов, что в
a2c_research.evaluation.evaluate. Поэтому return на кадре совпадает с цифрой в таблице, а видео
в режиме sampled можно честно сопоставлять с графиками. (Проверяется тестом.)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import A2C, PPO

from a2c_research.evaluation import isolated_rng

ALGOS = {"a2c": A2C, "ppo": PPO}
SUPPORTED_FORMATS = {".mp4", ".webm", ".gif"}


def _as_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Нужен RGB-кадр (высота, ширина, 3), получено {array.shape}.")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _writer_kwargs(suffix: str, fps: float) -> dict[str, object]:
    if suffix == ".mp4":
        return {"fps": fps, "codec": "libx264", "pixelformat": "yuv420p", "macro_block_size": None}
    if suffix == ".webm":
        return {"fps": fps, "codec": "libvpx-vp9", "macro_block_size": None}
    if suffix == ".gif":
        return {"duration": 1.0 / fps, "loop": 0}
    raise ValueError(f"Формат {suffix!r} не поддерживается. Доступно: {sorted(SUPPORTED_FORMATS)}")


class MultiFormatWriter:
    """Пишет один поток кадров сразу в несколько форматов (кадры не копятся в памяти)."""

    def __init__(self, base_path: str | Path, formats: Iterable[str] = (".mp4", ".gif"),
                 fps: int = 30, gif_stride: int = 2):
        base = Path(base_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        self.gif_stride, self._count = gif_stride, 0
        self.paths: list[Path] = []
        self._writers: dict[str, object] = {}
        for suffix in formats:
            if suffix not in SUPPORTED_FORMATS:
                raise ValueError(f"Формат {suffix!r} не поддерживается")
            path = Path(str(base) + suffix)
            writer_fps = fps / gif_stride if suffix == ".gif" else fps
            self._writers[suffix] = imageio.get_writer(path, **_writer_kwargs(suffix, writer_fps))
            self.paths.append(path)

    def append(self, frame: np.ndarray) -> None:
        frame = _as_rgb_uint8(frame)
        for suffix, writer in self._writers.items():
            if suffix == ".gif":
                if self._count % self.gif_stride == 0:
                    writer.append_data(np.ascontiguousarray(frame[::2, ::2]))
            else:
                writer.append_data(frame)
        self._count += 1

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()

    def __enter__(self) -> "MultiFormatWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Строки текста в левом верхнем углу (шрифт по умолчанию — только латиница)."""
    image = Image.fromarray(_as_rgb_uint8(frame))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()
    y = 6
    for text in lines:
        draw.text((8, y), text, fill=(0, 0, 0), font=font)
        y += 22
    return np.asarray(image)


def episode_status(env_id: str, terminated: bool, truncated: bool) -> str:
    """Чем закончился эпизод — словами, с учётом смысла среды."""
    if env_id == "CartPole-v1":
        return "TIME LIMIT (500 steps)" if truncated and not terminated else "POLE FELL"
    if env_id == "Acrobot-v1":
        return "GOAL REACHED" if terminated else "TIME LIMIT (goal not reached)"
    return "TIME LIMIT (200 steps)"


@dataclass
class LoadedRun:
    model: object | None        # None = случайная политика
    label: str
    env_id: str
    final_eval_seed: int
    run_dir: Path | None = None


def load_run(run: str | Path) -> LoadedRun:
    """Папка запуска (с config.json и model.zip) -> модель и подпись. Слово 'random' = случайная политика."""
    if str(run) == "random":
        return LoadedRun(None, "RANDOM policy (no training)", "", 2_000_000)
    run_dir = Path(run)
    metadata = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = metadata["config"]
    model = ALGOS[cfg["algo"]].load(run_dir / "model.zip", device="cpu")
    label = f"{cfg['algo'].upper()} | {cfg['name']} | seed {cfg['seed']}"
    return LoadedRun(model, label, cfg["env_id"], cfg["final_eval_seed"], run_dir)


def _action(env: gym.Env, raw):
    return int(raw) if isinstance(env.action_space, gym.spaces.Discrete) else raw


def record_panels(
    runs: list[LoadedRun],
    env_id: str,
    base_path: str | Path,
    episodes: int = 1,
    seed_start: int | None = None,
    deterministic: bool = False,
    fps: int = 30,
    formats: Iterable[str] = (".mp4", ".gif"),
) -> tuple[list[Path], list[list[tuple[float, int]]]]:
    """Одна или несколько политик рядом (слева направо) на ОДИНАКОВЫХ стартовых состояниях.

    seed_start по умолчанию = final_eval_seed первой модели: те же эпизоды, что в final_episodes.csv.
    Возвращает (пути файлов, [для каждой панели: [(return, длина)] по эпизодам]).
    У каждой панели собственное состояние генератора PyTorch, поэтому сэмплируемые действия
    не зависят от числа соседних панелей и совпадают с оценкой.
    """
    seed_start = runs[0].final_eval_seed if seed_start is None else seed_start
    n = len(runs)
    envs = [gym.make(env_id, render_mode="rgb_array") for _ in runs]
    results: list[list[tuple[float, int]]] = [[] for _ in runs]
    try:
        with isolated_rng(), MultiFormatWriter(base_path, formats=formats, fps=fps) as writer:
            frame = None
            for episode in range(episodes):
                seed = seed_start + episode
                observations, rng_states, totals, steps = [], [None] * n, [0.0] * n, [0] * n
                alive, status = [True] * n, [""] * n
                for i, (env, run) in enumerate(zip(envs, runs)):
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    observations.append(env.reset(seed=seed)[0])
                    if run.model is None:
                        env.action_space.seed(seed)
                    elif not deterministic:
                        rng_states[i] = torch.get_rng_state()
                last = [env.render() for env in envs]
                while any(alive):
                    panels = []
                    for i, (env, run) in enumerate(zip(envs, runs)):
                        if alive[i]:
                            if run.model is None:
                                raw = env.action_space.sample()
                            else:
                                if rng_states[i] is not None:
                                    torch.set_rng_state(rng_states[i])
                                raw, _ = run.model.predict(observations[i], deterministic=deterministic)
                                if rng_states[i] is not None:
                                    rng_states[i] = torch.get_rng_state()
                            observations[i], reward, terminated, truncated, _ = env.step(_action(env, raw))
                            totals[i] += float(reward)
                            steps[i] += 1
                            last[i] = env.render()
                            if terminated or truncated:
                                alive[i] = False
                                status[i] = episode_status(env_id, terminated, truncated)
                        lines = [runs[i].label, f"episode {episode + 1}/{episodes}   step {steps[i]}",
                                 f"return {totals[i]:.1f}"]
                        if status[i]:
                            lines.append(status[i])
                        panels.append(annotate(last[i], lines))
                    frame = np.concatenate(panels, axis=1)
                    writer.append(frame)
                for _ in range(fps):          # итоговый кадр держим ~1 секунду
                    writer.append(frame)
                for i in range(n):
                    results[i].append((totals[i], steps[i]))
        return list(writer.paths), results
    finally:
        for env in envs:
            env.close()
