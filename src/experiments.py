"""Какие эксперименты запускаем. Это единственный файл, который нужно править, чтобы «крутить» гиперпараметры.

Схема «один параметр за раз» (one-at-a-time): каждая серия (group) меняет ОДИН гиперпараметр,
остальные остаются по умолчанию. Так эффект параметра виден в чистом виде.

Формат: SWEEPS[алгоритм][серия] = список словарей «что переопределить».
Ключи — имена параметров SB3 (learning_rate, clip_range ...) либо полей ExperimentConfig
(n_envs, net_arch, activation).
"""
from __future__ import annotations

# Общие настройки: ОДИНАКОВЫЕ для PPO и A2C, чтобы сравнение было честным.
COMMON = dict(
    total_timesteps=100_000,  # бюджет в шагах среды
    eval_freq=5_000,
    n_eval_episodes=10,
)


def values(param: str, vals: list) -> list[dict]:
    return [{param: v} for v in vals]


# ----------------------------------------------------------------------------------------
# PPO
# Значения подобраны широко (по логарифмической шкале там, где это уместно), чтобы часть
# конфигураций заведомо «ломалась», а часть работала хорошо: так отличия наглядны.
# Дефолт SB3 для каждого параметра присутствует в списке — это контрольная точка серии.
# ----------------------------------------------------------------------------------------
PPO_SWEEPS: dict[str, list[dict]] = {
    "baseline": [{}],
    "learning_rate": values("learning_rate", [1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
    "clip_range": values("clip_range", [0.05, 0.1, 0.2, 0.3, 0.5]),
    "n_epochs": values("n_epochs", [1, 3, 10, 20, 40]),
    "n_steps": values("n_steps", [64, 128, 256, 512, 1024, 2048, 4096]),
    "batch_size": values("batch_size", [16, 32, 64, 128, 256, 512]),
    "gamma": values("gamma", [0.9, 0.95, 0.99, 0.999]),
    "gae_lambda": values("gae_lambda", [0.8, 0.9, 0.95, 0.99, 1.0]),
    "ent_coef": values("ent_coef", [0.0, 0.001, 0.01, 0.05, 0.1]),
    "vf_coef": values("vf_coef", [0.1, 0.25, 0.5, 1.0, 2.0]),
    "max_grad_norm": values("max_grad_norm", [0.1, 0.5, 1.0, 5.0]),
    "n_envs": values("n_envs", [1, 2, 4, 8]),
    "net_arch": values("net_arch", [[16, 16], [64, 64], [256, 256]]),
    "activation": values("activation", ["tanh", "relu"]),
}

# ----------------------------------------------------------------------------------------
# A2C — ЗАГОТОВКА ДЛЯ СОКОМАНДНИКА.
# Что нужно сделать: подобрать значения и дополнить серии (названия серий лучше оставить
# такими же, как у PPO, — тогда скрипт анализа сможет сравнивать алгоритмы по одинаковым сериям).
# Параметры, которых нет у PPO: use_rms_prop, rms_prop_eps. У A2C нет clip_range, n_epochs, batch_size.
# Дефолты SB3 для A2C: learning_rate=7e-4, n_steps=5, gae_lambda=1.0, ent_coef=0.0.
# ----------------------------------------------------------------------------------------
A2C_SWEEPS: dict[str, list[dict]] = {
    "baseline": [{}],
    "learning_rate": values("learning_rate", [1e-5, 1e-4, 7e-4, 3e-3, 1e-2]),  # TODO: уточнить
    "n_steps": values("n_steps", [5, 16, 64, 256]),                              # TODO: уточнить
    "gae_lambda": values("gae_lambda", [0.8, 0.9, 0.95, 1.0]),                   # TODO: уточнить
    "ent_coef": values("ent_coef", [0.0, 0.001, 0.01, 0.05]),                    # TODO: уточнить
    "use_rms_prop": values("use_rms_prop", [True, False]),                       # RMSprop vs Adam
}

SWEEPS = {"ppo": PPO_SWEEPS, "a2c": A2C_SWEEPS}
