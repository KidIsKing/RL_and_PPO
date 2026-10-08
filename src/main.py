from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from .video import save_video


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "videos"
MODEL_DIR = ROOT / "models"

ENV_ID = "CartPole-v1"
SEED = 0

# These match the main tutorial example.
TRAINING_TIMESTEPS = 10_000
VIDEO_FRAMES = 1_000
VIDEO_FPS = 20

# The tutorial also demonstrates vectorized training.
VECTOR_ENV_COUNT = 2
RUN_VECTOR_EXAMPLE = False


def capture_random_policy_video() -> None:
    """Reproduce the tutorial's random-action environment loop."""
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env.action_space.seed(SEED)

    obs, _ = env.reset(seed=SEED)
    frames: list[np.ndarray] = []

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        frame = env.render()
        frames.append(frame)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    for suffix in (".mp4", ".webm", ".gif"):
        save_video(frames, VIDEO_DIR / f"random_interaction{suffix}", VIDEO_FPS)


def train_ppo() -> PPO:
    """Train the PPO agent using the tutorial's 10,000-step budget."""
    env = gym.make(ENV_ID, render_mode="rgb_array")

    # Keep the tutorial seed while using Gymnasium's current reset API.
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=SEED,
        tensorboard_log=str(ROOT / "logs"),
    )

    model.learn(total_timesteps=TRAINING_TIMESTEPS)
    model.save(MODEL_DIR / "ppo_cartpole")

    env.close()
    return model


def capture_policy_video(model: PPO) -> None:
    """Record the deterministic policy exactly as frames rendered by Gymnasium."""
    env = gym.make(ENV_ID, render_mode="rgb_array")

    obs, _ = env.reset(seed=SEED)
    frames: list[np.ndarray] = []

    for _ in range(VIDEO_FRAMES):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        frame = env.render()
        frames.append(frame)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    for suffix in (".mp4", ".webm", ".gif"):
        save_video(frames, VIDEO_DIR / f"ppo_policy{suffix}", VIDEO_FPS)


def run_vectorized_example() -> None:
    """Reproduce the tutorial's vectorized-environment section."""
    env = make_vec_env(
        ENV_ID,
        n_envs=VECTOR_ENV_COUNT,
        seed=SEED,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=SEED,
        tensorboard_log=str(ROOT / "logs_vec"),
    )

    model.learn(total_timesteps=TRAINING_TIMESTEPS)

    obs = env.reset()
    frames: list[np.ndarray] = []

    for _ in range(VIDEO_FRAMES):
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)

        frame = env.render()
        frames.append(frame)

    env.close()

    for suffix in (".mp4", ".webm", ".gif"):
        save_video(
            frames,
            VIDEO_DIR / f"ppo_policy_vec_env{suffix}",
            VIDEO_FPS,
        )


def main() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("1/3  Recording random-policy video...")
    capture_random_policy_video()

    print("2/3  Training PPO...")
    model = train_ppo()

    print("3/3  Recording trained-policy video...")
    capture_policy_video(model)

    if RUN_VECTOR_EXAMPLE:
        print("Vectorized-environment example...")
        run_vectorized_example()

    print()
    print("Done.")
    print(f"Videos: {VIDEO_DIR}")
    print(f"Model:  {MODEL_DIR / 'ppo_cartpole.zip'}")


if __name__ == "__main__":
    main()
