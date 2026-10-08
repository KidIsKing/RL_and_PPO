# Lux RL tutorial — local baseline

This is the first clean local stage of the project.

It follows the **CartPole + PPO** part of Stone Tao's
**"RL with Lux: 1 - Intro to RL"** tutorial, but updates the old
Gym/SB3 notebook code to the modern Gymnasium/SB3 API.

The project deliberately does **not** introduce hyperparameter-search code yet.

## 1. Requirements

Use **Python 3.10**.

Create and activate a virtual environment:

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the environment:

```bash
python --version
python -c "import gymnasium, stable_baselines3, imageio; print(gymnasium.__version__, stable_baselines3.__version__, imageio.__version__)"
```

## 2. Run

From the project root:

```bash
python -m src.main
```

The script will:

1. run a random policy and save a short replay;
2. train PPO for 10,000 environment interactions;
3. run the deterministic trained policy;
4. save the same frames as:
   - `videos/*.mp4`
   - `videos/*.webm`
   - `videos/*.gif`

The MP4 and WebM files are the preferred formats for normal playback.
GIF is convenient for quick inspection.

## 3. What is intentionally different from the old notebook?

The original tutorial uses:

- Python 3.7
- `gym==0.21`
- `stable-baselines3==1.7.0`
- OpenCV `VideoWriter` with a `VP80` WebM codec tag
- old four-value `env.step(...)` API

The local version uses:

- Python 3.10
- Gymnasium
- current SB3 2.9.0
- the modern five-value `env.step(...)` API
- RGB frames passed directly to imageio/FFmpeg
- H.264 MP4 and VP9 WebM writers

This is intentional: the goal is to reproduce the **experiment and visualization**, not preserve outdated dependency hacks.

## 4. Project stage

At this stage there is only one learning setup:

```text
CartPole-v1
    ↓
PPO + MlpPolicy
    ↓
10,000 training timesteps
    ↓
deterministic evaluation
    ↓
RGB frames
    ↓
MP4 / WebM / GIF
```

The next stage can turn this into a research project for A2C vs PPO with:

- a central experiment configuration;
- explicit common and algorithm-specific parameters;
- controlled seeds;
- repeatable experiment IDs;
- TensorBoard/CSV results;
- evaluation metrics;
- automated comparison plots.
