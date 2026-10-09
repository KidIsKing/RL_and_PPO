"""Анализ результатов: метрики стабильности, графики кривых обучения, график чувствительности.

Запуск (из корня проекта):
    python -m src.analyze                       # читает results/, пишет в analysis/
    python -m src.analyze --results results_quick --out analysis_quick

Метрики одного запуска (по кривой eval_mean):
    final              средняя оценка за последние LAST_K точек (итоговое качество)
    auc                средняя оценка по всей кривой / 500  (0..1; скорость + качество обучения)
    solved             достигнут ли порог THRESHOLD хотя бы раз
    steps_to_threshold число шагов до первого достижения порога (NaN, если не достигнут)
    min_after_solve    минимум оценки ПОСЛЕ первого достижения порога (низкий = «срыв», нестабильность)

Метрики конфигурации — агрегат по сидам: среднее и std (разброс между сидами = стабильность).
Чувствительность параметра = разброс средних AUC между значениями этого параметра (max - min).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # рисуем в файлы, окно не нужно
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THRESHOLD = 475.0   # 95% от максимума CartPole-v1
MAX_RETURN = 500.0  # потолок CartPole-v1
LAST_K = 5          # сколько последних оценок усредняем для «итогового» качества

KEYS = ["algo", "group", "name", "sweep_index", "seed"]


def load_runs(results_dir: Path) -> pd.DataFrame:
    frames = []
    for eval_path in sorted(results_dir.glob("*/*/*/seed_*/eval.csv")):
        run_dir = eval_path.parent
        meta = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
        df = pd.read_csv(eval_path)
        df = df[df["timesteps"] <= meta["total_timesteps"]]  # отбрасываем «перебег» из-за размера rollout
        for key in ("algo", "group", "name", "sweep_index", "seed"):
            df[key] = meta[key]
        frames.append(df)
    if not frames:
        raise SystemExit(f"В {results_dir} нет завершённых запусков (ищу */*/*/seed_*/eval.csv).")
    return pd.concat(frames, ignore_index=True)


def run_metrics(g: pd.DataFrame) -> dict:
    g = g.sort_values("timesteps")
    y, t = g["eval_mean"].to_numpy(), g["timesteps"].to_numpy()
    hit = np.where(y >= THRESHOLD)[0]
    return {
        "final": float(y[-LAST_K:].mean()),
        "auc": float(y.mean() / MAX_RETURN),
        "solved": bool(len(hit)),
        "steps_to_threshold": float(t[hit[0]]) if len(hit) else np.nan,
        "min_after_solve": float(y[hit[0]:].min()) if len(hit) else np.nan,
    }


def per_run_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(KEYS):
        rows.append({**dict(zip(KEYS, key)), **run_metrics(g)})
    return pd.DataFrame(rows)


def per_config_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (algo, group, name, idx), g in runs.groupby(["algo", "group", "name", "sweep_index"]):
        rows.append({
            "algo": algo, "group": group, "name": name, "sweep_index": idx, "n_seeds": len(g),
            "final_mean": g["final"].mean(), "final_std": g["final"].std(ddof=0),
            "auc_mean": g["auc"].mean(), "auc_std": g["auc"].std(ddof=0),
            "solved_rate": g["solved"].mean(),
            "median_steps_to_threshold": g["steps_to_threshold"].median(),
            "mean_min_after_solve": g["min_after_solve"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["algo", "group", "sweep_index"]).reset_index(drop=True)


def _label(name: str) -> str:
    return name.split("=", 1)[1] if "=" in name else name


def plot_group(df: pd.DataFrame, runs: pd.DataFrame, algo: str, group: str, out_dir: Path) -> None:
    sub = df[(df["algo"] == algo) & (df["group"] == group)]
    run_sub = runs[(runs["algo"] == algo) & (runs["group"] == group)]
    order = sub[["sweep_index", "name"]].drop_duplicates().sort_values("sweep_index")["name"].tolist()
    cmap = plt.get_cmap("viridis")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    for i, name in enumerate(order):
        color = cmap(i / max(1, len(order) - 1))
        curve = sub[sub["name"] == name].groupby("timesteps")["eval_mean"].agg(["mean", "std"]).reset_index()
        curve["std"] = curve["std"].fillna(0.0)
        ax1.plot(curve["timesteps"], curve["mean"], color=color, label=_label(name), linewidth=1.8)
        ax1.fill_between(curve["timesteps"], (curve["mean"] - curve["std"]).clip(lower=0),
                         (curve["mean"] + curve["std"]).clip(upper=MAX_RETURN), color=color, alpha=0.15)

        finals = run_sub[run_sub["name"] == name]["final"].to_numpy()
        ax2.bar(i, finals.mean(), yerr=finals.std(), color=color, alpha=0.8, capsize=3)
        ax2.scatter(np.full(len(finals), i), finals, color="black", s=10, zorder=3)

    ax1.axhline(MAX_RETURN, color="gray", linestyle=":")
    ax1.set_xlabel("шаги среды")
    ax1.set_ylabel("награда за эпизод (оценка, deterministic)")
    ax1.set_title(f"{algo.upper()}: кривые обучения, серия «{group}» (среднее ± std по сидам)")
    ax1.legend(title=group, fontsize=8, loc="lower right")
    ax1.set_ylim(0, MAX_RETURN * 1.05)

    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels([_label(n) for n in order], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel(f"итоговая награда (посл. {LAST_K} оценок)")
    ax2.set_title("Итог по сидам (точки — отдельные сиды)")
    ax2.set_ylim(0, MAX_RETURN * 1.05)

    fig.tight_layout()
    fig.savefig(out_dir / f"{algo}_{group}.png", dpi=140)
    plt.close(fig)


def sensitivity_table(configs: pd.DataFrame, algo: str) -> pd.DataFrame:
    rows = []
    for group, g in configs[configs["algo"] == algo].groupby("group"):
        if group == "baseline" or len(g) < 2:
            continue
        rows.append({
            "group": group,
            "n_values": len(g),
            "auc_range": g["auc_mean"].max() - g["auc_mean"].min(),
            "final_range": g["final_mean"].max() - g["final_mean"].min(),
            "share_solved": float((g["solved_rate"] > 0).mean()),
            "worst_value": g.loc[g["auc_mean"].idxmin(), "name"],
            "best_value": g.loc[g["auc_mean"].idxmax(), "name"],
        })
    return pd.DataFrame(rows).sort_values("auc_range", ascending=False).reset_index(drop=True)


def plot_sensitivity(table: pd.DataFrame, algo: str, out_dir: Path) -> None:
    if table.empty:
        return
    t = table.sort_values("auc_range")
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(t) + 1.5))
    ax.barh(t["group"], t["auc_range"], color="tab:blue")
    ax.set_xlabel("чувствительность: max − min средней AUC по значениям параметра")
    ax.set_title(f"{algo.upper()}: насколько сильно параметр влияет на обучение")
    fig.tight_layout()
    fig.savefig(out_dir / f"{algo}_sensitivity.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Анализ результатов экспериментов")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="analysis")
    args = parser.parse_args()

    results_dir, out_dir = Path(args.results), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(results_dir)
    runs = per_run_table(df)
    configs = per_config_table(runs)
    runs.to_csv(out_dir / "run_metrics.csv", index=False)
    configs.to_csv(out_dir / "config_summary.csv", index=False)

    for algo in sorted(df["algo"].unique()):
        for group in df[df["algo"] == algo]["group"].unique():
            plot_group(df, runs, algo, group, out_dir)
        sens = sensitivity_table(configs, algo)
        sens.to_csv(out_dir / f"{algo}_sensitivity.csv", index=False)
        plot_sensitivity(sens, algo, out_dir)

    print(f"Готово. Таблицы и графики: {out_dir.resolve()}")
    print(configs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
