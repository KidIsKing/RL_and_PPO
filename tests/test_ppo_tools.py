"""Тесты PPO-инструментов: проверка окружения, выбор видео, выборка результатов, точность видео."""

import csv
import json
import shutil
from pathlib import Path

import pytest

from a2c_research.config import environment_config, make_config
from a2c_research.train import run_experiment
from ppo_tools import batch_videos, doctor, subset

ROOT = Path(__file__).resolve().parents[1]


def video_api():
    """Видео-зависимости необязательны (requirements-video.txt): без них эти тесты пропускаются."""
    pytest.importorskip("imageio")
    pytest.importorskip("pygame")
    from ppo_tools import video
    return video


# ---------------------------------------------------------------- doctor
def test_version_problems_detects_python_patch_and_packages():
    wanted = {"python": "3.10.22", "torch": "2.10.0"}
    assert doctor.version_problems(wanted, dict(wanted)) == []
    problems = doctor.version_problems(wanted, {"python": "3.10.11", "torch": "2.10.0+cu128"})
    assert len(problems) == 2
    assert "uv python install 3.10.22" in problems[0]
    assert "requirements-frozen.txt" in problems[1]


def test_core_is_identical_to_teammate_version():
    assert doctor.core_manifest_problems() == []


def test_core_manifest_flags_edits_missing_and_extra_files(tmp_path):
    core = tmp_path / "core"
    shutil.copytree(doctor.CORE_DIR, core, ignore=shutil.ignore_patterns("__pycache__"))
    assert doctor.core_manifest_problems(core, doctor.MANIFEST) == []
    (core / "metrics.py").write_text("# edited\n", encoding="utf-8")
    (core / "evaluation.py").unlink()
    (core / "extra.py").write_text("x = 1\n", encoding="utf-8")
    text = " ".join(doctor.core_manifest_problems(core, doctor.MANIFEST))
    assert "metrics.py: изменён" in text
    assert "evaluation.py: файл ядра отсутствует" in text
    assert "extra.py: лишний файл" in text
    assert "нет файла" in doctor.core_manifest_problems(core, tmp_path / "none.json")[0]


def test_doctor_reports_version_mismatch_without_running_anything(tmp_path, capsys):
    plan = json.loads((ROOT / "configs" / "main_search_v1.json").read_text(encoding="utf-8"))
    plan["versions"]["python"] = "3.10.1"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    assert doctor.main(["--plan", str(path)]) == 1
    output = capsys.readouterr().out
    assert "ОШИБКА" in output and "python: в плане 3.10.1" in output


# ---------------------------------------------------------------- план видео и пути
def test_plan_videos_picks_min_default_max_and_grid_corners():
    specs = {s.group: s for s in batch_videos.plan_videos("ppo")}
    lr = specs["learning_rate"]
    assert [row["learning_rate"] for row in lr.overrides] == [1e-5, 3e-4, 1e-2]
    clip = specs["clip_range"]
    assert [row["clip_range"] for row in clip.overrides] == [0.05, 0.2, 0.5]
    corners = specs["grid_lr_clip"]
    assert corners.kind == "corners"
    assert [(r["learning_rate"], r["clip_range"]) for r in corners.overrides] == [
        (1e-4, 0.05), (1e-4, 0.5), (1e-2, 0.05), (1e-2, 0.5)]
    assert "baseline" not in specs
    assert len(specs["activation"].overrides) == 2


def test_plan_videos_ladder_and_unknown_group():
    ladder = {s.group: s for s in batch_videos.plan_videos("ppo", ["learning_rate"], "ladder")}
    assert [row["learning_rate"] for row in ladder["learning_rate"].overrides] == [1e-5, 3e-4, 1e-3, 1e-2]
    assert batch_videos.plan_videos("ppo", ["no_such_group"]) == []


def test_run_dirs_follow_the_real_result_layout(tmp_path):
    spec = batch_videos.plan_videos("ppo", ["learning_rate"])[0]
    cart = batch_videos.run_dirs(spec, "ppo", "CartPole-v1", 3, tmp_path)
    acro = batch_videos.run_dirs(spec, "ppo", "Acrobot-v1", 3, tmp_path)
    assert cart[1] == tmp_path / "ppo" / "learning_rate" / "learning_rate=0.0003" / "seed_3"
    assert acro[1] == tmp_path / "Acrobot-v1" / "ppo" / "learning_rate" / "learning_rate=0.0003" / "seed_3"
    # путь совпадает с тем, что строит обучение
    assert cart[1] == make_config("ppo", "learning_rate", {"learning_rate": 3e-4}, 3,
                                  env_id="CartPole-v1").run_dir(tmp_path)


# ---------------------------------------------------------------- subset
def fake_run(root: Path, env: str, algo: str, group: str, name: str, seed: int) -> Path:
    cfg = make_config(algo, group, {}, seed, env_id=env)
    run = cfg.run_dir(root)
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"config": cfg.to_dict()}), encoding="utf-8")
    (run / "eval.csv").write_text("timesteps\n0\n", encoding="utf-8")
    return run


def test_subset_selects_algorithm_and_environment_and_never_overwrites(tmp_path):
    src = tmp_path / "src"
    for env in ("CartPole-v1", "Acrobot-v1"):
        for algo in ("a2c", "ppo"):
            for seed in (0, 1):
                fake_run(src, env, algo, "baseline", "default", seed)
    runs = subset.select_runs(src, {"a2c"})
    assert len(runs) == 4
    assert len(subset.select_runs(src, {"a2c"}, {"Acrobot-v1"})) == 2
    dst = tmp_path / "dst"
    assert subset.copy_runs(src, dst, runs, link=True) == 4
    assert (dst / "Acrobot-v1" / "a2c" / "baseline" / "default" / "seed_1" / "eval.csv").exists()
    assert not list(dst.glob("**/ppo"))
    with pytest.raises(FileExistsError):
        subset.copy_runs(src, dst, runs)
    assert len(list(src.glob("**/config.json"))) == 8  # источник не тронут


# ---------------------------------------------------------------- видео = эпизоды из итоговой оценки
def tiny(env_id: str):
    return environment_config(env_id, algo="ppo", phase="pilot", total_timesteps=128, eval_freq=32,
                              n_eval_episodes=1, final_eval_episodes=2,
                              algo_params={"n_steps": 64, "batch_size": 32})


def final_returns(run_dir: Path, policy: str) -> list[float]:
    with (run_dir / "final_episodes.csv").open(encoding="utf-8") as stream:
        return [float(r["return"]) for r in csv.DictReader(stream) if r["policy"] == policy]


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Acrobot-v1", "Pendulum-v1"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_video_episodes_reproduce_final_evaluation_returns(tmp_path, env_id, deterministic):
    video = video_api()
    cfg = tiny(env_id)
    run_experiment(cfg, tmp_path / "results")
    run_dir = cfg.run_dir(tmp_path / "results")
    run = video.load_run(run_dir)
    assert run.env_id == env_id
    paths, results = video.record_panels([run, run], env_id, tmp_path / "v" / "clip", episodes=2,
                                   deterministic=deterministic, formats=(".gif",))
    expected = final_returns(run_dir, "greedy" if deterministic else "sampled")
    assert [r for r, _ in results[0]] == pytest.approx(expected)
    assert results[0] == results[1]  # две одинаковые панели не мешают друг другу
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_random_policy_and_mp4_output(tmp_path):
    video = video_api()
    random_run = video.load_run("random")
    paths, results = video.record_panels([random_run], "CartPole-v1", tmp_path / "rnd", episodes=1,
                                   seed_start=5, formats=(".mp4",))
    assert paths[0].suffix == ".mp4" and paths[0].stat().st_size > 0
    assert results[0][0][1] >= 1


# ---------------------------------------------------------------- устойчивость к Windows и понятные подсказки
def test_core_check_ignores_crlf_line_endings(tmp_path):
    core = tmp_path / "core"
    shutil.copytree(doctor.CORE_DIR, core, ignore=shutil.ignore_patterns("__pycache__"))
    for path in core.iterdir():
        if path.is_file():
                        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert doctor.core_manifest_problems(core, doctor.MANIFEST) == []


def test_layout_notes_flag_stray_folders(tmp_path):
    for name in ("configs/x", "src", "tests", "A2C_RL", "my_tmp", "results_quick", "analysis_ppo"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "CORE_MANIFEST.json").write_text("{}", encoding="utf-8")
    text = " ".join(doctor.layout_notes(tmp_path))
    assert "A2C_RL: клон" in text and "my_tmp" in text and "results_quick" in text
    assert "analysis_ppo" not in text
    assert "configs/main_search_v1.json: ожидаемый" in text


def test_subset_explains_missing_runs_and_resolves_repo_root(tmp_path):
    empty = tmp_path / "A2C_RL" / "results_search"
    empty.mkdir(parents=True)
    (empty / "report.md").write_text("x", encoding="utf-8")
    assert "нет ни одного config.json" in subset.diagnose_empty(empty, {"a2c"})
    repo = tmp_path / "A2CvsPPO"
    fake_run(repo / "results_search", "CartPole-v1", "a2c", "baseline", "default", 0)
    assert subset.resolve_source(repo) == repo / "results_search"
    assert len(subset.select_runs(subset.resolve_source(repo), {"a2c"})) == 1
    assert "ни один не относится" in subset.diagnose_empty(subset.resolve_source(repo), {"ppo"})


def test_subset_link_falls_back_to_copy_when_hard_links_impossible(tmp_path, monkeypatch):
    src = tmp_path / "src"
    fake_run(src, "CartPole-v1", "a2c", "baseline", "default", 0)

    def refuse(*_):
        raise OSError("cross-device link")
    monkeypatch.setattr(subset.os, "link", refuse)
    runs = subset.select_runs(src, {"a2c"})
    assert subset.copy_runs(src, tmp_path / "dst", runs, link=True) == 1
    assert list((tmp_path / "dst").rglob("eval.csv"))
