"""Проверка окружения ПЕРЕД долгим запуском: совпадает ли оно с тем, под которое заморожен план.

    python -m ppo_tools.doctor
    python -m ppo_tools.doctor --plan configs/ppo_cartpole_all_v1.json

Зачем. Замороженный план хранит точные версии Python и библиотек. При несовпадении
`a2c_research.study` откажется запускать обучение, а если бы запустил, результаты попали бы
в ДРУГОЙ «страт» анализа (версии входят в условия сопоставимости) и не сравнивались бы
с A2C сокомандника. Лучше узнать об этом за минуту, чем после ночи расчётов.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = ROOT / "src" / "a2c_research"
MANIFEST = ROOT / "CORE_MANIFEST.json"
DEFAULT_PLAN = ROOT / "configs" / "main_search_v1.json"

FIX_PYTHON = (
    "Нужен Python именно этой версии (до патча). Варианты:\n"
    "      uv python install {want}   затем   uv venv --python {want} .venv\n"
    "      conda create -n ppo python={want}"
)


def version_problems(wanted: dict, actual: dict) -> list[str]:
    """Сравнивает версии из плана и из текущего окружения; возвращает читаемые расхождения."""
    problems = []
    for name, want in wanted.items():
        have = actual.get(name)
        if have == want:
            continue
        if name == "python":
            problems.append(f"python: в плане {want}, у вас {have}.\n    " + FIX_PYTHON.format(want=want))
        else:
            problems.append(f"{name}: в плане {want}, у вас {have}.  "
                            f"Исправить: pip install -r requirements-frozen.txt")
    return problems


def core_manifest_problems(core_dir: Path = CORE_DIR, manifest: Path = MANIFEST) -> list[str]:
    """Файлы ядра должны байт-в-байт совпадать с версией сокомандника (иначе протокол «уплыл»)."""
    if not manifest.exists():
        return [f"нет файла {manifest.name}: нечем подтвердить, что ядро не изменено"]
    expected = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    problems = []
    for name, digest in expected.items():
        path = core_dir / name
        if not path.exists():
            problems.append(f"{name}: файл ядра отсутствует")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"{name}: изменён относительно версии сокомандника")
    extra = {p.name for p in core_dir.glob("*.py")} - set(expected)
    problems += [f"{name}: лишний файл в ядре" for name in sorted(extra)]
    return problems


def report(title: str, problems: list[str]) -> bool:
    print(f"[{'OK' if not problems else 'ОШИБКА'}] {title}")
    for text in problems:
        print(f"    - {text}")
    return not problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args(argv)

    try:
        from a2c_research.io import runtime_metadata
        runtime = runtime_metadata()
    except Exception as error:  # нет пакета или библиотеки
        print(f"[ОШИБКА] не удалось собрать описание окружения: {error!r}\n"
              "    Установите проект: pip install -r requirements-frozen.txt && "
              "pip install --no-deps --no-build-isolation -e .")
        return 1

    ok = True
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    ok &= report(f"версии окружения совпадают с планом {args.plan.name}",
                 version_problems(plan["versions"], runtime["versions"]))
    ok &= report("код обучения совпадает с замороженным планом",
                 [] if plan["training_sha256"] == runtime["training_sha256"] else
                 ["training_sha256 плана и кода различаются: ядро a2c_research изменено или "
                  "запущено под другой версией Python (ast.dump зависит от версии)"])
    ok &= report("ядро идентично версии сокомандника (CORE_MANIFEST.json)", core_manifest_problems())

    if ok:
        try:
            from a2c_research.study import compile_study
            configs = compile_study(plan)
            ok &= report(f"план читается: {len(configs)} позиций запусков", [])
        except Exception as error:
            ok &= report("план читается", [str(error)])
    cpus = os.cpu_count() or 1
    print(f"\nПроцессоров: {cpus}. Для --workers обычно берут не больше (число ядер − 1); "
          f"один процесс обучения использует один поток PyTorch.")
    print("Итог:", "всё готово к запуску." if ok else "исправьте пункты выше до запуска.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
