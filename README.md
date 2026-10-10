# PPO vs A2C на CartPole: стабильность обучения и чувствительность к гиперпараметрам

Студенческое исследование. Код основан на учебнике Stone Tao «RL with Lux: 1 — Intro to RL»
(часть CartPole + PPO), перенесён на современный Gymnasium / Stable-Baselines3 2.9.

## Установка (Python 3.10.11)

Windows PowerShell:

    py -3.10 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

macOS / Linux:

    python3.10 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

## Быстрый старт

    python -m src.main                  # демо: случайная политика -> обучение PPO -> видео (1-3 минуты)
    python -m src.sweep --quick         # проверка всего конвейера: 20k шагов, 2 сида, 2 серии -> results_quick/
    python -m src.analyze --results results_quick --out analysis_quick

## Основной рабочий процесс

    python -m src.sweep --list                       # какие серии есть и сколько это запусков
    python -m src.sweep                              # все серии PPO, 5 сидов (параллельно, 5 процессов)
    python -m src.sweep --groups clip_range --seeds 10
    python -m src.analyze                            # таблицы и графики -> analysis/
    python -m src.render --run results/ppo/baseline/default/seed_0   # видео для выбранного запуска
    python -m src.train --algo ppo --set learning_rate=0.001 --set clip_range=0.1 --video   # один запуск вручную

Прерванный свип можно запустить заново: готовые запуски пропускаются.

## Структура

    src/config.py       ExperimentConfig — один объект описывает запуск целиком
    src/experiments.py  ЧТО запускаем: серии гиперпараметров PPO и заготовка для A2C
    src/algos.py        реестр алгоритмов (ppo, a2c), запись фактических гиперпараметров
    src/envs.py         создание сред
    src/callbacks.py    оценка политики каждые N шагов -> кривая обучения + диагностика
    src/train.py        run_experiment(cfg): обучение, сохранение результатов
    src/sweep.py        параллельный запуск серий
    src/analyze.py      метрики стабильности, графики, чувствительность параметров
    src/video.py        запись видео с подписями (mp4 / gif / webm)
    src/render.py       видео для уже обученной модели
    src/main.py         демо

## Что сохраняется для каждого запуска

    results/<algo>/<серия>/<конфигурация>/seed_<k>/
        config.json   конфиг + фактические гиперпараметры + версии библиотек
        eval.csv      кривая обучения: eval_sampled_* (основная метрика), eval_greedy_*, диагностика (approx_kl, clip_fraction, ...)
        summary.json  итоговая оценка и время обучения
        model.zip     обученная модель

## Важные замечания

- **Какая оценка основная.** Для кривых и сравнения алгоритмов используется `eval_sampled_*` (действия сэмплируются из
  политики — это то, что реально оптимизируют PPO и A2C). `eval_greedy_*` (argmax) нужна для видео: у необученной сети
  greedy-политика у некоторых сидов случайно держит шест все 500 шагов, поэтому на кривых она вводит в заблуждение.
- **Пропуск готовых запусков.** Запуск пропускается, только если сохранённый `config.json` совпадает с текущим конфигом.
  Если поменялся код оценки/логирования (как здесь), удали папки `results*/` и пересчитай.
- **«Решил задачу»** = оценка >= 475 держится 2 контрольные точки подряд (одиночный всплеск не считается).
