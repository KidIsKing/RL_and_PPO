# PPO в исследовании «A2C и PPO: стабильность обучения и чувствительность к гиперпараметрам»

PPO-сторона проекта на **том же протоколе**, что и A2C сокомандника ([akogarechau/A2C_RL](https://github.com/akogarechau/A2C_RL)).
Результаты, полученные здесь, сопоставимы с его A2C по построению, а не «на глаз».

## Как обеспечена сопоставимость

- `src/a2c_research/` — ядро протокола (среды, бюджеты, оценка, метрики, анализ), **скопировано без изменений**
  из репозитория сокомандника. `python -m ppo_tools.doctor` сверяет файлы с `CORE_MANIFEST.json`.
- `configs/main_search_v1.json` — тот же замороженный план: CartPole, Acrobot, Pendulum; seed 0–4; 1 620 позиций
  (810 для PPO).
- `src/ppo_tools/` — то, что добавлено здесь: проверка окружения, видео, выборка результатов.

| | Ваша прежняя версия | Теперь (общий протокол) |
|---|---|---|
| Среды | CartPole | CartPole, Acrobot, Pendulum |
| Бюджет | 50 000 шагов | 51 200 / 102 400 / 204 800 |
| Оценка по ходу | 10 эпизодов, SB3 `evaluate_policy` | 10 sampled + 10 deterministic, фиксированные seed, изоляция генераторов |
| Итоговая оценка | 30 эпизодов | 30 + 30 на отдельных seed |
| Метрики | итог, кривая | AUC, просадки, порог/удержание, бутстреп-интервалы, сетки, взаимодействия |
| Воспроизводимость | — | хеши конфигов и артефактов, замок `.running`, точные версии |

## 1. Установка (важно: версии должны совпасть точно)

Нужны Python **3.10.22**, torch 2.10.0, SB3 2.9.0, Gymnasium 1.4.0, NumPy 2.2.6, pandas 2.3.3, Matplotlib 3.10.8.
Версии входят в условия сопоставимости: при другой версии план откажется запускаться, а результаты не попали бы
в одну группу анализа с A2C.

```
conda create -n ppo python=3.10.22
conda activate ppo
pip install -r requirements-frozen.txt
pip install --no-deps --no-build-isolation -e ".[dev,video]"
python -m ppo_tools.doctor
```

`doctor` покажет, что именно не совпало, и как исправить. Если torch у вас вида `2.10.0+cu128`, поставьте
CPU-сборку `2.10.0`. Держите путь к проекту коротким (Windows).

## 2. Запуск

Все команды из корня проекта.

```
python -m a2c_research.sweep --algo ppo --quick --workers 1            # проверка за минуты, НЕ для статьи
python -m a2c_research.study --plan configs/main_search_v1.json --algos ppo --list
python -m a2c_research.study --plan configs/main_search_v1.json --algos ppo --workers 5
```

- Основное ядро PPO: 810 запусков. По его логам ≈ 38 CPU-часов (оценка, не замер на вашем ПК); при 5 процессах
  порядка 8 часов и больше. Можно частями: `--envs CartPole-v1`, `--groups learning_rate`, `--seeds 0 1`.
- Прервали? Запустите ту же команду: завершённые запуски проверяются и пропускаются. Если остался файл `.running`
  и процессов точно нет, удалите его.
- Дополнительные серии PPO на CartPole (675 позиций, ядро переиспользуется):
  `python -m a2c_research.study --plan configs/ppo_cartpole_all_v1.json --workers 5`
  (только PPO-специфичные: добавьте `--groups clip_range n_epochs batch_size grid_lr_clip grid_lr_epochs`).
  Эти серии у сокомандника объявлены, но не выполнены.

## 3. Анализ

PPO отдельно:
```
python -m a2c_research.analyze --results results_search --study-plan configs/main_search_v1.json --out analysis_ppo
```

Сравнение с A2C сокомандника. Его каталог содержит и PPO, а независимые копии одного запуска анализ
**исключает обе** («Conflicting duplicate run identity»; проверено). Поэтому берите у него только A2C:
```
python -m ppo_tools.subset --src ../A2C_RL/results_search --dst results_a2c_only --algos a2c --link
python -m a2c_research.analyze --results results_search results_a2c_only --study-plan configs/main_search_v1.json --out analysis_compare
```
Его результаты лежат в ветке `a2c` репозитория RATARETO/A2CvsPPO (каталог `results_search`).
Описание метрик и файлов: его `docs/METHODOLOGY.md` и `docs/RESULTS.md`.

## 4. Видео

```
python -m ppo_tools.batch_videos --results results_search --env CartPole-v1 --list
python -m ppo_tools.batch_videos --results results_search --env CartPole-v1 --seed 0
python -m ppo_tools.render --compare random results_search/ppo/baseline/default/seed_0
```
Серия: минимум | по умолчанию | максимум (`--mode ladder` — четыре ступени), сетка — четыре угла. По умолчанию
sampled и те же начальные состояния, что в итоговой оценке: return на кадре равен числу в `final_episodes.csv`.
Результат — `videos/`.

## 5. Проверки

`python -m pytest` — 93 теста (78 ядра + 15 инструментов); `python -m ruff check src tests`.

## Ограничения

- Проверено на Linux с точно такими же версиями; на Windows не запускалось.
- Подтверждающей фазы (seed 1000+) нет: её не определил и сокомандник.
- Фиксированные бюджеты не доказывают полную сходимость и общее превосходство алгоритма.
- Старый `stress_test` (эпизоды длиннее 500) не перенесён.
