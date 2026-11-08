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

## 1. Установка

Нужны Python **3.10.22**, torch 2.10.0, SB3 2.9.0, Gymnasium 1.4.0, NumPy 2.2.6, pandas 2.3.3, Matplotlib 3.10.8.
Версии (включая патч Python) входят в условия сопоставимости: при другой версии замороженный план откажется
запускаться, а результаты не попали бы в одну группу анализа с A2C.

Работайте в обычном `venv`. Нужен только интерпретатор 3.10.22; выберите один вариант.

**Вариант A (рекомендуется): интерпретатор из уже созданного окружения conda.**
Если когда-то выполнено `conda create -n ppo python=3.10.22`, файл уже на диске. Conda активировать не нужно.
Git Bash (подставьте своё имя пользователя):

```
/c/Users/<имя>/.conda/envs/ppo/python.exe -m venv .venv
source .venv/Scripts/activate          # PowerShell: .venv\Scripts\Activate.ps1
python --version                       # должно быть 3.10.22
python -m pip install -r requirements-frozen.txt
python -m pip install -r requirements-video.txt       # только для видео; без него остальное работает
python -m pip install --no-deps --no-build-isolation -e .
python -m ppo_tools.doctor
```

**Вариант B: без conda, через uv** (не гарантирую, что 3.10.22 там есть):

```
python -m pip install uv
uv python list --all-versions | grep 3.10.22    # если пусто — вариант B не подходит
uv python install 3.10.22
uv venv --python 3.10.22 .venv
source .venv/Scripts/activate
python -m pip install -r requirements-frozen.txt
python -m pip install -r requirements-video.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m ppo_tools.doctor
```

**Вариант C: только если 3.10.22 недоступен (например, 3.10.11).** Проект работает, но план нужно заморозить заново
под ваши версии (библиотеки всё равно точно по `requirements-frozen.txt`). `doctor` по основному плану покажет
расхождение по Python, это ожидаемо:

```
python -m a2c_research.study --prepare configs/ppo_local_v1.json --suite core --algos ppo
python -m ppo_tools.doctor --plan configs/ppo_local_v1.json
python -m a2c_research.study --plan configs/ppo_local_v1.json --workers 5
```

Ограничение: запуски окажутся в другой группе анализа, чем A2C сокомандника на 3.10.22. Инструмент не сопоставит
их напрямую. Этот вариант на Windows не проверялся.

`--no-deps` не ставит дополнительные зависимости, поэтому видео-пакеты идут отдельным файлом. Если torch вида
`2.10.0+cu128`, поставьте CPU-сборку `2.10.0`. Если установка оборвалась, проверьте `python -m pip list`. Держите
путь к проекту коротким.

## 2. Запуск

Все команды из корня проекта, внутри активированного `.venv`. Порядок работы:

```
# 0. Проверка окружения, ядра и структуры папок (все пункты должны быть OK)
python -m ppo_tools.doctor

# 1. Пробный запуск за минуты (НЕ для статьи); потом папку results_quick можно удалить
python -m a2c_research.sweep --algo ppo --quick --workers 1

# 2. Что будет запущено и что уже готово (в конце: Compatible completed / pending)
python -m a2c_research.study --plan configs/main_search_v1.json --algos ppo --list

# 3. Основное ядро PPO: 810 запусков, 3 среды (≈ 38 CPU-часов, порядка 8 часов и больше при 5 процессах)
python -m a2c_research.study --plan configs/main_search_v1.json --algos ppo --workers 5 2>&1 | tee run_main.log
#    частями: --envs CartPole-v1   (или Acrobot-v1, Pendulum-v1);  --groups learning_rate;  --seeds 0 1

# 4. Дополнительные серии CartPole (405 новых запусков, ядро переиспользуется), две части:
#    4а. серии, у которых есть аналоги у A2C (165 запусков), нужны для сравнения
python -m a2c_research.study --plan configs/ppo_cartpole_all_v1.json --groups gamma normalize_advantage vf_coef max_grad_norm net_arch activation n_envs grid_gamma_lambda --workers 5 2>&1 | tee run_extra_a.log
#    4б. серии, специфичные для PPO, без аналогов у A2C (240 запусков)
python -m a2c_research.study --plan configs/ppo_cartpole_all_v1.json --groups clip_range n_epochs batch_size grid_lr_clip grid_lr_epochs --workers 5 2>&1 | tee run_extra_b.log
#    весь дополнительный план сразу: убрать --groups

# 5. Резервная копия результатов (папка в .gitignore, Git её не хранит); положите архив на другой диск или в облако
tar -czf results_search_backup_2026-11-08.tar.gz results_search analysis_ppo
```

- Прервали? Запустите ту же команду: завершённые запуски проверяются и пропускаются. Если остался файл `.running`
  и процессов точно нет, удалите его. Отключите сон компьютера; один процесс оставьте свободным.
- Дополнительные серии есть только для CartPole. Для Acrobot и Pendulum выполняется основное ядро (п. 3).
- Эти серии у сокомандника объявлены (для PPO), но не выполнены.

## 3. Анализ

PPO отдельно:
```
python -m a2c_research.analyze --results results_search --study-plan configs/main_search_v1.json --out analysis_ppo
```

Сравнение с A2C сокомандника. Готовые запуски A2C лежат в ветке `a2c` репозитория RATARETO/A2CvsPPO
(каталог `results_search`, около 350 МБ); в самом A2C_RL готовых запусков нет. Клонируйте их **рядом** с проектом,
не внутри. Его каталог содержит и PPO, а независимые копии одного запуска анализ **исключает обе**
(«Conflicting duplicate run identity»; проверено), поэтому берите у него только A2C:
```
git clone --depth 1 --branch a2c --single-branch https://github.com/RATARETO/A2CvsPPO.git ../A2CvsPPO
python -m ppo_tools.subset --src ../A2CvsPPO --dst results_a2c_only --algos a2c --link
python -m a2c_research.analyze --results results_search results_a2c_only --study-plan configs/main_search_v1.json --out analysis_compare
```
Путь `--src` считается от папки запуска команды (запускайте из корня проекта). `--link` при невозможности жёстких
ссылок (другой диск) сам копирует файлы. Если `results_a2c_only` осталась от неудачной попытки, удалите её.
Описание метрик и файлов: его `docs/METHODOLOGY.md` и `docs/RESULTS.md`.

## 4. Видео

Видео пишется только для серий, у которых есть запуски; для остальных будет «ПРОПУСК ... нет модели»: это не ошибка,
досчитайте серию (п. 4 раздела 2).

```
# что будет записано (ничего не пишет)
python -m ppo_tools.batch_videos --results results_search --env CartPole-v1 --list

# все готовые серии CartPole, seed 0; потом только выбранные серии
python -m ppo_tools.batch_videos --results results_search --env CartPole-v1 --seed 0
python -m ppo_tools.batch_videos --results results_search --env CartPole-v1 --seed 0 --groups gamma vf_coef clip_range n_epochs

# другие среды (там готовы только серии основного ядра)
python -m ppo_tools.batch_videos --results results_search --env Acrobot-v1 --seed 0
python -m ppo_tools.batch_videos --results results_search --env Pendulum-v1 --seed 0

# четыре ступени вместо «мин | по умолчанию | макс»; детерминированная политика вместо sampled
python -m ppo_tools.batch_videos --results results_search --env CartPole-v1 --groups learning_rate --mode ladder
python -m ppo_tools.batch_videos --results results_search --env CartPole-v1 --greedy

# одна модель, либо «до и после» (случайная политика и обученная)
python -m ppo_tools.render --run results_search/ppo/baseline/default/seed_0
python -m ppo_tools.render --compare random results_search/ppo/baseline/default/seed_0
```
Серия: минимум | по умолчанию | максимум, сетка — четыре угла. По умолчанию sampled и те же начальные состояния,
что в итоговой оценке: return на кадре равен числу в `final_episodes.csv`. Результат — `videos/` (mp4 и gif).

## 5. Структура проекта

В корне должно быть: `configs/` (3 файла), `src/` (`a2c_research`, `ppo_tools`), `tests/`, `CORE_MANIFEST.json`,
`README.md`, `pyproject.toml`, `requirements-frozen.txt`, `requirements-video.txt`; локально ещё `.venv/`,
`results_search/`, `analysis_*/`, `videos/`. Всё лишнее (клон `A2C_RL`, `results_quick`) `doctor` перечислит в пункте
«структура проекта». Окончания строк (CRLF после Git на Windows) `doctor` игнорирует: на протокол они не влияют.

## 6. Проверки

`python -m pytest` — 97 тестов (78 ядра + 19 инструментов; без видео-пакетов видео-тесты пропускаются);
`python -m ruff check src tests`.

## Ограничения

- Проверено на Linux с точно такими же версиями; на Windows ядро и анализ у вас отработали (810 запусков PPO).
- Подтверждающей фазы (seed 1000+) нет: её не определил и сокомандник.
- Фиксированные бюджеты не доказывают полную сходимость и общее превосходство алгоритма.
- Старый `stress_test` (эпизоды длиннее 500) не перенесён.
