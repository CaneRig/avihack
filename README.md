# Авито × Innoglobalhack: дополнительные микрокатегории и черновики объявлений

По тексту объявления о ремонте (исходная микрокатегория 101) решение находит услуги из микрокатегорий 102–106 и для каждой предлагает черновик объявления. Выход строго в формате организаторов: `{"drafts": [{"mcId": int, "title": str|null, "text": str|null}]}`.

Реализованы шаги 0–2 плана (`docs/08_repo_and_roadmap.md`): данные и оценка, слой 1 классификации, черновики. Код — по техническим заданиям из `docs/`.

## С чего начать

| Кто вы | Что читать |
| --- | --- |
| Хотите понять решение за 5 минут | [docs/00_overview.md](docs/00_overview.md) |
| Пишете код | [docs/08_repo_and_roadmap.md](docs/08_repo_and_roadmap.md) — порядок шагов, затем ТЗ нужного шага |
| Запускаете эксперименты | [docs/06_experiments.md](docs/06_experiments.md) — реестр, куда дописывать результаты |
| Готовите окружение | [docs/09_downloads.md](docs/09_downloads.md) — команды для пакетов и моделей |

## Документы

| Файл | Содержание |
| --- | --- |
| [docs/00_overview.md](docs/00_overview.md) | Что известно о данных, выбранная архитектура, отброшенные варианты, риски |
| [docs/01_spec_data_eval.md](docs/01_spec_data_eval.md) | ТЗ: подготовка данных, разбиения, инструмент оценки |
| [docs/02_spec_classification.md](docs/02_spec_classification.md) | ТЗ: определение микрокатегорий |
| [docs/03_spec_drafts.md](docs/03_spec_drafts.md) | ТЗ: генерация и проверка черновиков |
| [docs/04_spec_service_demo.md](docs/04_spec_service_demo.md) | ТЗ: единая функция предсказания, пакетный запуск, демо, замеры |
| [docs/05_presentation_plan.md](docs/05_presentation_plan.md) | План слайдов и графиков |
| [docs/06_experiments.md](docs/06_experiments.md) | Реестр экспериментов |
| [docs/07_mentor_questions.md](docs/07_mentor_questions.md) | Вопросы к ментору и принятые предположения |
| [docs/08_repo_and_roadmap.md](docs/08_repo_and_roadmap.md) | Структура репозитория и порядок реализации на 24 часа |
| [docs/09_downloads.md](docs/09_downloads.md) | Список загрузок с командами |
| [docs/10_sources.md](docs/10_sources.md) | Источники |

Исходные материалы организаторов лежат в `task/` и `dataset/public/`. Папка `scratch/` — разведочные скрипты, в git не попадает.

## Как запустить

Все команды — из корня репозитория, Python из `.venv` (на Windows добавьте `PYTHONUTF8=1`).

```bash
# 0. Проверка окружения: пакеты, GPU, Ollama по JSON-схеме, BERTA, длины текстов в токенах
.venv/Scripts/python scripts/check_env.py

# 1. Таблицы data/processed/ и сводка по данным
.venv/Scripts/python scripts/prepare_data.py

# 2. Ориентиры D1 и перебор слоя 1 (C1) с записью в reports/runs.jsonl (около 25 минут на CPU)
.venv/Scripts/python scripts/run_step1.py

# 3. Обучить слой 1 и сохранить в artifacts/tfidf/
.venv/Scripts/python scripts/train.py

# 4. Предсказание для файла (CSV/Parquet/JSONL со столбцом description; режимы classify | fast | full)
.venv/Scripts/python scripts/predict_batch.py dataset/public/public-test-dataset.parquet out.jsonl --mode classify

# 5. Оценка файла предсказаний на срезе (gold_raw, gold_consensus, silver_cv, silver_fold0, challenge)
.venv/Scripts/python scripts/evaluate.py out.jsonl.bot.jsonl --slice gold_raw

# 6. Черновики: эксперимент G1 (нужен запущенный Ollama с моделью из docs/09_downloads.md)
.venv/Scripts/python scripts/run_g1.py
```

Тесты (pytest в окружении нет, файлы запускаются напрямую): `PYTHONPATH=src .venv/Scripts/python tests/test_core.py`, `tests/test_format.py`.
