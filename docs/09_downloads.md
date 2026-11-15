# Список загрузок

Всё скачивает и устанавливает человек. Команды выполняются из корня репозитория. Размеры — со страниц моделей; имена файлов моделей проверены по страницам Hugging Face 15.11.2026, установкой не проверялись.

Что уже есть на машине (проверено запуском):

- `uv` 0.13.7 и окружение `.venv` на Python 3.13 с пакетами `torch` 2.14.1+cu130, `transformers` 5.23.0, `scikit-learn` 1.9.1, `pandas` 3.1.1, `pyarrow`, `scipy`, `numpy`, `tqdm`, `pyyaml`, `huggingface_hub` 2.4.0;
- `torch` видит RTX 2080 Super Max-Q (8 GB), вычисления в fp16 на GPU работают;
- Ollama установлен; свободно 117 GB на диске.

В `.venv` нет: `accelerate`, `razdel`, `pymorphy3`, `gradio`, `httpx`.

Внимание: при проверке команда `ollama list` не ответила за две минуты. Перед загрузкой моделей убедитесь, что приложение Ollama запущено.

## Обязательное (около 5,7 GB)

1. Недостающие пакеты (десятки MB). Версии не закреплены намеренно: проверить их существование на дату хакатона не удалось; после установки зафиксируйте состав окружения второй командой.

```bash
uv pip install accelerate razdel pymorphy3 pymorphy3-dicts-ru gradio httpx
```

```bash
uv pip freeze > requirements.lock.txt
```

2. Энкодер для классификации, `sergeyzh/BERTA` (0,52 GB, MIT). Если команды `huggingface-cli` нет, та же команда пишется как `hf download`.

```bash
huggingface-cli download sergeyzh/BERTA --local-dir models/BERTA
```

3. LLM для проверки спорных случаев и генерации, T-lite-it-2.1 в квантовании Q4_K_M (5,03 GB, Apache-2.0), сразу в хранилище Ollama:

```bash
ollama pull hf.co/t-tech/T-lite-it-2.1-GGUF:Q4_K_M
```

## Желательное для экспериментов

| Что | Зачем | Размер | Команда |
| --- | --- | --- | --- |
| `deepvk/USER2-base` (Apache-2.0) | Энкодер с длинным контекстом, эксперимент C2b | 0,6 GB | `huggingface-cli download deepvk/USER2-base --local-dir models/USER2-base` |
| `sergeyzh/rubert-mini-frida` (MIT) | Маленький энкодер для CPU, эксперимент C2b | 0,13 GB | `huggingface-cli download sergeyzh/rubert-mini-frida --local-dir models/rubert-mini-frida` |
| Qwen3.5-9B, Q4_K_M (Apache-2.0) | Запасная LLM, эксперименты C4 и G6 | 5,68 GB | `ollama pull hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M` |
| Gemma 4 E4B, Q4_K_M (Apache-2.0) | Лёгкая запасная LLM, эксперименты C4 и G6 | 4,98 GB | `ollama pull hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M` |
| T-lite-it-2.1, Q5_K_M | Лучшее качество на карте 12 GB, эксперимент S3 | 5,85 GB | `ollama pull hf.co/t-tech/T-lite-it-2.1-GGUF:Q5_K_M` |

## Необязательное

| Что | Зачем | Размер | Команда |
| --- | --- | --- | --- |
| `cointegrated/rubert-base-cased-nli-threeway` | NLI-проверка черновиков, эксперимент G5. Лицензия на странице не указана | 0,71 GB | `huggingface-cli download cointegrated/rubert-base-cased-nli-threeway --local-dir models/rubert-nli` |
| `llama.cpp` (сборка для Windows с CUDA 12) | Замена Ollama, если понадобится тонкая настройка слотов | — | Скачать архив со страницы https://github.com/ggml-org/llama.cpp/releases. Имя архива для CUDA 12 подтвердить не удалось: последний выпуск на 13.11.2026 не содержал сборок |
| `cleanlab` | Поиск ошибок разметки, эксперимент C9 (можно обойтись без него) | небольшой | `uv pip install cleanlab` |

## Если что-то не ставится

- На второй машине (RTX 3060) `torch` ставится командой с pytorch.org для Windows, Pip и той версии CUDA, которую поддерживает драйвер; на текущей машине работает сборка `cu130`.
- `ollama pull hf.co/...` не работает → скачать файл командой `huggingface-cli download t-tech/T-lite-it-2.1-GGUF T-lite-it-2.1-Q4_K_M.gguf --local-dir models/T-lite-it-2.1-GGUF` и зарегистрировать его в Ollama через `Modelfile` со строкой `FROM` и путём к файлу.
- T-lite недоступна → основной становится Qwen3.5-9B (режим рассуждений отключить).
