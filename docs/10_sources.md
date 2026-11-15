# Источники

Собраны веб-поиском 15.11.2026. Пометка «не проверено» — утверждение взято из поисковой выдачи или пересказа, страница целиком не читалась.

## Энкодеры

- BERTA — карточка модели, результаты ruMTEB (классификация 69,8; многометочная 51,0), префиксы, лицензия MIT: https://huggingface.co/sergeyzh/BERTA
- rubert-mini-frida — 32,3 млн параметров, ruMTEB 63,1 / 46,3: https://huggingface.co/sergeyzh/rubert-mini-frida
- USER2-base и USER2-small — контекст 8 192 токена, префикс `classification: `, ruMTEB 61,67 / 36,61: https://huggingface.co/deepvk/USER2-base , https://huggingface.co/deepvk/USER2-small
- USER-bge-m3 — сравнение с bge-m3; замечание, что многометочная часть ruMTEB состоит из двух наборов: https://huggingface.co/deepvk/USER-bge-m3
- RuModernBERT — FlashAttention необязателен: https://huggingface.co/deepvk/RuModernBERT-base
- EmbeddingGemma — не поддерживает fp16: https://huggingface.co/google/embeddinggemma-300m
- Qwen3-Embedding-0.6B: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- GigaEmbeddings (3 млрд параметров, слишком велика): https://arxiv.org/html/2510.22369v1
- SetFit — метод для малого числа примеров: https://huggingface.co/blog/setfit , https://arxiv.org/pdf/2302.08957

## Локальные LLM

- T-lite-it-2.1 GGUF — файлы, лицензия Apache-2.0, русские бенчмарки по квантованиям (арена 83,9 для BF16; 71,7 для Q4_K_M; 78,6–78,9 для Q5): https://huggingface.co/t-tech/T-lite-it-2.1-GGUF
- Qwen3.5-9B GGUF: https://huggingface.co/unsloth/Qwen3.5-9B-GGUF ; отключение режима рассуждений: https://unsloth.ai/docs/models/qwen3.5.md
- Gemma 4 E4B GGUF: https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF ; лицензия Apache-2.0: https://the-decoder.com/googles-gemma-4-is-now-available-with-apache-2-0-licensing-for-the-first-time/
- GigaChat3 Lightning — в официальном репозитории нет квантования Q4: https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-GGUF
- Gemma 3 12B QAT и Saiga Gemma3 12B — файлы 7–8 GB: https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf , https://huggingface.co/IlyaGusev/saiga_gemma3_12b_gguf
- Бенчмарк MERA (свериться с актуальной таблицей): https://mera.a-ai.ru/en

## Запуск LLM

- Ollama, вывод по JSON-схеме: https://docs.ollama.com/capabilities/structured-outputs
- Ollama, одновременные запросы и их цена по памяти (не проверено): https://www.glukhov.org/post/2025/05/how-ollama-handles-parallel-requests/
- llama.cpp, сервер: параметры `--json-schema`, `response_format`, `-np`: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md
- llama.cpp, выпуски: https://github.com/ggml-org/llama.cpp/releases
- vLLM, установка (нет поддержки Windows без WSL): https://docs.vllm.ai/en/v0.19.1/getting_started/installation/gpu/

## Методы

- cleanlab, многометочная разметка: https://docs.cleanlab.ai/v2.3.0/tutorials/multilabel_classification.html , https://cleanlab.ai/blog/multilabel/
- FrugalGPT, каскады моделей: https://arxiv.org/pdf/2305.05176
- Коррекция сдвига априорных вероятностей (EM с калибровкой): https://arxiv.org/abs/1901.06852v4
- Обучение на метках документа с оценкой фрагментов (Angelidis, Lapata): https://arxiv.org/abs/1711.09645v2
- Классификация русских текстов: RuBERT против TF-IDF: https://arxiv.org/pdf/2206.07427
- SummaC, проверка согласованности через NLI по предложениям (сбалансированная точность 74,4 %): https://arxiv.org/pdf/2111.09525
- NLI-модели: https://huggingface.co/cointegrated/rubert-base-cased-nli-threeway , https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
- Метрика организаторов: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

## Что не подтверждено источниками и опирается на расчёт или опыт

- Скорости обучения и вывода на RTX 2080 Super, расход видеопамяти LLM с контекстом.
- Типичная доля объявлений, уходящих к LLM в каскаде (10–30 %).
- Приёмы для отрицаний на уровне признаков и стартовые параметры TF-IDF (кроме тех, что измерены разведочным бейзлайном).
