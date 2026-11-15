"""Шаг 0: проверка окружения (пакеты, GPU, Ollama по JSON-схеме, BERTA, длины текстов в токенах)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT: Path = Path(__file__).resolve().parents[1]
OLLAMA: str = "http://localhost:11434"
LLM: str = "hf.co/t-tech/T-lite-it-2.1-GGUF:Q4_K_M"


def ok(name: str, detail: str = "") -> None:
    """Печатает строку «ОК» для пройденной проверки."""
    print(f"ОК  {name} {detail}")


def check_packages() -> None:
    """Проверяет, что нужные пакеты импортируются, и печатает версии."""
    import importlib

    for m in ["torch", "transformers", "sklearn", "pandas", "pyarrow", "razdel", "pymorphy3", "gradio", "httpx", "accelerate", "yaml"]:
        mod = importlib.import_module(m)
        ok(f"пакет {m}", getattr(mod, "__version__", ""))


def check_gpu() -> None:
    """Проверяет CUDA и вычисления fp16 на GPU."""
    import torch

    assert torch.cuda.is_available(), "CUDA недоступна"
    x = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    assert torch.isfinite(x @ x).all()
    ok("GPU fp16", torch.cuda.get_device_name(0))


def check_ollama() -> None:
    """Просит LLM ответить по JSON-схеме и проверяет разбор ответа."""
    schema = {"type": "object", "properties": {"quote": {"type": "string"}, "status": {"type": "string", "enum": ["offered", "absent"]}}, "required": ["quote", "status"]}
    body = {
        "model": LLM,
        "messages": [{"role": "user", "content": "Объявление: «Устанавливаем розетки и выключатели». Предлагается ли электрика? Дай цитату и статус."}],
        "format": schema, "stream": False,
        "options": {"temperature": 0, "seed": 1, "num_ctx": 4096},
    }
    t = time.time()
    r = httpx.post(f"{OLLAMA}/api/chat", json=body, timeout=300).json()
    out = json.loads(r["message"]["content"])
    assert out["status"] in ("offered", "absent")
    ok("Ollama JSON-схема", f"{out} {time.time()-t:.1f}с, токенов вх/вых {r.get('prompt_eval_count')}/{r.get('eval_count')}")


def token_lengths() -> dict[str, float]:
    """Загружает BERTA, проверяет эмбеддинг на GPU и считает длины текстов в токенах."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ROOT / "models/BERTA")
    model = AutoModel.from_pretrained(ROOT / "models/BERTA", dtype=torch.float16).cuda().eval()
    enc = tok(["categorize: тест"], return_tensors="pt").to("cuda")
    with torch.no_grad():
        h = model(**enc).last_hidden_state
    assert torch.isfinite(h).all()
    ok("BERTA fp16", f"hidden {tuple(h.shape)}")
    tr = pd.read_json(ROOT / "dataset/public/public-train-annotated.jsonl", lines=True)
    te = pd.read_parquet(ROOT / "dataset/public/public-test-dataset.parquet")
    res: dict[str, float] = {}
    for name, df in [("train", tr), ("test", te)]:
        texts = (df["title"].fillna("") + "\n" + df["description"].fillna("")).tolist()
        n = np.array([len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]])
        # окна по 512 с перекрытием 64: шаг 448
        win = np.maximum(1, np.ceil(np.maximum(n - 64, 1) / 448)).astype(int)
        print(f"  {name}: токены p50={np.percentile(n,50):.0f} p90={np.percentile(n,90):.0f} p99={np.percentile(n,99):.0f} max={n.max()} | >512: {(n>512).mean():.1%} | окон p50={np.percentile(win,50):.0f} p95={np.percentile(win,95):.0f} | >4 окон: {(win>4).mean():.1%} | >2048 ток.: {(n>2048).mean():.1%}")
        res[name] = float(np.percentile(n, 50))
    ok("длины в токенах")
    return res


if __name__ == "__main__":
    for f in (check_packages, check_gpu, check_ollama, token_lengths):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print(f"ОШИБКА {f.__name__}: {e!r}")
            sys.exit(1)
