"""Клиент локальной LLM (Ollama): вывод по JSON-схеме, учёт токенов и времени. Адрес и модель — в конфиге."""
from __future__ import annotations

import json
import time
from typing import Any

import httpx


class LLMError(RuntimeError):
    """Сервер LLM недоступен или вернул невалидный ответ."""


class LLMClient:
    """Тонкая обёртка над /api/chat Ollama; безопасна для вызова из нескольких потоков."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.c: dict[str, Any] = cfg["llm"]
        self.http: httpx.Client = httpx.Client(base_url=self.c["base_url"], timeout=self.c["timeout"])

    def available(self) -> bool:
        """Отвечает ли сервер (быстрая проверка перед генерацией)."""
        try:
            return self.http.get("/api/tags", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False

    def chat_json(self, system: str, user: str, schema: dict[str, Any],
                  temperature: float | None = None, **opts: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Один вызов: (разобранный JSON, статистика {tokens_in, tokens_out, seconds}). opts перекрывают options Ollama."""
        c = self.c
        options = {"temperature": c["temperature"] if temperature is None else temperature, "top_p": c["top_p"],
                   "seed": c["seed"], "num_ctx": c["num_ctx"], "num_predict": c["num_predict"], **opts}
        body: dict[str, Any] = {"model": c["model"], "stream": False, "format": schema, "options": options,
                                "keep_alive": c["keep_alive"],
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if c.get("think") is not None:
            body["think"] = c["think"]
        t0 = time.time()
        trace: list[dict[str, Any]] = []  # по одному элементу на попытку: диагностика пустых ответов
        for attempt in range(self.c["retries"] + 1):  # пустой или битый ответ -> повтор с другим зерном
            body["options"]["seed"] = options["seed"] + attempt
            try:
                r = self.http.post("/api/chat", json=body)
                r.raise_for_status()
                j = r.json()
                trace.append({"done_reason": j.get("done_reason"), "eval_count": j.get("eval_count"),
                              "prompt_eval_count": j.get("prompt_eval_count"), "len": len(j.get("message", {}).get("content", ""))})
                data = json.loads(j["message"]["content"])
                break
            except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
                if attempt == self.c["retries"]:
                    err = LLMError(repr(e))
                    err.trace = trace  # type: ignore[attr-defined]
                    raise err from e
        return data, {"tokens_in": j.get("prompt_eval_count", 0), "tokens_out": j.get("eval_count", 0),
                      "seconds": time.time() - t0, "attempts": len(trace), "trace": trace}
