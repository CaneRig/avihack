"""Общая часть сборки ответа для бота: обработка одной строки и запись CSV testcase,microcats,descriptions.

Используется и scripts/make_submission.py, и демо-сервером app/server.py.
"""
from __future__ import annotations

import csv
import json
from typing import Any, Iterable, TextIO

from .predict import Predictor

JUDGE_KEYS = ("candidates", "sources", "roles", "strict_roles", "arbiter", "disagree", "fallback", "errors")


def process_row(pr: Predictor, tc: str, desc: str, drafts_by: str) -> dict[str, Any]:
    """Одна строка: категории и черновики. drafts_by: llm | template | none (для Predictor нужен режим full при llm)."""
    ans, info = pr.predict_batch([{"description": desc, "title": "", "mcId": 101}], progress=False)[0]
    drafts = ans["drafts"]
    if drafts_by == "template" and drafts:
        items = pr.drafter().make(desc, "", [d["mcId"] for d in drafts], use_llm=False)
        drafts = [{"mcId": i["mcId"], "title": i["title"], "text": i["text"]} for i in items]
    return {"testcase": tc, "drafts": drafts, "drafts_by": drafts_by, "source": info.get("source"), "error": info.get("error"),
            "judge": {k: (info.get("judge") or {}).get(k) for k in JUDGE_KEYS}}


def write_submission_csv(f: TextIO, testcases: Iterable[str], done: dict[str, dict], with_101: bool) -> None:
    """Пишет CSV для бота: microcats и descriptions — JSON-списки."""
    w = csv.writer(f)
    w.writerow(["testcase", "microcats", "descriptions"])
    for tc in testcases:
        dr = done[tc]["drafts"]
        cats = ([101] if with_101 else []) + [d["mcId"] for d in dr]
        texts = [((d.get("title") or "") + "\n" + d["text"]).strip() for d in dr if d.get("text")]
        w.writerow([tc, json.dumps(cats), json.dumps(texts, ensure_ascii=False)])
