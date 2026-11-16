"""Общая часть сборки ответа для бота: обработка одной строки и запись CSV testcase,microcats,descriptions.

Используется и scripts/make_submission.py, и демо-сервером app/server.py.
"""
from __future__ import annotations

import csv
import json
from typing import Any, Iterable, TextIO

from .predict import Predictor

JUDGE_KEYS = ("candidates", "sources", "roles", "strict_roles", "arbiter", "disagree", "fallback", "errors")


def draft_origin(item: dict[str, Any] | None, drafts_by: str) -> dict[str, Any]:
    """Чем получен текст: by = llm | regen | template; violations и причина отката (поведение проверок не меняется)."""
    if drafts_by != "llm":
        return {"by": "template", "violations": [], "reason": "выбран режим template"}
    if item is None:
        return {"by": None, "violations": [], "reason": ""}
    src = item.get("source")
    if src == "llm":
        return {"by": "llm", "violations": [], "reason": ""}
    if src == "llm_regen":
        return {"by": "regen", "violations": item.get("viol_first") or [], "reason": "первая попытка не прошла проверки"}
    viol = item.get("violations") or []
    if item.get("error"):
        reason = f"ошибка LLM: {item['error']}"
    elif viol:
        reason = "проверки не пройдены дважды: " + "; ".join(viol)
    else:
        reason = "LLM была недоступна (Ollama не ответила за 3 с), использован шаблон"
    return {"by": "template", "violations": viol, "reason": reason}


def process_row(pr: Predictor, tc: str, desc: str, drafts_by: str) -> dict[str, Any]:
    """Одна строка: категории и черновики. drafts_by: llm | template | none (для Predictor нужен режим full при llm)."""
    ans, info = pr.predict_batch([{"description": desc, "title": "", "mcId": 101}], progress=False)[0]
    drafts = ans["drafts"]
    items = info.get("drafts") or []  # записи Drafter.make (в режиме full): source, violations, error
    if drafts_by == "template" and drafts:
        items = pr.drafter().make(desc, "", [d["mcId"] for d in drafts], use_llm=False)
        drafts = [{"mcId": i["mcId"], "title": i["title"], "text": i["text"]} for i in items]
    by_cat = {i["mcId"]: i for i in items if isinstance(i, dict) and "mcId" in i}
    for d in drafts:
        d.update(draft_origin(by_cat.get(d["mcId"]), drafts_by))
    return {"testcase": tc, "drafts": drafts, "drafts_by": drafts_by, "source": info.get("source"), "error": info.get("error"),
            "judge": {k: (info.get("judge") or {}).get(k) for k in JUDGE_KEYS}}


def row_problems(r: dict[str, Any]) -> list[str]:
    """Технические сбои строки, которые нельзя выдавать за «услуг нет»: ошибка, сбой судьи, откат на корни, черновик без текста."""
    out = []
    if r.get("error"):
        out.append(f"ошибка: {r['error']}")
    j = r.get("judge") or {}
    if j.get("errors"):
        out.append(f"сбой судьи: {j['errors']}")
    if r.get("source") == "keywords_fallback" or j.get("fallback"):
        out.append("категории найдены по ключевым корням (модель не ответила)")
    if r.get("drafts_by") != "none" and any(not d.get("text") for d in r.get("drafts") or []):
        out.append("у категории нет текста черновика")
    return out


def write_submission_csv(f: TextIO, testcases: Iterable[str], done: dict[str, dict], with_101: bool) -> None:
    """Пишет CSV для бота: microcats и descriptions — JSON-списки."""
    w = csv.writer(f)
    w.writerow(["testcase", "microcats", "descriptions"])
    for tc in testcases:
        dr = done[tc]["drafts"]
        cats = ([101] if with_101 else []) + [d["mcId"] for d in dr]
        texts = [((d.get("title") or "") + "\n" + d["text"]).strip() for d in dr if d.get("text")]
        w.writerow([tc, json.dumps(cats), json.dumps(texts, ensure_ascii=False)])
