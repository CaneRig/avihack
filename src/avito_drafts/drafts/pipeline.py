"""Черновики для найденных категорий: фрагменты -> LLM -> проверки -> (при провале) шаблон."""
from __future__ import annotations

import re
from typing import Any

from ..config import ROOT
from ..llm import LLMClient, LLMError
from .checks import run_checks
from .fragments import Fragment, split_fragments
from .template import load_categories, template_draft

SCHEMA: dict[str, Any] = {"type": "object", "required": ["title", "text"],
                          "properties": {"title": {"type": "string"}, "text": {"type": "string"}}}


def shorten_title(t: str, limit: int) -> str:
    """Укорачивает длинный заголовок: часть до «:», «—» или запятой, иначе обрезка по границе слова."""
    if len(t) <= limit:
        return t
    head = re.split(r"[:—,(]", t, maxsplit=1)[0].strip()
    return head if 10 <= len(head) <= limit else t[:limit].rsplit(" ", 1)[0].rstrip(" ,:;—-")


class Drafter:
    """Генератор черновиков; одна LLM для всех категорий, шаблон — запасной вариант."""

    def __init__(self, cfg: dict[str, Any], l1: Any = None) -> None:
        self.cfg: dict[str, Any] = cfg
        self.client: LLMClient = LLMClient(cfg)
        self.system: str = (ROOT / cfg["paths"]["prompts"] / "draft_system.md").read_text(encoding="utf-8")
        self.cats: dict[int, dict[str, str]] = load_categories()
        self.l1: Any = l1  # TfidfLR для оценки фрагментов (необязательно)

    def frags(self, text: str) -> list[Fragment]:
        """Фрагменты исходника; слой 1 (если есть) помогает отнести фрагмент к категории."""
        scorer = (lambda ts: [dict(zip(range(102, 107), map(float, r))) for r in self.l1.predict_proba(ts)]) if self.l1 else None
        return split_fragments(text, self.cfg, scorer)

    def user_prompt(self, title: str, source: str, cat: int, frags: list[Fragment]) -> str:
        """Часть подсказки с исходником (общая для категорий) и хвост с целевой категорией и её фрагментами."""
        c = self.cats[cat]
        target = [f.text for f in frags if cat in f.cats and not f.header]
        general = [f.text for f in frags if f.general]
        head = (f"Заголовок исходного объявления: {title}\n" if title else "") + f'Исходное объявление:\n"""\n{source}\n"""\n\n'
        tail = (f"Целевая категория: {c['name']} (название только ориентир, слова из него в текст не переносить).\n"
                "Фрагменты исходника по этой категории:\n" + ("\n".join(f"- {t}" for t in target) or "- (явных фрагментов нет)")
                + ("\nОбщие условия, которые можно перенести:\n" + "\n".join(f"- {t}" for t in general[:5]) if general else "")
                + "\n\nСоставь черновик объявления только по этой категории.")
        return head + tail

    def make(self, description: str, title: str, cats: list[int], use_llm: bool = True) -> list[dict[str, Any]]:
        """Черновики для списка категорий. Запись: mcId, title, text, source, violations, warnings, tokens, seconds."""
        source = f"{title}\n{description}" if title else description
        frags = self.frags(source)
        llm_ok = use_llm and self.client.available()
        out: list[dict[str, Any]] = []
        for cat in cats:
            rec: dict[str, Any] = {"mcId": cat, "source": "template", "violations": [], "warnings": [], "tokens_in": 0, "tokens_out": 0, "seconds": 0.0}
            tfr = [f.text for f in frags if cat in f.cats and not f.header]
            if llm_ok:
                try:
                    data, st = self.client.chat_json(self.system, self.user_prompt(title, description, cat, frags), SCHEMA)
                    t, x = shorten_title(data.get("title", "").strip(), self.cfg["drafts"]["title_max"]), data.get("text", "").strip()
                    res = run_checks(t, x, title, source, cat, tfr, self.cfg)
                    rec.update(st, violations=res["violations"], warnings=res["warnings"], llm_title=t, llm_text=x)
                    if not res["violations"]:
                        rec.update(source="llm", title=t, text=x)
                except LLMError as e:
                    rec["error"] = repr(e)
            if rec["source"] == "template":
                rec["title"], rec["text"] = template_draft(cat, frags, self.cfg)
            out.append(rec)
        return out
