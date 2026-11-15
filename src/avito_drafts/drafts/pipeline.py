"""Черновики для найденных категорий: фрагменты -> LLM -> проверки -> (при провале) шаблон."""
from __future__ import annotations

import re
from typing import Any

from ..config import CATS, ROOT
from ..keywords import _RX
from ..text import for_model
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
        self.regen_tpl: str = (ROOT / cfg["paths"]["prompts"] / "draft_regen.md").read_text(encoding="utf-8")
        self.cats: dict[int, dict[str, str]] = load_categories()
        self.l1: Any = l1  # TfidfLR для оценки фрагментов (необязательно)

    def frags(self, text: str) -> list[Fragment]:
        """Фрагменты исходника; слой 1 (если есть) помогает отнести фрагмент к категории."""
        scorer = (lambda ts: [dict(zip(range(102, 107), map(float, r))) for r in self.l1.predict_proba(ts)]) if self.l1 else None
        return split_fragments(text, self.cfg, scorer)

    def user_prompt(self, title: str, source: str, cat: int, frags: list[Fragment], variant: str = "b") -> str:
        """Подсказка. Варианты (G2): a — только объявление; b — объявление + фрагменты категории (общая часть сначала);
        c — только фрагменты категории и общие условия, без объявления."""
        c = self.cats[cat]
        target = [f.text for f in frags if cat in f.cats and not f.header]
        general = [f.text for f in frags if f.general][: self.cfg["drafts"]["max_general_prompt"]]
        name = f"Целевая категория: {c['name']} (название только ориентир, слова из него в текст не переносить).\n"
        head = (f"Заголовок исходного объявления: {title}\n" if title else "") + f'Исходное объявление:\n"""\n{source}\n"""\n\n'
        if variant == "a":
            return head + name + "Составь черновик объявления только по этой категории."
        listing = "\n".join(f"- {t}" for t in target) or "- (явных фрагментов нет)"
        gen = ("\nОбщие условия, которые можно перенести:\n" + "\n".join(f"- {t}" for t in general)) if general else ""
        if variant == "c":
            return (name + "Исходное объявление не показано; вот его фрагменты по этой категории:\n" + listing + gen
                    + "\n\nСоставь черновик объявления только по этой категории.")
        return head + name + "Фрагменты исходника по этой категории:\n" + listing + gen + "\n\nСоставь черновик объявления только по этой категории."

    def feedback(self, title: str, text: str, cat: int, violations: list[str]) -> str:
        """Замечания для повторного вызова: список нарушений, у «чужих услуг» — названия категорий и найденные слова."""
        low, lines = for_model(f"{title}\n{text}"), []
        for v in violations:
            if v.startswith("чужие услуги"):
                for c in CATS:
                    if c != cat and (hits := [m.group(0) for m in _RX[c].finditer(low)]):
                        lines.append(f"- упомянуты работы другой категории «{self.cats[c]['name']}»: {', '.join(sorted(set(hits)))}. Убери их")
            else:
                lines.append(f"- {v}")
        return self.regen_tpl.replace("{violations}", "\n".join(lines)[: self.cfg["drafts"]["regen_max_viol_chars"]])

    def make(self, description: str, title: str, cats: list[int], use_llm: bool = True,
             variant: str | None = None, regen: bool | None = None) -> list[dict[str, Any]]:
        """Черновики для списка категорий. Запись: mcId, title, text, source (llm | llm_regen | template), violations
        (последней попытки), viol_first (первой), warnings, tokens, seconds (сумма вызовов), sec1/tin1/tout1 (первый вызов)."""
        d = self.cfg["drafts"]
        variant, regen = variant or d["prompt_variant"], d["regen"] if regen is None else regen
        source = f"{title}\n{description}" if title else description
        frags = self.frags(source)
        llm_ok = use_llm and self.client.available()
        out: list[dict[str, Any]] = []
        for cat in cats:
            rec: dict[str, Any] = {"mcId": cat, "source": "template", "violations": [], "warnings": [], "tokens_in": 0, "tokens_out": 0,
                                   "seconds": 0.0, "calls": 0, "viol_first": None, "trace": []}
            tfr = [f.text for f in frags if cat in f.cats and not f.header]
            if llm_ok:
                prompt = self.user_prompt(title, description, cat, frags, variant)
                t = x = ""
                for k in range(2 if regen else 1):
                    try:
                        if k == 0:
                            data, st = self.client.chat_json(self.system, prompt, SCHEMA)
                        else:
                            data, st = self.client.chat_json(self.system, prompt + self.feedback(t, x, cat, rec["violations"]), SCHEMA,
                                                             temperature=d["regen_temperature"])
                    except LLMError as e:
                        rec["error"] = repr(e)
                        rec["trace"].append(getattr(e, "trace", []))
                        break
                    t, x = shorten_title(data.get("title", "").strip(), d["title_max"]), data.get("text", "").strip()
                    res = run_checks(t, x, title, source, cat, tfr, self.cfg)
                    rec["calls"] += 1
                    rec["tokens_in"] += st["tokens_in"]
                    rec["tokens_out"] += st["tokens_out"]
                    rec["seconds"] += st["seconds"]
                    rec["trace"].append(st["trace"])
                    rec.update(violations=res["violations"], warnings=res["warnings"], llm_title=t, llm_text=x)
                    if k == 0:
                        rec.update(viol_first=res["violations"], sec1=st["seconds"], tin1=st["tokens_in"], tout1=st["tokens_out"],
                                   title1=t, text1=x, warn1=res["warnings"])
                    if not res["violations"]:
                        rec.update(source="llm" if k == 0 else "llm_regen", title=t, text=x)
                        break
            if rec["source"] == "template":
                rec["title"], rec["text"] = template_draft(cat, frags, self.cfg)
            out.append(rec)
        return out
