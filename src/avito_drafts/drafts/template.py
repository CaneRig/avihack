"""Шаблонный черновик: название услуги из справочника + фрагменты исходника в исходных формулировках (не выдумывает по построению)."""
from __future__ import annotations

import re
from typing import Any

import yaml

from ..config import ROOT
from .fragments import Fragment

_NUMERIC: re.Pattern[str] = re.compile(r"\d|₽|руб|%|скидк|акци|подар")


def load_categories() -> dict[int, dict[str, str]]:
    """Справочник категорий из prompts/categories.yaml."""
    with open(ROOT / "prompts" / "categories.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clause(f: Fragment) -> str:
    """Текст пункта; у фрагмента про ремонт в целом берётся часть после двоеточия (если её нет — пусто)."""
    if not f.whole:
        return f.text.rstrip(":")
    return f.text.split(":", 1)[1].strip() if ":" in f.text else ""


def template_draft(cat: int, frags: list[Fragment], cfg: dict[str, Any]) -> tuple[str, str]:
    """Заголовок и текст: вводная фраза, пункты целевой категории, затем до max_general общих условий без цен."""
    info = load_categories()[cat]
    target = [_clause(f) for f in frags if cat in f.cats and not f.header]
    target = [t for t in target if t]
    general = [f.text for f in frags if f.general and not f.cats and not _NUMERIC.search(f.text.lower())]
    text = info["intro"] + "\n" + "\n".join(f"— {t}" for t in target) if target else info["intro"].rstrip(":") + "."
    text = "\n".join([text, *general[: cfg["drafts"]["max_general"]]])
    limit = cfg["drafts"]["text_max"]
    if len(text) > limit:  # режем по границе строки
        text = text[:limit].rsplit("\n", 1)[0]
    return info["template_title"], text
