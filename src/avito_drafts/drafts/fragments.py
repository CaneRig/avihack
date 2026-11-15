"""Шаг 1 ТЗ 3: разбиение объявления на фрагменты и пометка их категорией / общими условиями / ремонтом в целом."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from razdel import sentenize

from ..config import CATS
from ..keywords import find_roots
from ..text import for_model

_BULLET: re.Pattern[str] = re.compile(r"^(?:[\s•·\-–—*➤►▪■●○✔✅☑➡▶🔨🔧🔹🔸👉️]+|\d+[.)]\s+)+(?=\S)")
_GENERAL: re.Pattern[str] = re.compile(r"опыт|гаранти|договор|смет|выезд|замер|регион|район|график|оплат|скидк|материал|инструмент|консультац|бесплатн")
_WHOLE: re.Pattern[str] = re.compile(r"под ключ|комплексн|капитальн|косметическ")
_PHONE: re.Pattern[str] = re.compile(r"звон|пиши|связ|обращайтесь|телефон|whatsapp|ватсап")


@dataclass
class Fragment:
    """Фрагмент исходника: текст без маркера списка, его категории, признаки общего условия и ремонта в целом."""

    text: str
    cats: set[int] = field(default_factory=set)
    general: bool = False
    whole: bool = False
    header: bool = False


def split_fragments(text: str, cfg: dict[str, Any], l1_scorer: Callable[[list[str]], list[dict[int, float]]] | None = None) -> list[Fragment]:
    """Режет текст на строки/пункты (длинные — на предложения), помечает категории; заголовки блоков распространяют категорию."""
    out: list[Fragment] = []
    limit = cfg["drafts"]["frag_split_chars"]
    for line in text.splitlines():
        s = _BULLET.sub("", line).strip()
        if not s:
            continue
        parts = [x.text.strip() for x in sentenize(s)] if len(s) > limit else [s]
        out += [Fragment(p) for p in parts if p]
    block: set[int] = set()  # категории текущего блока, заданного заголовком вида «Сантехника:»
    scores = l1_scorer([f.text for f in out]) if (l1_scorer and out) else [{} for _ in out]
    for f, sc in zip(out, scores):
        low = for_model(f.text)
        roots = {c for c, v in find_roots(f.text).items() if v}
        f.header = low.rstrip().endswith(":") and len(low) < 80
        if f.header:
            block = roots
            f.cats = set(roots)
            continue
        f.cats = roots | block
        if l1_scorer and not roots:
            f.cats |= {c for c in CATS if sc.get(c, 0) >= cfg["drafts"]["frag_l1_thr"]}
        f.general = bool(_GENERAL.search(low)) and not f.cats
        f.whole = bool(_WHOLE.search(low))
        if _PHONE.search(low) and not f.cats:
            f.general = False
    return out
