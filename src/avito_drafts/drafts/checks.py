"""Шаг 3 ТЗ 3: детерминированные проверки черновика относительно исходника (без LLM)."""
from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

import pymorphy3

from ..config import CATS, ROOT
from ..keywords import find_roots
from ..text import for_model

_NUM: re.Pattern[str] = re.compile(r"\d+(?:[.,]\d+)?")
_FORBID_ALWAYS: re.Pattern[str] = re.compile(r"в рамках ремонт|в объявлени|как указано|исходн\w+ объявлен|ремонт\w*[^.\n]{0,30}под ключ")
_FORBID_IF_ABSENT: tuple[str, ...] = ("отдельно", "по отдельности")
_PHONE: re.Pattern[str] = re.compile(r"\+?\d[\d\s\-()]{8,}\d|https?://|www\.|@\w+")
_WORD: re.Pattern[str] = re.compile(r"[а-яё]{4,}")
_CONTENT_POS: set[str] = {"NOUN", "VERB", "INFN", "ADJF", "ADJS", "PRTF", "PRTS", "GRND"}
_MORPH: pymorphy3.MorphAnalyzer = pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=1)
def allowed_words() -> frozenset[str]:
    """Список служебной лексики объявлений из prompts/allowed_words.txt (леммы)."""
    p = ROOT / "prompts" / "allowed_words.txt"
    return frozenset(for_model(w.strip()) for w in p.read_text(encoding="utf-8").splitlines() if w.strip())


@lru_cache(maxsize=50000)
def _lemma(w: str) -> tuple[str, str]:
    """(лемма, часть речи) слова по pymorphy3."""
    p = _MORPH.parse(w)[0]
    return p.normal_form.replace("ё", "е"), str(p.tag.POS)


def content_lemmas(text: str) -> list[str]:
    """Леммы содержательных слов (сущ., глаг., прил.) длиной от 4 букв."""
    return [lm for w in _WORD.findall(for_model(text)) for lm, pos in [_lemma(w)] if pos in _CONTENT_POS]


def _numbers(s: str) -> set[str]:
    """Числа текста в нормальной записи («1 000» -> «1000», «5,5» -> «5.5»)."""
    s = re.sub(r"(?<=\d)[  ](?=\d{3}\b)", "", s)
    return {n.replace(",", ".") for n in _NUM.findall(s)}


def _shingles(s: str, n: int = 5) -> set[tuple[str, ...]]:
    """Множество n-грамм слов."""
    w = for_model(s).split()
    return {tuple(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


def run_checks(title: str, text: str, source_title: str, source: str, target: int, target_frags: list[str],
               cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Возвращает {"violations": [...], "warnings": [...]}; каждое — короткая строка с цитатой из черновика."""
    d = cfg["drafts"]
    v: list[str] = []
    w: list[str] = []
    both, src_low, dr_low = f"{title}\n{text}", for_model(source), for_model(f"{title}\n{text}")
    # числа
    new_nums = _numbers(both) - _numbers(source)
    if new_nums:
        v.append(f"числа: {sorted(new_nums)}")
    # запрещённые формулировки
    if m := _FORBID_ALWAYS.search(dr_low):
        v.append(f"запрещено: «{m.group(0)}»")
    for f in _FORBID_IF_ABSENT:
        if f in dr_low and f not in src_low:
            v.append(f"запрещено без исходника: «{f}»")
    # чужие услуги
    foreign = [c for c in CATS if c != target and find_roots(both)[c]]
    if foreign:
        v.append(f"чужие услуги: {foreign}")
    # новые слова
    src_lem = set(content_lemmas(source)) | allowed_words()
    stems = {x[: d["stem_len"]] for x in src_lem}
    dr = content_lemmas(both)
    new = sorted({x for x in dr if x not in src_lem and x[: d["stem_len"]] not in stems})
    if len(new) > max(d["new_words_abs"], math.ceil(d["new_words_share"] * len(dr))):
        v.append(f"новые слова: {new}")
    # форма
    if len(title) > d["title_max"] or not title.strip():
        v.append(f"заголовок {len(title)} симв.")
    if re.search(r"[\"«»!]", title):
        v.append("заголовок: кавычки или «!»")
    if len(text) > d["text_max"] or not text.strip():
        v.append(f"текст {len(text)} симв.")
    if _PHONE.search(both):
        v.append("контакты или ссылка")
    if for_model(title).strip() == for_model(source_title).strip() and source_title.strip():
        v.append("заголовок равен исходному")
    sh = _shingles(text)
    if sh and len(sh & _shingles(source)) / len(sh) > d["overlap_max"]:
        v.append("текст копирует исходник")
    # предупреждения: длина и покрытие
    if len(text) < d["text_min"]:
        w.append(f"короткий текст {len(text)}")
    if target_frags:
        dst = {x[: d["stem_len"]] for x in content_lemmas(text)}
        cov = sum(1 for fr in target_frags if (ls := content_lemmas(fr)) and sum(x[: d["stem_len"]] in dst for x in ls) / len(ls) >= 0.5)
        if cov / len(target_frags) < d["coverage_min"]:
            w.append(f"покрытие {cov}/{len(target_frags)}")
    return {"violations": v, "warnings": w}
