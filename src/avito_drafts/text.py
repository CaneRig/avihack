"""Единая нормализация текста для обучения, вывода и демо."""
from __future__ import annotations

import re
import unicodedata

_INVISIBLE: re.Pattern[str] = re.compile("[\u200b-\u200f\u2060\ufeff\u00ad\u202a-\u202e]")
_LAT2CYR: dict[str, str] = {
    **dict(zip("aceopxykmthb", "асеорхукмтнв")),  # h->н, b->в, t->т
}
_LAT2CYR.update({k.upper(): v.upper() for k, v in list(_LAT2CYR.items())})
_LAT2CYR["H"] = "Н"
_LAT2CYR["B"] = "В"
_RUN: re.Pattern[str] = re.compile(r"[A-Za-zА-Яа-яЁё]+")
_CYR: re.Pattern[str] = re.compile(r"[А-Яа-яЁё]")
_LAT: re.Pattern[str] = re.compile(r"[A-Za-z]")
_NONALNUM: re.Pattern[str] = re.compile(r"[\W_]+")


def _fix_word(m: re.Match[str]) -> str:
    """Заменяет латинские буквы-двойники в слове, где смешаны кириллица и латиница."""
    w = m.group(0)
    if _CYR.search(w) and _LAT.search(w):
        return "".join(_LAT2CYR.get(ch, ch) for ch in w)
    return w


def normalize(s: str | None) -> str:
    """NFKC, удаление невидимых символов, замена букв-двойников, схлопывание пробелов и пустых строк."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _INVISIBLE.sub("", s)
    s = _RUN.sub(_fix_word, s)
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in s.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def make_text(title: str | None, description: str | None, use_title: bool = True) -> str:
    """Склеивает нормализованные заголовок и описание в поле text (без заголовка — только описание)."""
    d = normalize(description)
    t = normalize(title) if use_title else ""
    return f"{t}\n{d}" if t else d


def make_key(description: str | None) -> str:
    """Ключ повтора: нормализованное описание в нижнем регистре только из букв и цифр."""
    return _NONALNUM.sub("", normalize(description).lower())


def for_model(s: str) -> str:
    """Подготовка для TF-IDF: нижний регистр и «ё» -> «е»."""
    return s.lower().replace("ё", "е")


def has_garbage_only(s: str, min_chars: int) -> bool:
    """Истина, если описание слишком короткое или без букв (ответ — пустой список)."""
    return len(s) < min_chars or not re.search(r"[A-Za-zА-Яа-яЁё]{3}", s)
