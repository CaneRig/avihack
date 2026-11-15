"""Слой «роль упоминания»: кандидаты (ключевые корни и слой 1) -> LLM называет роль каждого упоминания -> категория снимается по роли.

Режимы: single — один мягкий судья; panel — мягкий и строгий судьи, при расхождении по категории решает арбитр.
Заголовок объявления в классификации не участвует: корни, строки и запрос берутся только из описания.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from .config import CATS, ROOT
from .keywords import _RX
from .llm import LLMClient
from .text import for_model, normalize

NAMES: dict[int, str] = {102: "штукатурные работы (штукатурка, выравнивание стен, откосы)",
                         103: "плиточные работы (плитка, керамогранит, мозаика)",
                         104: "сантехника (водоснабжение, канализация, водяное отопление, водяной тёплый пол)",
                         105: "электрика (проводка, розетки, освещение, щиток, электрический тёплый пол)",
                         106: "малярные работы и обои (покраска, шпаклёвка, поклейка обоев)"}
RESTR = re.compile(r"только комплекс|исключительно|отдельн\w+ (вид\w+ )?(мелк\w+ )?(работ|услуг|задач|комнат)|не бер[её]м|не выполня|не занима|минимальн\w+ (объ[её]м|заказ)|частичн|как комплекс|также отдельн|по отдельности")
ROLES: list[str] = ["service", "package_only", "team", "quality_control", "example_of_done_work", "negation_or_comparison",
                    "other_party_or_materials", "object_not_work", "other_not_service"]
ROLE: dict[str, Any] = {"type": "object", "properties": {"reason": {"type": "string"}, "role": {"type": "string", "enum": ROLES}},
                        "required": ["reason", "role"]}
SUFFIX = " || Ответь строго в формате JSON: для каждого вида работ reason и role."  # без перевода строки в конце: иначе Ollama отдаёт пустой ответ
ARB_SUFFIX = " || Ответь строго в формате JSON: quote, reason, answer."
ARB_SCHEMA: dict[str, Any] = {"type": "object", "properties": {"quote": {"type": "string"}, "reason": {"type": "string"},
                                                              "answer": {"type": "string", "enum": ["yes", "no"]}},
                              "required": ["quote", "reason", "answer"]}


def _split(description: str | None) -> tuple[list[str], list[str]]:
    """Строки нормализованного описания и их версии для поиска (нижний регистр)."""
    lines = [l.strip() for l in re.split(r"\n+", normalize(description)) if l.strip()]
    return lines, [for_model(l) for l in lines]


def _ctx(lines: list[str], i: int) -> str:
    """Ближайший заголовок блока выше строки i: строка на «:» или короткая строка капсом."""
    for j in range(i - 1, max(-1, i - 12), -1):
        if lines[j].rstrip().endswith(":") or (len(lines[j]) < 70 and lines[j].upper() == lines[j] and re.search(r"[А-Я]{4}", lines[j])):
            return lines[j][:120]
    return ""


def _fmt_line(lines: list[str], i: int, rx: re.Pattern[str] | None, low: str) -> str:
    """Строка i с заголовком блока; длинная строка обрезается вокруг корня (или с начала)."""
    ctx = _ctx(lines, i)
    l = lines[i]
    if len(l) > 320:
        m = rx.search(low) if rx is not None else None
        a = max(0, m.start() - 160) if m else 0
        l = ("…" if a else "") + l[a:a + 320] + "…"
    return (f"[под заголовком «{ctx}»] " if ctx else "") + l


def snippets(description: str | None, max_lines: int = 6, intro_chars: int = 350) -> tuple[dict[int, list[str]], list[str], list[str]]:
    """Строки с упоминанием по категориям (с ближайшим заголовком блока), фразы об условиях, все строки описания."""
    lines, low = _split(description)
    out: dict[int, list[str]] = {}
    for c in CATS:
        idx = [i for i, l in enumerate(low) if _RX[c].search(l)]
        if idx:
            out[c] = [_fmt_line(lines, i, _RX[c], low[i]) for i in idx[:max_lines]]
    restr = [l[:240] for l, lw in zip(lines, low) if RESTR.search(lw)][:4]
    return out, restr, lines


def build_user(description: str | None, sn: dict[int, list[str]], restr: list[str], lines: list[str], intro_chars: int = 350) -> str:
    """Запрос к эксперту: начало описания, фразы об условиях, строки-основания по категориям-кандидатам."""
    user = f"Начало объявления: {' '.join(lines)[:intro_chars]}\n"
    if restr:
        user += "Фразы об условиях работы:\n" + "\n".join("- " + x for x in restr) + "\n"
    for c, parts in sn.items():
        user += f"\nc{c} ({NAMES[c]}) — строки с упоминанием:\n" + "\n".join("- " + p for p in parts) + "\n"
    return user.strip() + SUFFIX


def schema_for(cats: list[int]) -> dict[str, Any]:
    """JSON-схема ответа: по объекту {reason, role} на каждую категорию-кандидата."""
    return {"type": "object", "properties": {f"c{c}": ROLE for c in cats}, "required": [f"c{c}" for c in cats]}


def _stage(st: dict[str, Any] | None) -> dict[str, Any]:
    """Токены и секунды одного этапа."""
    st = st or {}
    return {"tokens_in": int(st.get("tokens_in", 0)), "tokens_out": int(st.get("tokens_out", 0)), "seconds": float(st.get("seconds", 0.0))}


class Judge:
    """Судья: кандидаты -> роли от мягкого (и строгого) эксперта -> арбитр на расхождениях -> итоговые категории."""

    def __init__(self, cfg: dict[str, Any], llm: LLMClient | None = None, l1: Any = None) -> None:
        self.cfg = cfg
        self.j: dict[str, Any] = cfg["judge"]
        self.llm: LLMClient = llm or LLMClient(cfg)
        self.mode: str = self.j.get("mode", "panel")
        self.system: str = (ROOT / self.j["prompt"]).read_text(encoding="utf-8")
        self.remove: set[str] = set(self.j["remove_roles"])
        self.l1 = l1
        self.use_l1: bool = bool(self.j.get("tfidf_candidates")) and l1 is not None  # нет модели слоя 1 -> источник молча выключен
        if self.mode == "panel":
            self.strict: str = (ROOT / self.j["prompt_strict"]).read_text(encoding="utf-8")
            self.arbiter: str = (ROOT / self.j["prompt_arbiter"]).read_text(encoding="utf-8")
        else:
            self.remove -= set(self.j.get("single_keep_roles", []))

    def candidates(self, description: str | None) -> tuple[list[int], dict[str, str], dict[int, list[str]], list[str], list[str], dict[str, float]]:
        """(кандидаты, источники, строки-основания, фразы об условиях, все строки, вероятности слоя 1 по описанию)."""
        sn, restr, lines = snippets(description, self.j["max_lines"], self.j["intro_chars"])
        src = {str(c): "root" for c in sn}
        probs: dict[str, float] = {}
        if self.use_l1 and lines:
            desc = "\n".join(lines)
            p = self.l1.predict_proba([desc])[0]
            probs = {str(c): float(p[k]) for k, c in enumerate(CATS)}
            extra = [c for c in CATS if c not in sn and probs[str(c)] >= self.j["tfidf_candidate_threshold"]]
            if extra:
                pl = self.l1.predict_proba(lines)  # строки оцениваются по отдельности
                for c in extra:
                    k = CATS.index(c)
                    if len(desc) < self.j["tfidf_short_chars"]:
                        idx = list(range(len(lines)))
                    else:
                        idx = sorted(np.argsort(-pl[:, k], kind="stable")[:self.j["tfidf_lines"]].tolist())
                    sn[c] = [_fmt_line(lines, i, None, "") for i in idx]
                    src[str(c)] = "tfidf"
        sn = {c: sn[c] for c in CATS if c in sn}
        return list(sn), src, sn, restr, lines, probs

    def _expert(self, system: str, user: str, cands: list[int]) -> tuple[dict[int, tuple[str | None, str]] | None, dict[str, Any], str | None]:
        """Один вызов эксперта: ({c: (роль, причина)} или None при сбое/пустом ответе, этап, ошибка)."""
        try:
            data, st = self.llm.chat_json(system, user, schema_for(cands), temperature=self.j["temperature"], num_predict=self.j["num_predict"])
        except Exception as e:  # noqa: BLE001  любой сбой вызова
            return None, _stage(None), repr(e)[:200]
        if not isinstance(data, dict) or not data:
            return None, _stage(st), "empty"
        out: dict[int, tuple[str | None, str]] = {}
        for c in cands:
            r = data.get(f"c{c}")
            out[c] = (r["role"], str(r.get("reason", ""))) if isinstance(r, dict) and r.get("role") in ROLES else (None, "")
        return out, _stage(st), None

    def _arbitrate(self, c: int, description: str, sn: dict[int, list[str]], restr: list[str], lines: list[str]) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
        """Арбитр по одной категории: ({quote, reason, answer} или None, этап, ошибка)."""
        desc = normalize(description)
        if len(desc) <= self.j["arbiter_full_text_chars"]:
            body = "Текст объявления:\n" + desc
        else:
            body = ("Начало объявления: " + " ".join(lines)[:self.j["intro_chars"]] + "\n"
                    + ("Фразы об условиях работы:\n" + "\n".join("- " + x for x in restr) + "\n" if restr else "")
                    + "Строки с упоминанием работы:\n" + "\n".join("- " + p for p in sn.get(c, [])))
        user = body + f"\n\nРабота: {NAMES[c]}." + ARB_SUFFIX
        try:
            data, st = self.llm.chat_json(self.arbiter, user, ARB_SCHEMA, temperature=0.0, num_predict=self.j["arbiter_num_predict"])
        except Exception as e:  # noqa: BLE001
            return None, _stage(None), repr(e)[:200]
        if not isinstance(data, dict) or data.get("answer") not in ("yes", "no"):
            return None, _stage(st), "bad_answer"
        return {k: str(data.get(k, "")) for k in ("quote", "reason", "answer")}, _stage(st), None

    def judge(self, title: str | None, description: str | None) -> tuple[list[int], dict[str, Any]]:
        """(категории, сведения). Заголовок на результат не влияет (параметр оставлен для совместимости)."""
        cand, src, sn, restr, lines, probs = self.candidates(description)
        root = [c for c in cand if src[str(c)] == "root"]
        zero = {"tokens_in": 0, "tokens_out": 0, "seconds": 0.0}
        info: dict[str, Any] = {"mode": self.mode, "candidates": cand, "sources": src, "tfidf_probs": probs, "lines": {str(c): v for c, v in sn.items()},
                                "restr": restr, "roles": {}, "reasons": {}, "strict_roles": {}, "strict_reasons": {}, "arbiter": {},
                                "disagree": [], "fallback": False, "errors": {}, "calls": {"soft": 0, "strict": 0, "arbiter": 0},
                                "stages": {"soft": dict(zero), "strict": dict(zero), "arbiter": dict(zero)}}

        def totals() -> None:
            for k in ("tokens_in", "tokens_out", "seconds"):
                info[k] = sum(s[k] for s in info["stages"].values())

        totals()
        if not cand:
            return [], info
        user = build_user(description, sn, restr, lines, self.j["intro_chars"])
        soft, st, err = self._expert(self.system, user, cand)
        info["calls"]["soft"] = 1
        info["stages"]["soft"] = st
        totals()
        if soft is None:  # мягкий судья упал или пуст -> ответ по корням
            info.update(fallback=True)
            info["errors"]["soft"] = err
            return root, info
        for c, (role, reason) in soft.items():
            info["roles"][str(c)], info["reasons"][str(c)] = role, reason
        keep = {c: soft[c][0] is None or soft[c][0] not in self.remove for c in cand}  # нет ключа категории -> оставлена
        if self.mode == "panel":
            strict, st, err = self._expert(self.strict, user, cand)
            info["calls"]["strict"] = 1
            info["stages"]["strict"] = st
            if strict is None:
                info["errors"]["strict"] = err  # строгий упал -> решение мягкого
            else:
                for c, (role, reason) in strict.items():
                    info["strict_roles"][str(c)], info["strict_reasons"][str(c)] = role, reason
                for c in cand:
                    sk = strict[c][0] is None or strict[c][0] not in self.remove
                    if sk == keep[c]:
                        continue
                    info["disagree"].append(c)
                    a, st, err = self._arbitrate(c, description or "", sn, restr, lines)
                    info["calls"]["arbiter"] += 1
                    for k in ("tokens_in", "tokens_out", "seconds"):
                        info["stages"]["arbiter"][k] += st[k]
                    if a is None:
                        info["errors"][f"arbiter_c{c}"] = err  # арбитр упал -> решение мягкого
                    else:
                        info["arbiter"][str(c)] = a
                        keep[c] = a["answer"] == "yes"
            totals()
        return [c for c in cand if keep[c]], info
