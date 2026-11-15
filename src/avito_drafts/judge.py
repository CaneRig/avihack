"""Слой «роль упоминания»: ключевые корни находят строки, LLM называет роль каждого упоминания, категория снимается по конкретной причине."""
from __future__ import annotations

import re
from typing import Any

from .config import CATS, ROOT
from .keywords import _RX, find_roots
from .llm import LLMClient
from .text import for_model, make_text, normalize

NAMES: dict[int, str] = {102: "штукатурные работы", 103: "плиточные работы", 104: "сантехника", 105: "электрика",
                         106: "малярные работы и обои"}
RESTR = re.compile(r"только комплекс|исключительно|отдельн\w+ (вид\w+ )?(мелк\w+ )?(работ|услуг|задач|комнат)|не бер[её]м|не выполня|не занима|минимальн\w+ (объ[её]м|заказ)|частичн|как комплекс|также отдельн|по отдельности")
ROLES: list[str] = ["service", "package_only", "team", "quality_control", "example_of_done_work", "negation_or_comparison",
                    "other_party_or_materials", "object_not_work", "other_not_service"]
ROLE: dict[str, Any] = {"type": "object", "properties": {"reason": {"type": "string"}, "role": {"type": "string", "enum": ROLES}},
                        "required": ["reason", "role"]}
SUFFIX = " || Ответь строго в формате JSON: для каждого вида работ reason и role."  # без перевода строки в конце: иначе Ollama отдаёт пустой ответ


def snippets(description: str | None, max_lines: int = 6, intro_chars: int = 350) -> tuple[dict[int, list[str]], list[str], list[str]]:
    """Строки с упоминанием по категориям (с ближайшим заголовком блока), фразы об условиях, все строки описания."""
    lines = [l.strip() for l in re.split(r"\n+", normalize(description)) if l.strip()]
    low = [for_model(l) for l in lines]
    out: dict[int, list[str]] = {}
    for c in CATS:
        idx = [i for i, l in enumerate(low) if _RX[c].search(l)]
        if not idx:
            continue
        parts = []
        for i in idx[:max_lines]:
            ctx = ""
            for j in range(i - 1, max(-1, i - 12), -1):  # ближайший заголовок выше: строка на «:» или короткая строка капсом
                if lines[j].rstrip().endswith(":") or (len(lines[j]) < 70 and lines[j].upper() == lines[j] and re.search(r"[А-Я]{4}", lines[j])):
                    ctx = lines[j][:120]
                    break
            l = lines[i]
            if len(l) > 320:
                m = _RX[c].search(low[i])
                a = max(0, m.start() - 160)
                l = ("…" if a else "") + l[a:a + 320] + "…"
            parts.append((f"[под заголовком «{ctx}»] " if ctx else "") + l)
        out[c] = parts
    restr = [l[:240] for l, lw in zip(lines, low) if RESTR.search(lw)][:4]
    return out, restr, lines


def build_user(title: str | None, description: str | None, max_lines: int = 6, intro_chars: int = 350
               ) -> tuple[str, dict[int, list[str]], list[str]]:
    """Запрос к судье и его основания: (текст запроса, строки по категориям, фразы об условиях)."""
    sn, restr, lines = snippets(description, max_lines, intro_chars)
    intro = " ".join(lines)[:intro_chars]
    user = (f"Заголовок объявления: {title}\n" if isinstance(title, str) and title else "") + f"Начало объявления: {intro}\n"
    if restr:
        user += "Фразы об условиях работы:\n" + "\n".join("- " + x for x in restr) + "\n"
    for c, parts in sn.items():
        user += f"\nc{c} ({NAMES[c]}) — строки с упоминанием:\n" + "\n".join("- " + p for p in parts) + "\n"
    return user.strip() + SUFFIX, sn, restr


def schema_for(cats: list[int]) -> dict[str, Any]:
    """JSON-схема ответа: по объекту {reason, role} на каждую категорию-кандидата."""
    return {"type": "object", "properties": {f"c{c}": ROLE for c in cats}, "required": [f"c{c}" for c in cats]}


class Judge:
    """Судья: кандидаты по корням -> роль упоминания от LLM -> категория снимается, только если роль в remove_roles."""

    def __init__(self, cfg: dict[str, Any], llm: LLMClient | None = None) -> None:
        self.cfg = cfg
        self.j: dict[str, Any] = cfg["judge"]
        self.llm: LLMClient = llm or LLMClient(cfg)
        self.system: str = (ROOT / self.j["prompt"]).read_text(encoding="utf-8")
        self.remove: set[str] = set(self.j["remove_roles"])

    def judge(self, title: str | None, description: str | None) -> tuple[list[int], dict[str, Any]]:
        """(категории, сведения). Нет корней -> пусто без LLM; сбой LLM или нет ключа категории -> остаётся кандидат по корню."""
        hits = find_roots(make_text(title, description, True))
        cand = [c for c in CATS if hits[c]]
        info: dict[str, Any] = {"candidates": cand, "roles": {}, "reasons": {}, "lines": {}, "fallback": False,
                                "tokens_in": 0, "tokens_out": 0, "seconds": 0.0}
        if not cand:
            return [], info
        user, sn, restr = build_user(title, description, self.j["max_lines"], self.j["intro_chars"])
        info["lines"] = {str(c): v for c, v in sn.items()}
        info["restr"] = restr
        if not sn:  # корень только в заголовке: строк-оснований нет, судить нечего
            return cand, info
        try:
            data, st = self.llm.chat_json(self.system, user, schema_for(list(sn)), temperature=self.j["temperature"],
                                          num_predict=self.j["num_predict"])
        except Exception as e:  # noqa: BLE001  любой сбой вызова -> ответ по корням
            info.update(fallback=True, error=repr(e)[:200])
            return cand, info
        info.update(tokens_in=st["tokens_in"], tokens_out=st["tokens_out"], seconds=st["seconds"], resp=data)
        if not isinstance(data, dict) or not data:
            info["fallback"] = True
            return cand, info
        cats = []
        for c in cand:
            r = data.get(f"c{c}") if c in sn else None
            if isinstance(r, dict) and r.get("role") in ROLES:
                info["roles"][str(c)] = r["role"]
                info["reasons"][str(c)] = str(r.get("reason", ""))
                if r["role"] not in self.remove:
                    cats.append(c)
            else:
                cats.append(c)
                if c in sn:
                    info["fallback"] = True
        return cats, info
