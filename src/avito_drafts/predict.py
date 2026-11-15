"""Единая функция предсказания: объявление -> {"drafts": [{"mcId", "title", "text"}]} плюс служебные сведения."""
from __future__ import annotations

import time
from typing import Any

from tqdm import tqdm

from .config import CATS, ROOT, load_config
from .keywords import rule_predict
from .layer1 import TfidfLR
from .text import has_garbage_only, make_text, normalize

Row = dict[str, Any]  # description, mcId (необязательно, 101), title (необязательно)


class Predictor:
    """Держит загруженные модели и отвечает на объявления в режимах classify / fast / full."""

    def __init__(self, cfg: dict[str, Any] | None = None, mode: str | None = None) -> None:
        self.cfg: dict[str, Any] = cfg or load_config()
        self.mode: str = mode or self.cfg["mode"]
        try:
            self.l1: TfidfLR | None = TfidfLR.load(ROOT / self.cfg["paths"]["tfidf_dir"])
        except Exception:  # noqa: BLE001  без модели слоя 1 работают правило по корням и судья
            self.l1 = None
        self._drafter: Any = None
        self._judge: Any = None
        self._llm_ok: bool | None = None

    def judge(self) -> Any:
        """Ленивое создание судьи роли упоминания."""
        if self._judge is None:
            from .judge import Judge
            self._judge = Judge(self.cfg, l1=self.l1)
        return self._judge

    def judge_ready(self) -> bool:
        """Включён ли судья и отвечает ли сервер LLM (проверка один раз на пакет)."""
        if not self.cfg.get("judge", {}).get("enabled"):
            return False
        return self.judge().llm.available()

    def decide(self, p: dict[int, float], mc_id: int) -> list[int]:
        """Категории с вероятностью не ниже порога, без исходной, по возрастанию."""
        thr = self.cfg["layer1"]["threshold"]
        return sorted(c for c in CATS if p[c] >= thr and c != mc_id)

    def drafter(self) -> Any:
        """Ленивое создание генератора черновиков (нужен только в full и fast)."""
        if self._drafter is None:
            from .drafts.pipeline import Drafter
            self._drafter = Drafter(self.cfg, self.l1)
        return self._drafter

    def predict(self, description: str, mc_id: int = 101, title: str | None = None,
                mode: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Одно объявление: ответ в формате организаторов и служебный объект."""
        return self.predict_batch([{"description": description, "mcId": mc_id, "title": title}], mode, progress=False)[0]

    def predict_batch(self, rows: list[Row], mode: str | None = None,
                      progress: bool = True) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Пакет: категории — судья (classify/full) или правило по корням (fast, LLM недоступна), затем черновики.

        Точные повторы (заголовок, описание) судятся один раз; сбой строки не валит пакет.
        """
        mode = mode or self.mode
        use_title = self.cfg["layer1"]["use_title"]
        descs = [normalize(r.get("description")) for r in rows]
        titles = [normalize(r.get("title")) for r in rows]  # заголовок нужен черновикам; в классификации его нет
        texts = [make_text(t, d, use_title) for t, d in zip(titles, descs)]
        t0 = time.time()
        P = self.l1.predict_proba(texts) if self.l1 is not None else [None] * len(rows)
        t_cls = (time.time() - t0) / max(len(rows), 1)
        use_judge = mode != "fast" and self.judge_ready()
        judged: dict[str, tuple[list[int], dict[str, Any]]] = {}
        out: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for r, d, t, p, tx in tqdm(list(zip(rows, descs, titles, P, texts)), disable=not progress, desc=mode):
            info: dict[str, Any] = {"mode": mode, "probs": {}, "timing": {"classify": t_cls}}
            if has_garbage_only(d, self.cfg["data"]["min_chars"]):
                out.append(({"drafts": []}, info))
                continue
            mc_id = int(r.get("mcId") or 101)
            if p is not None:
                info["probs"] = dict(zip(CATS, map(float, p)))
            if use_judge:
                if d not in judged:  # заголовок на категории не влияет: ключ — описание
                    judged[d] = self.judge().judge(r.get("title"), d)
                found, jinfo = judged[d]
                info["judge"] = jinfo
                info["source"] = "keywords_fallback" if jinfo.get("fallback") else ("panel" if jinfo.get("mode") == "panel" else "judge")
                cats = sorted(c for c in found if c != mc_id)
            elif mode != "fast" and self.cfg.get("judge", {}).get("enabled"):  # судья включён, но LLM недоступна
                info["source"] = "keywords_fallback"
                cats = sorted(c for c in rule_predict(tx) if c != mc_id)
            elif self.cfg.get("judge", {}).get("enabled") or p is None:
                info["source"] = "keywords"
                cats = sorted(c for c in rule_predict(tx) if c != mc_id)
            else:  # судья выключен в конфиге: прежний слой 1
                info["source"] = "tfidf"
                cats = self.decide(info["probs"], mc_id)
            items: list[dict[str, Any]] = [{"mcId": c, "title": None, "text": None} for c in cats]
            if mode != "classify" and cats:
                try:
                    items = self.drafter().make(d, t, cats, use_llm=(mode == "full"))
                    info["drafts"] = items
                except Exception as e:  # noqa: BLE001  сбой строки: оставляем ответ классификатора
                    info["error"] = repr(e)
            out.append(({"drafts": [{"mcId": i["mcId"], "title": i["title"], "text": i["text"]} for i in items]}, info))
        return out
