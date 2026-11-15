"""Единая функция предсказания: объявление -> {"drafts": [{"mcId", "title", "text"}]} плюс служебные сведения."""
from __future__ import annotations

import time
from typing import Any

from tqdm import tqdm

from .config import CATS, ROOT, load_config
from .layer1 import TfidfLR
from .text import has_garbage_only, make_text, normalize

Row = dict[str, Any]  # description, mcId (необязательно, 101), title (необязательно)


class Predictor:
    """Держит загруженные модели и отвечает на объявления в режимах classify / fast / full."""

    def __init__(self, cfg: dict[str, Any] | None = None, mode: str | None = None) -> None:
        self.cfg: dict[str, Any] = cfg or load_config()
        self.mode: str = mode or self.cfg["mode"]
        self.l1: TfidfLR = TfidfLR.load(ROOT / self.cfg["paths"]["tfidf_dir"])
        self._drafter: Any = None

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
        """Пакет: слой 1 батчем на всех строках, затем черновики по строкам; сбой строки не валит пакет."""
        mode = mode or self.mode
        use_title = self.cfg["layer1"]["use_title"]
        descs = [normalize(r.get("description")) for r in rows]
        titles = [normalize(r.get("title")) if use_title else "" for r in rows]
        texts = [make_text(t, d, use_title) for t, d in zip(titles, descs)]
        t0 = time.time()
        P = self.l1.predict_proba(texts)
        t_cls = (time.time() - t0) / max(len(rows), 1)
        out: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for r, d, t, p in tqdm(list(zip(rows, descs, titles, P)), disable=not progress, desc=mode):
            info: dict[str, Any] = {"mode": mode, "probs": {}, "timing": {"classify": t_cls}}
            if has_garbage_only(d, self.cfg["data"]["min_chars"]):
                out.append(({"drafts": []}, info))
                continue
            info["probs"] = dict(zip(CATS, map(float, p)))
            cats = self.decide(info["probs"], int(r.get("mcId") or 101))
            items: list[dict[str, Any]] = [{"mcId": c, "title": None, "text": None} for c in cats]
            if mode != "classify" and cats:
                try:
                    items = self.drafter().make(d, t, cats, use_llm=(mode == "full"))
                    info["drafts"] = items
                except Exception as e:  # noqa: BLE001  сбой строки: оставляем ответ классификатора
                    info["error"] = repr(e)
            out.append(({"drafts": [{"mcId": i["mcId"], "title": i["title"], "text": i["text"]} for i in items]}, info))
        return out
