"""Формат ответа: на всех строках gold_raw и challenge — валидная схема организаторов (режимы classify и fast)."""
from __future__ import annotations

from typing import Any

from avito_drafts.config import CATS
from avito_drafts.data import load
from avito_drafts.predict import Predictor


def check(resp: dict[str, Any], mode: str, mc_id: int = 101) -> None:
    """Проверяет схему ответа: ключи, диапазон mcId, уникальность и порядок, null/не-null для title и text."""
    assert set(resp) == {"drafts"}
    ids = [d["mcId"] for d in resp["drafts"]]
    assert ids == sorted(set(ids)) and all(i in CATS and i != mc_id for i in ids)
    for d in resp["drafts"]:
        assert set(d) == {"mcId", "title", "text"}
        filled = bool(d["title"]) and bool(d["text"])
        assert filled == (mode != "classify") and (filled or (d["title"] is None and d["text"] is None))


def test_format() -> None:
    """classify и fast на gold_raw и challenge."""
    p = Predictor()
    rows = [{"description": r.description, "title": r.title} for n in ("gold_raw", "challenge") for r in load(n).itertuples()]
    for mode in ("classify", "fast"):
        res = p.predict_batch(rows, mode, progress=False)
        assert len(res) == len(rows)
        for resp, _ in res:
            check(resp, mode)
    assert p.predict("", 101)[0] == {"drafts": []} and p.predict("роаи", 101)[0] == {"drafts": []}


if __name__ == "__main__":
    test_format()
    print("ОК формат")
