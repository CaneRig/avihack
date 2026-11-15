"""Шаг 1: ориентиры («всегда пусто», правило по корням — D1) и перебор слоя 1 (C1). Результаты — в reports/runs.jsonl."""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_drafts import evalkit as E  # noqa: E402
from avito_drafts.config import CATS, load_config  # noqa: E402
from avito_drafts.data import load  # noqa: E402
from avito_drafts.keywords import rule_predict  # noqa: E402
from avito_drafts.layer1 import TfidfLR  # noqa: E402
from avito_drafts.text import make_text  # noqa: E402

Y: list[str] = [f"y{c}" for c in CATS]


def texts(df: pd.DataFrame, use_title: bool) -> list[str]:
    """Тексты таблицы с заголовком или без."""
    return [make_text(t, d, use_title) for t, d in zip(df.title, df.description)]


def to_preds(ids: list[Any], P: np.ndarray, thr: float) -> dict[str, list[int]]:
    """Матрица вероятностей -> {id: [mcId]} по порогу."""
    return {str(i): [c for c, v in zip(CATS, row) if v >= thr] for i, row in zip(ids, P)}


def report(exp: str, cfg: dict[str, Any], slices: dict[str, dict[str, list[int]]], t0: float, note: str) -> None:
    """Оценивает предсказания по срезам, печатает и пишет в журнал."""
    for name, pr in slices.items():
        r = E.evaluate(E.load_slice(name, cfg), pr, cfg)
        print(f"{exp:<4}{note:<34}{E.fmt(r)}")
        E.log_run(r, exp, cfg, seconds=round(time.time() - t0, 1), note=note)


def main() -> None:
    """Запускает D1 и C1."""
    cfg = load_config()
    silver, raw, cons, ch = (load(n, cfg) for n in ("silver", "gold_raw", "gold_consensus", "challenge"))
    t0 = time.time()
    # ориентиры
    report("D1", cfg, {"gold_raw": {str(r): [] for r in raw.row_id}}, t0, "всегда пусто")
    rule = lambda df, col: {str(i): rule_predict(t) for i, t in zip(df[col], df.text)}  # noqa: E731
    report("D1", cfg, {"gold_raw": rule(raw, "row_id"), "gold_consensus": rule(cons, "key"), "challenge": rule(ch, "row_id"),
                       "silver_cv": rule(silver, "item_id")}, t0, "правило по корням")
    # C1: перебор
    train_gold = silver[~silver.in_public_test].reset_index(drop=True)
    grid = [(ng, True) for ng in cfg["layer1"]["grid_char_ngram"]] + [(cfg["layer1"]["char_ngram"], False)]
    Cs = cfg["layer1"]["grid_C"]
    for ng, use_title in grid:
        t0 = time.time()
        over = {"char_ngram": ng}
        tr_t = lambda df: texts(df, use_title)  # noqa: E731
        # перекрёстная проверка по 5 частям
        oof = {C: np.zeros((len(silver), 5)) for C in Cs}
        for f in range(cfg["data"]["n_folds"]):
            tr, va = silver[silver.fold != f], silver[silver.fold == f]
            for C, m in TfidfLR.fit_many(cfg, tr_t(tr), tr[Y].to_numpy(), Cs, **over).items():
                oof[C][va.index] = m.predict_proba(tr_t(va))
        # обучение на всём трейне без текстов, близких к публичному тесту
        full = TfidfLR.fit_many(cfg, tr_t(train_gold), train_gold[Y].to_numpy(), Cs, **over)
        for C in Cs:
            c2 = copy.deepcopy(cfg)
            c2["layer1"].update(C=C, char_ngram=ng, use_title=use_title)
            thr = cfg["layer1"]["threshold"]
            sl = {"silver_cv": to_preds(silver.item_id, oof[C], thr),
                  "silver_fold0": to_preds(silver.item_id, oof[C], thr)}
            for name, df, col in (("gold_raw", raw, "row_id"), ("gold_consensus", cons, "key"), ("challenge", ch, "row_id")):
                sl[name] = to_preds(df[col], full[C].predict_proba(tr_t(df)), thr)
            report("C1", c2, sl, t0, f"C={C} char={tuple(ng)} title={use_title}")
        if use_title and tuple(ng) == tuple(cfg["layer1"]["char_ngram"]):
            full[cfg["layer1"]["C"]].save(Path(__file__).resolve().parents[1] / cfg["paths"]["tfidf_dir"])


if __name__ == "__main__":
    main()
