"""Инструмент оценки: micro-F1 как у организаторов (sklearn), бутстреп по группам, сравнение, журнал запусков."""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from .config import CATS, config_hash, load_config, path
from .data import load

SLICES: list[str] = ["gold_raw", "gold_consensus", "silver_cv", "silver_fold0", "challenge"]


@dataclass
class SliceData:
    """Эталон среза: идентификаторы, метки (n,5), маска учитываемых ячеек (n,5), группы для бутстрепа."""

    name: str
    ids: list[str]
    y: np.ndarray
    mask: np.ndarray
    groups: np.ndarray
    df: pd.DataFrame | None


def load_slice(name: str, cfg: dict[str, Any] | None = None) -> SliceData:
    """Собирает SliceData по названию среза."""
    cfg = cfg or load_config()
    if name not in SLICES:
        raise ValueError(f"неизвестный срез {name}")
    table = name if name in ("gold_raw", "gold_consensus", "challenge") else "silver"
    df = load(table, cfg)
    if name == "silver_fold0":
        df = df[df.fold == 0].reset_index(drop=True)
    y = df[[f"y{c}" for c in CATS]].to_numpy(int)
    mask = df[[f"m{c}" for c in CATS]].to_numpy(int) if name == "gold_consensus" else np.ones_like(y)
    idcol = {"gold_raw": "row_id", "gold_consensus": "key", "challenge": "row_id"}.get(name, "item_id")
    gcol = "cluster_id" if table == "silver" else "key"
    return SliceData(name, df[idcol].astype(str).tolist(), y, mask, df[gcol].to_numpy(), df)


def pred_matrix(sl: SliceData, preds: dict[str, list[int]]) -> np.ndarray:
    """Предсказания {id: [mcId]} -> бинарная матрица (n,5). Дубли схлопываются, чужие mcId отбрасываются с предупреждением.

    Для gold_consensus допускаются id строк gold_raw (берётся первая строка с тем же key).
    """
    if sl.name == "gold_consensus" and not set(sl.ids) <= set(preds):
        raw = load("gold_raw")
        preds = {**{k: preds[str(r)] for r, k in zip(raw.row_id[::-1], raw.key[::-1]) if str(r) in preds}, **preds}
    miss = [i for i in sl.ids if i not in preds]
    if miss:
        raise KeyError(f"нет предсказаний для {len(miss)} id, например {miss[:3]}")
    P = np.zeros_like(sl.y)
    for r, i in enumerate(sl.ids):
        for m in set(preds[i]):
            if m in CATS:
                P[r, CATS.index(m)] = 1
            else:
                warnings.warn(f"mcId {m} вне 102–106 отброшен")
    return P


def _counts(sl: SliceData, P: np.ndarray) -> np.ndarray:
    """Счётчики TP/FP/FN по группам: (n_групп, 3)."""
    m = sl.mask.astype(bool)
    tp = ((P == 1) & (sl.y == 1) & m).sum(1)
    fp = ((P == 1) & (sl.y == 0) & m).sum(1)
    fn = ((P == 0) & (sl.y == 1) & m).sum(1)
    _, inv = np.unique(sl.groups, return_inverse=True)
    out = np.zeros((inv.max() + 1, 3))
    np.add.at(out, inv, np.stack([tp, fp, fn], 1))
    return out


def _f1(c: np.ndarray) -> np.ndarray:
    """F1 по суммарным счётчикам (последняя ось = tp,fp,fn); совпадает с micro-F1 sklearn."""
    tp, fp, fn = c[..., 0], c[..., 1], c[..., 2]
    return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def _boot_idx(n: int, cfg: dict[str, Any]) -> np.ndarray:
    """Индексы выборок групп с возвращением (одни и те же для парного сравнения)."""
    return np.random.default_rng(cfg["eval"]["bootstrap_seed"]).integers(0, n, (cfg["eval"]["bootstrap_n"], n))


def evaluate(sl: SliceData, preds: dict[str, list[int]], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Все метрики среза: micro-F1 (sklearn), P/R, F1 по категориям, ошибки, доли пустых, доверительный интервал."""
    cfg = cfg or load_config()
    P = pred_matrix(sl, preds)
    m = sl.mask.astype(bool)
    yt, yp = sl.y * m, P * m  # маскированные ячейки обнуляются в обеих матрицах (не дают TP/FP/FN)
    kw = {"average": "micro", "zero_division": 0}
    per = [f1_score(sl.y[:, j][m[:, j]], P[:, j][m[:, j]], zero_division=0) for j in range(len(CATS))]
    boot = _f1(_counts(sl, P)[_boot_idx(len(np.unique(sl.groups)), cfg)].sum(1))
    lo, hi = np.percentile(boot, cfg["eval"]["ci"])
    empty_true = (sl.y * sl.mask).sum(1) == 0
    return {
        "slice": sl.name, "n": len(sl.ids), "micro_f1": float(f1_score(yt, yp, **kw)),
        "ci": [float(lo), float(hi)], "precision": float(precision_score(yt, yp, **kw)),
        "recall": float(recall_score(yt, yp, **kw)),
        "f1_by_cat": dict(zip(map(str, CATS), map(float, per))),
        "fp": int(((yp == 1) & (yt == 0)).sum()), "fn": int(((yp == 0) & (yt == 1)).sum()),
        "empty_pred_share": float((P.sum(1) == 0).mean()), "empty_true_share": float(empty_true.mean()),
        "fp_on_empty_true": int(P[empty_true].sum()),
    }


def compare(sl: SliceData, pa: dict[str, list[int]], pb: dict[str, list[int]], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Парный бутстрап: разность micro-F1 (Б минус А), её интервал и доля выборок, где Б лучше."""
    cfg = cfg or load_config()
    ca, cb = _counts(sl, pred_matrix(sl, pa)), _counts(sl, pred_matrix(sl, pb))
    idx = _boot_idx(len(ca), cfg)
    d = _f1(cb[idx].sum(1)) - _f1(ca[idx].sum(1))
    lo, hi = np.percentile(d, cfg["eval"]["ci"])
    return {"diff": float(_f1(cb.sum(0)) - _f1(ca.sum(0))), "ci": [float(lo), float(hi)], "b_better_share": float((d > 0).mean())}


def log_run(res: dict[str, Any], exp: str, cfg: dict[str, Any], seconds: float | None = None, note: str = "") -> None:
    """Дописывает строку в reports/runs.jsonl."""
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "exp": exp, "config_hash": config_hash(cfg),
           "seconds": seconds, "note": note, **res}
    p = path(cfg, "runs")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_preds(p: str) -> dict[str, list[int]]:
    """Читает JSONL {"id", "mcIds"} в словарь."""
    with open(p, encoding="utf-8") as f:
        return {str(r["id"]): list(r["mcIds"]) for r in map(json.loads, filter(str.strip, f))}


def fmt(r: dict[str, Any]) -> str:
    """Одна строка сводки для консоли."""
    return (f"{r['slice']:<15} n={r['n']:<5} F1={r['micro_f1']:.3f} [{r['ci'][0]:.3f}–{r['ci'][1]:.3f}] "
            f"P={r['precision']:.3f} R={r['recall']:.3f} fp={r['fp']} fn={r['fn']} "
            f"пуст.пред={r['empty_pred_share']:.2f} лишних на пустых={r['fp_on_empty_true']}")
