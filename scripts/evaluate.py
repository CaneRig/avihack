"""Оценка файла предсказаний (JSONL {"id","mcIds"}) на срезе; --b — парное сравнение с вариантом Б."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_drafts import evalkit as E  # noqa: E402
from avito_drafts.config import load_config  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("--slice", required=True, choices=E.SLICES)
    ap.add_argument("--exp", default="")
    ap.add_argument("--b", help="предсказания варианта Б")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    cfg = load_config()
    sl = E.load_slice(a.slice, cfg)
    r = E.evaluate(sl, E.read_preds(a.pred), cfg)
    print(E.fmt(r))
    if a.exp:
        E.log_run(r, a.exp, cfg, note=a.note)
    if a.b:
        print("Б против А:", E.compare(sl, E.read_preds(a.pred), E.read_preds(a.b), cfg))
