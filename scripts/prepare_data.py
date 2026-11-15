"""Шаг 1: строит data/processed/*.parquet и печатает сводку."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_drafts.config import CATS  # noqa: E402
from avito_drafts.data import build  # noqa: E402

if __name__ == "__main__":
    t = build()
    for n, d in t.items(): print(n, len(d))
    s, g, c = t["silver"], t["gold_raw"], t["gold_consensus"]
    y = [f"y{k}" for k in CATS]
    print("пустых: silver %.3f gold_raw %.3f consensus %.3f" % ((s[y].sum(1) == 0).mean(), (g[y].sum(1) == 0).mean(), (c[y].sum(1) == 0).mean()))
    print("частоты silver", s[y].mean().round(3).to_dict()); print("частоты gold_raw", g[y].mean().round(3).to_dict())
    print("кластеров silver", s.cluster_id.nunique(), "макс. размер", s.cluster_id.value_counts().max(), "| in_public_test", int(s.in_public_test.sum()))
    print("фолды", s.fold.value_counts().sort_index().tolist(), "| конфликт-групп", int(c.has_conflict.sum()), "| ничьих ячеек", int((1 - c[[f"m{k}" for k in CATS]]).values.sum()))
