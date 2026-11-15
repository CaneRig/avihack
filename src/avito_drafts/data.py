"""Сборка таблиц data/processed: silver, gold_raw, gold_consensus, challenge."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from .config import CATS, NAME_TO_ID, load_config, path
from .text import make_key, make_text


def _labels(ids: list[int]) -> dict[str, int]:
    """Список mcId -> колонки y102..y106 (0/1)."""
    return {f"y{c}": int(c in ids) for c in CATS}


def _clusters(texts: list[str], sim: float) -> np.ndarray:
    """Связные компоненты графа «косинус TF-IDF по словам >= sim» (блоками, без полной матрицы)."""
    X = TfidfVectorizer(lowercase=True).fit_transform(texts).tocsr()
    rows, cols = [], []
    for s in range(0, X.shape[0], 1000):
        S = (X[s:s + 1000] @ X.T).toarray()
        r, c = np.nonzero(S >= sim)
        rows.append(r + s); cols.append(c)
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = csr_matrix((np.ones(len(r)), (r, c)), shape=(X.shape[0],) * 2)
    return connected_components(g, directed=False)[1]


def build(cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Строит и сохраняет все таблицы; возвращает их словарём."""
    cfg = cfg or load_config()
    out = path(cfg, "processed"); out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_json(path(cfg, "raw_train"), lines=True)
    te = pd.read_parquet(path(cfg, "raw_test"))
    ch = pd.read_json(path(cfg, "challenge"), lines=True)

    def prep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["title"] = df["title"].fillna("")
        df["text"] = [make_text(t, d) for t, d in zip(df.title, df.description)]
        df["key"] = df.description.map(make_key)
        return df

    tr, te, ch = prep(tr), prep(te), prep(ch)
    for c in CATS: tr[f"y{c}"] = tr.found_microcategories.map(lambda l: int(c in [x["mcId"] for x in l]))
    te_ids = te.categories.map(lambda l: [NAME_TO_ID[n] for n in l])
    for c in CATS: te[f"y{c}"] = te_ids.map(lambda l: int(c in l))
    ch_ids = ch.expected.map(list)
    for c in CATS: ch[f"y{c}"] = ch_ids.map(lambda l: int(c in l))
    ycols = [f"y{c}" for c in CATS]

    # silver: уникальные по key
    n_dups = tr.groupby("key").size().rename("n_dups")
    silver = tr.drop_duplicates("key").merge(n_dups, left_on="key", right_index=True).reset_index(drop=True)
    silver = silver[["item_id", "title", "description", "text", "key", "n_dups", *ycols]]

    gold_raw = te.reset_index(drop=True).rename_axis("row_id").reset_index()[["row_id", "title", "description", "text", "key", *ycols]]

    # gold_consensus: большинство голосов, ничья -> маска 0
    rows = []
    for k, g in gold_raw.groupby("key", sort=False):
        n = len(g); r = g.iloc[0][["title", "description", "text", "key"]].to_dict(); conflict = False
        for c in CATS:
            s = int(g[f"y{c}"].sum()); conflict |= 0 < s < n
            r[f"y{c}"] = int(2 * s > n); r[f"m{c}"] = int(2 * s != n)
        r.update(n_dups=n, has_conflict=conflict); rows.append(r)
    cons = pd.DataFrame(rows)
    cons = cons[["title", "description", "text", "key", "n_dups", "has_conflict", *ycols, *[f"m{c}" for c in CATS]]]

    # кластеры по объединению уникальных текстов трейна и теста
    all_keys = pd.concat([silver[["key", "text"]], cons[["key", "text"]]]).drop_duplicates("key").reset_index(drop=True)
    cl = pd.Series(_clusters(all_keys.text.tolist(), cfg["data"]["cluster_sim"]), index=all_keys.key)
    silver["cluster_id"] = silver.key.map(cl)
    gold_clusters = set(cons.key.map(cl))
    silver["in_public_test"] = silver.cluster_id.isin(gold_clusters)
    silver["fold"] = -1
    folds = GroupKFold(n_splits=cfg["data"]["n_folds"], shuffle=True, random_state=cfg["data"]["seed"])
    for f, (_, idx) in enumerate(folds.split(silver, groups=silver.cluster_id)):
        silver.loc[silver.index[idx], "fold"] = f

    challenge = ch.reset_index(drop=True).rename_axis("row_id").reset_index()[["row_id", "id", "phenomenon", "title", "description", "text", "key", *ycols]]
    tabs = {"silver": silver, "gold_raw": gold_raw, "gold_consensus": cons, "challenge": challenge}
    for n, t in tabs.items(): t.to_parquet(out / f"{n}.parquet", index=False)
    return tabs


def load(name: str, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Читает таблицу из data/processed."""
    return pd.read_parquet(path(cfg or load_config(), "processed") / f"{name}.parquet")
