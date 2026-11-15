"""Эксперимент G1: базовая подсказка, T-lite. Доля черновиков, прошедших проверки с первого раза; нарушения; время и токены."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_drafts import evalkit as E  # noqa: E402
from avito_drafts.config import CATS, ROOT, load_config  # noqa: E402
from avito_drafts.data import load  # noqa: E402
from avito_drafts.drafts.pipeline import Drafter  # noqa: E402
from avito_drafts.layer1 import TfidfLR  # noqa: E402


def pairs(cfg: dict, n_train: int) -> list[tuple[str, int, str, str, int]]:
    """Положительные пары (источник, mcId, title, description, mcId): все из gold_consensus и случайные из трейна."""
    g, s = load("gold_consensus", cfg), load("silver", cfg)
    out = [("gold", c, r.title, r.description, 0) for r in g.itertuples() for c in CATS if getattr(r, f"y{c}") and getattr(r, f"m{c}")]
    rng = np.random.default_rng(cfg["data"]["seed"])
    s = s[~s.in_public_test].sample(n_train, random_state=int(rng.integers(1 << 30)))
    return out + [("train", c, r.title, r.description, 0) for r in s.itertuples() for c in CATS if getattr(r, f"y{c}") and rng.random() < 0.25][:n_train]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0, help="для отладки: взять только первые N пар")
    a = ap.parse_args()
    cfg = load_config()
    d = Drafter(cfg, TfidfLR.load(ROOT / cfg["paths"]["tfidf_dir"]))
    ps = pairs(cfg, a.n_train)[: a.limit or None]
    rows, viol, t0 = [], Counter(), time.time()
    for i, (src, cat, title, desc, _) in enumerate(ps):
        r = d.make(desc, title, [cat])[0]
        rows.append({"set": src, "title_src": title, "description": desc, **r})
        viol.update(v.split(":")[0].split(" ")[0] for v in r["violations"])
        print(i, len(ps), r["source"], r["violations"], f"{r['seconds']:.1f}с", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "reports"
    df.to_csv(out / "g1_drafts.csv", index=False, encoding="utf-8-sig")
    ok = df[df.tokens_in > 0]
    res = {"slice": "gold_consensus_pairs+train_sample", "n": len(df), "first_pass_ok_share": float((ok.source == "llm").mean()),
           "template_share": float((df.source == "template").mean()), "violations": dict(viol),
           "llm_errors": int(df.get("error", pd.Series(dtype=object)).notna().sum()),
           "sec_median": float(ok.seconds.median()), "sec_p95": float(ok.seconds.quantile(0.95)),
           "tokens_in_mean": float(ok.tokens_in.mean()), "tokens_out_mean": float(ok.tokens_out.mean()),
           "short_text_warn_share": float(ok.warnings.map(lambda w: any("короткий" in x for x in w)).mean()),
           "coverage_warn_share": float(ok.warnings.map(lambda w: any("покрытие" in x for x in w)).mean())}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    E.log_run(res, "G1", cfg, seconds=round(time.time() - t0, 1), note="базовая подсказка, T-lite Q4_K_M, без перегенерации")
