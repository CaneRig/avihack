"""Эксперименты G3 (перегенерация) и G2 (варианты подсказки) на одной фиксированной выборке пар из gold_consensus.

Один прогон с регенерацией даёт обе строки G3: результат «без исправления» — это первый вызов того же прогона
(та же подсказка, то же зерно), «с исправлением» — итог после повторного вызова."""
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


def make_sample(cfg: dict) -> list[dict]:
    """По sample_per_cat положительных пар на категорию, фиксированное зерно; сохраняется в sample_file."""
    d, g = cfg["drafts"], load("gold_consensus", cfg)
    rng = np.random.default_rng(d["sample_seed"])
    out = []
    for c in CATS:
        idx = [i for i, r in enumerate(g.itertuples()) if getattr(r, f"y{c}") and getattr(r, f"m{c}")]
        out += [{"row": int(i), "cat": c, "key": __import__("hashlib").md5(str(g.iloc[i]["key"]).encode()).hexdigest()[:10]} for i in sorted(rng.choice(idx, d["sample_per_cat"], replace=False))]
    p = ROOT / d["sample_file"]
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def vtype(v: str) -> str:
    return v.split(":")[0].split(" ")[0]


def summarize(df: pd.DataFrame, regen: bool) -> dict:
    """Метрики одного режима; при regen=False берётся только первый вызов."""
    ok = df[df.tokens_in > 0]
    first_ok = df.viol_first.map(lambda v: v == [])
    if regen:
        src, sec, tin, tout = df.source, ok.seconds, ok.tokens_in, ok.tokens_out
        viol = Counter(vtype(v) for vs in df.violations for v in vs)
        warn = ok.warnings
    else:
        src = df.apply(lambda r: "llm" if r.viol_first == [] else "template", axis=1)
        sec, tin, tout = df.sec1.dropna(), df.tin1.dropna(), df.tout1.dropna()
        viol = Counter(vtype(v) for vs in df.viol_first.dropna() for v in vs)
        warn = df.warn1.dropna()
    n = len(df)
    res = {"slice": "gold_consensus_sample", "n": n, "regen": regen,
           "passed_first_share": float(first_ok.mean()),
           "fixed_by_regen_share": float((src == "llm_regen").mean()),
           "template_share": float((src == "template").mean()),
           "violations_" + ("final_attempt" if regen else "first"): dict(viol),
           "violations_first": dict(Counter(vtype(v) for vs in df.viol_first.dropna() for v in vs)),
           "llm_errors": int(df.get("error", pd.Series(dtype=object)).notna().sum()),
           "sec_mean": float(sec.mean()), "sec_median": float(sec.median()), "tokens_in_mean": float(tin.mean()),
           "tokens_out_mean": float(tout.mean()),
           "coverage_warn_share": float(warn.map(lambda w: any("покрытие" in x for x in w)).mean()),
           "short_text_warn_share": float(warn.map(lambda w: any("короткий" in x for x in w)).mean())}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="b", choices=["a", "b", "c"])
    ap.add_argument("--exp", default="G2", help="через запятую: G2, G3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--make-sample", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    cfg = load_config()
    sp = ROOT / cfg["drafts"]["sample_file"]
    if a.make_sample or not sp.exists():
        make_sample(cfg)
        print("sample saved", sp)
        if a.make_sample:
            sys.exit()
    sample = json.loads(sp.read_text(encoding="utf-8"))[: a.limit or None]
    g = load("gold_consensus", cfg)
    d = Drafter(cfg, TfidfLR.load(ROOT / cfg["paths"]["tfidf_dir"]))
    rows, t0 = [], time.time()
    for i, s in enumerate(sample):
        r_ = g.iloc[s["row"]]
        r = d.make(r_["description"], r_["title"], [s["cat"]], variant=a.variant, regen=True)[0]
        rows.append({"key": s["key"], "variant": a.variant, "title_src": r_["title"], "description": r_["description"], **r})
        print(i, len(sample), r["source"], r["viol_first"], r["violations"] if r["source"] == "template" else "", f"{r['seconds']:.1f}с", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / f"g_run_{a.variant}.csv", index=False, encoding="utf-8-sig")
    sec = round(time.time() - t0, 1)
    for exp in a.exp.split(","):
        if exp == "G3":
            r0 = summarize(df, False)
            E.log_run({**r0, "variant": a.variant}, "G3", cfg, seconds=sec, note=f"БЕЗ перегенерации (первый вызов того же прогона), вариант {a.variant}. {a.note}")
            print(json.dumps(r0, ensure_ascii=False, indent=1))
        r1 = summarize(df, True)
        E.log_run({**r1, "variant": a.variant}, exp, cfg, seconds=sec, note=f"С перегенерацией, вариант подсказки {a.variant}. {a.note}")
        print(json.dumps(r1, ensure_ascii=False, indent=1))
