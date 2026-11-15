"""Файл для бота по приватному набору: CSV testcase,description -> CSV testcase,microcats,descriptions.

Считает порциями и дописывает ответы в <out>.jsonl, поэтому прерванный запуск продолжается с места остановки.
Пример: python scripts/make_submission.py .nocommit/private-test-dataset.csv .nocommit/sub_single.csv --judge-mode single --mode full
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_drafts.config import load_config  # noqa: E402
from avito_drafts.predict import Predictor  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="CSV со столбцами testcase, description")
    ap.add_argument("output", help="CSV для бота")
    ap.add_argument("--mode", default="full", choices=["classify", "fast", "full"], help="classify: без черновиков; fast: шаблонные; full: LLM")
    ap.add_argument("--judge-mode", default=None, choices=["single", "panel"], help="один мягкий судья или три эксперта (по умолчанию из конфигурации)")
    ap.add_argument("--variant", default="c", choices=["a", "b", "c"], help="вариант подсказки черновика; c (только фрагменты) не зависит от длины объявления")
    ap.add_argument("--with-101", action="store_true", help="добавлять 101 в microcats каждой строки")
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="только первые N строк (проверка)")
    a = ap.parse_args()

    cfg = load_config()
    if a.judge_mode:
        cfg["judge"]["mode"] = a.judge_mode
    cfg["drafts"]["prompt_variant"] = a.variant
    df = pd.read_csv(a.input)
    if a.limit:
        df = df.head(a.limit)
    cache = Path(a.output + ".jsonl")
    done: dict[str, dict] = {}
    if cache.exists():
        done = {j["testcase"]: j for j in map(json.loads, filter(str.strip, cache.read_text(encoding="utf-8").splitlines()))}
    todo = df[~df.testcase.isin(done)]
    pr = Predictor(cfg, mode=a.mode)
    t0 = time.time()
    with open(cache, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), a.chunk):
            part = todo.iloc[s:s + a.chunk]
            res = pr.predict_batch([{"description": d, "title": "", "mcId": 101} for d in part.description], progress=False)
            for tc, (ans, info) in zip(part.testcase, res):
                j = {"testcase": tc, "drafts": ans["drafts"], "source": info.get("source"), "error": info.get("error"),
                     "judge": {k: info.get("judge", {}).get(k) for k in ("candidates", "sources", "roles", "strict_roles", "arbiter", "disagree", "fallback", "errors")}}
                done[tc] = j
                f.write(json.dumps(j, ensure_ascii=False, default=str) + "\n")
            f.flush()
            print(f"{len(done)}/{len(df)}  {time.time() - t0:.0f} с", flush=True)
    with open(a.output, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["testcase", "microcats", "descriptions"])
        for tc in df.testcase:
            dr = done[tc]["drafts"]
            cats = ([101] if a.with_101 else []) + [d["mcId"] for d in dr]
            texts = [((d.get("title") or "") + "\n" + d["text"]).strip() for d in dr if d.get("text")]
            w.writerow([tc, json.dumps(cats), json.dumps(texts, ensure_ascii=False)])
    n_cat = sum(len(done[t]["drafts"]) for t in df.testcase)
    empty = sum(not done[t]["drafts"] for t in df.testcase)
    src = pd.Series([done[t]["source"] for t in df.testcase]).value_counts().to_dict()
    print(f"строк {len(df)}, категорий {n_cat}, без категорий {empty} ({empty / len(df):.0%}), источники {src}, {time.time() - t0:.0f} с -> {a.output}")
