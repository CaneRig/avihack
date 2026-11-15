"""Файл для бота по приватному набору: CSV testcase,description -> CSV testcase,microcats,descriptions.

Ответы дописываются в <out>.jsonl по мере готовности, поэтому прерванный запуск продолжается с места остановки
(уже посчитанные строки не пересчитываются, даже если сменить --drafts или --workers).
Пример: python scripts/make_submission.py .nocommit/private-test-dataset.csv .nocommit/sub_panel.csv --judge-mode panel --drafts template --workers 4 --with-101
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_drafts.config import load_config  # noqa: E402
from avito_drafts.predict import Predictor  # noqa: E402

JUDGE_KEYS = ("candidates", "sources", "roles", "strict_roles", "arbiter", "disagree", "fallback", "errors")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="CSV со столбцами testcase, description")
    ap.add_argument("output", help="CSV для бота")
    ap.add_argument("--drafts", default="llm", choices=["llm", "template", "none"],
                    help="llm: черновики пишет LLM (медленно); template: шаблон из фраз исходника, без LLM; none: без черновиков")
    ap.add_argument("--judge-mode", default=None, choices=["single", "panel"], help="один мягкий судья или три эксперта (по умолчанию из конфигурации)")
    ap.add_argument("--variant", default="c", choices=["a", "b", "c"], help="вариант подсказки черновика; c (только фрагменты) не зависит от длины объявления")
    ap.add_argument("--with-101", action="store_true", help="добавлять 101 в microcats каждой строки")
    ap.add_argument("--workers", type=int, default=1, help="одновременных объявлений; имеет смысл при OLLAMA_NUM_PARALLEL >= workers")
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
    pr = Predictor(cfg, mode="full" if a.drafts == "llm" else "classify")
    lock = threading.Lock()
    t0 = time.time()

    def work(tc: str, desc: str) -> dict:
        ans, info = pr.predict_batch([{"description": desc, "title": "", "mcId": 101}], progress=False)[0]
        drafts = ans["drafts"]
        if a.drafts == "template" and drafts:
            items = pr.drafter().make(desc, "", [d["mcId"] for d in drafts], use_llm=False)
            drafts = [{"mcId": i["mcId"], "title": i["title"], "text": i["text"]} for i in items]
        return {"testcase": tc, "drafts": drafts, "drafts_by": a.drafts, "source": info.get("source"), "error": info.get("error"),
                "judge": {k: (info.get("judge") or {}).get(k) for k in JUDGE_KEYS}}

    with open(cache, "a", encoding="utf-8") as f:
        def save(j: dict) -> None:
            with lock:
                done[j["testcase"]] = j
                f.write(json.dumps(j, ensure_ascii=False, default=str) + "\n")
                f.flush()
                if len(done) % 5 == 0 or len(done) == len(df):
                    print(f"{len(done)}/{len(df)}  {time.time() - t0:.0f} с", flush=True)

        rows = list(zip(todo.testcase, todo.description))
        if rows:
            save(work(*rows[0]))  # первая строка в одном потоке: прогрев модели и ленивых загрузок
        if a.workers <= 1:
            for r in rows[1:]:
                save(work(*r))
        else:
            with ThreadPoolExecutor(a.workers) as ex:
                for fut in as_completed([ex.submit(work, *r) for r in rows[1:]]):
                    save(fut.result())

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
