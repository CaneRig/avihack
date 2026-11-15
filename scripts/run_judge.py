"""Прогон судьи роли упоминания на срезах gold_consensus, gold_raw, challenge; ответы кэшируются в reports/judge_<срез>.jsonl.

Повторная оценка читает кэш и LLM не вызывает. gold_raw берёт ответы по ключу текста (тексты повторяются).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_drafts import evalkit as E  # noqa: E402
from avito_drafts.config import CATS, ROOT, load_config  # noqa: E402
from avito_drafts.judge import Judge  # noqa: E402
from avito_drafts.llm import LLMClient  # noqa: E402


TAG = ""  # суффикс кэша для сравнения моделей (--tag)


def cache_path(name: str) -> Path:
    return ROOT / "reports" / f"judge{TAG}_{name}.jsonl"


def read_cache(name: str) -> dict[str, dict]:
    p = cache_path(name)
    if not p.exists():
        return {}
    return {j["key"]: j for j in map(json.loads, filter(str.strip, p.read_text(encoding="utf-8").splitlines()))}


def row_key(r) -> str:
    from avito_drafts.text import make_key
    return str(getattr(r, "key", None) or make_key(r.description))


def judge_slice(name: str, cfg: dict, judge: Judge | None, shared: dict[str, dict]) -> dict[str, dict]:
    """Ответы судьи по ключу текста для среза; недостающие считаются (если есть судья) и дописываются в кэш."""
    sl = E.load_slice(name, cfg)
    cache = read_cache(name)
    p = cache_path(name)
    t0, n = time.time(), 0
    with open(p, "a", encoding="utf-8") as f:
        for r in sl.df.itertuples():
            k = row_key(r)
            if k in cache:
                continue
            if k in shared:  # тот же текст уже судили на другом срезе
                rec = shared[k]
            elif judge is None:
                raise SystemExit(f"нет кэша для {name}/{k[:20]}, а LLM не используется")
            else:
                cats, info = judge.judge(r.title, r.description)
                rec = {"key": k, "cats": cats, "roles": info["roles"], "reasons": info["reasons"], "lines": info["lines"],
                       "fallback": info["fallback"], "error": info.get("error"), "tin": info["tokens_in"],
                       "tout": info["tokens_out"], "sec": round(info["seconds"], 1), "called": bool(info["tokens_in"] or info.get("error"))}
                n += 1
                print(f"{name} {n} {time.time() - t0:.0f}c cats={cats} sec={rec['sec']}", flush=True)
            cache[k] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    return cache


def report(name: str, cfg: dict, cache: dict[str, dict], exp: str, note: str) -> None:
    sl = E.load_slice(name, cfg)
    keys = [row_key(r) for r in sl.df.itertuples()]
    preds = {i: cache[k]["cats"] for i, k in zip(sl.ids, keys)}
    res = E.evaluate(sl, preds, cfg)
    P = E.pred_matrix(sl, preds)
    m = sl.mask.astype(bool)
    res["tp"] = int(((P == 1) & (sl.y == 1) & m).sum())
    res["exact_docs"] = int((P == sl.y).all(1).sum())
    uniq = {k: cache[k] for k in set(keys)}
    called = [v for v in uniq.values() if v.get("called")]
    ok = [v for v in called if not v.get("error") and not v.get("fallback")]
    res["llm_calls"] = len(called)
    res["llm_errors"] = sum(bool(v.get("error")) for v in called)
    res["fallback_docs"] = sum(bool(v.get("fallback")) for v in uniq.values())
    if called:
        res["sec_median"] = float(np.median([v["sec"] for v in called]))
        res["tokens_in_mean"] = float(np.mean([v["tin"] for v in called]))
        res["tokens_out_mean"] = float(np.mean([v["tout"] for v in called]))
    print(E.fmt(res), f"| TP={res['tp']} точных объявлений {res['exact_docs']}/{res['n']} вызовов {res['llm_calls']} "
          f"ошибок {res['llm_errors']} медиана {res.get('sec_median')} токены {res.get('tokens_in_mean')}/{res.get('tokens_out_mean')}")
    E.log_run(res, exp, cfg, note=note)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="C17")
    ap.add_argument("--note", default="Судья роли упоминания (T-lite Q4_K_M, t=0), правило: снимают роли из judge.remove_roles. ")
    ap.add_argument("--cached", action="store_true", help="только оценка по кэшу, без вызовов LLM")
    ap.add_argument("--slices", nargs="+", default=["gold_consensus", "gold_raw", "challenge"])
    ap.add_argument("--model", help="другая модель судьи (имя в Ollama)")
    ap.add_argument("--tag", default="", help="суффикс файлов кэша, например q5 -> reports/judge_q5_<срез>.jsonl")
    ap.add_argument("--no-think", action="store_true", help="отключить рассуждения (Qwen3.5)")
    a = ap.parse_args()
    cfg = load_config()
    if a.tag:
        TAG = "_" + a.tag
    if a.model:
        cfg["llm"]["model"] = a.model
    if a.no_think:
        cfg["llm"]["think"] = False
    judge = None if a.cached else Judge(cfg, LLMClient(cfg))
    shared: dict[str, dict] = {}
    for name in a.slices:
        cache = judge_slice(name, cfg, judge, shared)
        shared.update(cache)
        report(name, cfg, cache, a.exp, a.note)
