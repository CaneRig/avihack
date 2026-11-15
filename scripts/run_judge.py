"""Прогон судьи (мягкий + строгий + арбитр) на срезах gold_consensus, gold_raw, challenge; полные ответы кэшируются в reports/judge_<tag>_<срез>.jsonl.

Из одного кэша без новых вызовов считаются три варианта: только мягкий судья, только строгий, panel.
Повторная оценка (--cached) читает кэш и LLM не вызывает. gold_raw берёт ответы по ключу текста (тексты повторяются).
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
from avito_drafts.config import CATS, ROOT, load_config, path  # noqa: E402
from avito_drafts.judge import Judge  # noqa: E402
from avito_drafts.layer1 import TfidfLR  # noqa: E402
from avito_drafts.llm import LLMClient  # noqa: E402

TAG = ""  # суффикс кэша (--tag)
VARIANTS = [("soft", "только мягкий судья"), ("strict", "только строгий судья"), ("panel", "panel: мягкий + строгий + арбитр")]


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
                rec = {"key": k, "cats": cats, "candidates": info["candidates"], "sources": info["sources"], "tfidf_probs": info["tfidf_probs"],
                       "roles": info["roles"], "reasons": info["reasons"], "strict_roles": info["strict_roles"],
                       "strict_reasons": info["strict_reasons"], "arbiter": info["arbiter"], "disagree": info["disagree"],
                       "lines": info["lines"], "fallback": info["fallback"], "errors": info["errors"], "calls": info["calls"],
                       "stages": {s: {a: round(b, 1) for a, b in v.items()} for s, v in info["stages"].items()},
                       "tin": info["tokens_in"], "tout": info["tokens_out"], "sec": round(info["seconds"], 1),
                       "called": sum(info["calls"].values()) > 0}
                n += 1
                print(f"{name} {n} {time.time() - t0:.0f}c cats={cats} dis={info['disagree']} err={info['errors']} sec={rec['sec']}", flush=True)
            cache[k] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    return cache


def variant_cats(rec: dict, variant: str, remove: set[str]) -> list[int]:
    """Итоговые категории записи для варианта: soft / strict / panel (strict при сбое строгого = soft)."""
    if variant == "panel" or rec.get("fallback") or "strict_roles" not in rec:
        return rec["cats"]
    roles = rec["roles"]
    if variant == "strict" and rec["strict_roles"]:
        roles = rec["strict_roles"]
    return [c for c in rec["candidates"] if roles.get(str(c)) not in remove]  # нет роли (ключа) -> категория оставлена


def prf(Y: np.ndarray, P: np.ndarray, M: np.ndarray) -> dict:
    """F1 и счётчики по маске ячеек."""
    tp = int(((P == 1) & (Y == 1) & (M == 1)).sum())
    fp = int(((P == 1) & (Y == 0) & (M == 1)).sum())
    fn = int(((P == 0) & (Y == 1) & (M == 1)).sum())
    return {"f1": 2 * tp / max(1, 2 * tp + fp + fn), "fp": fp, "fn": fn, "exact": int((((P == Y) | (M == 0)).all(1)).sum()), "n": len(Y)}


def fmt_prf(r: dict) -> str:
    return f"F1 {r['f1']:.3f} лишних {r['fp']:2d} пропущенных {r['fn']:2d} точных {r['exact']}/{r['n']}"


def report(name: str, cfg: dict, cache: dict[str, dict], exp: str, note: str) -> None:
    sl = E.load_slice(name, cfg)
    keys = [row_key(r) for r in sl.df.itertuples()]
    remove = set(cfg["judge"]["remove_roles"])
    uniq = {k: cache[k] for k in set(keys)}
    called = [v for v in uniq.values() if v.get("called")]
    calls = {s: sum(v.get("calls", {}).get(s, 0) for v in called) for s in ("soft", "strict", "arbiter")}
    errors = sum(len(v.get("errors") or {}) for v in called)
    stats = {"llm_calls": calls, "llm_errors": errors, "fallback_docs": sum(bool(v.get("fallback")) for v in uniq.values())}
    if called:
        stats["sec_median"] = float(np.median([v["sec"] for v in called]))
        stats["tokens_in_mean"] = float(np.mean([v["tin"] for v in called]))
        stats["tokens_out_mean"] = float(np.mean([v["tout"] for v in called]))
    print(f"== {name}: вызовов {calls}, ошибок {errors}, объявлений с запасным ответом {stats['fallback_docs']}, "
          f"медиана секунд на объявление {stats.get('sec_median')}")
    mats = {}
    for var, label in VARIANTS:
        preds = {i: variant_cats(cache[k], var, remove) for i, k in zip(sl.ids, keys)}
        res = E.evaluate(sl, preds, cfg)
        P = E.pred_matrix(sl, preds)
        mats[var] = P
        res["tp"] = int(((P == 1) & (sl.y == 1) & sl.mask.astype(bool)).sum())
        res["exact_docs"] = int(((P == sl.y) | (sl.mask == 0)).all(1).sum())
        res.update(stats)
        res["variant"] = var
        print(f"[{var:6s}]", E.fmt(res), f"| TP={res['tp']} точных {res['exact_docs']}/{res['n']}")
        E.log_run(res, exp, cfg, note=f"{note}вариант: {label}; тег {TAG.lstrip('_')}. ")
    P = mats["panel"]
    # вклад кандидатов, пришедших только от TF-IDF
    cells = [(r, c) for r, k in enumerate(keys) for c in cache[k].get("candidates", []) if cache[k]["sources"].get(str(c)) == "tfidf"]
    cells = [(r, c) for r, c in cells if sl.mask[r, CATS.index(c)]]
    kept = sum(P[r, CATS.index(c)] for r, c in cells)
    gold1 = sum(sl.y[r, CATS.index(c)] for r, c in cells)
    right = sum(P[r, CATS.index(c)] == sl.y[r, CATS.index(c)] for r, c in cells)
    print(f"ячеек-кандидатов только от TF-IDF: {len(cells)}; по эталону должны быть выданы: {gold1}; оставлены судьёй: {kept}; "
          f"итог верен в {right} из {len(cells)}; верных выдач (TP) среди них: {sum(P[r, CATS.index(c)] == 1 and sl.y[r, CATS.index(c)] == 1 for r, c in cells)}")
    if name != "challenge":
        return
    sp = {j["id"]: j["split"] for j in map(json.loads, filter(str.strip, open(ROOT / cfg["paths"]["challenge"], encoding="utf-8")))}
    S = np.array([sp[i] for i in sl.df["id"]])
    for var, _ in VARIANTS:
        for part, m in (("всего", np.ones(len(S), bool)), ("dev", S == "dev"), ("holdout", S == "holdout")):
            print(f"  {var:6s} {part:8s}", fmt_prf(prf(sl.y[m], mats[var][m], sl.mask[m])))
    print("ошибки panel (id, категория, эталон, мягкий / строгий / арбитр):")
    for r, k in enumerate(keys):
        rec = cache[k]
        for j, c in enumerate(CATS):
            if P[r, j] != sl.y[r, j]:
                print(f"  {sl.df['id'].iloc[r]} [{S[r]}] c{c} эталон {sl.y[r, j]} выдано {P[r, j]} | {rec['roles'].get(str(c))} / "
                      f"{rec.get('strict_roles', {}).get(str(c))} / {(rec.get('arbiter') or {}).get(str(c), {}).get('answer')} | источник {rec.get('sources', {}).get(str(c))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="C25")
    ap.add_argument("--note", default="Судья роли упоминания, схема из трёх экспертов (T-lite Q4_K_M, t=0), заголовок не используется, кандидаты: корни + TF-IDF. ")
    ap.add_argument("--cached", action="store_true", help="только оценка по кэшу, без вызовов LLM")
    ap.add_argument("--slices", nargs="+", default=["gold_consensus", "gold_raw", "challenge"])
    ap.add_argument("--model", help="другая модель судьи (имя в Ollama)")
    ap.add_argument("--tag", default="", help="суффикс файлов кэша, например panel -> reports/judge_panel_<срез>.jsonl")
    ap.add_argument("--prompt", help="другой файл системной подсказки мягкого судьи")
    ap.add_argument("--mode", help="panel | single (по умолчанию из конфигурации)")
    ap.add_argument("--no-think", action="store_true", help="отключить рассуждения (Qwen3.5)")
    a = ap.parse_args()
    cfg = load_config()
    if a.tag:
        TAG = "_" + a.tag
    if a.model:
        cfg["llm"]["model"] = a.model
    if a.prompt:
        cfg["judge"]["prompt"] = a.prompt
    if a.mode:
        cfg["judge"]["mode"] = a.mode
    if a.no_think:
        cfg["llm"]["think"] = False
    judge = None
    if not a.cached:
        try:
            l1 = TfidfLR.load(path(cfg, "tfidf_dir"))
        except Exception:  # noqa: BLE001  нет модели слоя 1 -> источник TF-IDF выключен
            l1 = None
        judge = Judge(cfg, LLMClient(cfg), l1)
    shared: dict[str, dict] = {}
    for name in a.slices:
        cache = judge_slice(name, cfg, judge, shared)
        shared.update(cache)
        report(name, cfg, cache, a.exp, a.note)
