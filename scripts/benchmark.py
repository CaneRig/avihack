"""Замеры для защиты: качество, время, токены и оценка стоимости для выбранной конфигурации -> reports/benchmark.json.

Примеры:
  python scripts/benchmark.py --slice challenge --judge-mode panel single --drafts llm
  python scripts/benchmark.py --slice gold_consensus --judge-mode panel --drafts none
  python scripts/benchmark.py --input .nocommit/private-test-dataset.csv --judge-mode single --drafts template --limit 50
Срез с эталоном (challenge, gold_consensus, gold_raw) даёт ещё и micro-F1; файл без эталона (--input) — только время и расход.
Стоимость — оценка: мощность, тариф и цена аренды задаются параметрами и в измерения не входят.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_drafts import evalkit as E  # noqa: E402
from avito_drafts.config import ROOT, load_config  # noqa: E402
from avito_drafts.predict import Predictor  # noqa: E402


def stat(xs: list[float]) -> dict[str, float] | None:
    """Медиана, 95-й перцентиль, среднее и число наблюдений."""
    if not xs:
        return None
    a = np.asarray(xs, float)
    return {"median": round(float(np.median(a)), 3), "p95": round(float(np.percentile(a, 95)), 3), "mean": round(float(a.mean()), 3), "n": int(a.size)}


def gpu() -> dict[str, object]:
    """Название видеокарты и занятая память по nvidia-smi; пусто, если утилиты нет."""
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,power.draw", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0].split(", ")
        return {"name": o[0], "memory_used_mb": float(o[1]), "memory_total_mb": float(o[2]), "power_draw_w": float(o[3])}
    except Exception:  # noqa: BLE001
        return {}


def run(cfg: dict, rows: list[dict], judge_mode: str, drafts: str, variant: str) -> tuple[list[dict], dict]:
    """Прогон одной конфигурации: ответы по строкам и сводка времени и расхода."""
    cfg = json.loads(json.dumps(cfg))
    cfg["judge"]["mode"] = judge_mode
    cfg["drafts"]["prompt_variant"] = variant
    pr = Predictor(cfg, mode="full" if drafts == "llm" else "classify")
    recs, cache, peak, watts = [], {}, 0.0, []
    t_all = time.time()
    for i, r in enumerate(rows):
        t0 = time.time()
        if r["description"] in cache:  # точный повтор: считаем один раз, как в пакетном запуске
            ans, info, dup = *cache[r["description"]], True
        else:
            ans, info = pr.predict_batch([{"description": r["description"], "title": "", "mcId": 101}], progress=False)[0]
            dup = False
            if drafts == "template" and ans["drafts"]:
                info["drafts"] = pr.drafter().make(r["description"], "", [d["mcId"] for d in ans["drafts"]], use_llm=False)
            cache[r["description"]] = (ans, info)
        recs.append({"id": r["id"], "cats": [d["mcId"] for d in ans["drafts"]], "seconds": time.time() - t0, "dup": dup, "info": info})
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            g = gpu()
            peak = max(peak, g.get("memory_used_mb", 0.0))
            if g.get("power_draw_w"):
                watts.append(g["power_draw_w"])
            print(f"  {judge_mode}/{drafts}: {i + 1}/{len(rows)}  {time.time() - t_all:.0f} с", flush=True)
    wall = time.time() - t_all
    uniq = [x for x in recs if not x["dup"]]
    J = [x["info"].get("judge") or {} for x in uniq]
    called = [j for j in J if sum((j.get("calls") or {}).values())]
    D = [d for x in uniq for d in (x["info"].get("drafts") or [])]
    llm_d = [d for d in D if d.get("calls")]
    tin = sum(j.get("tokens_in", 0) for j in J) + sum(d.get("tokens_in", 0) for d in D)
    tout = sum(j.get("tokens_out", 0) for j in J) + sum(d.get("tokens_out", 0) for d in D)
    llm_sec = sum(j.get("seconds", 0.0) for j in J) + sum(d.get("seconds", 0.0) for d in D)
    stages = {}
    for st in ("soft", "strict", "arbiter"):
        cs = [j["stages"][st] for j in called if j["calls"].get(st)]
        n = sum(j["calls"].get(st, 0) for j in called)
        stages[st] = {"calls": n, "seconds_per_ad": stat([c["seconds"] for c in cs]),
                      "tokens_in_per_call": round(sum(c["tokens_in"] for c in cs) / n, 1) if n else None,
                      "tokens_out_per_call": round(sum(c["tokens_out"] for c in cs) / n, 1) if n else None}
    src = pd.Series([d.get("source") for d in D]).value_counts().to_dict() if D else {}
    summ = {
        "judge_mode": judge_mode, "drafts": drafts, "ads": len(rows), "unique_ads": len(uniq), "wall_seconds": round(wall, 1),
        "ads_per_minute": round(len(rows) / wall * 60, 1),
        "seconds_per_ad": stat([x["seconds"] for x in uniq]),
        "share_ads_with_llm_call": round(len(called) / max(len(uniq), 1), 3),
        "share_ads_without_categories": round(sum(not x["cats"] for x in recs) / len(recs), 3),
        "decision_sources": pd.Series([x["info"].get("source") or "none" for x in uniq]).value_counts().to_dict(),
        "judge": {"llm_calls_per_ad_mean": round(sum(sum(j["calls"].values()) for j in called) / max(len(uniq), 1), 2),
                  "seconds_per_judged_ad": stat([j["seconds"] for j in called]), "stages": stages,
                  "disagreements": sum(len(j.get("disagree") or []) for j in J),
                  "fallback_ads": sum(bool(j.get("fallback")) for j in J), "errors": sum(len(j.get("errors") or {}) for j in J),
                  "tfidf_only_candidates": sum(v == "tfidf" for j in J for v in (j.get("sources") or {}).values())},
        "drafts_stats": {"count": len(D), "by_source": src,
                         "passed_first_share": round(sum(d.get("viol_first") == [] for d in llm_d) / len(llm_d), 3) if llm_d else None,
                         "template_share": round(src.get("template", 0) / len(D), 3) if D else None,
                         "seconds_per_draft": stat([d["seconds"] for d in llm_d]),
                         "llm_calls_per_draft_mean": round(sum(d["calls"] for d in llm_d) / len(llm_d), 2) if llm_d else None,
                         "tokens_in_per_draft": round(sum(d["tokens_in"] for d in llm_d) / len(llm_d), 1) if llm_d else None,
                         "tokens_out_per_draft": round(sum(d["tokens_out"] for d in llm_d) / len(llm_d), 1) if llm_d else None,
                         "errors": sum("error" in d for d in D)},
        "tokens": {"in_total": int(tin), "out_total": int(tout), "in_per_ad": round(tin / len(uniq), 1), "out_per_ad": round(tout / len(uniq), 1),
                   "out_tokens_per_llm_second": round(tout / llm_sec, 1) if llm_sec else None},
        "gpu": {"peak_memory_mb": peak or None, "power_draw_w_mean": round(float(np.mean(watts)), 1) if watts else None},
    }
    return recs, summ


def quality(sl: E.SliceData, recs: list[dict], cfg: dict) -> dict:
    """Метрики среза с эталоном; для трудных примеров — ещё и по частям dev и holdout."""
    preds = {r["id"]: r["cats"] for r in recs}
    q = E.evaluate(sl, preds, cfg)
    P = E.pred_matrix(sl, preds)
    q["exact_ads"] = int(((P == sl.y) | (sl.mask == 0)).all(1).sum())
    if sl.name == "challenge":
        lines = (ROOT / "data/challenge/seed_examples.jsonl").read_text(encoding="utf-8").splitlines()
        sp = {j["id"]: j.get("split", "dev") for j in map(json.loads, filter(str.strip, lines))}
        part = np.array([sp.get(x, "dev") for x in sl.df.id])
        for name in ("dev", "holdout"):
            m = part == name
            tp, fp, fn = (int(x[m].sum()) for x in ((P == 1) & (sl.y == 1), (P == 1) & (sl.y == 0), (P == 0) & (sl.y == 1)))
            q[name] = {"n": int(m.sum()), "micro_f1": round(2 * tp / max(1, 2 * tp + fp + fn), 3), "fp": fp, "fn": fn,
                       "exact_ads": int((P == sl.y)[m].all(1).sum())}
    return q


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", choices=["challenge", "gold_consensus", "gold_raw"], help="срез с эталоном")
    ap.add_argument("--input", help="CSV без эталона: столбцы testcase (или id) и description")
    ap.add_argument("--judge-mode", nargs="+", default=["panel"], choices=["panel", "single"])
    ap.add_argument("--drafts", default="llm", choices=["llm", "template", "none"])
    ap.add_argument("--variant", default="c", choices=["a", "b", "c"], help="вариант подсказки черновика")
    ap.add_argument("--limit", type=int, default=0, help="только первые N строк")
    ap.add_argument("--out", default="reports/benchmark.json")
    ap.add_argument("--watts", type=float, default=0.0, help="мощность под нагрузкой, Вт; 0 — взять среднюю по nvidia-smi (только видеокарта)")
    ap.add_argument("--kwh-price", type=float, default=7.0, help="тариф, руб. за кВт·ч (оценка)")
    ap.add_argument("--gpu-rent", type=float, default=60.0, help="аренда видеокарты, руб. в час (оценка)")
    a = ap.parse_args()
    if bool(a.slice) == bool(a.input):
        ap.error("нужен ровно один источник: --slice или --input")

    cfg = load_config()
    sl = None
    if a.slice:
        sl = E.load_slice(a.slice, cfg)
        rows = [{"id": i, "description": d} for i, d in zip(sl.ids, sl.df.description)]
    else:
        df = pd.read_csv(a.input)
        idc = "testcase" if "testcase" in df else "id" if "id" in df else None
        rows = [{"id": str(df[idc].iloc[k]) if idc else str(k), "description": d} for k, d in enumerate(df.description)]
    if a.limit:
        rows = rows[:a.limit]
    report = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": a.slice or a.input, "model": cfg["llm"]["model"], "num_ctx": cfg["llm"]["num_ctx"],
              "gpu": gpu(), "cost_assumptions": {"kwh_price_rub": a.kwh_price, "gpu_rent_rub_per_hour": a.gpu_rent,
                                                 "note": "стоимость — оценка, не измерение; мощность по nvidia-smi учитывает только видеокарту"},
              "runs": []}
    for jm in a.judge_mode:
        recs, summ = run(cfg, rows, jm, a.drafts, a.variant)
        if sl is not None:
            summ["quality"] = quality(sl, recs, cfg) if len(rows) == len(sl.ids) else {"note": "срез обрезан параметром --limit, метрика не считается"}
        w = a.watts or summ["gpu"]["power_draw_w_mean"] or 0.0
        hours = summ["wall_seconds"] / 3600
        summ["cost_estimate_per_1000_ads_rub"] = {
            "electricity": round(w / 1000 * hours * a.kwh_price / len(rows) * 1000, 2) if w else None,
            "gpu_rent": round(hours * a.gpu_rent / len(rows) * 1000, 2), "watts_used": w or None}
        report["runs"].append(summ)
        q = summ.get("quality") or {}
        print(f"== {jm} / черновики {a.drafts}: {summ['ads']} объявлений за {summ['wall_seconds']} с ({summ['ads_per_minute']} в минуту); "
              f"на объявление медиана {summ['seconds_per_ad']['median']} с, p95 {summ['seconds_per_ad']['p95']} с; "
              f"вызовов судьи на объявление {summ['judge']['llm_calls_per_ad_mean']}; токенов на объявление {summ['tokens']['in_per_ad']}/{summ['tokens']['out_per_ad']}"
              + (f"; micro-F1 {q['micro_f1']:.3f} (лишних {q['fp']}, пропущенных {q['fn']}, точно {q['exact_ads']}/{q['n']})" if "micro_f1" in q else ""))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("->", out)
