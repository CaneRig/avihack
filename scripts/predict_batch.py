"""Пакетное предсказание: CSV/Parquet/JSONL -> JSONL {"id","response"} и файл для бота {"id","mcIds"}."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_drafts.predict import Predictor  # noqa: E402
from avito_drafts.text import make_key  # noqa: E402


def read_any(p: str) -> pd.DataFrame:
    """Читает таблицу по расширению файла."""
    s = Path(p).suffix.lower()
    return pd.read_csv(p) if s == ".csv" else pd.read_parquet(p) if s == ".parquet" else pd.read_json(p, lines=True)


def write_bot_file(path: str, ids: list[object], answers: list[dict]) -> None:
    """Файл для бота: JSONL со строками {"id", "mcIds"}. Формат не подтверждён (вопрос 5), правится здесь."""
    with open(path, "w", encoding="utf-8") as f:
        for i, a in zip(ids, answers):
            f.write(json.dumps({"id": i, "mcIds": [d["mcId"] for d in a["drafts"]]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output", help="JSONL с ответами")
    ap.add_argument("--bot", help="файл для бота (по умолчанию <output>.bot.jsonl)")
    ap.add_argument("--mode", default=None, choices=["classify", "fast", "full"])
    ap.add_argument("--no-title", action="store_true", help="скрыть заголовок")
    a = ap.parse_args()
    df = read_any(a.input)
    if "id" not in df:
        df["id"] = range(len(df))
    if a.no_title or "title" not in df:
        df["title"] = ""
    if "mcId" not in df:
        df["mcId"] = 101
    df["_k"] = [make_key(d) + "|" + make_key(t) + f"|{m}" for d, t, m in zip(df.description, df.title, df.mcId)]
    uniq = df.drop_duplicates("_k")
    pr = Predictor(mode=a.mode)
    t0 = time.time()
    res = pr.predict_batch(uniq[["description", "title", "mcId"]].to_dict("records"))
    by_key = {k: r for k, r in zip(uniq._k, res)}
    answers = [by_key[k][0] for k in df._k]
    with open(a.output, "w", encoding="utf-8") as f:
        for i, ans in zip(df.id, answers):
            f.write(json.dumps({"id": i.item() if hasattr(i, "item") else i, "response": ans}, ensure_ascii=False) + "\n")
    write_bot_file(a.bot or a.output + ".bot.jsonl", [i.item() if hasattr(i, "item") else i for i in df.id], answers)
    errs = sum("error" in r[1] for r in res)
    print(f"строк {len(df)}, уникальных {len(uniq)}, ошибок {errs}, {time.time() - t0:.1f} с -> {a.output}")
