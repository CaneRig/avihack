"""Демо-сервер: один HTML, JSON API, фоновая обработка (одна задача одновременно).

Запуск: .venv\\Scripts\\python app\\server.py   (адрес и порт берутся из configs/default.yaml, раздел demo)
"""
from __future__ import annotations

import copy
import io
import json
import re
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402
import pandas as pd  # noqa: E402

from avito_drafts.config import ROOT as PROJECT_ROOT, load_config  # noqa: E402
from avito_drafts.predict import Predictor  # noqa: E402
from avito_drafts.submission import process_row, row_problems, write_submission_csv  # noqa: E402

SUBMISSIONS = APP / "submissions"
MAX_BODY = 64 * 1024 * 1024
ID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}(_[0-9]+)?$")

CFG = load_config()
LOCK = threading.Lock()
JOB: dict[str, Any] = {"status": "idle"}  # idle | running | done | error
PREDICTORS: dict[str, Predictor] = {}


def get_predictor(judge_mode: str, drafts: str) -> Predictor:
    """Predictor на каждый режим судьи создаётся один раз (режим судьи фиксируется при создании судьи)."""
    if judge_mode not in PREDICTORS:
        cfg = copy.deepcopy(CFG)
        cfg["judge"]["mode"] = judge_mode
        cfg["drafts"]["prompt_variant"] = "c"
        PREDICTORS[judge_mode] = Predictor(cfg, mode="classify")
    return PREDICTORS[judge_mode]


def health() -> dict[str, Any]:
    """Состояние: отвечает ли Ollama, есть ли модель, есть ли модель слоя 1."""
    base, model = CFG["llm"]["base_url"], CFG["llm"]["model"]
    res: dict[str, Any] = {"ollama": False, "model_present": False, "model": model, "tfidf": (PROJECT_ROOT / CFG["paths"]["tfidf_dir"] / "model.joblib").exists(), "message": ""}
    try:
        r = httpx.get(base + "/api/tags", timeout=3)
        r.raise_for_status()
        names = {m.get("name") for m in r.json().get("models", [])} | {m.get("model") for m in r.json().get("models", [])}
        res["ollama"] = True
        res["model_present"] = model in names or (model + ":latest") in names
        if not res["model_present"]:
            res["message"] = f"Ollama работает, но модели {model} нет в списке (ollama list)."
    except Exception:  # noqa: BLE001
        res["message"] = f"Ollama не отвечает по адресу {base}. Категории будут искаться по ключевым корням без LLM."
    if not res["tfidf"]:
        res["message"] += " Нет artifacts/tfidf/model.joblib: обучите модель (prepare_data.py и train.py)."
    return res


def new_submission_dir() -> tuple[str, Path]:
    SUBMISSIONS.mkdir(exist_ok=True)
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    sid, n = base, 1
    while (SUBMISSIONS / sid).exists():
        n += 1
        sid = f"{base}_{n}"
    d = SUBMISSIONS / sid
    d.mkdir()
    return sid, d


def run_job(kind: str, name: str, rows: list[tuple[str, str]], params: dict[str, Any]) -> None:
    """Фоновая обработка: строки по очереди, ошибка строки не останавливает файл."""
    t0 = time.time()
    try:
        h = health()
        warning = h["message"] if (not h["ollama"] or not h["model_present"]) else ""
        pr = get_predictor(params["judge"], params["drafts"])
        pr.mode = "full" if params["drafts"] == "llm" else "classify"
        results: list[dict[str, Any]] = []
        for tc, desc in rows:
            try:
                r = process_row(pr, tc, desc, params["drafts"])
            except Exception as e:  # noqa: BLE001
                r = {"testcase": tc, "drafts": [], "drafts_by": params["drafts"], "source": "error", "error": repr(e), "judge": {}}
            r["description"] = desc
            r["problems"] = row_problems(r)
            results.append(r)
            with LOCK:
                JOB["done"] = len(results)
        dur = time.time() - t0
        sid, d = new_submission_dir()
        meta = {"id": sid, "name": name, "kind": kind, "created": datetime.now().isoformat(timespec="seconds"), "params": params,
                "rows": len(rows), "duration": round(dur, 1), "warning": warning,
                "errors": sum(1 for r in results if row_problems(r)), "with_categories": sum(1 for r in results if r["drafts"])}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(d / "results.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        done = {r["testcase"]: r for r in results}
        with open(d / "out.csv", "w", encoding="utf-8", newline="") as f:
            write_submission_csv(f, [r["testcase"] for r in results], done, params["with_101"])
        with LOCK:
            JOB.update(status="done", submission=sid, warning=warning, finished=time.time())
    except Exception as e:  # noqa: BLE001
        with LOCK:
            JOB.update(status="error", error=f"{type(e).__name__}: {e}", finished=time.time())


def start_job(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    params = {"judge": body.get("judge", "panel"), "drafts": body.get("drafts", "template"), "with_101": bool(body.get("with_101", True))}
    if params["judge"] not in ("single", "panel") or params["drafts"] not in ("template", "llm"):
        return 400, {"error": "Неверные параметры режима."}
    if body.get("text") is not None:
        text = str(body["text"]).strip()
        if not text:
            return 400, {"error": "Текст пустой."}
        kind, name, rows = "text", "Один текст", [("text-1", text)]
    else:
        try:
            df = pd.read_csv(io.StringIO(str(body.get("csv", "")).lstrip("﻿")), dtype=str, keep_default_na=False)
        except Exception as e:  # noqa: BLE001
            return 400, {"error": f"Не удалось прочитать CSV: {e}"}
        if not {"testcase", "description"} <= set(df.columns):
            return 400, {"error": "В файле должны быть столбцы testcase и description."}
        if df.empty:
            return 400, {"error": "В файле нет строк."}
        dup = df.testcase[df.testcase.duplicated()].unique().tolist()
        if dup:
            return 400, {"error": f"Повторяющиеся testcase: {', '.join(dup[:5])}. Идентификаторы должны быть уникальны."}
        kind, name, rows = "file", str(body.get("name") or "файл.csv"), list(zip(df.testcase, df.description))
    with LOCK:
        if JOB.get("status") == "running":
            return 409, {"error": "Уже идёт другая обработка. Дождитесь её завершения."}
        JOB.clear()
        JOB.update(status="running", kind=kind, name=name, total=len(rows), done=0, started=time.time(), params=params)
    threading.Thread(target=run_job, args=(kind, name, rows, params), daemon=True).start()
    return 200, {"ok": True}


def status() -> dict[str, Any]:
    with LOCK:
        j = dict(JOB)
    now = time.time()
    if j.get("started"):
        j["elapsed"] = round(j.get("finished", now) - j["started"], 1)
    return j


def list_submissions() -> list[dict[str, Any]]:
    out = []
    for d in sorted(SUBMISSIONS.glob("*"), reverse=True) if SUBMISSIONS.exists() else []:
        try:
            out.append(json.loads((d / "meta.json").read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return out


def load_submission(sid: str) -> dict[str, Any] | None:
    d = SUBMISSIONS / sid
    if not ID_RE.match(sid) or not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (d / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"meta": meta, "rows": rows, "cat_names": CAT_NAMES_RU}


CAT_NAMES_RU = {101: "Ремонт (исходная категория)", 102: "Штукатурные работы", 103: "Плитка", 104: "Сантехника", 105: "Электрика", 106: "Малярные работы и обои"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:  # тише в консоли
        if "/api/status" not in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj: Any, code: int = 200) -> None:
        self.send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            self.send(200, (APP / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif p == "/api/health":
            self.json(health())
        elif p == "/api/status":
            self.json(status())
        elif p == "/api/submissions":
            self.json(list_submissions())
        elif m := re.fullmatch(r"/api/submissions/([^/]+)(/csv)?", p):
            sub = load_submission(m.group(1))
            if sub is None:
                self.json({"error": "Отправка не найдена."}, 404)
            elif m.group(2):
                data = (SUBMISSIONS / m.group(1) / "out.csv").read_bytes()
                self.send(200, data, "text/csv; charset=utf-8", {"Content-Disposition": f'attachment; filename="submission_{m.group(1)}.csv"'})
            else:
                self.json(sub)
        else:
            self.json({"error": "Не найдено."}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/run":
            return self.json({"error": "Не найдено."}, 404)
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return self.json({"error": "Файл слишком большой."}, 413)
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:  # noqa: BLE001
            return self.json({"error": "Неверный запрос."}, 400)
        code, resp = start_job(body)
        self.json(resp, code)


if __name__ == "__main__":
    demo = CFG.get("demo", {})
    host, port = demo.get("host", "127.0.0.1"), int(demo.get("port", 7860))
    try:
        srv = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        sys.exit(f"Не удалось занять {host}:{port} ({e}). Возможно, порт уже занят другой программой или прежним запуском сервера.")
    print(f"Демо: http://{host}:{port}  (остановка: Ctrl+C)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Остановлено.")
