"""Обучение слоя 1 на уникальных текстах трейна и сохранение в artifacts/tfidf/ (время обучения печатается)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_drafts.config import CATS, load_config, path  # noqa: E402
from avito_drafts.data import load  # noqa: E402
from avito_drafts.layer1 import TfidfLR  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    s = load("silver", cfg)
    if cfg["train"]["exclude_public_test"]:
        s = s[~s.in_public_test]
    t0 = time.time()
    m = TfidfLR.from_cfg(cfg).fit(s.text.tolist(), s[[f"y{c}" for c in CATS]].to_numpy())
    m.save(path(cfg, "tfidf_dir"))
    print(f"слой 1: {len(s)} текстов, {time.time() - t0:.0f} с -> {path(cfg, 'tfidf_dir')}")
