"""Загрузка конфигурации и пути проекта."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT: Path = Path(__file__).resolve().parents[2]
CATS: list[int] = [102, 103, 104, 105, 106]
CAT_NAMES: dict[int, str] = {
    102: "Штукатурные работы",
    103: "Плиточные работы",
    104: "Сантехника",
    105: "Электрика",
    106: "Поклейка обоев и малярные работы",
}
NAME_TO_ID: dict[str, int] = {v: k for k, v in CAT_NAMES.items()}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Читает YAML-конфиг (по умолчанию configs/default.yaml) в словарь."""
    p = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict[str, Any]) -> str:
    """Короткий хеш конфигурации для записи в журнал запусков."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]


def path(cfg: dict[str, Any], key: str) -> Path:
    """Возвращает абсолютный путь из раздела paths конфига."""
    return ROOT / cfg["paths"][key]
