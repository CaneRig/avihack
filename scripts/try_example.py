"""Быстрая проверка черновиков на примере из постановки (шаг 2): fast и full без слоя 1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avito_drafts.config import load_config  # noqa: E402
from avito_drafts.drafts.pipeline import Drafter  # noqa: E402

EX: str = ("Делаем ремонт квартир под ключ, а также отдельно выполняем сантехнические работы: разводку труб, установку раковин и унитазов, "
           "замену смесителей и подключение стиральных машин. Помогаем подобрать материалы и согласовываем стоимость до начала работ.")

if __name__ == "__main__":
    d = Drafter(load_config())
    for llm in (False, True):
        print("LLM" if llm else "шаблон", json.dumps(d.make(EX, "", [104], use_llm=llm), ensure_ascii=False, indent=1))
