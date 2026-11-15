"""Проверки: метрика на примере организаторов, нормализация."""
import numpy as np

from avito_drafts.evalkit import SliceData, evaluate
from avito_drafts.text import make_key, normalize


def test_metric_example() -> None:
    """Эталон {102,104}, предсказание {102,105}: TP=1, FP=1, FN=1, F1=0.5."""
    sl = SliceData("x", ["a"], np.array([[1, 0, 1, 0, 0]]), np.ones((1, 5), int), np.array(["a"]), None)
    r = evaluate(sl, {"a": [102, 105]})
    assert (r["fp"], r["fn"], r["micro_f1"]) == (1, 1, 0.5)


def test_homoglyphs() -> None:
    """Смешанное слово чинится, чисто латинское не трогается."""
    assert normalize("закупaeм") == "закупаем" and normalize("shop m-9245") == "shop m-9245"
    assert make_key("А  б,в!") == make_key("аБв")


if __name__ == "__main__":
    test_metric_example()
    test_homoglyphs()
    print("ОК")
