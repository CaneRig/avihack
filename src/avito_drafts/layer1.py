"""Слой 1: TF-IDF (символы внутри слов + слова) и пять логистических регрессий."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import copy

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from .config import CATS
from .text import for_model


class TfidfLR:
    """Пять независимых бинарных логистических регрессий над общими TF-IDF-признаками."""

    def __init__(self, C: float = 10.0, char_ngram: tuple[int, int] = (2, 5), word_ngram: tuple[int, int] = (1, 2),
                 min_df: int = 2, max_iter: int = 1000) -> None:
        self.C: float = C
        self.max_iter: int = max_iter
        self.vec: FeatureUnion = FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(char_ngram), sublinear_tf=True, min_df=min_df, dtype=np.float32)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=tuple(word_ngram), sublinear_tf=True, min_df=min_df, dtype=np.float32)),
        ])
        self.models: list[LogisticRegression] = []

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any], **over: Any) -> "TfidfLR":
        """Создаёт модель из раздела layer1 конфига; over перекрывает значения (для перебора)."""
        c = {**cfg["layer1"], **over}
        return cls(c["C"], c["char_ngram"], c["word_ngram"], c["min_df"], c["max_iter"])

    def fit(self, texts: list[str], Y: np.ndarray) -> "TfidfLR":
        """Обучает векторизатор и по одной регрессии на категорию; Y — матрица (n,5)."""
        X = self.vec.fit_transform([for_model(t) for t in texts])
        self.models = [LogisticRegression(C=self.C, max_iter=self.max_iter).fit(X, Y[:, j]) for j in range(len(CATS))]
        return self

    @classmethod
    def fit_many(cls, cfg: dict[str, Any], texts: list[str], Y: np.ndarray, Cs: list[float], **over: Any) -> dict[float, "TfidfLR"]:
        """Обучает модели для нескольких C, один раз построив TF-IDF (для перебора)."""
        base = cls.from_cfg(cfg, **over)
        X = base.vec.fit_transform([for_model(t) for t in texts])
        out: dict[float, TfidfLR] = {}
        for C in Cs:
            m = copy.copy(base)
            m.C = C
            m.models = [LogisticRegression(C=C, max_iter=base.max_iter).fit(X, Y[:, j]) for j in range(len(CATS))]
            out[C] = m
        return out

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Вероятности категорий, матрица (n,5)."""
        X = self.vec.transform([for_model(t) for t in texts])
        return np.stack([m.predict_proba(X)[:, 1] for m in self.models], 1)

    def save(self, d: Path) -> None:
        """Сохраняет модель в d/model.joblib."""
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, d / "model.joblib", compress=3)

    @staticmethod
    def load(d: Path) -> "TfidfLR":
        """Загружает модель из d/model.joblib."""
        return joblib.load(d / "model.joblib")
