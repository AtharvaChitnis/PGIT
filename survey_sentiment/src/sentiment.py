"""Sentiment feature extraction for survey auto-fill."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """Wraps VADER and returns tabular sentiment features."""

    FEATURE_COLUMNS = ("compound", "pos", "neu", "neg", "sentiment_label")

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def score_text(self, text: str) -> dict[str, float | str]:
        scores = self._analyzer.polarity_scores(text or "")
        compound = float(scores["compound"])
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return {
            "compound": compound,
            "pos": float(scores["pos"]),
            "neu": float(scores["neu"]),
            "neg": float(scores["neg"]),
            "sentiment_label": label,
        }

    def transform(self, texts: Iterable[str]) -> pd.DataFrame:
        rows = [self.score_text(str(text)) for text in texts]
        return pd.DataFrame(rows)

    def feature_matrix(self, texts: Iterable[str]) -> np.ndarray:
        """Numeric features only (for sklearn models)."""
        frame = self.transform(texts)
        return frame[["compound", "pos", "neu", "neg"]].to_numpy(dtype=float)


class VaderFeatureTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer that extracts VADER scores from text."""

    def __init__(self) -> None:
        self._analyzer: SentimentAnalyzer | None = None

    def fit(self, X, y=None):  # noqa: N803, ANN001
        self._analyzer = SentimentAnalyzer()
        return self

    def transform(self, X):  # noqa: N803, ANN001
        if self._analyzer is None:
            self._analyzer = SentimentAnalyzer()
        texts = [str(t) for t in np.asarray(X).ravel()]
        return self._analyzer.feature_matrix(texts)
