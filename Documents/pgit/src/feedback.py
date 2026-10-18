"""Bridge between the campaign model and the survey_sentiment module.

Two uses of free-text customer feedback:

1. ``SentimentFeatures``: an sklearn transformer that turns a comment column into numeric
   features (VADER compound/pos/neu/neg + has_comment). Add a ``comment`` column to the
   training data and the campaign pipeline picks it up automatically.
2. ``analyze_comment``: survey auto-fill (satisfaction, recommend, ...) for display in the app.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config

SENTIMENT_COLUMNS = ["compound", "pos", "neu", "neg", "has_comment"]


@lru_cache(maxsize=1)
def _default_analyzer():
    """VADER wrapper from the survey_sentiment package (imported lazily; optional dependency)."""
    from survey_sentiment.src.sentiment import SentimentAnalyzer

    return SentimentAnalyzer()


class SentimentFeatures(BaseEstimator, TransformerMixin):
    """Comment text -> [compound, pos, neu, neg, has_comment].

    Empty or missing comments get zeros and ``has_comment = 0`` so "no comment" is not
    confused with "neutral comment".
    """

    def fit(self, X, y=None):  # noqa: N803, ANN001
        return self

    def transform(self, X):  # noqa: N803, ANN001
        analyzer = _default_analyzer()
        texts = pd.Series(np.asarray(X).ravel()).fillna("").astype(str)
        out = np.zeros((len(texts), len(SENTIMENT_COLUMNS)), dtype=float)
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            s = analyzer.score_text(text)
            out[i] = [s["compound"], s["pos"], s["neu"], s["neg"], 1.0]
        return out

    def get_feature_names_out(self, input_features=None):  # noqa: ANN001
        return np.array(SENTIMENT_COLUMNS, dtype=object)


def load_survey_filler(model_path: str | Path = config.SURVEY_MODEL_PATH):
    """Return a trained SurveyFiller, or None if the model file / deps are unavailable."""
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        from survey_sentiment.src.survey_filler import SurveyFiller

        return SurveyFiller(path)
    except Exception:  # noqa: BLE001 - any import/unpickle problem -> fall back to rules
        return None


def analyze_comment(comment: str, filler: Any | None = None) -> dict[str, Any]:
    """Sentiment scores + auto-filled survey answers for one comment.

    Uses the trained model when ``filler`` is given, otherwise the rule-based mapping
    from VADER compound score.
    """
    if filler is not None:
        result = filler.fill(comment)
        result["method"] = "model"
        return result

    from survey_sentiment.src.survey_filler import SurveyFiller

    stub = SimpleNamespace(sentiment=_default_analyzer())
    return SurveyFiller.rule_based_fill(stub, comment)  # type: ignore[arg-type]
