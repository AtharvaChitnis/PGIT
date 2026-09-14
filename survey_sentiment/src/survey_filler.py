"""Fill structured survey answers from free-text comments via sentiment ML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .sentiment import SentimentAnalyzer
from .train import SURVEY_TARGETS, load_bundle


class SurveyFiller:
    """Predict Likert / yes-no survey fields from a free-text comment."""

    def __init__(self, model_path: str | Path) -> None:
        bundle = load_bundle(model_path)
        self.pipeline = bundle["pipeline"]
        self.encoders = bundle["encoders"]
        self.targets = bundle.get("targets", SURVEY_TARGETS)
        self.metrics = bundle.get("metrics", {})
        self.sentiment = SentimentAnalyzer()

    def fill(self, comment: str) -> dict[str, Any]:
        """Return sentiment scores plus predicted survey answers."""
        sentiment = self.sentiment.score_text(comment)
        encoded = self.pipeline.predict([comment])[0]
        answers: dict[str, Any] = {}
        for i, name in enumerate(self.targets):
            answers[name] = self.encoders[name].inverse_transform([encoded[i]])[0]
            # Cast numeric Likert labels back to int when possible
            try:
                answers[name] = int(answers[name])
            except (TypeError, ValueError):
                pass

        return {
            "comment": comment,
            "sentiment": sentiment,
            "survey_answers": answers,
        }

    def fill_many(self, comments: list[str]) -> pd.DataFrame:
        rows = []
        for comment in comments:
            result = self.fill(comment)
            row = {
                "comment": result["comment"],
                "sentiment_label": result["sentiment"]["sentiment_label"],
                "compound": result["sentiment"]["compound"],
                **result["survey_answers"],
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def rule_based_fill(self, comment: str) -> dict[str, Any]:
        """Simple fallback: map VADER compound score to survey scales."""
        sentiment = self.sentiment.score_text(comment)
        compound = float(sentiment["compound"])

        if compound >= 0.6:
            satisfaction = quality = repurchase = support = 5
            recommend = "yes"
        elif compound >= 0.2:
            satisfaction = quality = repurchase = support = 4
            recommend = "yes"
        elif compound > -0.2:
            satisfaction = quality = repurchase = support = 3
            recommend = "yes"
        elif compound > -0.6:
            satisfaction = quality = repurchase = support = 2
            recommend = "no"
        else:
            satisfaction = quality = repurchase = support = 1
            recommend = "no"

        return {
            "comment": comment,
            "sentiment": sentiment,
            "survey_answers": {
                "overall_satisfaction": satisfaction,
                "recommend": recommend,
                "product_quality": quality,
                "repurchase_intent": repurchase,
                "support_rating": support,
            },
            "method": "rule_based",
        }
