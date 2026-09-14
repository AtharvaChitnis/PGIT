"""Train models that map free-text sentiment to survey answers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .sentiment import VaderFeatureTransformer

# Survey fields the model learns to fill from free-text comments.
SURVEY_TARGETS = [
    "overall_satisfaction",
    "recommend",
    "product_quality",
    "repurchase_intent",
    "support_rating",
]


class TextFeatureBuilder(BaseEstimator, TransformerMixin):
    """Combine VADER sentiment scores with TF-IDF text features."""

    def __init__(self, max_features: int = 250) -> None:
        self.max_features = max_features
        self.vader_ = VaderFeatureTransformer()
        self.tfidf_ = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )

    def fit(self, X, y=None):  # noqa: N803, ANN001
        texts = [str(t) for t in np.asarray(X).ravel()]
        self.vader_.fit(texts, y)
        self.tfidf_.fit(texts)
        return self

    def transform(self, X):  # noqa: N803, ANN001
        texts = [str(t) for t in np.asarray(X).ravel()]
        sentiment = self.vader_.transform(texts)
        tfidf = self.tfidf_.transform(texts)
        return sparse.hstack([sparse.csr_matrix(sentiment), tfidf], format="csr")


def load_training_data(csv_path: str | Path) -> tuple[pd.Series, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    missing = [c for c in ["comment", *SURVEY_TARGETS] if c not in df.columns]
    if missing:
        raise ValueError(f"Training CSV missing columns: {missing}")
    return df["comment"], df[SURVEY_TARGETS].copy()


def encode_targets(
    y: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, LabelEncoder]]:
    encoders: dict[str, LabelEncoder] = {}
    encoded_cols = []
    for col in SURVEY_TARGETS:
        enc = LabelEncoder()
        encoded_cols.append(enc.fit_transform(y[col].astype(str)))
        encoders[col] = enc
    return np.column_stack(encoded_cols), encoders


def build_model(classifier: str = "logistic") -> Pipeline:
    if classifier == "forest":
        base = RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced_subsample",
        )
    else:
        base = LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=42,
        )

    return Pipeline(
        steps=[
            ("features", TextFeatureBuilder()),
            ("model", MultiOutputClassifier(base)),
        ]
    )


def train(
    csv_path: str | Path,
    classifier: str = "logistic",
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict[str, Any]:
    comments, targets = load_training_data(csv_path)
    y_encoded, encoders = encode_targets(targets)

    X_train, X_test, y_train, y_test = train_test_split(
        comments.to_numpy(),
        y_encoded,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = build_model(classifier=classifier)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    per_target: dict[str, dict[str, float]] = {}
    reports: dict[str, str] = {}
    for i, name in enumerate(SURVEY_TARGETS):
        acc = accuracy_score(y_test[:, i], y_pred[:, i])
        f1 = f1_score(y_test[:, i], y_pred[:, i], average="weighted", zero_division=0)
        per_target[name] = {"accuracy": float(acc), "f1_weighted": float(f1)}
        labels = encoders[name].classes_
        reports[name] = classification_report(
            y_test[:, i],
            y_pred[:, i],
            labels=list(range(len(labels))),
            target_names=[str(c) for c in labels],
            zero_division=0,
        )

    return {
        "pipeline": pipeline,
        "encoders": encoders,
        "metrics": per_target,
        "reports": reports,
        "targets": SURVEY_TARGETS,
    }


def save_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": bundle["pipeline"],
            "encoders": bundle["encoders"],
            "targets": bundle["targets"],
            "metrics": bundle["metrics"],
        },
        path,
    )
    return path


def load_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)
