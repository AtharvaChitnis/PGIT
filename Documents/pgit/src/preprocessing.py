"""Feature typing and preprocessing pipelines."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def infer_feature_types(
    X: pd.DataFrame, text_column: str | None = None
) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) column names.

    Anything non-numeric is categorical, which also covers pandas' newer ``str`` dtype
    (``select_dtypes("object")`` misses it).
    """
    cols = [c for c in X.columns if c != text_column]
    numeric = [
        c for c in cols
        if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])
    ]
    categorical = [c for c in cols if c not in numeric]
    return numeric, categorical


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    text_column: str | None = None,
) -> ColumnTransformer:
    """Scale numerics, one-hot categoricals, optionally add sentiment features from text."""
    transformers = [
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ]
    if text_column:
        from .feedback import SentimentFeatures  # local import: vaderSentiment is optional

        transformers.append(("sentiment", SentimentFeatures(), [text_column]))
    return ColumnTransformer(transformers, remainder="drop")


def build_smotenc_steps(numeric: list[str], categorical: list[str], random_state: int):
    """Steps for an imbalanced-learn pipeline: ordinal-encode -> SMOTENC -> scale/one-hot.

    SMOTENC needs categoricals as integer codes *before* resampling, so the usual
    one-hot step has to come after it. Requires ``imbalanced-learn``.
    """
    try:
        from imblearn.over_sampling import SMOTENC
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError("imbalance='smotenc' needs: pip install imbalanced-learn") from exc

    if not categorical:
        raise ValueError("SMOTENC requires at least one categorical feature")

    encode = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
            ("num", "passthrough", numeric),
        ]
    )
    cat_idx = list(range(len(categorical)))
    num_idx = list(range(len(categorical), len(categorical) + len(numeric)))
    smote = SMOTENC(categorical_features=cat_idx, random_state=random_state)
    post = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_idx),
            ("num", StandardScaler(), num_idx),
        ]
    )
    return [("encode", encode), ("smote", smote), ("post", post)]
