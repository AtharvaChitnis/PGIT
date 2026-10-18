"""Model zoo, search spaces and pipeline assembly."""

from __future__ import annotations

import importlib.util

import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import build_preprocessor, build_smotenc_steps, infer_feature_types

IMBALANCE_STRATEGIES = ("none", "class_weight", "smotenc")


def available_models() -> list[str]:
    names = ["logreg", "random_forest", "hist_gb"]
    if importlib.util.find_spec("xgboost") is not None:
        names.append("xgboost")
    return names


def build_estimator(
    name: str,
    imbalance: str = "none",
    random_state: int = config.RANDOM_STATE,
    pos_weight: float | None = None,
):
    weighted = imbalance == "class_weight"
    if name == "logreg":
        return LogisticRegression(
            max_iter=2000, class_weight="balanced" if weighted else None, random_state=random_state
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample" if weighted else None,
            n_jobs=1,  # parallelism is handled by the outer search
            random_state=random_state,
        )
    if name == "hist_gb":
        return HistGradientBoostingClassifier(
            class_weight="balanced" if weighted else None, random_state=random_state
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=random_state,
            scale_pos_weight=(pos_weight or 1.0) if weighted else 1.0,
        )
    raise ValueError(f"Unknown model '{name}'. Available: {available_models()}")


def param_distributions(name: str) -> dict:
    """Random-search spaces (keys are Pipeline parameter names)."""
    if name == "logreg":
        return {"model__C": loguniform(1e-3, 1e2)}
    if name == "random_forest":
        return {
            "model__n_estimators": randint(200, 600),
            "model__max_depth": randint(4, 20),
            "model__min_samples_leaf": randint(1, 30),
            "model__max_features": ["sqrt", 0.3, 0.5],
        }
    if name == "hist_gb":
        return {
            "model__learning_rate": loguniform(0.02, 0.3),
            "model__max_iter": randint(100, 500),
            "model__max_leaf_nodes": randint(8, 64),
            "model__min_samples_leaf": randint(10, 200),
            "model__l2_regularization": loguniform(1e-3, 10),
        }
    if name == "xgboost":
        return {
            "model__n_estimators": randint(100, 500),
            "model__learning_rate": loguniform(0.02, 0.3),
            "model__max_depth": randint(2, 9),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_weight": randint(1, 10),
            "model__reg_lambda": loguniform(1e-2, 10),
        }
    raise ValueError(f"Unknown model '{name}'")


def build_pipeline(
    name: str,
    X: pd.DataFrame,
    imbalance: str = "none",
    text_column: str | None = None,
    random_state: int = config.RANDOM_STATE,
    pos_weight: float | None = None,
) -> Pipeline:
    """Preprocessing + estimator. Final step is always called ``model``."""
    if imbalance not in IMBALANCE_STRATEGIES:
        raise ValueError(f"imbalance must be one of {IMBALANCE_STRATEGIES}")
    numeric, categorical = infer_feature_types(X, text_column)
    estimator = build_estimator(name, imbalance, random_state, pos_weight)

    if imbalance == "smotenc":
        if text_column:
            raise NotImplementedError("SMOTENC with a text column is not supported")
        try:
            from imblearn.pipeline import Pipeline as ImbPipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError("imbalance='smotenc' needs: pip install imbalanced-learn") from exc
        steps = build_smotenc_steps(numeric, categorical, random_state)
        return ImbPipeline([*steps, ("model", estimator)])

    pre = build_preprocessor(numeric, categorical, text_column)
    return Pipeline([("preprocessor", pre), ("model", estimator)])
