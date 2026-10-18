"""Metrics, threshold sweeps and expected-profit decision logic."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class Economics:
    """Campaign economics for one contact attempt.

    contact_cost:     cost of contacting one customer (call, mail, agent time)
    conversion_value: net value gained when a contacted customer subscribes

    Expected value of contacting a customer with response probability p:
        EV = p * conversion_value - contact_cost
    so, with calibrated probabilities, contact when p > contact_cost / conversion_value.
    """

    contact_cost: float
    conversion_value: float

    def __post_init__(self) -> None:
        if self.contact_cost < 0 or self.conversion_value <= 0:
            raise ValueError("contact_cost must be >= 0 and conversion_value > 0")

    @property
    def break_even_probability(self) -> float:
        return self.contact_cost / self.conversion_value

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def expected_value(proba, economics: Economics) -> np.ndarray:
    return np.asarray(proba, dtype=float) * economics.conversion_value - economics.contact_cost


def realized_profit(y_true, contacted, economics: Economics) -> float:
    """Profit actually obtained on labelled data for a given contact decision."""
    y = np.asarray(y_true).astype(int)
    flag = np.asarray(contacted).astype(bool)
    conversions = int((y[flag] == 1).sum())
    return float(conversions * economics.conversion_value - flag.sum() * economics.contact_cost)


def classification_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = proba >= threshold
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(y),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def threshold_sweep(
    y_true,
    proba,
    economics: Economics | None = None,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Precision / recall / F1 (and profit if economics given) across thresholds."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 0.991, 0.005), 3)
    y = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    rows = []
    for t in thresholds:
        pred = proba >= t
        tp = int(((y == 1) & pred).sum())
        fp = int(((y == 0) & pred).sum())
        fn = int(((y == 1) & ~pred).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = {"threshold": float(t), "precision": precision, "recall": recall,
               "f1": f1, "contacted": int(pred.sum())}
        if economics is not None:
            row["profit"] = tp * economics.conversion_value - (tp + fp) * economics.contact_cost
        rows.append(row)
    return pd.DataFrame(rows)


def best_threshold(sweep: pd.DataFrame, objective: str = "f1") -> tuple[float, float]:
    """(threshold, objective value) maximizing 'f1' or 'profit' over a sweep."""
    if objective not in {"f1", "profit"}:
        raise ValueError("objective must be 'f1' or 'profit'")
    if objective not in sweep.columns:
        raise ValueError(f"sweep has no '{objective}' column (pass economics for 'profit')")
    best = sweep.loc[sweep[objective].idxmax()]
    return float(best["threshold"]), float(best[objective])


def campaign_summary(y_true, proba, economics: Economics, threshold: float) -> dict[str, float]:
    """Profit of the model rule vs. contact-everyone / contact-nobody baselines."""
    y = np.asarray(y_true).astype(int)
    flag = np.asarray(proba) >= threshold
    model = realized_profit(y, flag, economics)
    everyone = realized_profit(y, np.ones(len(y), dtype=bool), economics)
    return {
        "threshold": float(threshold),
        "contacted": int(flag.sum()),
        "contacted_share": float(flag.mean()),
        "model_profit": model,
        "contact_everyone_profit": everyone,
        "contact_nobody_profit": 0.0,
        "uplift_vs_everyone": model - everyone,
    }
