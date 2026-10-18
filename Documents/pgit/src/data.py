"""Loading, splitting and (for smoke tests) synthesizing the Bank Marketing data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_bank_data(path: str | Path = config.DATA_PATH) -> pd.DataFrame:
    """Load bank-full.csv (semicolon separated in the Kaggle/UCI release)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download bank-full.csv from Kaggle/UCI into data/raw/, "
            "or run with --synthetic for a smoke test."
        )
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:  # comma separated variant
        df = pd.read_csv(path, sep=",")
    return df


def split_xy(
    df: pd.DataFrame,
    target: str = config.TARGET,
    drop: tuple[str, ...] = config.LEAKAGE_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Features/target with leakage columns removed. Target becomes 0/1 ints."""
    y = (df[target].astype(str).str.lower() == config.POSITIVE_LABEL).astype(int)
    X = df.drop(columns=[target, *[c for c in drop if c in df.columns]])
    return X, y


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    @property
    def X_dev(self) -> pd.DataFrame:
        """Train + validation: everything except the locked test set."""
        return pd.concat([self.X_train, self.X_val])

    @property
    def y_dev(self) -> pd.Series:
        return pd.concat([self.y_train, self.y_val])


def three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = config.VAL_SIZE,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Splits:
    """Stratified train / validation / test split (default 60 / 20 / 20)."""
    if not 0 < val_size + test_size < 1:
        raise ValueError("val_size + test_size must be between 0 and 1")
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=val_size / (1 - test_size),
        stratify=y_rest,
        random_state=random_state,
    )
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)


def make_synthetic_bank(n: int = 6000, seed: int = 0, with_comments: bool = False) -> pd.DataFrame:
    """Bank-Marketing-shaped synthetic data (same columns, imbalanced, learnable signal).

    Only for smoke tests / demos when the real CSV is not available. Do not report
    metrics from this data.
    """
    rng = np.random.default_rng(seed)
    jobs = ["admin.", "blue-collar", "technician", "services", "management", "retired",
            "student", "self-employed", "unemployed", "entrepreneur", "housemaid", "unknown"]
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 90, n),
            "job": rng.choice(jobs, n),
            "marital": rng.choice(["married", "single", "divorced"], n, p=[0.6, 0.28, 0.12]),
            "education": rng.choice(["primary", "secondary", "tertiary", "unknown"], n),
            "default": rng.choice(["no", "yes"], n, p=[0.98, 0.02]),
            "balance": rng.normal(1300, 3000, n).round().astype(int),
            "housing": rng.choice(["yes", "no"], n),
            "loan": rng.choice(["yes", "no"], n, p=[0.16, 0.84]),
            "contact": rng.choice(["cellular", "telephone", "unknown"], n, p=[0.65, 0.07, 0.28]),
            "day": rng.integers(1, 29, n),
            "month": rng.choice(months, n),
            "duration": rng.integers(5, 1500, n),
            "campaign": rng.integers(1, 20, n),
            "pdays": np.where(rng.random(n) < 0.82, -1, rng.integers(1, 500, n)),
            "previous": rng.integers(0, 6, n),
            "poutcome": rng.choice(["unknown", "failure", "success", "other"], n,
                                   p=[0.82, 0.11, 0.03, 0.04]),
        }
    )
    logit = (
        -4.0
        + 2.2 * (df["poutcome"] == "success")
        + 0.9 * df["month"].isin(["mar", "sep", "oct", "dec"])
        + 0.5 * (df["job"].isin(["student", "retired"]))
        - 0.6 * (df["housing"] == "yes")
        - 0.5 * (df["contact"] == "unknown")
        + 0.0002 * df["balance"]
        - 0.04 * df["campaign"]
        + 0.002 * df["duration"]  # leaky signal, dropped by split_xy
    )
    p = 1 / (1 + np.exp(-logit))
    df["y"] = np.where(rng.random(n) < p, "yes", "no")

    if with_comments:
        pos = ["great service, very happy", "love this bank, helpful staff", "excellent offer"]
        neg = ["terrible experience, very disappointed", "awful support, rude staff", "bad fees"]
        neu = ["it is okay", "no strong opinion", "average"]
        has = rng.random(n) < 0.5
        tone = np.where(df["y"] == "yes", rng.random(n) < 0.7, rng.random(n) < 0.35)
        comments = np.where(
            has,
            np.where(tone, rng.choice(pos, n), np.where(rng.random(n) < 0.5, rng.choice(neg, n), rng.choice(neu, n))),
            "",
        )
        df["comment"] = comments
    return df
