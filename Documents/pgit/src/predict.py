"""Load a trained bundle and turn probabilities into contact / don't-contact decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .evaluation import Economics, expected_value


def load_bundle(path: str | Path = config.BUNDLE_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train first: python -m src.train")
    return joblib.load(path)


def prepare_frame(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Select/order the training feature columns; fill an absent comment column with ''."""
    cols = bundle["feature_columns"]
    frame = df.copy()
    text_col = bundle.get("text_column")
    if text_col and text_col not in frame.columns:
        frame[text_col] = ""
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    return frame[cols]


def predict_proba(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    return bundle["model"].predict_proba(prepare_frame(bundle, df))[:, 1]


def recommend(
    bundle: dict,
    df: pd.DataFrame,
    economics: Economics,
    max_contacts: int | None = None,
) -> pd.DataFrame:
    """Per-customer decision by expected value.

    contact  <=>  p * conversion_value - contact_cost > 0   (i.e. p > break-even)
    With ``max_contacts`` (campaign capacity) only the highest-EV customers are contacted.
    """
    proba = predict_proba(bundle, df)
    ev = expected_value(proba, economics)
    contact = ev > 0
    if max_contacts is not None:
        keep = np.zeros(len(df), dtype=bool)
        order = np.argsort(-ev)[: max(int(max_contacts), 0)]
        keep[order] = True
        contact &= keep
    out = df.copy()
    out["response_probability"] = proba
    out["expected_value"] = ev
    out["contact"] = contact
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Score a CSV of customers.")
    p.add_argument("--input", required=True, help="CSV (';' or ',' separated) with customer features")
    p.add_argument("--output", default="scored_customers.csv")
    p.add_argument("--model", default=str(config.BUNDLE_PATH))
    p.add_argument("--contact-cost", type=float, required=True)
    p.add_argument("--conversion-value", type=float, required=True)
    p.add_argument("--max-contacts", type=int, default=None)
    args = p.parse_args()

    bundle = load_bundle(args.model)
    df = pd.read_csv(args.input, sep=None, engine="python")
    scored = recommend(bundle, df, Economics(args.contact_cost, args.conversion_value), args.max_contacts)
    scored.to_csv(args.output, index=False)
    print(f"{int(scored['contact'].sum())} of {len(scored)} customers recommended for contact -> {args.output}")


if __name__ == "__main__":
    main()
