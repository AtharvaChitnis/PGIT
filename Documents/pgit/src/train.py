"""Train / tune / select / evaluate with a strict validation protocol.

Protocol (the test set is touched exactly once, at the very end):

1. Stratified split: 60% train / 20% validation / 20% test.
2. Hyperparameter tuning: ``RandomizedSearchCV`` with stratified k-fold CV on TRAIN only
   (scoring: PR-AUC, which is informative under class imbalance).
3. Model selection: tuned candidates are compared on the VALIDATION set.
4. Threshold selection: out-of-fold predictions from k-fold CV on train+validation
   ("development" data) for the winning configuration. Maximizes F1 or expected profit.
5. Final fit on train+validation, optionally probability-calibrated, then ONE evaluation
   on the test set at the already-chosen threshold.

Usage:
    python -m src.train --data data/raw/bank-full.csv --models logreg hist_gb xgboost
    python -m src.train --synthetic --n-iter 3          # smoke test without the real data
    python -m src.train --objective profit --contact-cost 1 --conversion-value 10
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from . import config
from .data import load_bank_data, make_synthetic_bank, split_xy, three_way_split
from .evaluation import (
    Economics,
    best_threshold,
    campaign_summary,
    classification_metrics,
    threshold_sweep,
)
from .models import IMBALANCE_STRATEGIES, available_models, build_pipeline, param_distributions
from .preprocessing import infer_feature_types

# Illustrative only: real campaign economics must come from the business.
DEFAULT_ECONOMICS = Economics(contact_cost=1.0, conversion_value=10.0)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def run_experiment(
    df: pd.DataFrame,
    *,
    model_names: list[str] | None = None,
    imbalance: str = "none",
    n_iter: int = 20,
    cv_folds: int = config.CV_FOLDS,
    objective: str = "f1",
    economics: Economics = DEFAULT_ECONOMICS,
    calibrate: bool = True,
    text_column: str | None = None,
    random_state: int = config.RANDOM_STATE,
    n_jobs: int = -1,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full protocol. Returns a report dict (also written to ``out_dir`` if given)."""
    log = print if verbose else (lambda *a, **k: None)
    model_names = model_names or ["logreg", "hist_gb"]
    for name in model_names:
        if name not in available_models():
            raise ValueError(f"Model '{name}' unavailable. Installed: {available_models()}")

    use_text = bool(text_column and text_column in df.columns)
    text_column = text_column if use_text else None

    X, y = split_xy(df)
    splits = three_way_split(X, y, random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    pos_weight = float((splits.y_train == 0).sum() / max((splits.y_train == 1).sum(), 1))
    log(f"Split: train={len(splits.X_train)} val={len(splits.X_val)} test={len(splits.X_test)} "
        f"| positive rate train={splits.y_train.mean():.3f}")

    # ---- 2 & 3: tune on train (CV), compare on validation ---------------------------------
    candidates: dict[str, RandomizedSearchCV] = {}
    comparison = []
    for name in model_names:
        pipe = build_pipeline(name, splits.X_train, imbalance, text_column, random_state, pos_weight)
        search = RandomizedSearchCV(
            pipe,
            param_distributions(name),
            n_iter=n_iter,
            scoring="average_precision",
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            refit=True,
        )
        search.fit(splits.X_train, splits.y_train)
        val_proba = search.predict_proba(splits.X_val)[:, 1]
        val = classification_metrics(splits.y_val, val_proba, 0.5)
        candidates[name] = search
        comparison.append({
            "model": name,
            "cv_pr_auc": float(search.best_score_),
            "val_pr_auc": val["pr_auc"],
            "val_roc_auc": val["roc_auc"],
            "best_params": _jsonable(search.best_params_),
        })
        log(f"  {name:14s} cv PR-AUC={search.best_score_:.4f} | val PR-AUC={val['pr_auc']:.4f} "
            f"ROC-AUC={val['roc_auc']:.4f}")

    winner = max(comparison, key=lambda r: r["val_pr_auc"])["model"]
    log(f"Selected on validation PR-AUC: {winner}")

    # ---- 4: threshold from out-of-fold CV predictions on train+validation -----------------
    X_dev, y_dev = splits.X_dev, splits.y_dev
    final = clone(candidates[winner].best_estimator_)
    if calibrate:
        final = CalibratedClassifierCV(final, method="isotonic", cv=cv_folds)
    oof = cross_val_predict(final, X_dev, y_dev, cv=cv, method="predict_proba", n_jobs=n_jobs)[:, 1]
    sweep = threshold_sweep(y_dev, oof, economics)
    threshold, objective_value = best_threshold(sweep, objective)
    log(f"Threshold (objective={objective}, out-of-fold CV): {threshold:.3f} -> {objective_value:.3f}")

    # ---- 5: final fit, single test evaluation ---------------------------------------------
    final.fit(X_dev, y_dev)
    test_proba = final.predict_proba(splits.X_test)[:, 1]
    test_metrics = classification_metrics(splits.y_test, test_proba, threshold)
    test_campaign = campaign_summary(splits.y_test, test_proba, economics, threshold)
    log(f"TEST  PR-AUC={test_metrics['pr_auc']:.4f} ROC-AUC={test_metrics['roc_auc']:.4f} "
        f"P={test_metrics['precision']:.3f} R={test_metrics['recall']:.3f} F1={test_metrics['f1']:.3f}")

    numeric, categorical = infer_feature_types(X, text_column)
    report = {
        "protocol": {
            "split": {"train": len(splits.X_train), "val": len(splits.X_val), "test": len(splits.X_test)},
            "cv_folds": cv_folds,
            "n_iter": n_iter,
            "imbalance": imbalance,
            "calibrated": calibrate,
            "objective": objective,
            "random_state": random_state,
            "text_column": text_column,
        },
        "economics_illustrative": economics.to_dict(),
        "model_comparison": comparison,
        "selected_model": winner,
        "selected_params": candidates[winner].best_params_,
        "threshold": threshold,
        "threshold_objective_value_oof": objective_value,
        "test_metrics": test_metrics,
        "test_campaign": test_campaign,
    }
    report = _jsonable(report)

    bundle = {
        "model": final,
        "threshold": threshold,
        "feature_columns": list(X.columns),
        "numeric": numeric,
        "categorical": categorical,
        "text_column": text_column,
        "calibrated": calibrate,
        "categories": {c: sorted(X[c].astype(str).unique().tolist()) for c in categorical},
        "numeric_summary": {
            c: {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())}
            for c in numeric
        },
        "selected_model": winner,
        "economics_used_for_threshold": economics.to_dict(),
        "versions": {"sklearn": sklearn.__version__, "python": platform.python_version()},
    }

    if out_dir is not None:
        out = Path(out_dir)
        (out / "models").mkdir(parents=True, exist_ok=True)
        (out / "reports").mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out / "models" / "campaign_model.joblib")
        (out / "reports" / "metrics.json").write_text(json.dumps(report, indent=2))
        sweep.to_csv(out / "reports" / "threshold_sweep_oof.csv", index=False)
        pd.DataFrame(comparison).to_csv(out / "reports" / "model_comparison.csv", index=False)
        log(f"Saved model + reports under {out}")

    return {"report": report, "bundle": bundle, "sweep": sweep}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(config.DATA_PATH))
    p.add_argument("--synthetic", action="store_true", help="use generated data (smoke test only)")
    p.add_argument("--models", nargs="+", default=["logreg", "hist_gb"], help=f"from {available_models()}")
    p.add_argument("--imbalance", choices=IMBALANCE_STRATEGIES, default="none")
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--cv-folds", type=int, default=config.CV_FOLDS)
    p.add_argument("--objective", choices=["f1", "profit"], default="f1")
    p.add_argument("--contact-cost", type=float, default=DEFAULT_ECONOMICS.contact_cost)
    p.add_argument("--conversion-value", type=float, default=DEFAULT_ECONOMICS.conversion_value)
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--text-column", default=None, help="comment column to add sentiment features from")
    p.add_argument("--out", default=str(config.ROOT))
    args = p.parse_args()

    df = make_synthetic_bank(with_comments=bool(args.text_column)) if args.synthetic else load_bank_data(args.data)
    run_experiment(
        df,
        model_names=args.models,
        imbalance=args.imbalance,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
        objective=args.objective,
        economics=Economics(args.contact_cost, args.conversion_value),
        calibrate=not args.no_calibrate,
        text_column=args.text_column,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
