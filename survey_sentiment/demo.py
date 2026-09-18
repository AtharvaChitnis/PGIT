#!/usr/bin/env python3
"""Train the survey filler model and run a small demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from survey_sentiment.src.survey_filler import SurveyFiller  # noqa: E402
from survey_sentiment.src.train import save_bundle, train  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "sample_feedback.csv"
DEFAULT_MODEL = ROOT / "models" / "survey_filler.joblib"

DEMO_COMMENTS = [
    "I am so happy with this purchase — amazing quality and fast support!",
    "This was awful. Broken on arrival and nobody helped me.",
    "The product arrived on time and works as described. Average overall.",
    "Would definitely recommend. Great value and solid build.",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and demo a sentiment-based survey response filler."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--classifier",
        choices=["logistic", "forest"],
        default="logistic",
        help="Base classifier for multi-output survey prediction.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run inference with an existing model.",
    )
    args = parser.parse_args()

    if not args.skip_train:
        print(f"Training on {args.data} with {args.classifier} classifier...")
        bundle = train(args.data, classifier=args.classifier)
        save_bundle(bundle, args.model)
        print(f"Saved model bundle -> {args.model}\n")
        print("Hold-out metrics by survey field:")
        for field, scores in bundle["metrics"].items():
            print(
                f"  {field:22s}  "
                f"acc={scores['accuracy']:.3f}  "
                f"f1={scores['f1_weighted']:.3f}"
            )
        print()

    filler = SurveyFiller(args.model)
    print("Demo: auto-filled survey responses from free-text comments\n")
    for comment in DEMO_COMMENTS:
        result = filler.fill(comment)
        print("-" * 72)
        print(f"Comment : {result['comment']}")
        print(
            f"Sentiment: {result['sentiment']['sentiment_label']} "
            f"(compound={result['sentiment']['compound']:.3f})"
        )
        print("Survey answers:")
        print(json.dumps(result["survey_answers"], indent=2))
    print("-" * 72)

    batch = filler.fill_many(DEMO_COMMENTS)
    out_csv = ROOT / "models" / "demo_filled_surveys.csv"
    batch.to_csv(out_csv, index=False)
    print(f"\nBatch results written to {out_csv}")


if __name__ == "__main__":
    main()
