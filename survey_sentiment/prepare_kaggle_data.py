#!/usr/bin/env python3
"""Prepare survey-training data from the Kaggle Sentiment Analysis kernel inputs.

Kernel: satyaprakashshukl/sentiment-analysis
Datasets used by that kernel:
  1. tejasurya/latest-elected-uk-prime-minister-rishi-sunak  (tweet text)
  2. sauravmaheshkar/huggingface-bert-variants              (BERT model weights)

We train on the tweet text. Sentiment labels are produced with VADER and mapped
onto the same 1–5 star scale the kernel derives from
`nlptown/bert-base-multilingual-uncased-sentiment`. The 17GB BERT-variants
mirror is optional; HuggingFace can supply the model when needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from survey_sentiment.src.sentiment import SentimentAnalyzer  # noqa: E402

DEFAULT_TWEETS = (
    ROOT
    / "data"
    / "kaggle"
    / "latest-elected-uk-prime-minister-rishi-sunak"
    / "uk_pm.csv"
)
DEFAULT_OUT = ROOT / "data" / "kaggle_labeled_surveys.csv"


def compound_to_star(compound: float) -> int:
    """Map VADER compound score to a 1–5 star sentiment (kernel-compatible scale)."""
    if compound >= 0.6:
        return 5
    if compound >= 0.2:
        return 4
    if compound > -0.2:
        return 3
    if compound > -0.6:
        return 2
    return 1


def star_to_survey(star: int) -> dict:
    recommend = "yes" if star >= 3 else "no"
    # Support rating tracks sentiment but stays slightly compressed toward mid
    support = max(1, min(5, star if star != 3 else 3))
    return {
        "overall_satisfaction": star,
        "recommend": recommend,
        "product_quality": star,
        "repurchase_intent": star if star >= 3 else max(1, star - 0),
        "support_rating": support,
    }


def prepare(
    tweets_csv: Path,
    out_csv: Path,
    sample_size: int = 8000,
    english_only: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    if not tweets_csv.exists():
        raise FileNotFoundError(
            f"Missing {tweets_csv}. Download with:\n"
            "  kaggle datasets download -d "
            "tejasurya/latest-elected-uk-prime-minister-rishi-sunak "
            f"-p {tweets_csv.parent} --unzip"
        )

    usecols = ["text"]
    # language column may be present
    header = pd.read_csv(tweets_csv, nrows=0).columns.tolist()
    if "language" in header:
        usecols.append("language")

    df = pd.read_csv(tweets_csv, usecols=usecols)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 20]

    if english_only and "language" in df.columns:
        df = df[df["language"] == "en"]

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)

    analyzer = SentimentAnalyzer()
    rows = []
    for text in df["text"].tolist():
        sentiment = analyzer.score_text(text)
        star = compound_to_star(float(sentiment["compound"]))
        answers = star_to_survey(star)
        rows.append(
            {
                "comment": text[:2000],
                **answers,
                "sentiment_label": sentiment["sentiment_label"],
                "compound": sentiment["compound"],
                "star_sentiment": star,
            }
        )

    out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Training CSV keeps only the survey schema columns used by train.py
    train_cols = [
        "comment",
        "overall_satisfaction",
        "recommend",
        "product_quality",
        "repurchase_intent",
        "support_rating",
    ]
    out[train_cols].to_csv(out_csv, index=False)

    meta_path = out_csv.with_name(out_csv.stem + "_with_sentiment.csv")
    out.to_csv(meta_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tweets", type=Path, default=DEFAULT_TWEETS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-size", type=int, default=8000)
    parser.add_argument("--all-languages", action="store_true")
    args = parser.parse_args()

    frame = prepare(
        tweets_csv=args.tweets,
        out_csv=args.out,
        sample_size=args.sample_size,
        english_only=not args.all_languages,
    )
    print(f"Wrote {len(frame)} labeled rows -> {args.out}")
    print("Star sentiment distribution:")
    print(frame["star_sentiment"].value_counts().sort_index().to_string())
    print("Recommend distribution:")
    print(frame["recommend"].value_counts().to_string())


if __name__ == "__main__":
    main()
