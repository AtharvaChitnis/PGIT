#!/usr/bin/env python3
"""Download the two Kaggle datasets used by satyaprakashshukl/sentiment-analysis."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "kaggle"
KERNEL_DIR = ROOT / "kaggle_kernels"

# Datasets attached to the kernel.
DATASETS = {
    "tweets": "tejasurya/latest-elected-uk-prime-minister-rishi-sunak",
    # Optional: ~17GB offline BERT weight mirror. Prefer HuggingFace instead.
    "bert_variants": "sauravmaheshkar/huggingface-bert-variants",
}

KERNEL = "satyaprakashshukl/sentiment-analysis"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-bert-variants",
        action="store_true",
        help="Also download the ~17GB huggingface-bert-variants dataset.",
    )
    parser.add_argument(
        "--pull-kernel",
        action="store_true",
        help="Also pull the Kaggle kernel notebook source.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)

    tweets_slug = DATASETS["tweets"]
    tweets_dir = DATA_DIR / tweets_slug.split("/")[-1]
    run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            tweets_slug,
            "-p",
            str(tweets_dir),
            "--unzip",
        ]
    )

    if args.with_bert_variants:
        bert_slug = DATASETS["bert_variants"]
        bert_dir = DATA_DIR / bert_slug.split("/")[-1]
        print(
            "WARNING: downloading ~17GB of BERT weights. "
            "Prefer transformers.from_pretrained("
            "'nlptown/bert-base-multilingual-uncased-sentiment').",
            file=sys.stderr,
        )
        run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                bert_slug,
                "-p",
                str(bert_dir),
                "--unzip",
            ]
        )
    else:
        print(
            "Skipping huggingface-bert-variants (~17GB). "
            "Pass --with-bert-variants to download it."
        )

    if args.pull_kernel:
        run(
            [
                "kaggle",
                "kernels",
                "pull",
                KERNEL,
                "-p",
                str(KERNEL_DIR / "sentiment-analysis"),
            ]
        )


if __name__ == "__main__":
    main()
