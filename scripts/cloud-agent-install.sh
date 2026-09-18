#!/usr/bin/env bash
# Idempotent Cloud Agent install for the survey-sentiment project.
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

cd "$(dirname "$0")/.."

python3 -m pip install --user --upgrade pip
python3 -m pip install --user -r survey_sentiment/requirements.txt

# Public Kaggle datasets download without API secrets.
# Kernel CLI pull still needs auth; notebook source is already vendored.
if command -v kaggle >/dev/null 2>&1; then
  python3 survey_sentiment/download_kaggle_sources.py || {
    echo "WARN: Kaggle tweet download failed; using any existing local copy." >&2
  }
else
  echo "WARN: kaggle CLI missing after install." >&2
fi

# Prepare labeled training CSV when raw tweets are present.
if [[ -f survey_sentiment/data/kaggle/latest-elected-uk-prime-minister-rishi-sunak/uk_pm.csv ]]; then
  python3 survey_sentiment/prepare_kaggle_data.py --sample-size 8000
fi

# Smoke-train / demo against the prepared Kaggle labels when available.
DATA="survey_sentiment/data/sample_feedback.csv"
if [[ -f survey_sentiment/data/kaggle_labeled_surveys.csv ]]; then
  DATA="survey_sentiment/data/kaggle_labeled_surveys.csv"
fi
python3 survey_sentiment/demo.py --data "${DATA}" --classifier logistic

echo "Install complete."
