"""Project-wide constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "bank-full.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
BUNDLE_PATH = MODELS_DIR / "campaign_model.joblib"
SURVEY_MODEL_PATH = ROOT / "survey_sentiment" / "models" / "survey_filler.joblib"

TARGET = "y"
POSITIVE_LABEL = "yes"

# Known only during/after the call, so it leaks the outcome for pre-contact targeting.
LEAKAGE_COLUMNS = ("duration",)

# Optional free-text column (customer feedback) used by the merged sentiment features.
TEXT_COLUMN = "comment"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20
CV_FOLDS = 5
