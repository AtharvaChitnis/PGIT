# Survey Sentiment Auto-Fill

A small ML pipeline that turns free-text feedback into structured survey answers using sentiment analysis.

## What it does

1. Scores each comment with **VADER** sentiment (`compound`, `pos`, `neu`, `neg`)
2. Combines those scores with lightweight **TF-IDF** text features
3. Trains a **multi-output classifier** (logistic regression or random forest) to predict:

| Field | Type |
|---|---|
| `overall_satisfaction` | 1–5 Likert |
| `recommend` | yes / no |
| `product_quality` | 1–5 Likert |
| `repurchase_intent` | 1–5 Likert |
| `support_rating` | 1–5 Likert |

Given only a comment, the model fills the remaining survey fields.

## Kaggle training sources

Trains on the two datasets attached to
[`satyaprakashshukl/sentiment-analysis`](https://www.kaggle.com/code/satyaprakashshukl/sentiment-analysis):

| Dataset | Role |
|---|---|
| `tejasurya/latest-elected-uk-prime-minister-rishi-sunak` | Tweet text corpus (`uk_pm.csv`) |
| `sauravmaheshkar/huggingface-bert-variants` | Offline BERT weight mirror (~17GB, optional) |

The kernel scores tweets with `nlptown/bert-base-multilingual-uncased-sentiment` (1–5 stars).
This project labels the same tweet text with VADER mapped onto that 1–5 scale, then trains the survey filler.

```bash
# Requires Kaggle API credentials (KAGGLE_USERNAME + KAGGLE_KEY)
pip install kaggle
python3 survey_sentiment/download_kaggle_sources.py          # tweets only
python3 survey_sentiment/download_kaggle_sources.py --pull-kernel
python3 survey_sentiment/prepare_kaggle_data.py --sample-size 8000
python3 survey_sentiment/demo.py --data survey_sentiment/data/kaggle_labeled_surveys.csv
```

Skip the BERT-variants download unless you need offline weights
(`--with-bert-variants`). Prefer HuggingFace for the multilingual sentiment model.

## Quick start (bundled sample)

```bash
pip install -r survey_sentiment/requirements.txt
python3 survey_sentiment/demo.py
```

Optional flags:

```bash
python3 survey_sentiment/demo.py --classifier forest
python3 survey_sentiment/demo.py --skip-train   # reuse saved model
```

## Project layout

```text
survey_sentiment/
├── data/
│   ├── sample_feedback.csv              # small synthetic set
│   └── kaggle_labeled_surveys.csv       # prepared from UK PM tweets
├── kaggle_kernels/                      # pulled kernel notebook
├── src/
│   ├── sentiment.py                     # VADER feature extraction
│   ├── train.py                         # train / save multi-output model
│   └── survey_filler.py                 # inference + rule-based fallback
├── models/                              # saved joblib bundle + demo CSV
├── download_kaggle_sources.py
├── prepare_kaggle_data.py
├── demo.py
└── requirements.txt
```

## Example

Input comment:

> I am so happy with this purchase — amazing quality and fast support!

Typical filled answers:

- overall_satisfaction: 5
- recommend: yes
- product_quality: 5
- repurchase_intent: 5
- support_rating: 5

## Notes

- Hold-out metrics are printed after training.
- On the Kaggle-prepared 8k-row set, logistic regression reaches ~0.94 accuracy on Likert fields and ~0.99 on recommend.
- `SurveyFiller.rule_based_fill()` maps VADER compound → Likert values without ML (baseline).
- Raw Kaggle downloads under `data/kaggle/` are gitignored (re-fetch with the download script).
