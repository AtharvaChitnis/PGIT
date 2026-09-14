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

## Quick start

```bash
pip install -r survey_sentiment/requirements.txt
python survey_sentiment/demo.py
```

Optional flags:

```bash
python survey_sentiment/demo.py --classifier forest
python survey_sentiment/demo.py --skip-train   # reuse saved model
```

## Project layout

```text
survey_sentiment/
├── data/sample_feedback.csv   # labeled comments + survey answers
├── src/
│   ├── sentiment.py           # VADER feature extraction
│   ├── train.py               # train / save multi-output model
│   └── survey_filler.py       # inference + rule-based fallback
├── models/                    # saved joblib bundle + demo CSV
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

- Sample data is synthetic and educational.
- Hold-out metrics are printed after training; with only ~60 rows they will vary.
- `SurveyFiller.rule_based_fill()` maps VADER compound score to Likert values without ML, useful as a baseline.
