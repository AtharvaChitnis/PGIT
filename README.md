# Marketing Campaign Response Prediction

An imbalanced binary-classification project using the Kaggle Bank Marketing dataset. The goal is to predict whether a customer will subscribe to a term deposit after a marketing campaign.

## Project Objective

The initial objective is:

> Predict whether a customer will respond positively to a marketing campaign based on customer and campaign characteristics.

The broader project direction is to develop a marketing-research ML system that can support customer targeting, response prediction, segmentation, and eventually cost-sensitive campaign decisions.

## Dataset

**Dataset:** Bank Marketing / UCI Bank Marketing dataset  
**Source:** Kaggle  
**Target:** `y`

Target values:

- `no` = customer did not subscribe
- `yes` = customer subscribed

The dataset contains numerical and categorical variables, including customer demographics, financial information, contact information, and previous campaign outcomes.

### Class Imbalance

The target is significantly imbalanced, with the `no` class being much more frequent than the `yes` class.

Because of this imbalance, accuracy alone is not sufficient for evaluating the models. Precision, recall, F1-score, ROC-AUC, and confusion matrices are used.

## Data Preparation

The project uses:

- Train/test split with `stratify=y`
- Numerical feature standardization
- One-hot encoding for categorical features
- A preprocessing pipeline using `ColumnTransformer` and `Pipeline`
- `duration` excluded from the main realistic prediction experiment because it represents information available after/during the customer contact and can introduce information leakage when the goal is pre-contact targeting

The same train/test split was retained across the baseline model comparison to make the experiments comparable.

## Models

Four baseline classifiers were implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

### Baseline Results

Approximate results obtained from the current experiment:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 89.5% | 68.3% | 18.5% | 29.1% | — |
| Decision Tree | 83.6% | 30.8% | 32.5% | ~31.9% | — |
| Random Forest | 89.4% | 62.3% | 24.1% | 34.8% | 0.783 |
| XGBoost | 89.5% | 64.4% | 23.8% | ~34.7% | 0.801 |

These results show that XGBoost had the highest ROC-AUC among the baseline models tested, while the Decision Tree produced the highest recall at the default threshold.

## Error Analysis

The XGBoost baseline was examined using:

- False positives
- False negatives
- Actual vs. predicted positive responses
- Predicted probability distributions
- Borderline predictions around the classification threshold

The baseline XGBoost confusion matrix was:

| | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 11,768 | 209 |
| Actual Yes | 1,209 | 378 |

This highlighted a major issue: many actual `yes` customers were classified as `no`.

## Classification Threshold Experiment

The XGBoost model outputs a probability of response. The default classification threshold of 0.5 was not treated as inherently optimal.

For the baseline (unweighted) XGBoost model:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.1 | 28.6% | 70.1% | 40.6% |
| 0.2 | 47.9% | 52.6% | **50.1%** |
| 0.3 | 53.5% | 43.0% | 47.7% |
| 0.4 | 59.2% | 33.6% | 42.8% |
| 0.5 | 64.4% | 23.8% | 34.8% |
| 0.6 | 70.8% | 17.0% | 27.4% |
| 0.7 | 76.3% | 12.2% | 21.0% |

For the thresholds tested, 0.2 produced the highest F1 for the unweighted XGBoost model.

## Class Weighting

Class weighting was tested as a method for addressing class imbalance.

### Weighted Logistic Regression

- Accuracy: 75.6%
- Precision: 26.8%
- Recall: 62.6%
- F1: 37.5%
- ROC-AUC: 0.770

### Weighted Decision Tree

- Accuracy: 84.2%
- Precision: 32.1%
- Recall: 31.7%
- F1: 31.9%
- ROC-AUC: 0.614

### Weighted Random Forest

- Accuracy: 89.6%
- Precision: 66.5%
- Recall: 21.9%
- F1: 33.0%
- ROC-AUC: 0.786

### Weighted XGBoost

- Accuracy: 82.5%
- Precision: 36.0%
- Recall: 63.3%
- F1: 45.9%
- ROC-AUC: 0.803

The weighted XGBoost model showed a large increase in recall compared with the unweighted baseline, while precision and accuracy decreased.

## Weighted XGBoost Threshold Experiment

Threshold tuning was also tested after class weighting:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.1 | 12.0% | 99.6% | 21.5% |
| 0.2 | 14.0% | 95.2% | 24.4% |
| 0.3 | 18.2% | 85.2% | 30.0% |
| 0.4 | 25.3% | 74.7% | 37.8% |
| 0.5 | 36.0% | 63.3% | 45.9% |
| 0.6 | 46.3% | 53.2% | 49.5% |
| 0.7 | 52.1% | 44.8% | 48.2% |

A finer threshold search from 0.05 to 0.95 found:

- **Best tested threshold:** 0.61
- Precision: 47.1%
- Recall: 52.4%
- F1: 49.6%

This threshold was selected using the current test-set experiment for learning purposes; it should not be treated as a final production threshold. A rigorous final implementation should select the threshold using training/validation data and evaluate the chosen threshold once on an untouched test set.

## Current Findings

1. The target variable is strongly imbalanced.
2. Accuracy can be misleading because the majority class is `no`.
3. Logistic Regression is relatively conservative at the default threshold.
4. The Decision Tree identifies more positive cases but produces more false positives.
5. Random Forest provides a more balanced baseline than a single Decision Tree.
6. XGBoost currently has the strongest baseline ROC-AUC among the tested models.
7. Class weighting substantially changes minority-class recall for Logistic Regression and XGBoost, but does not guarantee improvement for every model.
8. Classification threshold selection has a major effect on precision and recall.
9. The best threshold depends on the objective of the marketing campaign.

## Business Objective

The project is intended to move beyond simply maximizing F1 or accuracy.

The eventual objective is:

> **Minimize marketing cost / maximize campaign value by choosing which customers to contact based on predicted response probability and the financial costs and benefits of the campaign.**

A cost-sensitive decision system can incorporate:

- Cost of contacting a customer
- Profit from a successful conversion
- Potential cost of missing a likely responder
- Campaign capacity or contact budget

This means the final threshold should ultimately be chosen using **expected business value**, not an arbitrary 0.5 threshold.

## Current Project Status

### Completed

- [x] Dataset loading and inspection
- [x] Exploratory data analysis
- [x] Target class imbalance analysis
- [x] Data preprocessing
- [x] Train/test split
- [x] Logistic Regression baseline
- [x] Decision Tree baseline
- [x] Random Forest baseline
- [x] XGBoost baseline
- [x] Confusion-matrix-based error analysis
- [x] Probability/threshold analysis
- [x] Class-weighting experiments

### Next Steps

- Define realistic campaign costs and benefits
- Optimize the threshold for expected marketing value
- Compare cost-sensitive performance across models
- Optionally investigate SMOTE/SMOTENC as an additional imbalance experiment
- Perform systematic hyperparameter tuning
- Use validation/cross-validation for model and threshold selection
- Evaluate the final model on an untouched test set
- Build a reusable prediction application/dashboard

## Tech Stack

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost

## Project Structure

A possible expanded project structure is:

```text
marketing-research-ml/
│
├── notebooks/
│   └── Marketing_Research_ML.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── app/
│   └── app.py
│
└── README.md
```

## Disclaimer

This project is an educational ML experiment based on a public marketing dataset. The reported financial/cost objective is a proposed framework; actual campaign economics must be supplied by the business using the model.

