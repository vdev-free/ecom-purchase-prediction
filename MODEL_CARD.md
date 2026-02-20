# Model Card — E-commerce Purchase Prediction (Day 102)

## Overview

Binary classifier to predict whether an e-commerce session will end with a purchase (`Revenue=1`).

## Intended Use

- Trigger targeted discounts or support for high-intent sessions.
- Prioritize real-time assistance under operational quota constraints (e.g., top-k%).

Not intended for:

- User-level long-term profiling or sensitive attribute inference.

## Data

Source: `data/processed/online_shoppers_clean.csv`  
Unit: one row = one user session  
Target: `Revenue` (1 purchase / 0 no purchase)

Class imbalance: positives ≈ 15.6% in test split.

## Model

Pipeline:

1. Feature engineering: `add_features`
2. Preprocessing: OneHotEncoder (categorical) + StandardScaler (numeric)
3. Classifier: Logistic Regression

## Evaluation

Primary: PR-AUC  
Secondary: ROC-AUC, Precision, Recall

Operating point (example cost setting):

- Cost model: FP_cost=1€, FN_cost=5€
- Recommended quota: 0.30 (top-30% sessions by probability)
- This minimizes total cost under the assumed costs.

## Explainability (SHAP)

Top features by mean(|SHAP|) on test sample:

- PageValues
- ExitRates
- Month (seasonality: Nov/May/Mar/Dec)
- BounceRates
- Administrative
- SpecialDay
- Weekend

Artifacts:

- `artifacts/shap_top_features.csv`

## Limitations / Risks

- Threshold/quota depends on business costs; must be re-evaluated if costs change.
- Potential drift over time (seasonality, marketing changes).
- Model probabilities may be uncalibrated; consider calibration if decisions depend on true probability values.

## Monitoring (next)

- Track PR-AUC/ROC-AUC and calibration over time.
- Track predicted positive rate under chosen quota.
- Drift checks for key features (e.g., PageValues, BounceRates).
