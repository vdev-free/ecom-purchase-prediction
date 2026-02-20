# E-commerce Purchase Prediction

## Business Problem (Day 96)

An e-commerce company wants to predict whether a user session will end with a purchase.

By predicting high-intent sessions, the company can:

- show targeted discounts
- trigger real-time support
- optimize marketing spend
- improve conversion rate

### Prediction Task

Binary classification:

- 1 → user completed purchase
- 0 → user did not purchase

Each row represents one user session.

### Success Metrics

Primary metric:

- PR-AUC (due to class imbalance)

Secondary metrics:

- ROC-AUC
- Precision
- Recall

---

# Full ML Project (Days 96–100)

## Day 96 — Problem Framing + Experiment Plan

Defined:

- Clear business objective
- Target variable: `Revenue`
- Primary metric: PR-AUC
- Secondary metrics: ROC-AUC, Precision, Recall
- Experimental roadmap

---

## Day 97 — EDA + Data Cleaning (Script-based)

Converted exploratory work into reproducible scripts:

- `data_cleaning.py`
- `run_data_check.py`
- `run_make_processed.py`

Key steps:

- Missing value checks
- Type corrections
- Basic sanity validation
- Saved cleaned dataset to:

```
data/processed/online_shoppers_clean.csv
```

Reproducible pipeline (not notebook-only).

---

## Day 98 — Baseline Model + MLflow Tracking

Implemented:

- Logistic Regression baseline
- Train/test split (stratified)
- Metrics:
  - PR-AUC
  - ROC-AUC
  - Precision
  - Recall
- Confusion matrix artifact

Tracked in MLflow:

- Parameters
- Metrics
- Artifacts

Run:

```bash
PYTHONPATH=src python -m ecom.train_baseline
```

MLflow UI:

```bash
mlflow ui
# http://127.0.0.1:5000
```

---

## Day 99 — Feature Engineering + Production Pipeline

Created production-ready pipeline:

Pipeline steps:

1. `add_features` (custom feature engineering)
2. ColumnTransformer:
   - OneHotEncoder (categorical)
   - StandardScaler (numerical)
3. LogisticRegression

Implemented via:

```
build_pipeline()
```

Ensures:

- No data leakage
- Reproducibility
- Single object for training & inference

---

## Day 100 — Cross-Validation + Hyperparameter Tuning

### Cross-Validation (Stratified 5-Fold)

Evaluated stability:

- CV ROC-AUC scores:
  - 0.86896
  - 0.88627
  - 0.88535
  - 0.88110
  - 0.89124
- Mean CV ROC-AUC: **0.88258**

Run:

```bash
PYTHONPATH=src python -m ecom.train_cv
```

---

### Hyperparameter Tuning (RandomizedSearchCV)

Tuned:

- LogisticRegression `C`

Search configuration:

- 5-fold Stratified CV
- 15 random samples
- Scoring: ROC-AUC

Results:

- Best C: `0.010826367338740546`
- Best CV ROC-AUC: **0.89245**
- Test ROC-AUC: **0.90487**

Saved:

```
models/best_model.joblib
```

Logged to MLflow experiment:

```
ecom-tuning
```

Run:

```bash
PYTHONPATH=src python -m ecom.train_tune
```

---

## Day 101 — Thresholding + Error Analysis (business operating point)

Model outputs probabilities, so we selected an operating threshold based on business constraints.

### Threshold sweep (fixed thresholds)

| threshold | pred1_rate | precision | recall |
| --------: | ---------: | --------: | -----: |
|       0.1 |      0.460 |     0.322 |  0.948 |
|       0.2 |      0.188 |     0.591 |  0.712 |
|       0.3 |      0.120 |     0.703 |  0.539 |
|       0.4 |      0.093 |     0.752 |  0.445 |
|       0.5 |      0.070 |     0.779 |  0.351 |

Saved artifact: `artifacts/threshold_sweep.csv` (also logged to MLflow).

### Quota-based thresholding (top-k strategy)

We often have an operational limit (e.g. discounts/support can be shown only to a fraction of sessions).
So we select the threshold to target a fixed quota.

Quota = 0.20:

- threshold ≈ 0.1908
- pred1_rate ≈ 0.2003
- precision ≈ 0.5644
- recall ≈ 0.7225
- confusion matrix: TN=1846 FP=213 FN=106 TP=276

### Cost-based operating point selection

Assumed costs:

- FP_cost = 1€
- FN_cost = 5€

Total cost = FP _ FP_cost + FN _ FN_cost

| quota | threshold | precision | recall |  FP |  FN | total_cost |
| ----: | --------: | --------: | -----: | --: | --: | ---------: |
|  0.10 |    0.3650 |     0.727 |  0.466 |  67 | 204 |       1087 |
|  0.20 |    0.1908 |     0.564 |  0.723 | 213 | 106 |        743 |
|  0.30 |    0.1427 |     0.442 |  0.848 | 409 |  58 |        699 |

✅ Recommended operating point (min cost): **quota = 0.30**  
Run is logged in MLflow experiment `ecom-thresholding`:

- `threshold_quota_0.2` (baseline quota run)
- `recommended_quota_0.30_fp1_fn5` (cost-based choice)

---

## Day 102 — Explainability (SHAP) + Model Card

### Goal

Make the model decision transparent and understandable.

The model outputs probabilities, but business stakeholders need to understand:

- which features drive predictions
- whether the model logic aligns with domain knowledge

---

## SHAP Explainability

We used SHAP (SHapley Additive exPlanations) for the trained Logistic Regression pipeline.

Pipeline steps:

1. Feature engineering (`add_features`)
2. Preprocessing (OneHotEncoder + StandardScaler)
3. Logistic Regression

SHAP was computed on the transformed feature space (after preprocessing), because this is the actual input to the model.

### Global Feature Importance

Feature importance was computed as:

mean(|SHAP value|) per feature

Top features:

- PageValues
- ExitRates
- Month (seasonality effects)
- BounceRates
- Administrative
- SpecialDay
- Weekend

Artifact:

- `artifacts/shap_top_features.csv`
- Logged to MLflow experiment `ecom-explainability`

Run:

```bash
PYTHONPATH=src python -m ecom.train_threshold
````
