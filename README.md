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

# Current Project Status (Production-Oriented)

✅ Reproducible data pipeline  
✅ Feature engineering inside sklearn Pipeline  
✅ Stratified cross-validation  
✅ Hyperparameter tuning  
✅ MLflow experiment tracking  
✅ Saved production-ready model artifact  

---

# Next Steps (Days 101–110)

- Threshold optimization (business-aligned precision/recall)
- Error analysis
- Model explainability (SHAP)
- Model packaging
- FastAPI inference service
- Dockerization
- CI pipeline
- Cloud deployment

---

# Project Structure

```
src/ecom/
  train_baseline.py
  train_cv.py
  train_tune.py
  pipeline.py
  features.py

data/
  processed/

models/
mlruns/
```

---

# How to Reproduce

1. Install dependencies
2. Prepare data
3. Run baseline
4. Run CV
5. Run tuning
6. Open MLflow UI

---

This project follows a production-style ML workflow,
not just notebook experimentation.
