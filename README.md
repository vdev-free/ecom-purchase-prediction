# E-commerce Purchase Prediction

## Business Problem

An e-commerce company wants to predict whether a user session will end with a purchase.

By predicting high-intent sessions, the company can:

- show targeted discounts
- trigger real-time support
- optimize marketing spend
- improve conversion rate

## Prediction Task

Binary classification:

- 1 → user completed purchase
- 0 → user did not purchase

Each row represents one user session.

## Success Metrics

Primary metric:
- PR-AUC (Precision-Recall AUC)

Secondary metrics:
- Precision
- Recall
- ROC-AUC

Why PR-AUC?

The dataset is expected to be imbalanced.
PR-AUC better reflects performance on the positive (purchase) class.

## Experiment Plan

| Step | Description | Goal |
|------|------------|------|
| 1 | Baseline Logistic Regression | Establish minimum performance |
| 2 | Add feature scaling | Improve linear model performance |
| 3 | Tree-based model (Random Forest) | Capture nonlinear patterns |
| 4 | Hyperparameter tuning | Improve best model |
| 5 | Threshold optimization | Align model with business goal |
| 6 | Model explainability (SHAP) | Understand key drivers |
