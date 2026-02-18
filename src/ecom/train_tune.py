from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from ecom.train_baseline import build_pipeline
import os
import joblib
import mlflow

DATA_PATH = Path("data/processed/online_shoppers_clean.csv")

def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Revenue"])
    y = df["Revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    clf = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_distributions = {
        "model__C": np.logspace(-3, 3, 30),
    }

    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        n_iter=15,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=2,
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ecom-tuning")

    with mlflow.start_run(run_name="logreg_random_search"):
        mlflow.log_param("tuning_method", "RandomizedSearchCV")
        mlflow.log_param("n_iter", 15)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "roc_auc")

        search.fit(X_train, y_train)

        best_params = search.best_params_
        best_cv = float(search.best_score_)

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_roc_auc", best_cv)

        best_model = search.best_estimator_

        proba = best_model.predict_proba(X_test)[:, 1]
        test_roc_auc = float(roc_auc_score(y_test, proba))

        mlflow.log_metric("test_roc_auc", test_roc_auc)

        os.makedirs("models", exist_ok=True)
        model_path = "models/best_model.joblib"
        joblib.dump(best_model, model_path)

        mlflow.log_artifact(model_path)

        print("Best params:", best_params)
        print("Best CV ROC-AUC:", best_cv)
        print("Test ROC-AUC:", test_roc_auc)
        print("Saved:", model_path)


if __name__ == "__main__":
    main()
