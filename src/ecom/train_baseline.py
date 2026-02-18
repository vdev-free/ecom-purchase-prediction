from pathlib import Path
import os

import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
)

from ecom.features import add_features


DATA_PATH = Path("data/processed/online_shoppers_clean.csv")


def build_pipeline() -> Pipeline:
    cat_cols = ["Month", "VisitorType"]
    features_step = FunctionTransformer(add_features, validate=False)

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            (
                "num",
                Pipeline(steps=[("scaler", StandardScaler())]),
                make_column_selector(dtype_include=["int64", "float64"]),
            ),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    clf = Pipeline(
        steps=[
            ("features", features_step),
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    return clf


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Revenue"])
    y = df["Revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    clf = build_pipeline()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ecom-features")

    with mlflow.start_run(run_name="logreg_with_features"):
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.3).astype(int)

        pr_auc = average_precision_score(y_test, proba)
        roc_auc = roc_auc_score(y_test, proba)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)

        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("threshold", 0.3)
        mlflow.log_param(
            "feature_engineering",
            "v1_total_pages_total_duration_duration_per_page",
        )
        mlflow.log_param("baseline_experiment", "ecom-baseline")
        mlflow.log_param("baseline_run", "logreg_baseline")
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        cm = confusion_matrix(y_test, pred)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm)
        ax.set_title("Confusion Matrix (threshold=0.3)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        for (i, j), v in zip([(0, 0), (0, 1), (1, 0), (1, 1)], cm.flatten()):
            ax.text(j, i, str(v), ha="center", va="center")

        plt.tight_layout()

        os.makedirs("artifacts", exist_ok=True)
        run_id = mlflow.active_run().info.run_id
        path = f"artifacts/confusion_matrix_{run_id}.png"
        plt.savefig(path)
        plt.close(fig)

        mlflow.log_artifact(path)


if __name__ == "__main__":
    main()
