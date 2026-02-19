from pathlib import Path
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

DATA_PATH = Path("data/processed/online_shoppers_clean.csv")
MODEL_PATH = Path("models/best_model.joblib")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Revenue"])
    y = df["Revenue"]

    _, X_test, _, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    model = joblib.load(MODEL_PATH)

    proba = model.predict_proba(X_test)[:, 1]
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    for t in thresholds:
        pred = (proba >= t).astype(int)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        pred_rate = float(pred.mean())
        print(f"t={t:.1f} pred1_rate={pred_rate:.3f} precision={precision:.3f} recall={recall:.3f}")

    quota = 0.20
    threshold_quota = float(np.quantile(proba, 1 - quota))
    pred_quota = (proba >= threshold_quota).astype(int)

    precision_q = precision_score(y_test, pred_quota, zero_division=0)
    recall_q = recall_score(y_test, pred_quota, zero_division=0)
    pred_rate_q = float(pred_quota.mean())

    # print("\n=== Quota thresholding ===")
    # print("Quota:", quota)
    # print("Threshold for quota:", threshold_quota)
    # print("Predicted 1 rate:", pred_rate_q)
    # print("Precision:", precision_q)
    # print("Recall:", recall_q)

    cm = confusion_matrix(y_test, pred_quota)
    tn, fp, fn, tp = cm.ravel()

    # print("\n=== Confusion matrix (quota=0.2) ===")
    # print(cm)
    # print(f"TN={tn} FP={fp} FN={fn} TP={tp}")

    os.makedirs("artifacts", exist_ok=True)

    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        pred_rate = float(pred.mean())
        rows.append(
            {"threshold": t, "pred1_rate": pred_rate, "precision": precision, "recall": recall}
        )

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv("artifacts/threshold_sweep.csv", index=False)
    # print("Saved: artifacts/threshold_sweep.csv")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ecom-thresholding")

    with mlflow.start_run(run_name="threshold_quota_0.2"):
    # params
      mlflow.log_param("quota", quota)
      mlflow.log_param("threshold_quota", threshold_quota)

    # metrics (quota result)
      mlflow.log_metric("pred1_rate", pred_rate_q)
      mlflow.log_metric("precision", float(precision_q))
      mlflow.log_metric("recall", float(recall_q))

    # confusion matrix numbers
      mlflow.log_metric("tn", float(tn))
      mlflow.log_metric("fp", float(fp))
      mlflow.log_metric("fn", float(fn))
      mlflow.log_metric("tp", float(tp))

    # log sweep csv
      mlflow.log_artifact("artifacts/threshold_sweep.csv")

    # save confusion matrix image
      fig, ax = plt.subplots(figsize=(4, 4))
      ax.imshow(cm)
      ax.set_title("Confusion Matrix (quota=0.2)")
      ax.set_xlabel("Predicted")
      ax.set_ylabel("Actual")
      ax.set_xticks([0, 1])
      ax.set_yticks([0, 1])
      for (i, j), v in zip([(0,0),(0,1),(1,0),(1,1)], cm.flatten()):
        ax.text(j, i, str(v), ha="center", va="center")
      plt.tight_layout()

      os.makedirs("artifacts", exist_ok=True)
      cm_path = "artifacts/confusion_matrix_quota_0.2.png"
      plt.savefig(cm_path)
      plt.close(fig)

      mlflow.log_artifact(cm_path)

   

    fp_cost = 1.0
    fn_cost = 5.0

    quotas = [0.10, 0.20, 0.30]

    print("\n=== Cost by quota ===")
    print(f"Assume FP_cost={fp_cost}€, FN_cost={fn_cost}€")

    for q in quotas:
        thr = float(np.quantile(proba, 1 - q))
        pred = (proba >= thr).astype(int)

        cm = confusion_matrix(y_test, pred)
        tn, fp, fn, tp = cm.ravel()

        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        pred_rate = float(pred.mean())

        total_cost = fp * fp_cost + fn * fn_cost

        print(
        f"quota={q:.2f} thr={thr:.4f} pred1_rate={pred_rate:.3f} "
        f"precision={precision:.3f} recall={recall:.3f} "
        f"FP={fp} FN={fn} cost={total_cost:.1f}"
        )

    best_quota = 0.30
    best_thr = float(np.quantile(proba, 1 - best_quota))
    best_pred = (proba >= best_thr).astype(int)

    cm_best = confusion_matrix(y_test, best_pred)
    tn_b, fp_b, fn_b, tp_b = cm_best.ravel()

    precision_b = precision_score(y_test, best_pred, zero_division=0)
    recall_b = recall_score(y_test, best_pred, zero_division=0)
    pred_rate_b = float(best_pred.mean())
    best_cost = fp_b * fp_cost + fn_b * fn_cost

    with mlflow.start_run(run_name="recommended_quota_0.30_fp1_fn5"):
     mlflow.log_param("strategy", "quota_thresholding")
     mlflow.log_param("fp_cost", fp_cost)
     mlflow.log_param("fn_cost", fn_cost)
     mlflow.log_param("quota", best_quota)
     mlflow.log_param("threshold_quota", best_thr)

     mlflow.log_metric("pred1_rate", pred_rate_b)
     mlflow.log_metric("precision", float(precision_b))
     mlflow.log_metric("recall", float(recall_b))
     mlflow.log_metric("total_cost", float(best_cost))

     mlflow.log_metric("tn", float(tn_b))
     mlflow.log_metric("fp", float(fp_b))
     mlflow.log_metric("fn", float(fn_b))
     mlflow.log_metric("tp", float(tp_b))

     print("Logged MLflow run: recommended_quota_0.30_fp1_fn5")

if __name__ == "__main__":
    main()
