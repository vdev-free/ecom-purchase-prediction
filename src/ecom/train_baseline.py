from pathlib import Path
import mlflow
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score

DATA_PATH = Path("data/processed/online_shoppers_clean.csv")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=['Revenue'])
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# print("Train size:", X_train.shape)
# print("Test size:", X_test.shape)
# print("Train positive rate:", y_train.mean())
# print("Test positive rate:", y_test.mean())

# Columns by type
cat_cols = ['Month', 'VisitorType']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)

model = LogisticRegression(max_iter=1000)

clf = Pipeline(
    steps=[
        ('preprocess', preprocess),
        ('model', model)
    ]
)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment('ecom-baseline')

with mlflow.start_run(run_name='logreg_baseline'):

  clf.fit(X_train, y_train)

  proba = clf.predict_proba(X_test)[:, 1]
  pred = (proba >= 0.3).astype(int)

  pr_auc = average_precision_score(y_test, proba)
  roc_auc = roc_auc_score(y_test, proba)
  precision = precision_score(y_test, pred)
  recall = recall_score(y_test, pred)

  mlflow.log_param("model", "logistic_regression")
  mlflow.log_param("threshold", 0.3)

  mlflow.log_metric("pr_auc", pr_auc)
  mlflow.log_metric("roc_auc", roc_auc)
  mlflow.log_metric("precision", precision)
  mlflow.log_metric("recall", recall)

  cm = confusion_matrix(y_test, pred)  # [[TN, FP], [FN, TP]]

  fig, ax = plt.subplots(figsize=(4, 4))
  ax.imshow(cm)
  ax.set_title("Confusion Matrix (threshold=0.3)")
  ax.set_xlabel("Predicted")
  ax.set_ylabel("Actual")
  ax.set_xticks([0, 1])
  ax.set_yticks([0, 1])

  # підписи чисел у клітинках
  for (i, j), v in zip([(0,0),(0,1),(1,0),(1,1)], cm.flatten()):
      ax.text(j, i, str(v), ha="center", va="center")

  plt.tight_layout()

  os.makedirs("artifacts", exist_ok=True)
  path = "artifacts/confusion_matrix.png"
  plt.savefig(path)
  plt.close(fig)

  mlflow.log_artifact(path)