from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/online_shoppers_clean.csv")
MODEL_PATH = Path("models/best_model.joblib")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Revenue"])
    y = df["Revenue"]

    # беремо test, щоб пояснювати "на нових даних"
    _, X_test, _, _ = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    model = joblib.load(MODEL_PATH)

    # ВАЖЛИВО: наша модель = Pipeline, тому трансформимо фічі тим самим препроцесом
    X_feat = model.named_steps["features"].transform(X_test)
    X_pre = model.named_steps["preprocess"].transform(X_feat)

    # Feature names після one-hot + числових
    feature_names = model.named_steps["preprocess"].get_feature_names_out()

    # Візьмемо невеликий семпл, щоб було швидко
    n = min(300, X_pre.shape[0])
    X_pre_sample = X_pre[:n]

    # SHAP для лінійної моделі (LogReg)
    clf = model.named_steps["model"]
    explainer = shap.LinearExplainer(clf, X_pre_sample, feature_names=feature_names)

    shap_values = explainer.shap_values(X_pre_sample)

    # важливість = середнє |SHAP| по фічі
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    top = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(25)
    )

    os.makedirs("artifacts", exist_ok=True)
    out_path = Path("artifacts/shap_top_features.csv")
    top.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(top.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
