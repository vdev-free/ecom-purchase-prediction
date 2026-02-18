from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import pandas as pd
from pathlib import Path
from ecom.train_baseline import build_pipeline

DATA_PATH = Path("data/processed/online_shoppers_clean.csv")

def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=['Revenue'])
    y = df['Revenue']

    X_train, _, y_train, _ =train_test_split(X, y, random_state=42, stratify=y, train_size=0.2)

    clf = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   
    scores = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=1)

    print("CV ROC-AUC scores:", scores)
    print("Mean ROC-AUC:", scores.mean())


if __name__ == '__main__':
    main()