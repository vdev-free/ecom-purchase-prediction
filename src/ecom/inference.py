from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

BUNDLE_PATH = Path("models/model_bundle.joblib")

def load_bundle(path: Path = BUNDLE_PATH) -> dict[str, Any]:
    bundle = joblib.load(path)
    return bundle

def predict_one(session: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    model = bundle['model']
    threshold = bundle['threshold']

    X = pd.DataFrame([session])
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(proba >= threshold)

    return{
        "probability": proba,
        "prediction": pred,
        "threshold": threshold,
        "model_version": bundle.get("model_version", "unknown"),
    }

if __name__ == "__main__":
    bundle = load_bundle()

    example_session = {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 1,
        "ProductRelated_Duration": 10.0,
        "BounceRates": 0.2,
        "ExitRates": 0.3,
        "PageValues": 0.0,
        "SpecialDay": 0.0,
        "Month": "May",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 1,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }

    out = predict_one(example_session, bundle)
    print(out)

    print("\n=== Sensitivity check: PageValues ===")
    for pv in [0.0, 10.0, 50.0, 100.0]:
        example_session["PageValues"] = pv
        out = predict_one(example_session, bundle)
        print(f"PageValues={pv:>5} -> proba={out['probability']:.4f} pred={out['prediction']}")