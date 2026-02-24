from pathlib import Path
import joblib
import os

MODEL_PATH = Path("models/best_model.joblib")
BUNDLE_PATH = Path("models/model_bundle.joblib")

def main() -> None:
    model = joblib.load(MODEL_PATH)
    bundle = {
        'model': model,
        'threshold': 0.1427,
        'model_version': 'v1',
    }

    os.makedirs('models', exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH)

    print("Saved bundle:", BUNDLE_PATH)
    print("Keys:", list(bundle.keys()))

if __name__ == '__main__':
    main()