"""CLI inference for a single patient record."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import BEST_MODEL_PATH, FEATURE_COLUMNS


def predict(patient: dict, model_path: Path | None = None) -> dict:
    pipeline = joblib.load(model_path or BEST_MODEL_PATH)
    row = {col: patient[col] for col in FEATURE_COLUMNS}
    X = pd.DataFrame([row])
    proba = float(pipeline.predict_proba(X)[0, 1])
    label = int(proba >= 0.5)
    risk = "High" if label == 1 else "Low"
    return {
        "risk_level": risk,
        "probability": round(proba, 4),
        "prediction": label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict heart disease risk")
    parser.add_argument("--json", type=str, help="Patient JSON string")
    parser.add_argument("--file", type=str, help="Path to patient JSON file")
    args = parser.parse_args()

    if args.file:
        patient = json.loads(Path(args.file).read_text())
    elif args.json:
        patient = json.loads(args.json)
    else:
        patient = {
            "age": 55,
            "sex": 1,
            "cp": 2,
            "trestbps": 140,
            "chol": 250,
            "fbs": 0,
            "restecg": 1,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.2,
            "slope": 1,
            "ca": 0,
            "thal": 2,
        }

    result = predict(patient)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
