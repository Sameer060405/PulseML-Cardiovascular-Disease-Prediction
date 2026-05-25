"""Feature engineering and sklearn preprocessing pipeline."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    FEATURE_COLUMNS,
    FEATURE_NAMES_PATH,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)


def build_preprocessor() -> ColumnTransformer:
    numeric_features = ["age_years", "height", "weight", "bmi", "ap_hi", "ap_lo"]
    categorical_features = [
        "gender",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
    ]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ],
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COL]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def fit_preprocessor(
    X_train: pd.DataFrame,
    save_path: Path | None = None,
) -> ColumnTransformer:
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    save_path = save_path or PREPROCESSOR_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, save_path)

    feature_names = preprocessor.get_feature_names_out().tolist()
    FEATURE_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_NAMES_PATH.write_text(json.dumps(feature_names, indent=2))

    return preprocessor


def load_preprocessor(path: Path | None = None) -> ColumnTransformer:
    return joblib.load(path or PREPROCESSOR_PATH)
