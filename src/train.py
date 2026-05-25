"""Train, tune, and evaluate multiple classifiers on the full dataset."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    BEST_MODEL_PATH,
    BEST_PARAMS_PATH,
    DATASET_NAME,
    METRICS_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TUNE_CV_FOLDS,
    TUNE_N_ITER,
    TUNE_SAMPLE_SIZE,
)
from src.data_loader import dataset_summary, load_dataset
from src.preprocess import build_preprocessor, split_data
from src.tuning import get_base_estimators, get_param_distributions


def _tune_sample(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if len(X_train) <= TUNE_SAMPLE_SIZE:
        return X_train, y_train
    X_sub, _, y_sub, _ = train_test_split(
        X_train,
        y_train,
        train_size=TUNE_SAMPLE_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    return X_sub, y_sub


def tune_and_fit(
    name: str,
    estimator,
    param_dist: dict,
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Pipeline, dict]:
    X_tune, y_tune = _tune_sample(X_train, y_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=TUNE_N_ITER,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=TUNE_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    search.fit(X_tune, y_tune)

    best = search.best_estimator_
    best.fit(X_train, y_train)

    return best, {
        "best_params": search.best_params_,
        "cv_roc_auc_tune": round(float(search.best_score_), 4),
    }


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune_info: dict,
) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        "tune_cv_roc_auc": tune_info.get("cv_roc_auc_tune"),
        "best_params": tune_info.get("best_params", {}),
    }


def train_all(force_download: bool = False) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(force_download=force_download)
    summary = dataset_summary(df)
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    neg, pos = np.bincount(y_train)
    scale_pos_weight = float(neg / pos)

    estimators = get_base_estimators(scale_pos_weight)
    param_dists = get_param_distributions(scale_pos_weight)

    results = []
    all_best_params = {}
    best_name = None
    best_score = -1.0
    best_pipeline = None

    for name in estimators:
        print(f"  Tuning {name} ...")
        pipeline, tune_info = tune_and_fit(
            name,
            estimators[name],
            param_dists[name],
            preprocessor,
            X_train,
            y_train,
        )
        metrics = evaluate_model(
            name, pipeline, X_test, y_test, X_train, y_train, tune_info
        )
        results.append(metrics)
        all_best_params[name] = tune_info

        joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib")

        composite = 0.6 * metrics["roc_auc"] + 0.4 * metrics["f1"]
        if composite > best_score:
            best_score = composite
            best_name = name
            best_pipeline = pipeline

    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    BEST_PARAMS_PATH.write_text(json.dumps(all_best_params, indent=2, default=str))

    report = {
        "dataset_name": DATASET_NAME,
        "dataset_rows": summary["rows"],
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "positive_rate": summary["positive_rate"],
        "hyperparameter_tuning": True,
        "tune_sample_size": min(TUNE_SAMPLE_SIZE, len(X_train)),
        "tune_n_iter": TUNE_N_ITER,
        "best_model": best_name,
        "models": sorted(results, key=lambda x: x["roc_auc"], reverse=True),
    }

    METRICS_PATH.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    report = train_all(force_download=True)
    print(json.dumps(report, indent=2))
