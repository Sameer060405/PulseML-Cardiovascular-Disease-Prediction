"""Hyperparameter search grids for each classifier."""

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def get_param_distributions(scale_pos_weight: float) -> dict:
    return {
        "logistic_regression": {
            "classifier__C": loguniform(1e-2, 1e2),
            "classifier__solver": ["lbfgs", "saga"],
        },
        "random_forest": {
            "classifier__n_estimators": randint(150, 400),
            "classifier__max_depth": [10, 14, 18, 22, None],
            "classifier__min_samples_leaf": randint(2, 12),
            "classifier__max_features": ["sqrt", "log2", 0.6],
        },
        "gradient_boosting": {
            "classifier__n_estimators": randint(100, 250),
            "classifier__learning_rate": uniform(0.03, 0.12),
            "classifier__max_depth": randint(3, 7),
            "classifier__subsample": uniform(0.75, 0.2),
        },
        "xgboost": {
            "classifier__n_estimators": randint(150, 400),
            "classifier__max_depth": randint(4, 10),
            "classifier__learning_rate": uniform(0.03, 0.15),
            "classifier__subsample": uniform(0.7, 0.25),
            "classifier__colsample_bytree": uniform(0.7, 0.25),
            "classifier__scale_pos_weight": [scale_pos_weight * 0.8, scale_pos_weight, scale_pos_weight * 1.2],
        },
        "hist_gradient_boosting": {
            "classifier__max_depth": randint(4, 12),
            "classifier__learning_rate": uniform(0.03, 0.12),
            "classifier__max_leaf_nodes": randint(31, 127),
            "classifier__l2_regularization": loguniform(1e-4, 1e-1),
        },
    }


def get_base_estimators(scale_pos_weight: float) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "xgboost": XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        ),
    }
