"""Generate EDA and model comparison figures."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.config import METRICS_PATH, MODELS_DIR, REPORTS_DIR
from src.data_loader import load_dataset
from src.preprocess import split_data

sns.set_theme(style="whitegrid", palette="muted")


def plot_eda(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    df["target"].value_counts().sort_index().plot(
        kind="bar",
        ax=ax,
        color=["#4C78A8", "#F58518"],
    )
    ax.set_title("Target Distribution (0 = No Disease, 1 = Cardiovascular Disease)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "target_distribution.png", dpi=150)
    plt.close(fig)

    numeric_cols = ["age_years", "bmi", "ap_hi", "ap_lo", "weight"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for ax, col in zip(axes, numeric_cols):
        sns.histplot(data=df, x=col, hue="target", kde=True, ax=ax, element="step")
        ax.set_title(col)
    axes[-1].axis("off")
    fig.suptitle("Feature Distributions by Disease Class", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_distributions.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(metrics_path: Path, out_dir: Path) -> None:
    report = json.loads(metrics_path.read_text())
    df = pd.DataFrame(report["models"])

    fig, ax = plt.subplots(figsize=(10, 6))
    df_melt = df.melt(
        id_vars=["model"],
        value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
        var_name="metric",
        value_name="score",
    )
    sns.barplot(data=df_melt, x="model", y="score", hue="metric", ax=ax)
    ax.set_title("Tuned Model Performance Comparison")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df.sort_values("cv_roc_auc_mean", ascending=False),
        x="model",
        y="cv_roc_auc_mean",
        ax=ax,
        color="#72B7B2",
    )
    ax.set_title("5-Fold Cross-Validation ROC-AUC (Full Train Set)")
    ax.set_ylim(0.5, 1.0)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "cv_roc_auc.png", dpi=150)
    plt.close(fig)


def plot_evaluation_curves(out_dir: Path) -> None:
    import joblib

    from src.config import BEST_MODEL_PATH

    df = load_dataset()
    _, X_test, _, y_test = split_data(df)
    pipeline = joblib.load(BEST_MODEL_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=axes[0])
    axes[0].set_title("ROC Curve — Best Tuned Model")
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, ax=axes[1])
    axes[1].set_title("Confusion Matrix — Best Tuned Model")
    fig.tight_layout()
    fig.savefig(out_dir / "best_model_evaluation.png", dpi=150)
    plt.close(fig)


def generate_all_figures() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    plot_eda(df, REPORTS_DIR)
    if METRICS_PATH.exists():
        plot_model_comparison(METRICS_PATH, REPORTS_DIR)
    if (MODELS_DIR / "best_model.joblib").exists():
        plot_evaluation_curves(REPORTS_DIR)


if __name__ == "__main__":
    generate_all_figures()
