"""One-command entry point: download data, tune models, generate figures."""

import json
from pathlib import Path

from src.config import DATA_PATH
from src.train import train_all
from src.visualize import generate_all_figures


def main() -> None:
    print("=" * 60)
    print("PulseML — Full pipeline (70k dataset + hyperparameter tuning)")
    print("=" * 60)

    if DATA_PATH.exists():
        print("Removing cached dataset to fetch fresh copy...")
        DATA_PATH.unlink()

    report = train_all(force_download=True)
    print("\nTraining complete. Best model:", report["best_model"])
    print(f"Dataset rows: {report['dataset_rows']:,}")
    print("\nTuned model metrics:")
    for m in report["models"]:
        print(
            f"  {m['model']:26s}  ROC-AUC={m['roc_auc']:.4f}  "
            f"F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}  CV-AUC={m['cv_roc_auc_mean']:.4f}"
        )

    print("\nGenerating figures...")
    generate_all_figures()
    print("Done. Launch app: streamlit run app/streamlit_app.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
