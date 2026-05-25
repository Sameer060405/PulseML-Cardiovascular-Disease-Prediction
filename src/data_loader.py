"""Download and prepare the public cardiovascular disease dataset (~70k rows)."""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from src.config import (
    DATA_PATH,
    DATASET_URL,
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COL,
)


def _download_raw(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATASET_URL} ...")
    urlretrieve(DATASET_URL, path)


def _clean_cardio(df: pd.DataFrame) -> pd.DataFrame:
    """Standard cleaning rules for the cardio_train dataset."""
    df = df.drop_duplicates().copy()

    # Fix swapped systolic/diastolic pressure
    swap_mask = df["ap_hi"] < df["ap_lo"]
    df.loc[swap_mask, ["ap_hi", "ap_lo"]] = df.loc[swap_mask, ["ap_lo", "ap_hi"]].values

    df = df[
        (df["ap_hi"] >= 90)
        & (df["ap_hi"] <= 250)
        & (df["ap_lo"] >= 60)
        & (df["ap_lo"] <= 150)
        & (df["height"] >= 140)
        & (df["height"] <= 220)
        & (df["weight"] >= 40)
        & (df["weight"] <= 200)
    ]

    df["age_years"] = (df["age"] / 365.25).round().astype(int)
    df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(1)
    df[TARGET_COL] = df["cardio"].astype(int)

    return df


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[FEATURE_COLUMNS + [TARGET_COL]].copy()
    out = out.dropna()
    for col in FEATURE_COLUMNS:
        if col == "bmi":
            out[col] = out[col].astype(float)
        elif col in ("age_years", "height", "weight", "ap_hi", "ap_lo"):
            out[col] = out[col].astype(int)
        else:
            out[col] = out[col].astype(int)
    return out.reset_index(drop=True)


def load_dataset(path: Path | None = None, force_download: bool = False) -> pd.DataFrame:
    path = path or DATA_PATH

    if force_download and path.exists():
        path.unlink()

    if not path.exists():
        raw_path = path.with_suffix(".raw.csv")
        _download_raw(raw_path)
        raw = pd.read_csv(raw_path, sep=";")
        df = _prepare_features(_clean_cardio(raw))
        df.to_csv(path, index=False)
        raw_path.unlink(missing_ok=True)
    else:
        df = pd.read_csv(path)

    missing = set(FEATURE_COLUMNS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    return df[FEATURE_COLUMNS + [TARGET_COL]].copy()


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "features": len(FEATURE_COLUMNS),
        "positive_rate": round(float(df[TARGET_COL].mean()), 4),
        "age_range": [int(df["age_years"].min()), int(df["age_years"].max())],
    }
