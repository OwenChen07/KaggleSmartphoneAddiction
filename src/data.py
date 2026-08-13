"""Data loading and column definitions for Playground S6E8.

The loaders stay faithful to what is on disk: missing values are preserved as
NaN and are handled inside the model pipelines, never here. Anything that fills
or encodes belongs in `preprocessing.py` so it is fit inside a CV fold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

ID_COL = "id"
TARGET = "addicted_label"

NUMERIC_COLS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]

CATEGORICAL_COLS = [
    "gender",
    "stress_level",
    "academic_work_impact",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

_DTYPES = {c: "float32" for c in NUMERIC_COLS} | {c: "object" for c in CATEGORICAL_COLS}


def load_train(sample: int | None = None, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (X, y). `sample` draws a random subset for fast iteration only —
    nothing sampled may be written to experiments/log.csv."""
    df = pd.read_csv(DATA_DIR / "train.csv", dtype=_DTYPES)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
    y = df[TARGET].to_numpy(dtype=np.int8)
    return df[FEATURE_COLS], y


def load_test() -> tuple[pd.DataFrame, np.ndarray]:
    """Return (X_test, test_ids)."""
    df = pd.read_csv(DATA_DIR / "test.csv", dtype=_DTYPES)
    return df[FEATURE_COLS], df[ID_COL].to_numpy()


def missingness_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Binary is-missing indicators for every feature column.

    Used to test whether missingness itself carries signal (MNAR) rather than
    being incidental. See reports/eda.md.
    """
    return X[FEATURE_COLS].isna().astype(np.int8).add_suffix("__isna")
