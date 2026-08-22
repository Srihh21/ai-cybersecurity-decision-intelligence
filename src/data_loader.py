from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from .config import ExperimentConfig


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def _find_label_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
    columns = list(columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        "Could not detect target column. Expected one of: " + ", ".join(candidates)
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    return _normalize_columns(pd.read_csv(path, low_memory=False))


def load_unsw_nb15_official_split(
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load the official pre-partitioned UNSW-NB15 train/test CSV files.

    Expected files under data/raw/:
      - UNSW_NB15_training-set.csv
      - UNSW_NB15_testing-set.csv

    The official split is preserved; the training and testing files are not
    concatenated or randomly re-split.
    """
    train_path = config.raw_dir / "UNSW_NB15_training-set.csv"
    test_path = config.raw_dir / "UNSW_NB15_testing-set.csv"

    train_df = _read_csv(train_path)
    test_df = _read_csv(test_path)

    label_col = _find_label_column(train_df.columns, config.label_candidates)
    if label_col not in test_df.columns:
        raise ValueError(f"Target column '{label_col}' is missing from the test set.")

    train_only = set(train_df.columns) - set(test_df.columns)
    test_only = set(test_df.columns) - set(train_df.columns)
    if train_only or test_only:
        raise ValueError(
            "Training/testing schemas do not match. "
            f"Train-only columns: {sorted(train_only)}; "
            f"Test-only columns: {sorted(test_only)}"
        )

    return train_df, test_df, label_col


def make_binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        unique = set(pd.Series(series).dropna().astype(float).unique().tolist())
        if unique.issubset({0.0, 1.0}):
            return series.fillna(0).astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    benign_tokens = {
        "benign",
        "normal",
        "normal.",
        "0",
        "false",
        "legitimate",
        "non_attack",
    }
    return (~normalized.isin(benign_tokens)).astype(int)


def dataset_summary(df: pd.DataFrame, label_col: str, split_name: str) -> pd.DataFrame:
    target = make_binary_target(df[label_col])
    return pd.DataFrame(
        {
            "split": [split_name] * 6,
            "metric": [
                "rows",
                "columns",
                "missing_cells",
                "duplicate_rows",
                "benign_rows",
                "attack_rows",
            ],
            "value": [
                len(df),
                len(df.columns),
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
                int((target == 0).sum()),
                int((target == 1).sum()),
            ],
        }
    )
