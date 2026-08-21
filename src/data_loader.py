from typing import Iterable, Tuple

import pandas as pd

from .config import ExperimentConfig


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def _find_label_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
    columns = list(columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError("Could not detect target column. Expected one of: " + ", ".join(candidates))


def load_dataset(config: ExperimentConfig) -> Tuple[pd.DataFrame, str]:
    csv_files = sorted(config.raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {config.raw_dir}. Place a public cybersecurity dataset CSV there first. "
            "Supported examples include CIC-IDS2017 and UNSW-NB15."
        )
    frames = []
    for path in csv_files:
        frame = pd.read_csv(path, low_memory=False)
        frames.append(_normalize_columns(frame))
    df = pd.concat(frames, ignore_index=True, sort=False)
    if config.max_rows and len(df) > config.max_rows:
        df = df.sample(config.max_rows, random_state=config.random_state).reset_index(drop=True)
    label_col = _find_label_column(df.columns, config.label_candidates)
    return df, label_col


def make_binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        unique = set(pd.Series(series).dropna().astype(float).unique().tolist())
        if unique.issubset({0.0, 1.0}):
            return series.fillna(0).astype(int)
    normalized = series.astype(str).str.strip().str.lower()
    benign_tokens = {"benign", "normal", "normal.", "0", "false", "legitimate", "non_attack"}
    return (~normalized.isin(benign_tokens)).astype(int)


def dataset_summary(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    target = make_binary_target(df[label_col])
    return pd.DataFrame({
        "metric": ["rows", "columns", "missing_cells", "duplicate_rows", "benign_rows", "attack_rows"],
        "value": [len(df), len(df.columns), int(df.isna().sum().sum()), int(df.duplicated().sum()), int((target == 0).sum()), int((target == 1).sum())],
    })
