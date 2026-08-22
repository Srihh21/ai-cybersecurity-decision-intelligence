from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import make_binary_target


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    numeric_features: List[str]
    categorical_features: List[str]


def clean_dataframe(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    cleaned = df.copy().replace([np.inf, -np.inf], np.nan)
    identifier_tokens = ("id", "timestamp", "flow_id")
    drop_cols = [
        c
        for c in cleaned.columns
        if c != label_col and (c in identifier_tokens or c.endswith("_id"))
    ]
    return cleaned.drop(columns=drop_cols, errors="ignore").reset_index(drop=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)]
    )


def _features_and_target(df: pd.DataFrame, label_col: str):
    cleaned = clean_dataframe(df, label_col)
    y = make_binary_target(cleaned[label_col]).reset_index(drop=True)

    # attack_cat describes the ground-truth attack family and would leak the
    # answer into a binary intrusion detector. Keep it for descriptive analysis,
    # but exclude it from model inputs along with the target label.
    leakage_cols = [label_col]
    if "attack_cat" in cleaned.columns:
        leakage_cols.append("attack_cat")

    X = cleaned.drop(columns=leakage_cols, errors="ignore").dropna(axis=1, how="all")
    return X.reset_index(drop=True), y


def prepare_official_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
) -> PreparedData:
    """Prepare UNSW-NB15 while preserving its official train/test partition."""
    X_train, y_train = _features_and_target(train_df, label_col)
    X_test, y_test = _features_and_target(test_df, label_col)

    train_columns = set(X_train.columns)
    test_columns = set(X_test.columns)
    if train_columns != test_columns:
        raise ValueError(
            "Prepared train/test feature schemas do not match. "
            f"Train-only: {sorted(train_columns - test_columns)}; "
            f"Test-only: {sorted(test_columns - train_columns)}"
        )

    X_test = X_test[X_train.columns]
    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=build_preprocessor(X_train),
        numeric_features=numeric,
        categorical_features=categorical,
    )
