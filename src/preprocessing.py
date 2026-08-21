from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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
    cleaned = df.copy().replace([np.inf, -np.inf], np.nan).drop_duplicates().reset_index(drop=True)
    identifier_tokens = ("id", "timestamp", "flow_id")
    drop_cols = [c for c in cleaned.columns if c != label_col and (c in identifier_tokens or c.endswith("_id"))]
    return cleaned.drop(columns=drop_cols, errors="ignore")

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=2))])
    return ColumnTransformer([("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)])

def prepare_data(df: pd.DataFrame, label_col: str, test_size: float, random_state: int) -> PreparedData:
    cleaned = clean_dataframe(df, label_col)
    y = make_binary_target(cleaned[label_col])
    X = cleaned.drop(columns=[label_col]).dropna(axis=1, how="all")
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return PreparedData(X_train, X_test, y_train, y_test, build_preprocessor(X_train), numeric, categorical)
