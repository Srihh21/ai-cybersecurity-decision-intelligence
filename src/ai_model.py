from pathlib import Path
from typing import Dict
import joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

def build_models(preprocessor, random_state: int) -> Dict[str, Pipeline]:
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample", n_jobs=-1, random_state=random_state),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state=random_state, n_jobs=-1),
    }
    return {name: Pipeline([("preprocessor", clone(preprocessor)), ("model", estimator)]) for name, estimator in estimators.items()}

def train_models(models, X_train, y_train):
    trained = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        trained[name] = pipeline
    return trained

def save_models(models, models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, models_dir / f"{name}.joblib")
