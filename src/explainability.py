from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

def _dense(matrix): return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)

def explain_model(pipeline, X_sample: pd.DataFrame, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    preprocessor = pipeline.named_steps["preprocessor"]; model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(X_sample); feature_names = preprocessor.get_feature_names_out(); dense = _dense(transformed)
    if dense.shape[0] > 500: dense = dense[:500]
    try:
        if not hasattr(model, "feature_importances_"): raise TypeError("Selected model is not tree-based")
        explainer = shap.TreeExplainer(model); shap_values = explainer.shap_values(dense)
        if isinstance(shap_values, list): shap_values = shap_values[-1]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3: shap_values = shap_values[:, :, -1]
        shap.summary_plot(shap_values, dense, feature_names=feature_names, show=False, max_display=15); plt.tight_layout(); plt.savefig(output_dir / "shap_summary.png", dpi=300, bbox_inches="tight"); plt.close()
        importance = np.abs(shap_values).mean(axis=0)
        top_table = pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance}).sort_values("mean_abs_shap", ascending=False).head(20)
        top_idx = np.argsort(np.abs(shap_values), axis=1)[:, -3:][:, ::-1]
        per_row = pd.DataFrame({"important_features": [", ".join(feature_names[idx].tolist()) for idx in top_idx]})
    except Exception:
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_); top_table = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(8, 6)); plot_df = top_table.head(15).sort_values(top_table.columns[-1]); ax.barh(plot_df["feature"], plot_df.iloc[:, -1]); ax.set_title("Feature Importance"); plt.tight_layout(); plt.savefig(output_dir / "feature_importance.png", dpi=300); plt.close()
        else: top_table = pd.DataFrame(columns=["feature", "importance"])
        per_row = pd.DataFrame({"important_features": ["Unavailable"] * len(dense)})
    return top_table.reset_index(drop=True), per_row
