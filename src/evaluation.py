from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve

def evaluate_models(models, X_test, y_test) -> Tuple[pd.DataFrame, Dict]:
    rows, predictions = [], {}
    for name, model in models.items():
        pred = model.predict(X_test); prob = model.predict_proba(X_test)[:, 1]
        rows.append({"model": name, "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0), "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0), "roc_auc": roc_auc_score(y_test, prob), "pr_auc": average_precision_score(y_test, prob)})
        predictions[name] = {"pred": pred, "prob": prob}
    metrics = pd.DataFrame(rows).sort_values(["f1", "pr_auc", "roc_auc"], ascending=False).reset_index(drop=True)
    return metrics, predictions

def plot_model_comparison(metrics: pd.DataFrame, output: Path) -> None:
    ax = metrics.set_index("model")[["precision", "recall", "f1", "roc_auc", "pr_auc"]].plot(kind="bar", figsize=(10, 6)); ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title("Model Performance Comparison"); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.savefig(output, dpi=300); plt.close()

def plot_confusion(y_true, pred, output: Path) -> None:
    cm = confusion_matrix(y_true, pred); fig, ax = plt.subplots(figsize=(5, 4)); image = ax.imshow(cm); fig.colorbar(image, ax=ax)
    for (i, j), value in np.ndenumerate(cm): ax.text(j, i, int(value), ha="center", va="center")
    ax.set_xticks([0, 1], labels=["Benign", "Attack"]); ax.set_yticks([0, 1], labels=["Benign", "Attack"]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix"); plt.tight_layout(); plt.savefig(output, dpi=300); plt.close()

def plot_roc_pr(y_true, prob, roc_output: Path, pr_output: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, prob); fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, prob):.3f}"); ax.plot([0, 1], [0, 1], linestyle="--"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve"); ax.legend(); plt.tight_layout(); plt.savefig(roc_output, dpi=300); plt.close()
    precision, recall, _ = precision_recall_curve(y_true, prob); fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(recall, precision, label=f"AP={average_precision_score(y_true, prob):.3f}"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall Curve"); ax.legend(); plt.tight_layout(); plt.savefig(pr_output, dpi=300); plt.close()
