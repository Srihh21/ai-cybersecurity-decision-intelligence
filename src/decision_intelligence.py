import numpy as np
import pandas as pd

def recommended_action(risk_level: str, uncertainty: float) -> str:
    if risk_level == "CRITICAL": return "Immediate Human Review / Isolate or Contain"
    if risk_level == "HIGH": return "Escalate and Investigate"
    if risk_level == "MODERATE": return "Investigate and Monitor"
    if uncertainty >= 0.70: return "Monitor / Human Review for Uncertainty"
    return "Monitor"

def priority_from_risk(risk_level: str) -> str:
    return {"CRITICAL": "P1", "HIGH": "P2", "MODERATE": "P3", "LOW": "P4"}[risk_level]

def generate_decision_outputs(predictions: np.ndarray, risk_df: pd.DataFrame, important_features: pd.Series | None = None) -> pd.DataFrame:
    out = risk_df.copy()
    out.insert(0, "prediction", np.asarray(predictions).astype(int))
    out["risk_level"] = out["risk_level"].astype(str)
    out["decision_priority"] = out["risk_level"].map(priority_from_risk)
    out["human_review_required"] = out["risk_level"].isin(["HIGH", "CRITICAL"]) | (out["uncertainty"] >= 0.70)
    out["recommended_action"] = [recommended_action(r, u) for r, u in zip(out["risk_level"], out["uncertainty"])]
    if important_features is not None:
        values = important_features.astype(str).tolist()
        if len(values) < len(out): values += ["Unavailable"] * (len(out) - len(values))
        out["important_features"] = values[:len(out)]
    else:
        out["important_features"] = "Unavailable"
    out["decision_reason"] = "Risk=" + out["risk_level"] + "; attack_probability=" + out["attack_probability"].round(3).astype(str) + "; uncertainty=" + out["uncertainty"].round(3).astype(str)
    return out
