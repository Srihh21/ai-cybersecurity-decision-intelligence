from typing import Dict
import numpy as np
import pandas as pd

def calculate_uncertainty(probability: np.ndarray) -> np.ndarray:
    return 1.0 - np.abs(np.asarray(probability) - 0.5) * 2.0

def calculate_decision_score(attack_probability: np.ndarray, weights: Dict[str, float], criticality: float = 0.50) -> pd.DataFrame:
    attack_probability = np.asarray(attack_probability, dtype=float)
    uncertainty = calculate_uncertainty(attack_probability)
    threat_severity = attack_probability.copy()
    criticality_arr = np.full_like(attack_probability, fill_value=criticality)
    score = (weights["attack_probability"] * attack_probability + weights["threat_severity"] * threat_severity + weights["uncertainty"] * uncertainty + weights["criticality"] * criticality_arr)
    return pd.DataFrame({"attack_probability": attack_probability, "threat_severity": threat_severity, "uncertainty": uncertainty, "criticality": criticality_arr, "decision_score": np.clip(score, 0, 1)})

def assign_risk_level(score: pd.Series, thresholds: Dict[str, float]) -> pd.Series:
    return pd.cut(score, bins=[-np.inf, thresholds["moderate"], thresholds["high"], thresholds["critical"], np.inf], labels=["LOW", "MODERATE", "HIGH", "CRITICAL"], right=False).astype(str)
