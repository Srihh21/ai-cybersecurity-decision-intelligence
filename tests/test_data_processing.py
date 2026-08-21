import pandas as pd
from src.data_loader import make_binary_target
from src.cyber_risk import assign_risk_level, calculate_decision_score

def test_binary_target_mapping():
    y = make_binary_target(pd.Series(["BENIGN", "Attack", "normal", "DoS"])); assert y.tolist() == [0, 1, 0, 1]

def test_decision_score_bounds():
    risk = calculate_decision_score([0.1, 0.5, 0.9], {"attack_probability": 0.50, "threat_severity": 0.20, "uncertainty": 0.15, "criticality": 0.15}); assert risk["decision_score"].between(0, 1).all()

def test_risk_levels():
    levels = assign_risk_level(pd.Series([0.1, 0.4, 0.7, 0.9]), {"moderate": 0.35, "high": 0.60, "critical": 0.80}); assert levels.tolist() == ["LOW", "MODERATE", "HIGH", "CRITICAL"]
