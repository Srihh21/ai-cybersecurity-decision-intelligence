from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    root_dir: Path = Path(__file__).resolve().parents[1]
    random_state: int = 42
    test_size: float = 0.20
    max_rows: int = 150_000
    shap_sample_size: int = 500

    label_candidates: Tuple[str, ...] = (
        "label", "class", "target", "attack", "attack_cat"
    )

    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "moderate": 0.35,
        "high": 0.60,
        "critical": 0.80,
    })

    decision_weights: Dict[str, float] = field(default_factory=lambda: {
        "attack_probability": 0.50,
        "threat_severity": 0.20,
        "uncertainty": 0.15,
        "criticality": 0.15,
    })

    @property
    def raw_dir(self) -> Path:
        return self.root_dir / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root_dir / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        return self.root_dir / "models"

    @property
    def figures_dir(self) -> Path:
        return self.root_dir / "reports" / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.root_dir / "reports" / "tables"

    @property
    def results_dir(self) -> Path:
        return self.root_dir / "reports" / "results"
