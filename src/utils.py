import json
import logging
import random
from pathlib import Path
from typing import Any
import numpy as np

def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger("ai_sdi")

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)

def ensure_directories(*paths: Path) -> None:
    for path in paths: path.mkdir(parents=True, exist_ok=True)

def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f: json.dump(data, f, indent=2, default=str)
