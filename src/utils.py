"""
Shared utilities: reproducibility, config loading, small helpers.

This module was MISSING in the first release (scripts imported `set_seed`,
`ensure_dir`, `load_config` from here). It is now provided.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds we can reach for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic CPU behaviour
        torch.use_deterministic_algorithms(False)  # LSTM cudnn not applicable on CPU
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed; return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file into a dict."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mean_std(values) -> tuple[float, float]:
    """Return (mean, sample-std) of a list of floats; std=0 for a single value."""
    a = np.asarray(list(values), dtype=float)
    if a.size <= 1:
        return float(a.mean()) if a.size else float("nan"), 0.0
    return float(a.mean()), float(a.std(ddof=1))
