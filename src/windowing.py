"""
Windowing and feature utilities shared by the loaders.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    window: int = 24,
    stride: int = 1,
    label_strategy: str = "last",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert (T, F) time series into (N, window, F) sliding windows.

    label_strategy:
      'last'     -> window label = label of its last step (paper default)
      'any'      -> positive iff any step is anomalous
      'majority' -> positive iff > 50% of steps are anomalous
    """
    if label_strategy not in {"last", "any", "majority"}:
        raise ValueError(f"Unknown label_strategy: {label_strategy}")
    T, F = X.shape
    n = (T - window) // stride + 1
    if n <= 0:
        raise ValueError(f"Series length {T} shorter than window {window}")
    Xw = np.empty((n, window, F), dtype=np.float32)
    yw = np.empty(n, dtype=np.int64)
    for i in range(n):
        s = i * stride
        e = s + window
        Xw[i] = X[s:e]
        seg = y[s:e]
        if label_strategy == "last":
            yw[i] = int(seg[-1])
        elif label_strategy == "any":
            yw[i] = int(seg.max() > 0)
        else:
            yw[i] = int(seg.mean() > 0.5)
    return Xw, yw


def remove_constant_features(
    X: pd.DataFrame, train_mask: np.ndarray, threshold: float = 1e-10
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns whose variance on the TRAINING rows is below threshold."""
    train_var = X.loc[train_mask].var(numeric_only=True)
    keep = train_var.index[train_var > threshold].tolist()
    dropped = [c for c in X.columns if c not in keep]
    return X[keep].copy(), dropped
