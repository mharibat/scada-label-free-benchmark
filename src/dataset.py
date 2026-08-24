"""
Standard one-class dataset container returned by every loader.

Convention for one-class (unsupervised) learning:
  * X_train / y_train  -> NORMAL-only training samples (y_train is all zeros).
  * X_val   / y_val    -> validation split WITH anomalies (for threshold tuning).
  * X_test  / y_test   -> held-out test split WITH anomalies (final evaluation).

`X_*` may be:
  * 3-D  (N, window, n_features)  -> windowed (BATADAL), or
  * 2-D  (N, n_features)          -> tabular (Gas Pipeline).

Detectors that need a flat matrix call `.flat()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class OneClassDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str] = field(default_factory=list)
    name: str = ""
    windowed: bool = False

    @staticmethod
    def _flat(a: np.ndarray) -> np.ndarray:
        return a.reshape(a.shape[0], -1) if a.ndim == 3 else a

    def flat(self):
        return (
            self._flat(self.X_train),
            self._flat(self.X_val),
            self._flat(self.X_test),
        )

    def info(self) -> dict:
        def rate(y):
            return round(float(np.mean(y)) * 100, 2)
        return {
            "dataset": self.name,
            "windowed": self.windowed,
            "n_features": self.X_train.shape[-1],
            "window": self.X_train.shape[1] if self.windowed else 1,
            "train_n": int(len(self.y_train)),
            "val_n": int(len(self.y_val)),
            "test_n": int(len(self.y_test)),
            "train_anom_%": rate(self.y_train),
            "val_anom_%": rate(self.y_val),
            "test_anom_%": rate(self.y_test),
        }
