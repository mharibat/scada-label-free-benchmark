"""
Gas Pipeline (Morris / Turnipseed, Mississippi State) loader.

This loader was MISSING in the first release even though Gas Pipeline is one of
the two core datasets. It parses the original ARFF file `IanArffDataset.arff`.

Protocol (paper Section III-B and V-B):
  * 17 telemetry features; 3 label columns (binary / categorized / specific).
  * Missing values encoded as '?' -> imputed with 0.
  * RobustScaler (handles large-magnitude fields such as 'address') then
    clip to [-5, +5] and Min-Max rescale to [0, 1].
  * No windowing (packet-level, not a regular time series).
  * Chronological 70/10/20 split. Training restricted to NORMAL packets only
    (one-class). Validation and test keep their natural anomaly rates.

Expected (paper): train ~149,996 normal packets; test 54,927 packets with
~21.9% anomalies (12,009 attacks).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from ..dataset import OneClassDataset

# 20 ARFF attributes in order; last 3 are labels.
ARFF_COLUMNS = [
    "address", "function", "length", "setpoint", "gain", "reset_rate",
    "deadband", "cycle_time", "rate", "system_mode", "control_scheme",
    "pump", "solenoid", "pressure_measurement", "crc_rate", "command_response",
    "time", "binary_result", "categorized_result", "specific_result",
]
FEATURE_COLUMNS = ARFF_COLUMNS[:17]   # 17 features
LABEL_COLUMN = "binary_result"


def _read_arff(path: Path) -> pd.DataFrame:
    """Minimal ARFF reader: skip @-header, parse @data rows, '?' -> NaN."""
    rows: List[List[str]] = []
    started = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not started:
                if s.lower() == "@data":
                    started = True
                continue
            if not s or s.startswith("%"):
                continue
            rows.append(s.split(","))
    df = pd.DataFrame(rows, columns=ARFF_COLUMNS)
    df = df.replace("?", np.nan)
    return df


def build_gas_pipeline(
    data_dir: str | Path = "data/gas",
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    clip: float = 5.0,
    verbose: bool = True,
) -> OneClassDataset:
    folder = Path(data_dir)
    arff = folder / "IanArffDataset.arff"
    if not arff.exists():
        raise FileNotFoundError(
            f"Need IanArffDataset.arff in {folder}. "
            "Run scripts/download_data.py --dataset gas_pipeline"
        )

    df = _read_arff(arff)
    y = (pd.to_numeric(df[LABEL_COLUMN], errors="coerce").fillna(0) > 0).astype(int).values

    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0.0).astype(np.float32).values   # '?' -> 0 imputation

    T = len(X)
    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)
    tr_idx = np.arange(0, n_train)
    va_idx = np.arange(n_train, n_train + n_val)
    te_idx = np.arange(n_train + n_val, T)

    # one-class: training uses NORMAL packets only
    tr_normal = tr_idx[y[tr_idx] == 0]

    # RobustScaler fit on training-normal, then clip + Min-Max to [0,1]
    rs = RobustScaler()
    rs.fit(X[tr_normal])

    def transform(A):
        A = rs.transform(A)
        A = np.clip(A, -clip, clip)
        return A

    Xtr = transform(X[tr_normal])
    Xva = transform(X[va_idx])
    Xte = transform(X[te_idx])

    mm = MinMaxScaler()
    mm.fit(Xtr)
    Xtr = mm.transform(Xtr).astype(np.float32)
    Xva = mm.transform(Xva).astype(np.float32)
    Xte = mm.transform(Xte).astype(np.float32)

    ds = OneClassDataset(
        X_train=Xtr, y_train=np.zeros(len(Xtr), dtype=np.int64),
        X_val=Xva, y_val=y[va_idx].astype(np.int64),
        X_test=Xte, y_test=y[te_idx].astype(np.int64),
        feature_names=FEATURE_COLUMNS, name="Gas Pipeline", windowed=False,
    )
    if verbose:
        for k, v in ds.info().items():
            print(f"  {k:>14}: {v}")
    return ds


def build_gas_supervised(
    data_dir: str | Path = "data/gas",
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    clip: float = 5.0,
):
    """
    SUPERVISED counterpart on the SAME chronological test split, for an
    internally-controlled 'unsupervised cost' measurement. Training uses the
    first 70% of packets WITH labels (attacks included); the test split is
    identical to the one-class loader's test split.

    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    folder = Path(data_dir)
    df = _read_arff(folder / "IanArffDataset.arff")
    y = (pd.to_numeric(df[LABEL_COLUMN], errors="coerce").fillna(0) > 0).astype(int).values
    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32).values

    T = len(X)
    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)
    tr_idx = np.arange(0, n_train)                       # WITH anomalies (supervised)
    va_idx = np.arange(n_train, n_train + n_val)
    te_idx = np.arange(n_train + n_val, T)

    rs = RobustScaler().fit(X[tr_idx])
    def tf(A):
        return np.clip(rs.transform(A), -clip, clip)
    Xtr, Xva, Xte = tf(X[tr_idx]), tf(X[va_idx]), tf(X[te_idx])
    mm = MinMaxScaler().fit(Xtr)
    return (mm.transform(Xtr).astype(np.float32), y[tr_idx].astype(np.int64),
            mm.transform(Xva).astype(np.float32), y[va_idx].astype(np.int64),
            mm.transform(Xte).astype(np.float32), y[te_idx].astype(np.int64))
