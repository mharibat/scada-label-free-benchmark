"""
BATADAL loader — reproduces the PAPER's protocol exactly.

Protocol (paper Section V-B):
  * dataset03 (8,761 normal rows)  -> ENTIRE training set (one-class, all normal).
  * dataset04 (4,177 rows, labelled) -> split CHRONOLOGICALLY 50/50 into
    validation and test.
  * Pre-processing:
      - strip whitespace from headers,
      - ATT_FLAG: replace -999 with 0, then (>0)->1,
      - drop constant features using TRAINING variance (expected: drop 7 -> keep 36),
      - Min-Max scaling fit on TRAINING only,
      - sliding windows W=24, stride=1, label = last step.

Expected shapes (paper): 8,738 / 2,065 / 2,066 windows (train/val/test),
anomaly rate 0% / ~2.0% / ~8.6%.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..dataset import OneClassDataset
from ..windowing import make_windows

LABEL_CANDIDATES = ["ATT_FLAG", "att_flag", "attack", "Attack"]


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for cand in LABEL_CANDIDATES:
        if cand in df.columns:
            df[cand] = df[cand].replace(-999, 0)
            df[cand] = (pd.to_numeric(df[cand], errors="coerce").fillna(0) > 0).astype(int)
            df.rename(columns={cand: "ATT_FLAG"}, inplace=True)
            break
    return df


def build_batadal(
    data_dir: str | Path = "data/batadal",
    window: int = 24,
    stride: int = 1,
    label_strategy: str = "last",
    drop_constant: bool = True,
    verbose: bool = True,
) -> OneClassDataset:
    folder = Path(data_dir)
    f03 = folder / "BATADAL_dataset03.csv"
    f04 = folder / "BATADAL_dataset04.csv"
    if not f03.exists() or not f04.exists():
        raise FileNotFoundError(
            f"Need BATADAL_dataset03.csv and BATADAL_dataset04.csv in {folder}. "
            "Run scripts/download_data.py --dataset batadal"
        )

    df03 = _read(f03)
    df04 = _read(f04)
    if "ATT_FLAG" not in df03.columns:
        df03["ATT_FLAG"] = 0  # dataset03 is attack-free

    time_cols = [c for c in df03.columns if c.upper() in ("DATETIME", "TIME", "TIMESTAMP")]
    feature_cols = [c for c in df03.columns if c not in time_cols + ["ATT_FLAG"]]

    def feats(df):
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return X.astype(np.float32)

    X03, y03 = feats(df03), df03["ATT_FLAG"].astype(int).values
    X04, y04 = feats(df04), df04["ATT_FLAG"].astype(int).values
    fit_end_rows = int(len(X03) * 0.80)

    # ---- drop constant features using TRAINING (dataset03) variance ----
    if drop_constant:
        train_var = X03.iloc[:fit_end_rows].var()
        keep = train_var.index[train_var > 1e-10].tolist()
        dropped = [c for c in feature_cols if c not in keep]
        X03, X04, feature_cols = X03[keep], X04[keep], keep
        if verbose:
            print(f"[BATADAL] dropped {len(dropped)} constant feature(s); kept {len(keep)}")

    # ---- Chronological normal-only fit/calibration split (80/20) ----
    # The calibration tail is not used to fit the scaler or detector.
    fit_end = fit_end_rows
    X03_fit, y03_fit = X03.iloc[:fit_end], y03[:fit_end]
    X03_cal, y03_cal = X03.iloc[fit_end:], y03[fit_end:]

    # ---- Min-Max scaling fit on detector-fitting normals only ----
    scaler = MinMaxScaler()
    X03_fit_s = scaler.fit_transform(X03_fit.values).astype(np.float32)
    X03_cal_s = scaler.transform(X03_cal.values).astype(np.float32)
    X04s = scaler.transform(X04.values).astype(np.float32)

    # ---- chronological 50/50 split of dataset04 into val / test ----
    half = len(X04s) // 2
    Xval_raw, yval_raw = X04s[:half], y04[:half]
    Xte_raw, yte_raw = X04s[half:], y04[half:]

    # ---- windows ----
    Xtr, ytr = make_windows(X03_fit_s, y03_fit, window, stride, label_strategy)
    Xcal, ycal = make_windows(X03_cal_s, y03_cal, window, stride, label_strategy)
    Xval, yval = make_windows(Xval_raw, yval_raw, window, stride, label_strategy)
    Xte, yte = make_windows(Xte_raw, yte_raw, window, stride, label_strategy)

    # training must be one-class (normal only); dataset03 is normal, but be safe
    normal = ytr == 0
    Xtr, ytr = Xtr[normal], ytr[normal]
    cal_normal = ycal == 0
    Xcal, ycal = Xcal[cal_normal], ycal[cal_normal]

    ds = OneClassDataset(
        X_train=Xtr, y_train=ytr, X_cal=Xcal, y_cal=ycal,
        X_val=Xval, y_val=yval, X_test=Xte, y_test=yte,
        feature_names=feature_cols, name="BATADAL", windowed=True,
    )
    if verbose:
        for k, v in ds.info().items():
            print(f"  {k:>14}: {v}")
    return ds

