"""
HAI (HIL-based Augmented ICS) loader — one-class, windowed.

Source : https://github.com/icsdataset/hai  (HAI 22.04). Files are Git-LFS;
the downloader fetches them from media.githubusercontent.com.

Protocol (matches the BATADAL treatment):
  * train1.csv (attack-free) -> entire one-class training set.
  * test1.csv (labelled)     -> split chronologically 50/50 into val / test.
  * drop constant features by training variance; Min-Max scaling fit on
    training; sliding windows (default W = 10 s so the flattened window
    dimension is comparable to BATADAL), window label = last step.

A tractable subset (train1 + test1) is used so the deep autoencoders train on
CPU in reasonable time; this is stated explicitly in the paper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..dataset import OneClassDataset
from ..windowing import make_windows


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_hai(
    data_dir: str | Path = "data/hai",
    window: int = 10,
    stride: int = 1,
    label_strategy: str = "last",
    drop_constant: bool = True,
    max_train: int = 15000,   # cap normal training windows for tractable CPU fitting
    seed: int = 42,
    verbose: bool = True,
) -> OneClassDataset:
    folder = Path(data_dir)
    ftr, fte = folder / "train1.csv", folder / "test1.csv"
    if not ftr.exists() or not fte.exists():
        raise FileNotFoundError(
            f"Need train1.csv and test1.csv in {folder}. "
            "Run scripts/download_data.py --dataset hai"
        )
    tr, te = _read(ftr), _read(fte)
    lab = [c for c in te.columns if c.lower() == "attack"][0]
    time_cols = [c for c in te.columns if c.lower() in ("timestamp", "time")]
    feat = [c for c in te.columns if c not in time_cols + [lab]]

    def X(df):
        return df[feat].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)

    Xtr_df = X(tr)
    ytr = pd.to_numeric(tr[lab], errors="coerce").fillna(0).astype(int).values
    Xte_df = X(te)
    yte = pd.to_numeric(te[lab], errors="coerce").fillna(0).astype(int).values
    fit_end_rows = int(len(Xtr_df) * 0.80)

    if drop_constant:
        v = Xtr_df.iloc[:fit_end_rows].var()
        keep = v.index[v > 1e-10].tolist()
        feat = keep
        Xtr_df, Xte_df = Xtr_df[keep], Xte_df[keep]
        if verbose:
            print(f"[HAI] kept {len(keep)} non-constant features")

    # Chronological 80/20 normal-only fit/calibration split. Preprocessing is
    # fitted on the detector-fitting prefix only.
    fit_end = fit_end_rows
    Xtr_fit_df, Xtr_cal_df = Xtr_df.iloc[:fit_end], Xtr_df.iloc[fit_end:]
    ytr_fit, ytr_cal = ytr[:fit_end], ytr[fit_end:]
    scaler = MinMaxScaler()
    Xtr_s = scaler.fit_transform(Xtr_fit_df.values).astype(np.float32)
    Xcal_s = scaler.transform(Xtr_cal_df.values).astype(np.float32)
    Xte_s = scaler.transform(Xte_df.values).astype(np.float32)

    # train1 is attack-free -> all training windows are normal
    Xtr, ytr_w = make_windows(Xtr_s, ytr_fit, window, stride, label_strategy)
    Xcal, ycal_w = make_windows(Xcal_s, ytr_cal, window, stride, label_strategy)
    normal = ytr_w == 0
    Xtr, ytr_w = Xtr[normal], ytr_w[normal]
    cal_normal = ycal_w == 0
    Xcal, ycal_w = Xcal[cal_normal], ycal_w[cal_normal]
    # subsample normal training windows for tractable CPU fitting (full val/test kept)
    if max_train and len(Xtr) > max_train:
        rng = np.random.RandomState(seed)
        sel = rng.choice(len(Xtr), max_train, replace=False)
        Xtr, ytr_w = Xtr[sel], ytr_w[sel]
        if verbose:
            print(f"[HAI] subsampled training windows to {max_train}")

    # test1 -> chronological 50/50 val/test
    half = len(Xte_s) // 2
    Xval, yval = make_windows(Xte_s[:half], yte[:half], window, stride, label_strategy)
    Xte, ytev = make_windows(Xte_s[half:], yte[half:], window, stride, label_strategy)

    ds = OneClassDataset(
        X_train=Xtr, y_train=ytr_w, X_cal=Xcal, y_cal=ycal_w,
        X_val=Xval, y_val=yval, X_test=Xte, y_test=ytev,
        feature_names=feat, name="HAI", windowed=True,
    )
    if verbose:
        for k, v in ds.info().items():
            print(f"  {k:>14}: {v}")
    return ds

