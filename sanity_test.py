"""
Smoke test — verifies the pipeline works WITHOUT the real datasets, on a
small synthetic one-class problem. Runs in a few seconds.

    python sanity_test.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

import numpy as np

from src.dataset import OneClassDataset
from src.detectors import make_detector
from src.metrics import binary_metrics, best_threshold_by_f1
from src.utils import set_seed


def make_synth(n_train=1500, n_eval=600, F=8, seed=42):
    set_seed(seed)
    rng = np.random.RandomState(seed)
    Xtr = rng.randn(n_train, F).astype(np.float32) * 0.5           # normal only
    Xev = rng.randn(n_eval, F).astype(np.float32) * 0.5
    y = np.zeros(n_eval, dtype=np.int64)
    k = n_eval // 5
    Xev[:k] += rng.randn(k, F) * 3.0                               # inject anomalies
    y[:k] = 1
    idx = rng.permutation(n_eval)
    return Xtr, Xev[idx], y[idx]


def main():
    print("=" * 60)
    print("SCADA one-class benchmark — sanity test (synthetic)")
    print("=" * 60)
    Xtr, Xev, y = make_synth()
    half = len(Xev) // 2
    ds = OneClassDataset(
        X_train=Xtr[:1200], y_train=np.zeros(1200, int),
        X_cal=Xtr[1200:], y_cal=np.zeros(len(Xtr) - 1200, int),
        X_val=Xev[:half], y_val=y[:half],
        X_test=Xev[half:], y_test=y[half:],
        feature_names=[f"f{i}" for i in range(Xtr.shape[1])],
        name="synthetic", windowed=False,
    )
    for key in ["mahalanobis", "pca", "isoforest", "ocsvm", "dense_ae_compact"]:
        det = make_detector(key, seed=42)
        if hasattr(det, "epochs"):
            det.epochs = 10
        det.fit(ds.X_train)
        vs, ts = det.score(ds.X_val), det.score(ds.X_test)
        thr, _ = best_threshold_by_f1(ds.y_val, vs)
        m = binary_metrics(ds.y_test, ts, thr)
        print(f"  {key:18s} F1={m['f1']:.3f}  AUC={m['auc']:.3f}  "
              f"(fit {getattr(det,'fit_time_',0):.2f}s)")
    Xtr_w = ds.X_train.reshape(len(ds.X_train), 4, 2)
    Xval_w = ds.X_val.reshape(len(ds.X_val), 4, 2)
    Xtest_w = ds.X_test.reshape(len(ds.X_test), 4, 2)
    for key in ["dense_ae", "cnn_ae", "lstm_ae", "transformer_ae"]:
        det = make_detector(key, seed=42, epochs=2)
        det.fit(Xtr_w)
        vs, ts = det.score(Xval_w), det.score(Xtest_w)
        thr, _ = best_threshold_by_f1(ds.y_val, vs)
        m = binary_metrics(ds.y_test, ts, thr)
        print(f"  {key:18s} F1={m['f1']:.3f}  AUC={m['auc']:.3f}  "
              f"(fit {getattr(det,'fit_time_',0):.2f}s)")
    print("\nSanity test complete (all detector families import, fit, and score)")


if __name__ == "__main__":
    main()

