"""Run leakage-aware threshold and operational-metric extensions.

The script fits every detector on normal training observations only, compares
the labelled validation-F1 threshold with two label-free normal-score
quantiles, and reports point, event, false-alarm, and moving-block uncertainty
metrics. It saves both summaries and per-seed values for auditability.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detectors import DETERMINISTIC, make_detector  # noqa: E402
from src.loaders import build_dataset  # noqa: E402
from src.metrics import (  # noqa: E402
    best_threshold_by_f1,
    binary_metrics,
    f1_point_adjust,
    moving_block_bootstrap_ci,
    operational_metrics,
    threshold_from_normal_scores,
)


DEFAULT_METHODS = {
    "batadal": ["mahalanobis", "pca", "ocsvm", "isoforest", "dense_ae", "cnn_ae", "lstm_ae", "transformer_ae"],
    "gas_pipeline": ["mahalanobis", "pca", "ocsvm", "isoforest", "dense_ae_compact"],
    "hai": ["mahalanobis", "pca", "ocsvm", "isoforest", "dense_ae", "cnn_ae", "lstm_ae", "transformer_ae"],
}

LABELS = {
    "mahalanobis": "Mahalanobis Distance",
    "pca": "PCA Reconstruction",
    "ocsvm": "One-Class SVM",
    "isoforest": "Isolation Forest",
    "dense_ae": "Dense-Autoencoder",
    "dense_ae_compact": "Dense-Autoencoder",
    "cnn_ae": "CNN-Autoencoder",
    "lstm_ae": "LSTM-Autoencoder",
    "transformer_ae": "Transformer-Autoencoder",
}

DOMAIN = {
    "batadal": {"points_per_day": 24.0, "block_length": 24},
    "gas_pipeline": {"points_per_day": None, "block_length": 1000},
    "hai": {"points_per_day": 86400.0, "block_length": 300},
}


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _summary(per_seed: list[dict]) -> dict:
    keys = ["f1", "precision", "recall", "fpr", "auc", "aucpr", "f1pa",
            "event_recall", "mean_delay_points", "false_alarm_events_per_10k",
            "false_alarm_events_per_day"]
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in per_seed if row.get(key) is not None and np.isfinite(row[key])]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def run_method(ds, dataset: str, key: str, seeds: list[int], epochs: int,
               n_boot: int, out_dir: Path) -> dict:
    run_seeds = [seeds[0]] if key in DETERMINISTIC else seeds
    by_policy = {"val_f1": [], "normal_cal_q99": [], "normal_cal_q995": []}
    fit_times = []
    n_params = 0
    for pos, seed in enumerate(run_seeds):
        print(f"[{dataset}] {key} seed={seed}", flush=True)
        det = make_detector(key, seed=seed, epochs=epochs)
        det.fit(ds.X_train)
        cal_scores = det.score(ds.X_cal)
        val_scores = det.score(ds.X_val)
        test_scores = det.score(ds.X_test)
        val_thr, val_best = best_threshold_by_f1(ds.y_val, val_scores)
        thresholds = {
            "val_f1": val_thr,
            "normal_cal_q99": threshold_from_normal_scores(cal_scores, 0.99),
            "normal_cal_q995": threshold_from_normal_scores(cal_scores, 0.995),
        }
        if pos == 0:
            np.savez_compressed(
                out_dir / f"{dataset}_{key}_seed{seed}_scores.npz",
                y_test=ds.y_test, test_scores=test_scores, thresholds=np.array(list(thresholds.values())),
            )
        for policy, threshold in thresholds.items():
            row = binary_metrics(ds.y_test, test_scores, threshold)
            row["seed"] = seed
            row["f1pa"] = f1_point_adjust(ds.y_test, test_scores, threshold)
            row["val_best_f1"] = val_best
            row.update(operational_metrics(
                ds.y_test, test_scores, threshold,
                points_per_day=DOMAIN[dataset]["points_per_day"],
            ))
            if pos == 0 and n_boot > 0:
                lo, hi = moving_block_bootstrap_ci(
                    ds.y_test, test_scores, threshold, metric="f1", n_boot=n_boot,
                    block_length=DOMAIN[dataset]["block_length"], seed=seed,
                )
                row["f1_block_ci95"] = [lo, hi]
            by_policy[policy].append(row)
        fit_times.append(float(getattr(det, "fit_time_", 0.0)))
        n_params = int(getattr(det, "n_params", 0))

    return {
        "key": key,
        "method": LABELS[key],
        "n_seeds": len(run_seeds),
        "n_params": n_params,
        "fit_time_sec_mean": float(np.mean(fit_times)),
        "policies": {
            policy: {"summary": _summary(rows), "per_seed": rows}
            for policy, rows in by_policy.items()
        },
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=sorted(DEFAULT_METHODS))
    p.add_argument("--data-dir", required=True)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-train", type=int, default=0,
                   help="HAI only; 0 means the full normal training sequence")
    p.add_argument("--n-boot", type=int, default=0,
                   help="Moving-block replicates for the representative first seed; 0 disables")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    kwargs = {"data_dir": args.data_dir, "verbose": True}
    if args.dataset == "hai":
        kwargs["max_train"] = args.max_train
    ds = build_dataset(args.dataset, **kwargs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    methods = args.methods or DEFAULT_METHODS[args.dataset]
    started = time.time()
    results = []
    for key in methods:
        results.append(run_method(ds, args.dataset, key, args.seeds, args.epochs, args.n_boot, out.parent))
        payload = {
            "dataset": args.dataset,
            "dataset_info": ds.info(),
            "seeds": args.seeds,
            "epochs": args.epochs,
            "max_train": args.max_train if args.dataset == "hai" else None,
            "elapsed_sec": time.time() - started,
            "results": results,
        }
        out.write_text(json.dumps(_clean(payload), indent=2), encoding="utf-8")
        print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()

