"""
One-class benchmark runner.

For each detector:
  1. fit on NORMAL training data,
  2. score validation and test,
  3. pick the F1-optimal threshold on VALIDATION (no test leakage),
  4. evaluate on TEST at that threshold,
  5. repeat over several seeds (deterministic detectors run once) and report
     mean +/- std of F1/AUC/AUCPR, plus a bootstrap 95% CI on the best seed.

This addresses two review points: the paper reported single-seed numbers with
no variance, and made ranking claims on differences smaller than run-to-run
noise.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .detectors import APPLICABLE, DETERMINISTIC, make_detector
from .metrics import binary_metrics, best_threshold_by_f1, bootstrap_ci, f1_point_adjust
from .utils import mean_std


def run_detector(ds, key: str, label: str, seeds: List[int], verbose=True, epochs=None) -> Dict:
    run_seeds = [seeds[0]] if key in DETERMINISTIC else seeds
    per_seed = []
    keep = None  # keep scores of the first seed for CI + threshold reporting
    for s in run_seeds:
        det = make_detector(key, seed=s, epochs=epochs)
        det.fit(ds.X_train)
        val_scores = det.score(ds.X_val)
        test_scores = det.score(ds.X_test)
        thr, val_f1 = best_threshold_by_f1(ds.y_val, val_scores)
        m = binary_metrics(ds.y_test, test_scores, thr)
        m["f1pa"] = f1_point_adjust(ds.y_test, test_scores, thr)
        m["val_f1"] = val_f1
        m["fit_time_sec"] = float(getattr(det, "fit_time_", 0.0))
        m["n_params"] = int(getattr(det, "n_params", 0))
        per_seed.append(m)
        if keep is None:
            keep = (test_scores, thr)
        if verbose:
            print(f"    seed {s}: F1={m['f1']:.4f} AUC={m['auc']:.4f} "
                  f"AUCPR={m['aucpr']:.4f} P={m['precision']:.3f} R={m['recall']:.3f}")

    f1s = [m["f1"] for m in per_seed]
    aucs = [m["auc"] for m in per_seed]
    aucprs = [m["aucpr"] for m in per_seed]
    f1_m, f1_s = mean_std(f1s)
    auc_m, auc_s = mean_std(aucs)
    aucpr_m, aucpr_s = mean_std(aucprs)

    test_scores, thr = keep
    ci_lo, ci_hi = bootstrap_ci(ds.y_test, test_scores, thr, metric="f1")

    best = per_seed[int(np.argmax(f1s))]
    return {
        "method": label,
        "key": key,
        "n_seeds": len(run_seeds),
        "deterministic": key in DETERMINISTIC,
        "f1_mean": f1_m, "f1_std": f1_s,
        "auc_mean": auc_m, "auc_std": auc_s,
        "aucpr_mean": aucpr_m, "aucpr_std": aucpr_s,
        "f1_ci95": [ci_lo, ci_hi],
        "precision": best["precision"], "recall": best["recall"],
        "f1pa": best["f1pa"],
        "n_params": best["n_params"],
        "fit_time_sec": float(np.mean([m["fit_time_sec"] for m in per_seed])),
        "per_seed_f1": f1s,
    }


def run_benchmark(ds, dataset_key: str, seeds: List[int], verbose=True, epochs=None) -> List[Dict]:
    results = []
    for key, label in APPLICABLE[dataset_key]:
        if verbose:
            print(f"\n[{label}]")
        results.append(run_detector(ds, key, label, seeds, verbose=verbose, epochs=epochs))
    results.sort(key=lambda r: r["f1_mean"], reverse=True)
    return results
