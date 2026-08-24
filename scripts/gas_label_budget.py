"""Gas Pipeline label-budget curve and chronological-vs-random split audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loaders.gas_pipeline import (  # noqa: E402
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    _read_arff,
)
from src.metrics import binary_metrics, threshold_from_normal_scores  # noqa: E402


CALIBRATION_QUANTILE = 0.995
TEMPORAL_BLOCK_SIZE = 1000


def _fit_eval(Xtr, ytr, Xcal_normal, Xte, yte, seed):
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, class_weight="balanced", random_state=seed,
    )
    clf.fit(Xtr, ytr)
    threshold = threshold_from_normal_scores(
        clf.predict_proba(Xcal_normal)[:, 1], CALIBRATION_QUANTILE
    )
    return binary_metrics(
        yte, clf.predict_proba(Xte)[:, 1], threshold
    )


def _scale_fit_transform(Xfit, Xcal, Xtest):
    """Fit every preprocessing step on the classifier-fitting partition only."""
    rs = RobustScaler().fit(Xfit)
    Xfit_r = np.clip(rs.transform(Xfit), -5, 5)
    Xcal_r = np.clip(rs.transform(Xcal), -5, 5)
    Xtest_r = np.clip(rs.transform(Xtest), -5, 5)
    mm = MinMaxScaler().fit(Xfit_r)
    return (
        mm.transform(Xfit_r).astype(np.float32),
        mm.transform(Xcal_r).astype(np.float32),
        mm.transform(Xtest_r).astype(np.float32),
    )


def _chronological_data(data_dir: str):
    """56/14/10/20 fit/calibration/unused-validation/test protocol."""
    df = _read_arff(Path(data_dir) / "IanArffDataset.arff")
    y = (pd.to_numeric(df[LABEL_COLUMN], errors="coerce").fillna(0) > 0).astype(int).values
    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32).values
    total = len(X)
    train_end = int(total * 0.70)
    fit_end = int(train_end * 0.80)
    test_start = train_end + int(total * 0.10)
    Xfit, yfit = X[:fit_end], y[:fit_end]
    Xcal, ycal = X[fit_end:train_end], y[fit_end:train_end]
    Xtest, ytest = X[test_start:], y[test_start:]
    Xfit, Xcal, Xtest = _scale_fit_transform(Xfit, Xcal, Xtest)
    return Xfit, yfit, Xcal[ycal == 0], Xtest, ytest


def _block_aware_attack_sample(attacks, fraction, seed):
    """Select attack packets by whole temporal blocks whenever possible.

    The nominal budget is a fraction of attack packets, but packets are added
    from randomly ordered 1,000-packet blocks to avoid treating adjacent attack
    packets as independent annotations. The realised count can slightly exceed
    the nominal target because the last selected block is retained in full.
    """
    attacks = np.asarray(attacks, dtype=int)
    if fraction >= 1.0:
        return attacks.copy(), np.unique(attacks // TEMPORAL_BLOCK_SIZE)
    target = max(1, int(round(len(attacks) * fraction)))
    blocks = np.unique(attacks // TEMPORAL_BLOCK_SIZE)
    rng = np.random.RandomState(seed)
    ordered = rng.permutation(blocks)
    selected = []
    selected_blocks = []
    total = 0
    for block in ordered:
        block_attacks = attacks[attacks // TEMPORAL_BLOCK_SIZE == block]
        if block_attacks.size == 0:
            continue
        selected.append(block_attacks)
        selected_blocks.append(int(block))
        total += len(block_attacks)
        if total >= target:
            break
    return np.sort(np.concatenate(selected)), np.asarray(selected_blocks, dtype=int)


def label_budget(data_dir: str, seeds: list[int]):
    Xtr, ytr, Xcal_normal, Xte, yte = _chronological_data(data_dir)
    normal = np.flatnonzero(ytr == 0)
    attacks = np.flatnonzero(ytr == 1)
    fractions = [0.01, 0.05, 0.10, 0.25, 1.00]
    rows = []
    for fraction in fractions:
        metrics = []
        counts = []
        block_counts = []
        for seed in seeds:
            labelled_attacks, labelled_blocks = _block_aware_attack_sample(
                attacks, fraction, seed
            )
            # Preserve chronological order; no validation attack labels are
            # consumed to tune a decision threshold.
            idx = np.sort(np.r_[normal, labelled_attacks])
            m = _fit_eval(Xtr[idx], ytr[idx], Xcal_normal, Xte, yte, seed)
            metrics.append(m)
            counts.append(int(len(labelled_attacks)))
            block_counts.append(int(len(labelled_blocks)))
            print(
                f"budget={fraction:.0%} seed={seed} attacks={len(labelled_attacks)} "
                f"blocks={len(labelled_blocks)} F1={m['f1']:.4f}", flush=True
            )
        rows.append({
            "attack_label_fraction": fraction,
            "nominal_n_labelled_attack_packets": max(1, int(round(len(attacks) * fraction))),
            "realised_n_labelled_attack_packets_mean": float(np.mean(counts)),
            "realised_fraction_mean": float(np.mean(counts) / len(attacks)),
            "selected_temporal_blocks_mean": float(np.mean(block_counts)),
            "f1_mean": float(np.mean([m["f1"] for m in metrics])),
            "f1_std": float(np.std([m["f1"] for m in metrics], ddof=1)) if len(metrics) > 1 else 0.0,
            "auc_mean": float(np.mean([m["auc"] for m in metrics])),
            "aucpr_mean": float(np.mean([m["aucpr"] for m in metrics])),
            "per_seed": metrics,
        })
    return rows


def random_split(data_dir: str, seeds: list[int]):
    df = _read_arff(Path(data_dir) / "IanArffDataset.arff")
    y = (pd.to_numeric(df[LABEL_COLUMN], errors="coerce").fillna(0) > 0).astype(int).values
    X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32).values
    results = []
    for seed in seeds:
        Xtr_all, Xhold, ytr_all, yhold = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=seed,
        )
        _Xva, Xte, _yva, yte = train_test_split(
            Xhold, yhold, test_size=2 / 3, stratify=yhold, random_state=seed,
        )
        Xtr, Xcal, ytr, ycal = train_test_split(
            Xtr_all, ytr_all, test_size=0.20, stratify=ytr_all, random_state=seed,
        )
        Xtr_m, Xcal_m, Xte_m = _scale_fit_transform(Xtr, Xcal, Xte)
        m = _fit_eval(
            Xtr_m, ytr, Xcal_m[ycal == 0],
            Xte_m, yte, seed,
        )
        results.append(m)
        print(f"random_split seed={seed} F1={m['f1']:.4f}", flush=True)
    return {
        "f1_mean": float(np.mean([m["f1"] for m in results])),
        "f1_std": float(np.std([m["f1"] for m in results], ddof=1)) if len(results) > 1 else 0.0,
        "auc_mean": float(np.mean([m["auc"] for m in results])),
        "per_seed": results,
    }


def chronological_full(data_dir: str, seeds: list[int]):
    Xtr, ytr, Xcal_normal, Xte, yte = _chronological_data(data_dir)
    results = [
        _fit_eval(Xtr, ytr, Xcal_normal, Xte, yte, seed)
        for seed in seeds
    ]
    return {
        "f1_mean": float(np.mean([m["f1"] for m in results])),
        "f1_std": float(np.std([m["f1"] for m in results], ddof=1)) if len(results) > 1 else 0.0,
        "auc_mean": float(np.mean([m["auc"] for m in results])),
        "per_seed": results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    payload = {
        "model": "HistGradientBoostingClassifier",
        "label_budget_definition": (
            "all normal training packets plus a nominal fraction of attack packets; "
            "attack packets are selected in 1000-packet temporal blocks, and no "
            "validation attack labels are used"
        ),
        "threshold_policy": (
            "99.5th percentile of predicted attack probability on a disjoint "
            "normal-only calibration tail"
        ),
        "calibration_attack_labels_used": 0,
        "class_weight": "balanced",
        "temporal_block_size_packets": TEMPORAL_BLOCK_SIZE,
        "chronological_protocol": "56% fit / 14% normal-only calibration / 10% unused holdout / 20% test",
        "preprocessing_fit_scope": "classifier-fitting partition only",
        "seeds": args.seeds,
        "label_budget": label_budget(args.data_dir, args.seeds),
        "split_audit": {
            "chronological_70_10_20": chronological_full(args.data_dir, args.seeds),
            "stratified_random_70_10_20": random_split(args.data_dir, args.seeds),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

