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
    build_gas_supervised,
)
from src.metrics import best_threshold_by_f1, binary_metrics  # noqa: E402


def _fit_eval(Xtr, ytr, Xva, yva, Xte, yte, seed):
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed,
    )
    clf.fit(Xtr, ytr)
    val_score = clf.predict_proba(Xva)[:, 1]
    threshold, _ = best_threshold_by_f1(yva, val_score)
    return binary_metrics(yte, clf.predict_proba(Xte)[:, 1], threshold)


def label_budget(data_dir: str, seeds: list[int]):
    Xtr, ytr, Xva, yva, Xte, yte = build_gas_supervised(data_dir=data_dir)
    normal = np.flatnonzero(ytr == 0)
    attacks = np.flatnonzero(ytr == 1)
    fractions = [0.01, 0.05, 0.10, 0.25, 1.00]
    rows = []
    for fraction in fractions:
        metrics = []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            n = max(1, int(round(len(attacks) * fraction)))
            labelled_attacks = rng.choice(attacks, n, replace=False)
            idx = np.r_[normal, labelled_attacks]
            rng.shuffle(idx)
            m = _fit_eval(Xtr[idx], ytr[idx], Xva, yva, Xte, yte, seed)
            metrics.append(m)
            print(f"budget={fraction:.0%} seed={seed} F1={m['f1']:.4f}", flush=True)
        rows.append({
            "attack_label_fraction": fraction,
            "n_labelled_attacks": max(1, int(round(len(attacks) * fraction))),
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
        Xtr, Xhold, ytr, yhold = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=seed,
        )
        Xva, Xte, yva, yte = train_test_split(
            Xhold, yhold, test_size=2 / 3, stratify=yhold, random_state=seed,
        )
        rs = RobustScaler().fit(Xtr)
        Xtr_r = np.clip(rs.transform(Xtr), -5, 5)
        Xva_r = np.clip(rs.transform(Xva), -5, 5)
        Xte_r = np.clip(rs.transform(Xte), -5, 5)
        mm = MinMaxScaler().fit(Xtr_r)
        m = _fit_eval(
            mm.transform(Xtr_r).astype(np.float32), ytr,
            mm.transform(Xva_r).astype(np.float32), yva,
            mm.transform(Xte_r).astype(np.float32), yte, seed,
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
    Xtr, ytr, Xva, yva, Xte, yte = build_gas_supervised(data_dir=data_dir)
    results = [_fit_eval(Xtr, ytr, Xva, yva, Xte, yte, seed) for seed in seeds]
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
        "label_budget_definition": "all normal training labels plus the stated fraction of attack labels",
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
