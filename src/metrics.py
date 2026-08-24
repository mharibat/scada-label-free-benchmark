"""
Evaluation metrics for one-class anomaly detection.

Methodological stance (unchanged from the paper):
  * Raw F1 is the PRIMARY point-decision metric (Kim et al., AAAI 2022,
    show point-adjusted F1 can turn random scores into apparent SOTA).
  * AUC and AUCPR are reported as threshold-independent rankers.
  * F1PA is provided ONLY as a transparency check, never as primary.

The operating threshold is always selected on the VALIDATION set and then
applied unchanged to the test set (no test-set leakage).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Full metric panel at a fixed threshold. Higher score => more anomalous."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    out: Dict[str, float] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": float(threshold),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, y_score))
        out["aucpr"] = float(average_precision_score(y_true, y_score))
    else:
        out["auc"] = float("nan")
        out["aucpr"] = float("nan")
    return out


def best_threshold_by_f1(
    y_true: np.ndarray, y_score: np.ndarray, n_thresholds: int = 500
) -> Tuple[float, float]:
    """
    Sweep candidate thresholds on a (validation) set and return the one that
    maximises F1. Candidates are the score quantiles, which is more robust than
    a fixed linspace when scores are on an arbitrary scale (e.g. Mahalanobis).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        # No positives in validation -> fall back to median score
        return float(np.median(y_score)), 0.0

    qs = np.linspace(0.0, 1.0, n_thresholds)
    cands = np.unique(np.quantile(y_score, qs))
    best_t, best_f1 = float(cands[len(cands) // 2]), -1.0
    for t in cands:
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, float(best_f1)


def threshold_from_normal_scores(y_score: np.ndarray, quantile: float = 0.995) -> float:
    """Label-free operating threshold calibrated only on normal training scores."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one")
    scores = np.asarray(y_score, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).any():
        raise ValueError("normal calibration scores are empty or non-finite")
    return float(np.nanquantile(scores, quantile))


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive-exclusive contiguous True segments."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def operational_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    points_per_day: float | None = None,
) -> Dict[str, float]:
    """Event recall, detection delay, and false-alarm burden at a fixed threshold.

    Attack and false-alarm episodes are contiguous runs. Delay is expressed in
    samples and, when the sampling rate is known, in days. False alarms are
    reported per 10k observations and optionally per day.
    """
    yt = np.asarray(y_true).astype(int)
    yp = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    attack_events = _segments(yt == 1)
    false_events = _segments((yt == 0) & (yp == 1))

    delays: list[int] = []
    detected = 0
    for start, end in attack_events:
        hits = np.flatnonzero(yp[start:end] == 1)
        if hits.size:
            detected += 1
            delays.append(int(hits[0]))

    normal_points = int(np.sum(yt == 0))
    out: Dict[str, float] = {
        "n_attack_events": int(len(attack_events)),
        "n_detected_events": int(detected),
        "event_recall": float(detected / len(attack_events)) if attack_events else float("nan"),
        "mean_delay_points": float(np.mean(delays)) if delays else float("nan"),
        "median_delay_points": float(np.median(delays)) if delays else float("nan"),
        "n_false_alarm_events": int(len(false_events)),
        "false_alarm_events_per_10k": float(len(false_events) * 10000 / normal_points)
        if normal_points else float("nan"),
    }
    if points_per_day and points_per_day > 0:
        observed_days = len(yt) / float(points_per_day)
        out["false_alarm_events_per_day"] = (
            float(len(false_events) / observed_days) if observed_days else float("nan")
        )
        out["mean_delay_days"] = (
            float(np.mean(delays) / points_per_day) if delays else float("nan")
        )
    return out


def f1_point_adjust(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    """
    Point-adjusted F1 (transparency only). If ANY point in a contiguous
    anomalous segment is flagged, the whole segment counts as detected.
    Reported ONLY alongside raw F1 (Kim et al. 2022).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    adjusted = y_pred.copy()
    in_seg, start, detected = False, 0, False
    for i, yt in enumerate(y_true):
        if yt == 1 and not in_seg:
            in_seg, start, detected = True, i, bool(y_pred[i] == 1)
        elif yt == 1 and in_seg:
            detected = detected or bool(y_pred[i] == 1)
        elif yt == 0 and in_seg:
            if detected:
                adjusted[start:i] = 1
            in_seg, detected = False, False
    if in_seg and detected:
        adjusted[start:] = 1
    return float(f1_score(y_true, adjusted, zero_division=0))


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    metric: str = "f1",
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Percentile bootstrap 95% CI for a test-set metric at a fixed threshold.
    Complements the across-seed variance for the deep models.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        if metric == "f1":
            v = f1_score(yt, (ys >= threshold).astype(int), zero_division=0)
        elif metric == "auc":
            v = roc_auc_score(yt, ys)
        elif metric == "aucpr":
            v = average_precision_score(yt, ys)
        else:
            raise ValueError(metric)
        vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def moving_block_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    metric: str = "f1",
    n_boot: int = 1000,
    block_length: int | None = None,
    seed: int = 42,
) -> Tuple[float, float]:
    """Moving-block percentile CI that preserves local temporal dependence."""
    rng = np.random.RandomState(seed)
    yt = np.asarray(y_true).astype(int)
    ys = np.asarray(y_score).astype(float)
    n = len(yt)
    if n == 0:
        return float("nan"), float("nan")
    block = int(block_length or max(2, round(n ** (1.0 / 3.0))))
    block = min(max(1, block), n)
    n_blocks = int(np.ceil(n / block))
    starts_max = max(1, n - block + 1)
    vals = []
    for _ in range(n_boot):
        starts = rng.randint(0, starts_max, size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        b_yt, b_ys = yt[idx], ys[idx]
        if len(np.unique(b_yt)) < 2:
            continue
        if metric == "f1":
            val = f1_score(b_yt, (b_ys >= threshold).astype(int), zero_division=0)
        elif metric == "auc":
            val = roc_auc_score(b_yt, b_ys)
        elif metric == "aucpr":
            val = average_precision_score(b_yt, b_ys)
        else:
            raise ValueError(metric)
        vals.append(float(val))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
