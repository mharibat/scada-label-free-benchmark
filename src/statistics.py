"""
Statistical comparison of detectors across datasets (Demšar, JMLR 2006).

  * Friedman test: is there any significant difference among methods?
  * Average ranks per method across datasets.
  * Nemenyi post-hoc critical difference (CD): two methods differ significantly
    at level alpha iff their average ranks differ by more than CD.
  * A critical-difference diagram.

Note: with a small number of datasets these tests have low power; that is the
honest situation this study is in and we report it as such.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

# Studentised-range q values at alpha=0.05 (Demšar 2006, Table 5) for k methods.
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
       7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def friedman(scores: pd.DataFrame) -> Tuple[float, float]:
    """scores: rows = datasets, cols = methods; entries = F1 (higher better)."""
    if scores.shape[0] < 3 or scores.shape[1] < 3:
        raise ValueError("Friedman needs >=3 datasets and >=3 methods")
    cols = [scores[c].values for c in scores.columns]
    stat, p = friedmanchisquare(*cols)
    return float(stat), float(p)


def average_ranks(scores: pd.DataFrame) -> pd.Series:
    """Average rank of each method across datasets (rank 1 = best F1)."""
    ranks = scores.rank(axis=1, ascending=False, method="average")
    return ranks.mean(axis=0).sort_values()


def critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    q = Q05.get(k, Q05[10])
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def nemenyi_pairs(scores: pd.DataFrame) -> Dict[str, object]:
    n, k = scores.shape
    ranks = average_ranks(scores)
    cd = critical_difference(k, n)
    names = list(ranks.index)
    sig = []  # pairs that DIFFER significantly
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = abs(ranks[names[i]] - ranks[names[j]])
            if diff > cd:
                sig.append((names[i], names[j], round(diff, 3)))
    return {"avg_ranks": ranks, "cd": cd, "significant_pairs": sig}


def cd_diagram(scores: pd.DataFrame, save_path: str, title: str = "") -> None:
    """Draw a Demšar critical-difference diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, k = scores.shape
    ranks = average_ranks(scores)
    cd = critical_difference(k, n)
    names = list(ranks.index)
    vals = ranks.values

    lo, hi = 1, k
    fig, ax = plt.subplots(figsize=(8, 2.6 + 0.35 * k))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, 1.28)
    ax.axis("off")
    # top axis
    y0 = 0.85
    ax.plot([lo, hi], [y0, y0], "k-", lw=1)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [y0, y0 + 0.03], "k-", lw=1)
        ax.text(r, y0 + 0.06, str(r), ha="center", fontsize=10)
    # CD bar
    ax.plot([lo, lo + cd], [y0 + 0.16, y0 + 0.16], "k-", lw=2.5)
    ax.text(lo + cd / 2, y0 + 0.19, f"CD = {cd:.2f}", ha="center", fontsize=10)
    # method leaders
    order = np.argsort(vals)
    half = (len(vals) + 1) // 2
    for rank_pos, idx in enumerate(order):
        x = vals[idx]
        left = rank_pos < half
        y = 0.6 - 0.12 * (rank_pos if left else (len(vals) - 1 - rank_pos))
        endx = lo - 0.45 if left else hi + 0.45
        ax.plot([x, x], [y0, y], "k-", lw=0.8)
        ax.plot([x, endx], [y, y], "k-", lw=0.8)
        ax.text(endx + (-0.05 if left else 0.05), y,
                f"{names[idx]} ({x:.2f})", ha="right" if left else "left",
                va="center", fontsize=9)
    # connect non-significant groups (rank diff <= CD)
    sorted_vals = vals[order]
    ybar = y0 - 0.06
    used = 0
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and (sorted_vals[j + 1] - sorted_vals[i]) <= cd:
            j += 1
        if j > i:
            ax.plot([sorted_vals[i] - 0.03, sorted_vals[j] + 0.03],
                    [ybar - 0.02 * used, ybar - 0.02 * used], "k-", lw=3)
            used += 1
    if title:
        ax.text((lo + hi) / 2, 1.24, title, ha="center", fontsize=11)
    plt.tight_layout()
    _ = used  # (number of connecting bars drawn)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def full_report(scores: pd.DataFrame, alpha: float = 0.05) -> dict:
    stat, p = friedman(scores)
    nem = nemenyi_pairs(scores)
    return {
        "n_datasets": int(scores.shape[0]),
        "n_methods": int(scores.shape[1]),
        "friedman_stat": stat,
        "friedman_p": p,
        "friedman_significant": bool(p < alpha),
        "avg_ranks": {k: round(float(v), 3) for k, v in nem["avg_ranks"].items()},
        "critical_difference": round(nem["cd"], 3),
        "significant_pairs": nem["significant_pairs"],
    }
