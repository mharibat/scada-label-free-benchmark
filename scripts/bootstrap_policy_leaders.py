"""Moving-block uncertainty for policy leaders in the revised benchmark.

The primary 95% interval uses 1,000 replicates. Half- and double-block
sensitivity checks use 500 replicates each. Intervals are computed from the
archived representative seed-42 score arrays and are explicitly descriptive;
they do not absorb model-selection or cross-site uncertainty.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import moving_block_bootstrap_ci  # noqa: E402


BLOCKS = {"batadal": 24, "gas_pipeline": 1000, "hai": 300}
POLICY_INDEX = {"val_f1": 0, "normal_cal_q995": 2}


def _leader(payload: dict, policy: str) -> dict:
    return max(
        payload["results"],
        key=lambda row: row["policies"][policy]["summary"]["f1_mean"],
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.results_dir)
    files = {
        "batadal": root / "rigorous_batadal.json",
        "gas_pipeline": root / "rigorous_gas.json",
        "hai": root / "rigorous_hai.json",
    }
    rows = []
    for dataset, result_file in files.items():
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        for policy, threshold_index in POLICY_INDEX.items():
            leader = _leader(payload, policy)
            score_file = root / f"{dataset}_{leader['key']}_seed42_scores.npz"
            saved = np.load(score_file)
            y = saved["y_test"]
            scores = saved["test_scores"]
            threshold = float(saved["thresholds"][threshold_index])
            nominal = BLOCKS[dataset]
            primary = moving_block_bootstrap_ci(
                y, scores, threshold, n_boot=1000,
                block_length=nominal, seed=42,
            )
            sensitivity = {}
            for label, length in (
                ("half", max(1, nominal // 2)),
                ("double", nominal * 2),
            ):
                sensitivity[label] = list(moving_block_bootstrap_ci(
                    y, scores, threshold, n_boot=500,
                    block_length=length, seed=42,
                ))
            rows.append({
                "dataset": dataset,
                "policy": policy,
                "leader": leader["method"],
                "seed": 42,
                "point_f1_mean_across_seeds": leader["policies"][policy]["summary"]["f1_mean"],
                "block_length": nominal,
                "replicates": 1000,
                "f1_ci95": list(primary),
                "block_sensitivity_500_replicates": sensitivity,
            })
            print(dataset, policy, leader["method"], primary, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

