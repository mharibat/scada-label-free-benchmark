# Label-Free Threshold Calibration and Label Efficiency in One-Class SCADA Anomaly Detection

This repository contains the code, configurations, retained numerical results,
and figure data for the cross-domain SCADA anomaly-detection benchmark by
Mohammad Haribat.

## Citation and permanent archive

- GitHub repository: https://github.com/mharibat/scada-label-free-benchmark
- Archived release: [10.5281/zenodo.22077541](https://doi.org/10.5281/zenodo.22077541)
- Version: `v1.0.0`

## What the benchmark evaluates

The study compares classical and neural one-class detectors on BATADAL, Gas
Pipeline, and HAI 22.04. It focuses on four methodological questions:

1. How detector rankings change between labelled validation-F1 calibration and
   normal-only q99/q99.5 calibration.
2. How point-wise scores differ from event coverage, detection delay, and
   false-alarm burden.
3. How performance changes as the available attack-label budget increases.
4. How a chronological split compares with a stratified random split.

The final stochastic protocol uses seeds 42, 43, and 44. HAI is treated as a
descriptive stress test because its retained test half contains four attack
events.

## Repository contents

```text
configs/        dataset configurations
docs/           detailed reproducibility guidance
figures/        final figures and consolidated result table
results/        rigorous JSON metrics and representative seed-42 score arrays
scripts/        final benchmark and label-budget entry points
src/            loaders, metrics, detectors, and experiment utilities
```

Raw third-party datasets are not redistributed in this public repository.
Place the files listed in `DATA_SOURCES.md` under `data/` before running the
experiments.

## Environment

The retained run used Python 3.11.16, NumPy 2.4.6, pandas 3.0.5, SciPy 1.17.1,
scikit-learn 1.9.0, PyTorch 2.13.0+cpu, XGBoost 3.2.0, and matplotlib 3.11.1.

```bash
python -m venv .venv
python -m pip install -r requirements-lock.txt
```

## Reproduce the final experiments

```bash
python scripts/run_rigorous_extension.py --dataset batadal --data-dir data/batadal --seeds 42 43 44 --epochs 20 --n-boot 200 --out results/rigorous_batadal.json
python scripts/run_rigorous_extension.py --dataset gas_pipeline --data-dir data/gas --seeds 42 43 44 --epochs 20 --n-boot 200 --out results/rigorous_gas.json
python scripts/run_rigorous_extension.py --dataset hai --data-dir data/hai --seeds 42 43 44 --epochs 10 --max-train 0 --n-boot 100 --out results/rigorous_hai.json
python scripts/gas_label_budget.py --data-dir data/gas --seeds 42 43 44 --out results/gas_label_budget.json
```

The archived JSON files contain per-seed scalar metrics. NPZ files retain test
labels, scores, and thresholds for seed 42; the other score arrays can be
regenerated from the fixed scripts and seeds.

## Headline results

| Dataset | Labelled validation-F1 leader | F1 | Normal-only q99.5 leader | F1 |
|---|---|---:|---|---:|
| BATADAL | Transformer-AE | 0.612 | Dense-AE | 0.651 |
| Gas Pipeline | Mahalanobis | 0.463 | Mahalanobis | 0.422 |
| HAI 22.04 | Mahalanobis | 0.212 | Isolation Forest | 0.494 |

The HAI q99.5 leader detects one of four attack events. The Gas-Pipeline
HistGradientBoosting label-budget curve reaches F1 values of 0.534, 0.585,
0.584, 0.606, and 0.608 at 1%, 5%, 10%, 25%, and 100% of attack labels.

## License

The repository code is released under the MIT License. Dataset copyright and
licensing remain with the original dataset creators.

