# Attack-label-free SCADA calibration benchmark

Reproducibility package for **Attack-Label-Free Threshold Calibration and
Label Efficiency in One-Class SCADA Anomaly Detection: A Reproducible
Cross-Domain Benchmark**.

The revision separates detector fitting from normal-only threshold calibration
before preprocessing is fitted. “Attack-label-free” means that no attack labels
are used to choose q99/q99.5; it does not mean that the normal reference period
is uncurated.

## Revised protocol

| Dataset | Normal fitting | Normal calibration | Labelled validation | Test | Features |
|---|---:|---:|---:|---:|---:|
| BATADAL | 6,985 windows | 1,730 | 2,065 | 2,066 | 36 |
| Gas Pipeline | 119,996 packets | 30,000 | 27,462 | 54,927 | 17 |
| HAI 22.04 | 74,871 windows | 18,712 | 43,191 | 43,191 | 59 |

The fitting/calibration split is chronological. Feature selection and scalers
are fitted on the fitting prefix only. Neural early stopping uses a
chronological normal tail inside that prefix.

## Headline revised results

| Dataset | Labelled validation-F1 leader | F1 | Held-out-normal q99.5 leader | F1 |
|---|---|---:|---|---:|
| BATADAL | Transformer-Autoencoder | 0.610 | Dense-Autoencoder | 0.635 |
| Gas Pipeline | PCA Reconstruction | 0.464 | Mahalanobis Distance | 0.416 |
| HAI 22.04 | Mahalanobis Distance | 0.273 | Isolation Forest | 0.108 |

HAI has only four test events. Its representative q99.5 moving-block 95%
interval is `[0.000, 0.509]`; the result is descriptive, not confirmatory.

## Corrected Gas-Pipeline label budget

HistGradientBoosting uses balanced class weights. Attack packets are selected
in 1,000-packet temporal blocks, the threshold is q99.5 on a disjoint
normal-only calibration period, and no validation attack labels are used.

| Nominal budget | Mean realised attack packets | Test F1 mean ± SD |
|---:|---:|---:|
| 1% | 502 | 0.392 ± 0.075 |
| 5% | 1,782 | 0.426 ± 0.157 |
| 10% | 3,462 | 0.432 ± 0.066 |
| 25% | 8,580 | 0.512 ± 0.066 |
| 100% | 33,740 | 0.590 ± 0.009 |

The 100% endpoint is identical to the chronological split audit. The matched
stratified-random audit is `0.699 ± 0.002`.

## Reproduce

```bash
python -m venv .venv
python -m pip install -r requirements-lock.txt
python scripts/download_data.py --dataset all

python scripts/run_rigorous_extension.py --dataset batadal --data-dir data/batadal --seeds 42 43 44 --epochs 20 --n-boot 0 --out results_revised/rigorous_batadal.json
python scripts/run_rigorous_extension.py --dataset gas_pipeline --data-dir data/gas --seeds 42 43 44 --epochs 20 --n-boot 0 --out results_revised/rigorous_gas.json
python scripts/run_rigorous_extension.py --dataset hai --data-dir data/hai --seeds 42 43 44 --epochs 10 --max-train 0 --n-boot 0 --out results_revised/rigorous_hai.json
python scripts/gas_label_budget.py --data-dir data/gas --seeds 42 43 44 45 46 --out results_revised/gas_label_budget.json
python scripts/bootstrap_policy_leaders.py --results-dir results_revised --out results_revised/bootstrap_policy_leaders.json
```

JSON files contain per-seed metrics. Representative seed-42 score arrays are
stored as compressed NPZ files.

## Data policy

The code is MIT-licensed. Raw datasets are third-party artifacts and remain
under their original terms. `DATA_SOURCES.md` records provenance, retrieval
URLs, access date, and SHA-256 values for the exact analysed files. The public
repository need not redistribute raw data.

Software archive: <https://doi.org/10.5281/zenodo.22077541>.

