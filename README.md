# Attack-label-free SCADA calibration benchmark

This repository contains the revised code and processed results for a
cross-domain benchmark on BATADAL, Gas Pipeline, and HAI 22.04.

The key methodological change is a strict chronological separation between
detector fitting and normal-only q99/q99.5 threshold calibration. The corrected
Gas-Pipeline label-budget experiment selects attack packets in temporal blocks
and never uses validation attack labels to tune its threshold.

See [`README_FINAL.md`](README_FINAL.md) for the protocol, headline results, and
reproduction commands; [`DATA_SOURCES.md`](DATA_SOURCES.md) for provenance and
SHA-256 values; and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the
audit rationale.

Code licence: MIT. Dataset files remain under their original third-party terms.

Archive: <https://doi.org/10.5281/zenodo.22077541>.

