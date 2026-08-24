# Version 2.0.0

This release implements the independent-calibration revision used by the final
IJCIP submission package.

- Splits the normal reference sequence chronologically into detector-fitting
  and held-out threshold-calibration periods before preprocessing is fitted.
- Excludes calibration samples from model fitting, neural early stopping,
  constant-feature screening, scaling, and threshold selection.
- Corrects the HAI feature schema to 59 fitting-prefix features.
- Rebuilds the Gas-Pipeline label-budget study with 1,000-packet temporal
  blocks, balanced class weights, a normal-only q99.5 threshold, and five seeds.
- Adds a matched chronological versus stratified-random split audit.
- Adds moving-block bootstrap intervals and half/double block-length
  sensitivity analyses for policy leaders.
- Records exact data provenance and SHA-256 checksums while keeping raw
  third-party datasets outside the public software archive.

The headline HAI result is explicitly descriptive because the test sequence
contains only four attack events.

