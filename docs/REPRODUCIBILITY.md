# Reproducibility and audit guide

## Independence of fitting and calibration

For BATADAL, Gas Pipeline, and HAI, the normal reference sequence is divided
chronologically before feature selection or scaling. The fitting prefix is used
for preprocessing and detector fitting. A disjoint normal tail is transformed
with those fitted objects and used only to estimate q99/q99.5. The labelled
validation set is used only for the comparison validation-F1 policy. Test labels
never select a model or threshold.

HAI feature `P1_PP04SP` is constant in the fitting prefix but varies later. It
is therefore excluded; retaining it would let calibration-period information
influence feature selection. The final HAI input has 59 features.

## Label-budget definition

The chronological supervised audit uses 56% of the original packet sequence
for fitting, the next 14% as calibration (normal packets only), the next 10% as
an unused protocol holdout, and the final 20% as test. Attack labels in the
calibration region are not inspected. Within the fitting region, the nominal
budget is the fraction of attack packets selected from randomly ordered
1,000-packet temporal blocks. All fitting-region normal packets are retained.

The stratified-random audit preserves the same 56/14/10/20 proportions and
preprocessing boundaries. Both 100% paths call the same fitting/evaluation
function.

## Uncertainty

Policy-leader moving-block percentile intervals use 1,000 replicates. Nominal
block lengths are 24 BATADAL windows (one day), 1,000 Gas packets, and 300 HAI
windows (five minutes). Half/double-block checks use 500 replicates. These
intervals describe local test-sequence dependence for representative seed 42;
they do not absorb model-selection, site, or dataset uncertainty.

## Integrity checks

1. Run `python sanity_test.py`.
2. Run the five reproduction commands in `README_FINAL.md`.
3. Compare the JSON summaries with the manuscript tables.
4. Verify raw inputs against `DATA_SOURCES.md`.
5. Verify the release archive against `MANIFEST_SHA256.txt`.

Python and package versions are pinned in `requirements-lock.txt`.

