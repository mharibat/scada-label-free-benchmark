# Contextual full-supervision reference

`gas_supervised_reference.json` records an earlier same-chronological-split
comparison among four fully supervised tree ensembles. Random Forest obtained
F1 = 0.623 in that comparison.

This file is retained for traceability only. It is not part of the controlled
attack-label-budget curve in Table 6. That curve uses HistGradientBoosting at
every label fraction and reports a three-seed mean of F1 = 0.608 at 100% of
attack labels. The two numbers use different estimator-selection procedures and
must not be presented as the same supervised ceiling.
