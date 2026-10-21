# RiskProp3D Level-I moment-matched law control

This directory retains the completed retrospective control comparing the
frozen HTC predictive law with an analytically derived moment-matched
Gaussian/wrapped-Gaussian surrogate.

Endpoint: **RETROSPECTIVE OUTCOME A — AGGREGATE HIGHER-ORDER LAW-SHAPE EFFECT
SUPPORTED**.

Start with `RESULTS_REPORT.md` or `RESULTS_REPORT_ZH.md`. Machine-readable
evidence is in `result.json`; all per-target data are in
`paired_track_metrics.h5`.

The audit used GPU 3 only. The concurrently running UniTraj-MTR MH0 training
on GPUs 0--2 was not interrupted. No model was retrained, no condition was
selected from matched-law results, the internal test remained sealed, and the
original G/HTC R0 trajectory-energy streams replayed with maximum absolute
error zero.

Two pre-outcome freezes that failed sampler-QA implementation checks are
retained as sibling directories `riskprop3d_level1_matched_law_invalid_freeze_v1`
and `riskprop3d_level1_matched_law_invalid_freeze_v2`. A serialization-only
HDF5 shim is documented in `packaging_shim.json`; it did not alter scientific
estimators or endpoint logic.
