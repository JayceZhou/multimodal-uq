# Complete descriptive attribute-stratum export

This file expands the frozen development attribute analysis without training, model execution, resampling, condition selection, or significance filtering. It contains every pre-defined stratum for range, speed, history length, LiDAR point-count proxy, current XY residual quartile, and class group.

## Coverage

- Population: development only. No held-out-degradation or final-validation attribute analysis was part of the frozen S4.2 archive, and none is introduced here.
- Conditions: clean S0; every original lowlight/dropout/beam S1, S2, and S3 condition; each full S1-S3 family aggregate; and the original all-degradation aggregate.
- Interfaces: stochastic HTC versus Gaussian pushforward under CV and MLP probes, and direct HTC-token versus Gaussian-token conditioning under the matched R3-P GT-supervised student.
- Metrics: trajectory energy and risk--coverage. Positive `baseline - candidate` favors HTC. All 95% intervals use 10,000 paired scene bootstrap resamples with seed 20260805.
- Rows: 1,596. Rows are not filtered by effect magnitude, direction, or interval. The status `suppressed_lt10_selected_scenes` appears only when a pre-defined stratum contains fewer than ten selected scenes; such rows are retained with blank estimates rather than deleted. There are 0 such metric rows.

## Provenance and scope status

`clean_s0` and `all_original_degradations` reproduce the two scopes already stored in `outputs/riskprop3d_final_confirmation/attributes/result.json`. Their values are checked exactly against that frozen JSON. Per-severity and per-family rows are a deterministic read-only expansion from the same frozen per-target metric archives, masks, seed pairing, effect direction, scene aggregation, and bootstrap seed. They are descriptive and do not create a new gate or family-wise claim.

`n_scenes` counts scenes containing at least one selected target; `n_inference_scenes` is the number entering the statistic. For risk--coverage, a scene requires at least two selected targets, so these counts can differ. `n_tracks` is the number of selected condition-target rows and is not treated as the inferential sample size.

## Stratum authority

Current-residual boundaries are the frozen clean label-cal quartiles: 0.0792848095298, 0.137456767261, and 0.227989066392 m. GT point counts and current residuals are evaluation-only descriptors and never model inputs.

## Missingness and validity

The frozen source reports `all_numeric_finite=True`. No invalid or NaN effect is silently removed. Suppressed rows remain present with an explicit status. All non-suppressed clean/all-degradation rows match the frozen aggregate file.
