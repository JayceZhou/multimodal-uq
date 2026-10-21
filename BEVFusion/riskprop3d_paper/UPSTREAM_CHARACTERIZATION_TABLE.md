# Frozen upstream predictive-law characterization

This file records the existing clean pre-forecast evidence used to characterize
the two trained upstream predictive laws in the manuscript. It introduces no
new model, split, threshold, or post-forecast analysis.

## Sources

- `outputs/geoprob3d_g0/result.json`
- `geoprob3d_g0/README.md`
- `tools/riskprop3d_r0.py` and `riskprop3d_r0/README.md` for the downstream
  process-noise contract

Implementation crosswalk:

- **G**: M1 diagonal-Gaussian linear state plus wrapped-Gaussian yaw;
- **HTC**: M2 diagonal multivariate Student-t linear state plus von Mises yaw.

Both laws share the frozen detector mean. The values below characterize the
complete trained predictive laws; they do not identify a causal effect of the
Student-t or von Mises family alone.

Both predictive laws, their three head seeds, and the associated upstream
checkpoints were fixed before any downstream forecasting outcome was
evaluated. No downstream metric selected a law, checkpoint, epoch, or seed.

## Clean pre-forecast characterization

All entries use the same matched frozen-query rows of the clean development
evaluation within each head seed. NLL, natural coverage, and raw-scale
statistics are computed directly from the probability-head outputs, before
any label-calibration temperature or other post-hoc scale correction. The
equal-coverage volume is a descriptive exception: each law is rescaled on
those same rows to its empirical 90% coverage before volume is compared. The
downstream label-calibration split is used for the forecasting process-noise
contract below, not for the upstream NLL or natural-coverage entries.

Positive NLL contrasts are `G - HTC` and favor HTC. Coverage gap is empirical
coverage minus the nominal 0.90. Ratio ranges are the three retained head-seed
values.

| Quantity | Frozen value | Interpretation boundary |
|---|---:|---|
| Full-geometry NLL contrast, G - HTC | 4.10243 (95% CI 1.97022, 7.42856) | The trained HTC law has lower joint geometry NLL |
| XY NLL contrast, G - HTC | 0.12799 (0.11494, 0.14169) | The trained HTC law has lower planar-location NLL |
| Yaw NLL contrast, G - HTC | 2.71034 (0.62140, 6.01430) | The trained HTC law has lower yaw NLL |
| Linear 90% coverage gap | G: -0.03827; HTC: +0.01330 | HTC is closer to nominal linear coverage on this split |
| Yaw 90% coverage gap | G: +0.02549; HTC: +0.03107 | Both over-cover; G is slightly closer to nominal yaw coverage |
| Median linear-scale ratio, HTC/G | 0.758--0.791 | HTC predicts smaller median raw linear scales across seeds |
| Equal-coverage 90% ellipsoid-volume ratio, HTC/G | 1.095--1.249 | After equalizing coverage, HTC static regions are wider across seeds |

The upstream evidence is therefore mixed in calibration and sharpness despite
the favorable HTC proper-score contrasts. The downstream stochastic result
must be stated as a comparison between these two trained predictive laws, not
as a pure distribution-family or parameter-level causal effect. Better
upstream NLL does not logically fix the downstream ordering because nonlinear
dynamics, additive process uncertainty, temporal aggregation, and
distributional evaluation may attenuate or reorder predictive-law quality.

## Frozen future process-noise contract

For each forecasting probe, clean label-calibration residuals fit a separate
stack of six zero-mean 2D Gaussian covariances, one `2 x 2` covariance per
future horizon. Eigenvalues are floored at `1e-4` for numerical stability. The
same probe-specific additive law and common noise draws are used for the
mean-state current interface, G propagation, and HTC propagation. No degraded
or final-validation outcome enters this fit. Thus, within a probe, the Level-I
comparison changes the current-state predictive law while holding the future
process-noise construction fixed.
