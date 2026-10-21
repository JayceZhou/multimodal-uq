# Paired upstream-to-downstream transfer export

This directory contains the read-only paired export used to analyze how the
frozen Gaussian (`G`, formerly `M1`) and heavy-tailed/circular (`HTC`, formerly
`M2`) predictive laws transfer from 3D perception to motion forecasting.

## Main files

- `paired_rows.h5`: compact columnar master file with all 1,708,500 rows.
- `paired_rows.csv.gz`: portable copy of the same 23 columns.
- `shards/*.npz`: 48 independently loadable population-condition-seed shards.
- `condition_seed_summary.csv`: one row per population, condition, severity,
  and head seed. Its means are descriptive track-weighted summaries, not the
  paper's scene-weighted bootstrap estimands; use `scene_token` for inferential
  aggregation.
- `schema.json`: field definitions and metric contracts.
- `manifest.json`: row ranges, source/output hashes, and pairing audit.

The pairing key is

```text
population + condition + severity + head_seed + row_id
```

where `row_id` is `sample_token:instance_token`. The manifest contains 48
unique population-condition-severity-seed groups, and `row_id` is unique
inside every group.

## Metric definitions

- `xy_nll_*` uses the raw, uncalibrated frozen probability-head output.
- `linear90_inside_*` is natural 90% inclusion for the complete eight-
  dimensional linear state. G uses a chi-square threshold with 8 degrees of
  freedom; HTC uses the row-specific `8 * F(8, nu)` threshold.
- `eq90_volume_*` is a legacy requested field name. Its value is consistently
  **XY ellipse area in square metres**, not an 8-D volume. For each law,
  condition, and head seed, the ellipse is empirically rescaled to 90%
  coverage on the identical matched rows:

  ```text
  area = pi * q90 * scale_x * scale_y
  ```

- `downstream_es_*` is copied exactly from the retained 64-sample R0/final-
  confirmation archives: fixed CV/MLP probes, shared process-noise draws, and
  common random numbers. No downstream distribution was resampled.

The additional source tokens, source-row index, XY squared standardized
residuals, and equal-coverage thresholds are included for auditability.

## Loading

```python
import h5py

with h5py.File("paired_rows.h5", "r") as data:
    population = data["population"][:].astype(str)
    delta_xy_nll = data["xy_nll_g"][:] - data["xy_nll_htc"][:]
    delta_es_mlp = (
        data["downstream_es_g_mlp"][:]
        - data["downstream_es_htc_mlp"][:]
    )
```

To reproduce the export without training or new forecast sampling:

```bash
cd /home/zhousc/multimodal-uq/BEVFusion
/home/zhousc/miniconda3/envs/bev/bin/python \
  tools/export_riskprop3d_upstream_downstream.py
```
