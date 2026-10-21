# RiskProp3D R0 minimal reproducible audit

## Question and boundary

R0 asks whether the frozen M2 Student-t state and von Mises yaw distribution
improves downstream probabilistic motion forecasting relative to deterministic
perception, M1 Gaussian propagation, detector score, and query density. It
does not train or alter BEVFusion, M1, or M2.

The retained state is:

- static probabilistic 3D geometry: closed by G0;
- M2 heavy-tail and circular risk signal: supported;
- downstream predictive utility: tested here;
- M3, set likelihood, health-conditioned artifacts, DC0 partials, official
  val, internal test, world models, maps, and interaction models: forbidden.

## Environment and inputs

```bash
cd /home/zhousc/multimodal-uq/BEVFusion
source /home/zhousc/miniconda3/bin/activate bev
```

R0 reuses the accepted F1-V3 clean train/label-cal authority, the ten G0 dev
conditions, the three frozen M1/M2 checkpoints, and nuScenes trainval metadata.
GT instance tokens are used only for association and targets. Each eligible
track has two to four consecutive matched historical frames (1--2 seconds),
the current matched detection, and six consecutive future GT positions (3
seconds). All states and future positions are transformed into the current
LiDAR frame. Corruption family and severity are never predictor inputs.

## Frozen predictors and propagation

Two predictors are evaluated:

- `CV`: constant velocity;
- `MLP`: a fixed 64-128-128-12 residual MLP trained once on clean train for 15
  epochs with seed 20260804, then frozen.

Both consume four padded historical 3D states, current state, history mask,
and class. The MLP predicts a residual over constant velocity. Clean
label-cal fits one shared 2x2 additive forecast-residual covariance per future
horizon and predictor. That same covariance is used for P0--P3, so only the
current perception distribution differs:

- P0: GT current-state oracle;
- P1: deterministic detector mean;
- P2: M1 diagonal Gaussian state and wrapped-Gaussian yaw;
- P3: M2 diagonal multivariate Student-t state and von Mises yaw.

Each probabilistic result uses 64 propagated samples. Three preregistered
replicates pair the three frozen head seeds with propagation seeds 20260806,
20260807, and 20260808. P2 and P3 linear states use common normal/uniform
random numbers; their different circular laws use their native samplers.
P0--P3 use common process-noise draws.

## Run

Use GPUs 0--3 only:

```bash
python tools/riskprop3d_r0.py prepare
python tools/riskprop3d_r0.py train --device 0
python tools/riskprop3d_r0.py evaluate --replicate 0 --device 0
python tools/riskprop3d_r0.py evaluate --replicate 1 --device 1
python tools/riskprop3d_r0.py evaluate --replicate 2 --device 2
python tools/riskprop3d_r0.py finalize
```

The three evaluation commands may run concurrently.

## Metrics and frozen gates

Trajectory energy score is the primary proper score. A shared-kernel mixture
trajectory NLL is diagnostic. The audit also reports predictive-mean and
best-of-64 ADE/FDE, miss rate at 2 m, 50/80/90% coverage, final-horizon ellipse
area, calibration--sharpness, high-FDE AUROC/AUPRC, risk--coverage, false-safe
rate, and a future ego-proximity proxy.

All primary confidence intervals use 10,000 paired-scene bootstrap resamples
with seed 20260805. Positive improvement means P3 is better.

1. On clean, P3 energy score must beat P2 and P1 with CI lower bounds above
   zero for both CV and MLP.
2. P3-over-P2 energy gain must be significant in at least two degradation
   families, non-significantly reversed in the third, and retain that pattern
   on S1/S2 without S3 for both predictors.
3. At empirical 90% final-position coverage, P3 median area must be at most
   1.10x P2; alternatively, at P2's natural 90% area P3 coverage must be
   higher. This must hold on clean and the three family aggregates.
4. Relative to P1 predictive-mean trajectories, P3 may increase ADE by at
   most 0.05 m and FDE by at most 0.10 m on clean and every family.
5. P3 must significantly improve high-FDE AUPRC, high-FDE risk--coverage, or
   false-safe rate over P2 on clean or on a non-beam-S3-only family aggregate.
6. After controlling score, query density, M1 final area, class, range,
   velocity, history length, and history-innovation residual, P3 final area
   must retain positive partial correlation and positive nested-model delta
   R2 with future FDE for both predictors.
7. All three replicates must have the same clean P3-over-P2 energy direction,
   and CV and MLP must agree on the direction of Gates 1--6.

Passing all gates yields `GO TO RISKPROP3D METHOD DEVELOPMENT`. Otherwise the
endpoint is `STOP PERCEPTION-TO-FORECAST PROPAGATION`.

## Result

**GO TO RISKPROP3D METHOD DEVELOPMENT** (`7/7` gates passed).

- On clean, P3 improved energy over P2 by 0.01614
  (95% CI 0.01102--0.02150) with CV and 0.01230
  (0.00850--0.01619) with MLP.
- Lowlight, dropout, and beam all improved for both predictors, including
  when S3 was excluded. Family gains ranged from 0.01411 to 0.04902; the
  S1/S2-only gains ranged from 0.01222 to 0.02963.
- Equal-coverage final-area ratios were 0.866--0.929 versus P2. The largest
  ADE/FDE increases over P1 were 0.01125 m and 0.01692 m.
- Clean high-FDE AUPRC improved by 0.03745 with CV and 0.03251 with MLP.
- M2 uncertainty retained incremental future-error information: partial
  correlation was 0.28988/0.27223 and delta R2 was 0.03278/0.02860 for
  CV/MLP.
- All three propagation replicates had positive clean directions. All
  126,936,582 retained numeric values were finite across 330,564 condition
  tracks. Official validation and internal test remained unread.

The complete machine-readable evidence is `outputs/riskprop3d_r0/result.json`.

## Code and retained outputs

- `tools/riskprop3d_r0.py`: track construction, shared predictor training,
  propagation, metrics, bootstrap inference, and final decision.
- `outputs/riskprop3d_r0/tracks/*.npz`: canonical clean train/label-cal and ten
  dev-condition track inputs.
- `outputs/riskprop3d_r0/predictor.pt`: the one frozen MLP and the two
  label-cal process-noise authorities.
- `outputs/riskprop3d_r0/metrics/replicate_*.npz`: paired per-track results.
- `outputs/riskprop3d_r0/result.json`: final gate evidence and endpoint.

Temporary logs and visualizations are not retained.

## Common problems

- A missing future annotation or fewer than two consecutive matched history
  frames excludes that track mechanically.
- Any GT-row/annotation mismatch above 1e-4 m stops preparation.
- Existing retained outputs are never overwritten.
- CUDA commands reject device indices outside 0--3.
