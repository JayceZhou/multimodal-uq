# RiskProp3D R2-Core mechanism audit

## Frozen question and boundary

R2-Core starts only because the read-only R1.5 audit reached
`GO TO RISKPROP3D-CORE`. It asks why the target-centric distributional state
encoder (DSE) works:

1. Is its gain merely extra model capacity?
2. Would the same encoder work equally well with M1 Gaussian/wrapped-Gaussian
   perception?
3. Does M2 tail information contribute beyond its covariance?
4. Does circular yaw concentration contribute beyond yaw mean?

R2-Core does not add a map, interaction, world model, planning or collision
loss, a risk head, a new detector/UQ head, or a new forecasting backbone. M1,
M2, BEVFusion, the R1 scene authority, official validation, and internal test
remain frozen. Distributional Interaction v1 remains closed.

## Frozen models

All six groups use the R1 history/class input, constant-velocity residual, the
same 6-mode diagonal-Gaussian trajectory decoder, 15 clean-train epochs, and
the same `mixture NLL + 0.1 * best-mode ADE` loss. Clean label-cal selects one
global scale temperature per group. No mechanism subset is a model input.

- `C0_mean_matched`: identical DSE/fusion/decoder parameterization to C2--C5,
  but its auxiliary branch receives only normalized mean state, yaw
  `sin/cos`, and constants. Every scale, moment, log determinant, tail, and
  concentration input is removed.
- `C1_raw_concat`: the R1 raw M2 scale/df/kappa concatenation baseline.
- `C2_gaussian_dse`: the identical DSE architecture receives M1 Gaussian
  linear variance, wrapped-Gaussian mean resultant `exp(-sigma_yaw^2/2)`, and
  constant Gaussian-limit tail inputs.
- `C3_no_tail`: receives full M2 mean, covariance/marginal moments, log
  determinant, radial/principal uncertainty, and circular concentration, but
  the two df/tail-specific inputs are constants. Student-t covariance may
  depend on df because covariance is explicitly retained; df variation is not
  exposed separately.
- `C4_no_circular`: receives full M2 linear moments and tail inputs plus yaw
  mean `sin/cos`, but circular concentration/resultant is fixed to one.
- `C5_full_dse`: full M2 Student-t moments/df and von Mises mean resultant.

C0/C2/C3/C4/C5 have exactly the same trainable parameter count and receive
identical initial weights within replicate. Only their frozen distribution
adapter differs. C1 retains its deliberately simpler raw-concat encoder.

## Frozen subsets

Subset thresholds are fixed before training and are evaluation-only:

- `high_current_residual`: current XY perception error above the clean
  label-cal 80th percentile;
- `far_range`: detector range at least 40 m;
- `sparse_lidar`: current GT annotation has at most five LiDAR points, used
  only as a scene-geometry sparsity proxy (not corruption-adjusted point
  support);
- `yaw_boundary`: current local detector yaw lies within 30 degrees of
  `-pi/pi`;
- `turning`: angle between the first and last GT future path segments exceeds
  30 degrees, with both segments longer than 0.5 m;
- `future_heading_change`: absolute GT annotation yaw change from current to
  the sixth future frame exceeds 30 degrees;
- `crossing`: the frozen R1 geometric crossing flag;
- `vehicle`: class in car/truck/construction-vehicle/bus/trailer;
- `two_wheeler`: motorcycle or bicycle;
- `high_fde`: same-replicate C0 predictive-mean FDE above 2 m.

GT residuals, future states, and annotation point counts are used only to
define post-prediction analysis subsets. They are never read by model forward
or training loss.

## Metrics and inference

Trajectory energy score remains primary. High-FDE AUPRC, false-safe rate,
risk--coverage, equal-coverage area, ADE/FDE, and seed directions are
secondary. Exact mixture trajectory NLL remains diagnostic and is always
reported, including directions adverse to C5.

Positive lower-is-better improvement is `ablation - C5`. Per-track
differences are averaged within scene, then across three paired replicates.
Primary 95% CIs use 10,000 scene bootstrap resamples with seed `20260805`.
Families are audited over S1/S2/S3 and again over S1/S2 without S3.

The replicates pair the frozen head seeds `20260731/20260801/20260802`, model
seeds `20260815/20260816/20260817`, and sampling seeds
`20260818/20260819/20260820`.

## Validity checks

Any failure yields `INVALID R2-CORE RUN` and no scientific endpoint.

- R1.5 must retain its valid GO endpoint and all R1 scene/checkpoint hashes
  must match the frozen source authorities;
- train/label-cal/dev remain scene-disjoint and all subset rows match R0/R1
  sample/instance/future authority;
- C0/C2/C3/C4/C5 parameter counts and within-replicate initialization match;
- all three preregistered seed triples and common evaluation rows match;
- every retained numeric value is finite;
- official validation and internal test remain unread.

## Frozen gates

1. **G1 capacity.** C5 over C0 energy must be significant on clean, in at
   least two full degradation families with no significant reversal in the
   third, and retain the same rule on S1/S2. C5 ADE/FDE is handled by G7.
2. **G2 distribution family.** C5 over C2 energy must be significant on clean,
   in at least two full families with no significant reversal, and retain the
   same rule on S1/S2.
3. **G3 tail contribution.** C5 over C3 clean energy must not be significantly
   reversed. At least one of high-current-residual, far-range, sparse-LiDAR,
   or high-FDE clean subsets must show a significant gain larger than the
   all-track point estimate. At least one full degradation family must improve
   significantly and none may significantly reverse.
4. **G4 circular contribution.** C5 over C4 clean energy must not be
   significantly reversed, and either overall clean or a yaw-sensitive subset
   must improve significantly. At least one of yaw-boundary, turning, or
   future-heading-change must show a significant gain larger than the
   all-track point estimate. Crossing, vehicle, and two-wheeler are reported
   as additional yaw-sensitive evidence.
5. **G5 risk utility.** For each key ablation C0--C4, C5 must significantly
   improve high-FDE AUPRC, false-safe rate, or risk--coverage on clean or an
   S1/S2-only family aggregate.
6. **G6 sharpness.** At empirical 90% coverage, C5 median final area must be at
   most 1.05x the sharpest C0--C4 ablation on clean, every full family, and
   every S1/S2-only family aggregate.
7. **G7 accuracy.** Relative to C0, C5 may increase predictive-mean ADE by at
   most 0.05 m and FDE by at most 0.10 m on clean and every full family.
8. **G8 seed stability.** All three clean-energy directions must be positive
   for C5 over every C0--C4 ablation.

## Frozen endpoints

- All G1--G8 pass: `GO TO R3 STRONG-HOST GENERALIZATION`.
- G1/G2/G5--G8 pass and exactly one or both of G3/G4 pass:
  `GO TO R3 WITH SCOPED MECHANISM CLAIM`; the claim names only supported
  tail/circular components.
- G1 passes but G2 fails: `REASSESS RISKPROP3D PAPER POSITIONING` because the
  evidence supports distribution-aware encoding but not M2 family structure.
- Otherwise: `STOP RISKPROP3D-CORE MECHANISM CLAIM`.

## Run

Use GPUs 0--3 only:

```bash
cd /home/zhousc/multimodal-uq/BEVFusion
source /home/zhousc/miniconda3/bin/activate bev

python tools/riskprop3d_r2_core.py prepare
python tools/riskprop3d_r2_core.py self-check
python tools/riskprop3d_r2_core.py train --replicate 0 --device 0
python tools/riskprop3d_r2_core.py train --replicate 1 --device 1
python tools/riskprop3d_r2_core.py train --replicate 2 --device 2
python tools/riskprop3d_r2_core.py evaluate --replicate 0 --device 0
python tools/riskprop3d_r2_core.py evaluate --replicate 1 --device 1
python tools/riskprop3d_r2_core.py evaluate --replicate 2 --device 2
python tools/riskprop3d_r2_core.py finalize
```

Replicates may run concurrently. Every retained output is written atomically
and never overwritten.

## Result

**STOP RISKPROP3D-CORE MECHANISM CLAIM** (`1/8` gates passed; the run is
valid).

- G1 failed. C5 improved clean energy over the capacity-matched mean-only C0
  by 0.00690, but its 95% CI crossed zero (-0.00150--0.01605). The three
  degradation families were positive, including without S3, but cannot repair
  the missing clean result.
- G2 failed. C5 over Gaussianized C2 was 0.00508 on clean
  (-0.00775--0.01753). Only dropout was significant; full beam had a negative
  point estimate of -0.00466 (-0.02123--0.00829).
- G3 failed despite a significant all-track clean C5-over-C3 gain of 0.00796
  (0.00080--0.01570). None of the frozen tail-sensitive subsets was
  significant: the largest point estimate was 0.02451 on high-FDE tracks
  (-0.00525--0.05561).
- G4 failed. C5 was non-significantly worse than C4 on clean by 0.00228, and
  no core yaw-sensitive subset improved. The two-wheeler subset significantly
  favored removing circular concentration (-0.04457;
  -0.10014---0.00186).
- G5 failed. C5 had a supported risk advantage over C0, C2, and C3, but not
  over raw concat C1 or no-circular C4. G6 also failed: equal-coverage area
  ratios to the sharpest ablation were 1.0699 on clean and 1.0525--1.0668 on
  the frozen family aggregates, above the 1.05 limit.
- G7 passed. The largest full-family predictive-mean ADE/FDE increases over
  C0 were only 0.00093/0.00164 m. G8 failed because clean energy directions
  were not stable across all three seeds; the third seed reversed C5 versus
  C1, C2, and C4, and the second reversed C5 versus C0.
- Diagnostic mixture NLL significantly favored C5 only against C0 on clean
  (gain 0.37174); comparisons with C1--C4 were non-significant. This audit
  does not change the R1.5 limitation that DSE mixture NLL is worse than MC;
  no claim of universal probabilistic-metric superiority is supported.
- All six validity checks passed. All 107,342,589 audited numeric values were
  finite across 330,564 condition tracks. Official validation and internal
  test remained unread.

The capacity-matched audit therefore does not support escalation to a strong
host or a causal heavy-tail/circular mechanism claim. The positive R1.5
DSE-over-concat/MC result remains a bounded empirical finding, but it is not
stable enough under new model seeds and matched ablations to establish
RiskProp3D-Core as the third-work mechanism.

Complete machine-readable evidence is
`outputs/riskprop3d_r2_core/result.json`.

## Retained outputs

- `outputs/riskprop3d_r2_core/subsets/*.npz`: frozen evaluation-only mechanism
  subsets aligned to the R1 scene rows;
- `outputs/riskprop3d_r2_core/prepare_manifest.json`: source, checkpoint,
  subset, and split hashes;
- `outputs/riskprop3d_r2_core/checkpoints/replicate_*.pt`: six trained variants,
  capacity/initialization evidence, and label-cal temperatures;
- `outputs/riskprop3d_r2_core/metrics/replicate_*.npz`: paired per-target
  metrics for ten conditions and three seeds;
- `outputs/riskprop3d_r2_core/result.json`: final validity, confidence
  intervals, gates, and endpoint.
