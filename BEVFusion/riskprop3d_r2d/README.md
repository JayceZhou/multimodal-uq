# RiskProp3D R2-D risk-preserving amortized propagation audit

## Frozen question and boundary

R2-D starts from two retained facts: R0 supports direct M2 Monte Carlo (MC)
propagation, while the valid R2-Core run closes the DSE heavy-tail/circular
mechanism claim. R2-D asks one narrower method question:

> Can a single learned transport preserve the future distribution, calibration,
> sharpness, and downstream risk utility of the accepted 64-sample M2
> stochastic pushforward?

R2-D does not revive C5 or distributional interaction and does not add a map,
strong host, world model, planning/collision loss, separate risk head, new
detector/UQ head, corruption input, official validation, or internal test.
The frozen R0 MLP is the only teacher host. CV remains part of the R0 evidence
but is not a distillation host because it does not represent the learned
64-forward computation that amortization is intended to replace.

## Frozen data and teacher

Train, label-cal, and the ten dev conditions reuse the R0 track order. The R1
scene files supply only the already hash-frozen head inputs missing from the R0
clean-train track archive. R0 and R1 sample token, instance token, input, and
future arrays must match row for row before training.

For each replicate, the teacher draws 64 current states from the accepted M2
Student-t/von Mises head, propagates them through the frozen R0 MLP, and adds
the frozen label-cal MLP process noise. Teacher generation therefore preserves
the complete empirical stochastic pushforward rather than fitting a Gaussian
teacher. Dev teacher replay must reproduce the retained R0 MLP P1/P3 metrics.

## Frozen groups

- `D0_deterministic`: the retained R0 MLP P1 distribution, consisting of its
  deterministic center trajectory plus frozen process noise.
- `D1_mc_teacher`: the retained R0 MLP P3 64-sample M2 pushforward.
- `D2_gt_only`: a single-pass six-mode diagonal-Gaussian student trained only
  against future GT.
- `D3_risk_distilled`: the identical student trained with GT, distribution
  transport, calibration--sharpness, and risk-preservation objectives.

D2 and D3 receive exactly the same input and identical initial weights within
replicate. Their input is the 64-dimensional R0 history/current/class vector
plus a 20-dimensional raw probability token: eight M2 means, eight log scales,
`log(df - 2)`, yaw-mean sine/cosine, and `log(1 + kappa)`. These coordinate
transforms are for numerical stability and carry no tail/circular mechanism
claim.

The student is one `84-96-64-150` MLP. Its 150 outputs parameterize six mode
weights and six residual future means/scales over constant velocity. It has
24,118 parameters, no more than the frozen R0 MLP's 26,380. Clean train is used
for 15 epochs; clean label-cal selects one global scale temperature per group.

## Frozen objectives

D2 minimizes only

```text
L_GT = trajectory mixture NLL / 12 + 0.1 * normalized best-mode ADE.
```

D3 minimizes

```text
L_D3 = L_GT
     + 1.00 * L_transport
     + 0.25 * L_moment
     + 0.25 * L_calsharp
     + 0.25 * L_risk.
```

All trajectory distances are normalized by clean-train residual scales.

- `L_transport` is a weighted sample energy-distance estimator between the
  six-mode student and all 64 teacher trajectories. It includes cross- and
  student-self terms; the teacher-self constant is omitted only during
  optimization.
- `L_moment` matches horizon-wise mean, log marginal variance, and XY
  correlation of the student mixture to the empirical teacher moments.
- `L_calsharp` matches teacher/GT Mahalanobis behavior and adds a one-sided
  penalty only when student final log determinant exceeds the teacher's.
- `L_risk` matches standardized final log-area risk scores and their batch
  correlation. Risk is derived from the trajectory distribution; there is no
  learned classifier or separate risk output.

Thus the method is not plain trajectory imitation. It explicitly constrains
distribution transport, calibration--sharpness, and teacher risk ordering.

## Seeds and evaluation

The replicates use accepted M2 head seeds `20260731/20260801/20260802`, train
teacher seeds `20260821/20260822/20260823`, model seeds
`20260824/20260825/20260826`, and student evaluation seeds
`20260827/20260828/20260829`. Dev D0/D1 replay retains the original R0
propagation seeds `20260806/20260807/20260808`. Bootstrap uses 10,000 paired
scene resamples with seed `20260805`.

Primary downstream quality remains trajectory energy score. Exact mixture NLL,
ADE/FDE, risk metrics, and all adverse directions remain diagnostic or
secondary. Teacher fidelity reports sample energy distance, trajectory first-
moment error, horizon-wise covariance error, and absolute 90% coverage gap.

Prediction runtime excludes the frozen upstream head for both methods. D1 time
starts from its M2 parameters and includes state sampling, 64 MLP propagations,
and process-noise construction. D3 time includes raw-token construction,
device transfer, and its one forward pass; downstream metric sampling is
excluded for both.

## Validity checks

Any failure yields `INVALID R2-D RUN` and no scientific endpoint.

- the R0 GO and valid R2-Core STOP authorities and all referenced hashes match;
- R0/R1 train, label-cal, and dev rows match and splits remain scene-disjoint;
- dev D0/D1 replay matches every retained R0 MLP P1/P3 metric within `1e-5`;
- D2/D3 parameter counts and within-replicate initialization hashes match;
- all three preregistered seed tuples and common rows are present;
- all retained numeric values are finite;
- corruption identity and current GT residual are absent from model inputs;
- official validation and internal test remain unread.

## Frozen gates

Positive improvement means D3 is better unless stated otherwise.

1. **G1 teacher fidelity.** On clean, D3 must significantly improve over D2
   in teacher energy distance, first-moment error, second-moment error, and
   absolute 90% coverage gap; every paired-scene 95% CI lower bound must exceed
   zero.
2. **G2 downstream proper score.** D3 over D2 energy must improve
   significantly on clean and in at least two full lowlight/dropout/beam
   families without significant reversal in the third. The same family rule
   must hold on S1/S2 without S3.
3. **G3 MC equivalence.** On clean and each full degradation family, D3 must
   either significantly beat D1 or pass paired equivalence. Equivalence uses
   a two-sided 90% bootstrap CI for `ES_D3 - ES_D1` wholly inside
   `+/- 0.02 * mean(ES_D1)`; a 95% CI upper bound below zero counts as direct
   superiority. S1/S2-only results are reported but not gated.
4. **G4 risk retention.** On clean, D1 must significantly improve over D0 and
   D3 must retain at least 90% of that gain for high-FDE AUPRC or
   risk--coverage. The paired-scene 95% CI lower bound of
   `gain_D3 - 0.9 * gain_D1` must exceed zero for at least one metric.
5. **G5 sharpness.** At empirical 90% coverage, D3 median final area must be at
   most `1.05x` D1 on clean and every full family. S1/S2-only ratios are also
   reported.
6. **G6 efficiency.** D3 measured prediction time must be at least `10x`
   faster than D1 on clean after averaging replicates, and its parameter count
   must be at most `1.10x` the frozen R0 MLP.

## Frozen endpoints

- All G1--G6 pass: `GO TO R3 STRONG-HOST AMORTIZED PROPAGATION`.
- Any valid gate failure: `STOP LEARNED AMORTIZED PROPAGATION`.

No intermediate endpoint permits tuning D3, changing the loss weights, adding
a risk head, or opening official validation.

## Run

Use GPUs 0--3 only:

```bash
cd /home/zhousc/multimodal-uq/BEVFusion
source /home/zhousc/miniconda3/bin/activate bev

python tools/riskprop3d_r2d.py prepare
python tools/riskprop3d_r2d.py self-check
python tools/riskprop3d_r2d.py train --replicate 0 --device 0
python tools/riskprop3d_r2d.py train --replicate 1 --device 1
python tools/riskprop3d_r2d.py train --replicate 2 --device 2
python tools/riskprop3d_r2d.py evaluate --replicate 0 --device 0
python tools/riskprop3d_r2d.py evaluate --replicate 1 --device 1
python tools/riskprop3d_r2d.py evaluate --replicate 2 --device 2
python tools/riskprop3d_r2d.py finalize
```

Replicates may run concurrently. Retained outputs are atomic and never
overwritten.

## Result

**STOP LEARNED AMORTIZED PROPAGATION** (`5/6` gates passed; the run is
valid).

- G1 passed. Relative to the identical GT-only D2, D3 reduced clean teacher
  energy distance by 0.53689 (95% CI 0.50198--0.57474), first-moment error by
  0.10111 (0.07879--0.12572), relative second-moment error by 0.43758
  (0.41599--0.45856), and absolute 90% coverage gap by 0.05178
  (0.04484--0.05903).
- G2 failed in the adverse direction. Clean D2-over-D3 energy improvement was
  -0.05941 (-0.08835---0.02657), so D3 was significantly worse. Full-family
  point estimates were -0.05061/-0.03198/-0.03185 for
  lowlight/dropout/beam; lowlight and beam were significantly adverse. After
  removing S3, all three families significantly favored D2. All three clean
  seed directions were adverse.
- G3 passed by direct superiority rather than equivalence. `ES_D3 - ES_D1`
  was -0.10957 on clean (95% CI -0.12076---0.09758), and D3 also
  significantly beat D1 in all three full degradation families by
  0.09929--0.11080.
- G4 passed. Clean high-FDE AUPRC gain over D0 was 0.54969 for D3 versus
  0.53290 for D1; the 90%-retention contrast was 0.07008
  (0.06196--0.07900). Risk--coverage also passed with a retention contrast of
  0.05371 (0.04455--0.06337).
- G5 passed. Equal-coverage D3/D1 area ratios were 0.9698 on clean and
  0.9703--0.9801 on the full family aggregates. G6 passed: measured clean
  prediction speedup was 38.55x, and D3 used 24,118 parameters, 0.914x the
  frozen 26,380-parameter R0 MLP.
- Diagnostic mixture NLL also favored D2 over D3 by 0.60502 on clean
  (0.49776--0.70776), although D3 remained substantially better than D1 on
  this diagnostic. Distribution preservation therefore did not create an
  incremental proper-score benefit over ordinary GT training.
- All six validity checks passed. D0/P1 and D1/P3 replay matched every R0
  metric exactly (`max_abs = 0.0`) for all conditions and replicates. All
  75,515,193 audited numeric values were finite across 330,564 condition
  tracks. Official validation and internal test remained unread.

The experiment shows that a compact one-forward model can closely preserve
the MC distribution, risk ranking, and sharpness while being much faster.
However, the same architecture trained only on GT is significantly better on
the frozen downstream proper score. The preservation objective is therefore
not supported as a necessary RiskProp3D method contribution, and no R3
strong-host escalation is authorized.

Complete machine-readable evidence is
`outputs/riskprop3d_r2d/result.json`.

## Retained outputs

- `outputs/riskprop3d_r2d/prepare_manifest.json`: frozen source, split,
  checkpoint, and row hashes;
- `outputs/riskprop3d_r2d/checkpoints/replicate_*.pt`: paired D2/D3 models,
  initialization evidence, loss contract, and label-cal temperatures;
- `outputs/riskprop3d_r2d/metrics/replicate_*.npz`: exact D0/D1 replay,
  D2/D3 metrics, teacher-fidelity evidence, and measured prediction times;
- `outputs/riskprop3d_r2d/result.json`: validity, paired confidence intervals,
  TOST evidence, gates, and endpoint.
