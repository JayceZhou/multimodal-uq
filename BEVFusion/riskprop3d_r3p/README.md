# RiskProp3D R3-P probability-token necessity audit

## Frozen question and boundary

R3-P is the final positioning experiment for the current third-work direction.
It asks only:

> Does perception uncertainty itself improve ordinary supervised forecasting,
> beyond model capacity, detector confidence, and Gaussian uncertainty?

R3-P is not new RiskProp3D method development. It adds no loss, module,
teacher/distillation, map, interaction, risk head, stronger host, mechanism
subset, detector/UQ head, official validation, or internal test. It reuses the
R2-D D2 student architecture and GT-only objective exactly.

## Frozen groups and token contract

All groups receive the same 64-dimensional R0 history/current/class vector and
a 20-dimensional token. They have the identical `84-96-64-150` architecture,
24,118 trainable parameters, within-replicate initial weights, six-mode
diagonal-Gaussian decoder, GT loss, batch order, 15 epochs, and clean
train/label-cal/dev rows.

The common mean authority is M2 mean state and yaw mean for all four groups.
This prevents M1/M2 mean differences from being misinterpreted as uncertainty
benefits.

- `P0_mean_only`: M2 mean state and yaw sine/cosine; all uncertainty
  coordinates are constants.
- `P1_confidence`: P0 plus detector score and `log(1 + query density)`; all
  probability coordinates remain constants.
- `P2_gaussian`: the common M2 mean/yaw plus M1 Gaussian log scales, a constant
  Gaussian-limit tail coordinate, and wrapped-Gaussian yaw precision
  `log(1 + 1 / sigma_yaw^2)`.
- `P3_m2_probability`: the exact R2-D D2 raw token: M2 mean state, eight log
  scales, `log(df - 2)`, yaw sine/cosine, and `log(1 + kappa)`.

Each group has its own clean-train input normalization because its token has a
different numerical meaning. Normalization buffers are not trainable;
trainable weights are initialized identically. Condition/family/severity and
current GT residual never enter a token.

## Frozen training and evaluation

Every group minimizes only

```text
L_GT = trajectory mixture NLL / 12 + 0.1 * normalized best-mode ADE.
```

Clean label-cal selects one global scale temperature per group. Evaluation
uses 64 common-random-number samples from each learned mixture. Trajectory
energy score is primary. Exact mixture NLL, ADE/FDE, high-FDE AUPRC,
risk--coverage, false-safe rate, and empirical equal-coverage area are always
reported.

High FDE is the same-replicate P0 predictive-mean FDE above 2 m. Risk ranking
uses predicted final log ellipse area. False-safe rate uses the lowest-risk
80% of tracks within scene; all risk differences are computed scene-wise
before paired bootstrap inference.

Positive lower-is-better improvement is `baseline - P3`. Per-track
differences are averaged within scene and then across three paired replicates.
Confidence intervals use 10,000 paired-scene bootstrap resamples with seed
`20260805`. Families are evaluated over S1/S2/S3 and again over S1/S2.

The replicates pair accepted head seeds `20260731/20260801/20260802`, model
seeds `20260830/20260831/20260832`, and sampling seeds
`20260833/20260834/20260835`.

## Validity checks

Any failure yields `INVALID R3-P RUN` and no scientific endpoint.

- R0, the valid R2-Core STOP, and the valid R2-D STOP authorities and all
  referenced source hashes remain frozen;
- R0/R1 train, label-cal, and dev rows match and splits remain scene-disjoint;
- all four groups have identical architecture/parameter count and matching
  within-replicate trainable initialization hashes;
- every group uses only the frozen GT-only loss, epochs, batch order, and
  label-cal temperature rule;
- all three preregistered seed tuples and common evaluation rows are present;
- every retained numeric value is finite;
- corruption identity and current GT residual are absent from model inputs;
- official validation and internal test remain unread.

## Frozen gates

1. **G1 mean-only necessity.** P3 over P0 energy must improve significantly on
   clean and in at least two full lowlight/dropout/beam families without
   significant reversal in the third. The same family rule must hold on S1/S2
   without S3.
2. **G2 beyond confidence.** P3 must significantly improve clean energy over
   P1. Full and S1/S2 family results are reported but not gated.
3. **G3 M2 necessity.** P3 must significantly improve clean energy over P2.
   Full and S1/S2 family results are reported but not gated. Failure forbids a
   heavy-tail/circular input-contribution claim.
4. **G4 risk increment.** On clean, the same one of high-FDE AUPRC,
   risk--coverage, or false-safe rate must significantly improve for P3 over
   both P0 and P1.
5. **G5 sharpness and accuracy.** At empirical 90% coverage, P3 median final
   area must be at most `1.05x` the sharpest P0/P1/P2 baseline on clean and
   every full family. Relative to P0, P3 may increase predictive-mean ADE by at
   most 0.05 m and FDE by at most 0.10 m on clean and every full family.
6. **G6 seed stability.** All three clean energy directions must be positive
   for P3 over each of P0, P1, and P2.

## Frozen endpoints

- All G1--G6 pass: `PERCEPTION-DISTRIBUTION CONDITIONING SUPPORTED`.
- Any valid gate failure: `STOP RISKPROP3D DIRECTION`.

There is no R4 rescue experiment after a valid STOP. A GO supports a robust
uncertainty-conditioned forecasting/benchmark positioning, not a claim of a
new propagation theory or automatic strong-CVPR-level novelty.

## Run

Use GPUs 0--3 only:

```bash
cd /home/zhousc/multimodal-uq/BEVFusion
source /home/zhousc/miniconda3/bin/activate bev

python tools/riskprop3d_r3p.py prepare
python tools/riskprop3d_r3p.py self-check
python tools/riskprop3d_r3p.py train --replicate 0 --device 0
python tools/riskprop3d_r3p.py train --replicate 1 --device 1
python tools/riskprop3d_r3p.py train --replicate 2 --device 2
python tools/riskprop3d_r3p.py evaluate --replicate 0 --device 0
python tools/riskprop3d_r3p.py evaluate --replicate 1 --device 1
python tools/riskprop3d_r3p.py evaluate --replicate 2 --device 2
python tools/riskprop3d_r3p.py finalize
```

Replicates may run concurrently. Retained outputs are atomic and never
overwritten.

## Result

**STOP RISKPROP3D DIRECTION** (`2/6` gates passed; the run is valid).

- G1 failed. P3 improved clean energy over mean-only P0 by only 0.00205
  (95% CI -0.01378--0.01559). Full-family point estimates were
  0.00247/0.00418/-0.00232 for lowlight/dropout/beam, and every CI crossed
  zero. The S1/S2-only family results were likewise non-significant.
- G2 failed. P3-over-confidence P1 clean energy was 0.00069
  (-0.01531--0.01321). Detector score/query density therefore cannot be
  excluded as an equally effective conditioning baseline on the primary
  score.
- G3 failed. P3 was non-significantly worse than Gaussian-token P2 on clean:
  the P2-minus-P3 improvement was -0.00140 (-0.01454--0.00943). No learned
  M2-over-M1 token necessity or heavy-tail/circular input-contribution claim
  is supported.
- G4 passed only through risk--coverage. P3 improved clean risk--coverage over
  P0 by 0.00699 (0.00065--0.01369) and over P1 by 0.00672
  (0.00027--0.01381). High-FDE AUPRC and false-safe rate did not pass against
  both baselines.
- G5 passed. The equal-coverage P3 area ratio to the sharpest baseline was
  1.0251 on clean and 1.0031--1.0265 on full family aggregates. P3 slightly
  improved clean ADE/FDE over P0 by 0.00304/0.00910 m; the largest adverse
  full-family changes were only 0.00431/0.00580 m on beam.
- G6 failed because one seed reversed P3 versus P2 (-0.00046). All three seeds
  favored P3 over P0 and P1, but those aggregate clean effects were not
  significant.
- Diagnostic mixture NLL significantly favored P3 over P0 by 0.32989
  (0.19746--0.45760) and over P1 by 0.29304 (0.16806--0.40656), but was
  non-significant against P2. These secondary results do not repair the failed
  primary necessity gates.
- All eight validity checks passed. All 67,729,089 audited numeric values were
  finite across 330,564 condition tracks. Official validation and internal
  test remained unread.

The final positioning audit therefore does not isolate an incremental
trajectory-energy benefit from the M2 probability token under ordinary GT
supervision. R0's direct stochastic-propagation phenomenon remains accepted
as its bounded result, but the learned RiskProp3D direction is closed and no
R4 rescue is authorized.

Complete machine-readable evidence is
`outputs/riskprop3d_r3p/result.json`.

## Retained outputs

- `outputs/riskprop3d_r3p/prepare_manifest.json`: frozen source, split, row,
  checkpoint, and seed authorities;
- `outputs/riskprop3d_r3p/checkpoints/replicate_*.pt`: four paired GT-only
  models, normalization buffers, initialization evidence, and label-cal
  temperatures;
- `outputs/riskprop3d_r3p/metrics/replicate_*.npz`: common-row per-track
  metrics for ten conditions and three seeds;
- `outputs/riskprop3d_r3p/result.json`: validity, paired confidence intervals,
  gates, evidence hashes, and final endpoint.
