# RiskProp3D R1.5 DSE-only transfer and mechanism audit

## Frozen question and boundary

R1.5 accepts the valid R1 endpoint `STOP RISKPROP3D METHOD ESCALATION` and
does not reinterpret it. It asks one narrower question:

> Does the already-trained distributional state encoder (DSE), without the
> unsupported interaction branch, retain a stable learning-based propagation
> advantage on clean and unseen Camera/LiDAR degradations?

This is a read-only audit. It uses the three frozen R1 metrics archives and R1
checkpoints; it trains no model, changes no hyperparameter, generates no new
trajectory prediction, and does not read official validation or internal test.
`R1_interaction` is a closed negative control and is not eligible for model
selection.

## Frozen comparisons and inference

The retained groups are:

- `B0`: deterministic mean-state predictor;
- `B1_MC`: 64-sample M2 Monte Carlo propagation through `B0`;
- `B2_concat`: raw uncertainty feature concatenation;
- `R1_DSE`: dedicated target-centric distributional state encoder.

All lower-is-better improvements are `baseline - R1_DSE`. Risk improvements
use candidate-minus-baseline for AUPRC/AUROC and baseline-minus-candidate for
risk--coverage and false-safe rate, matching R1. Per-track differences are
averaged within scene, then across the three paired replicates. All primary
95% confidence intervals use 10,000 scene bootstrap resamples with seed
`20260805`.

Family aggregates average the available scene result across severities before
bootstrap. Both S1/S2/S3 and S1/S2-only aggregates are frozen. High-FDE events
remain `B0` predictive-mean FDE above 2 m, and risk remains final predicted
log-area; no new risk label or score is introduced.

## Validity checks

Failure yields `INVALID R1.5 AUDIT` rather than a scientific endpoint.

- the source R1 result must be valid and retain its frozen STOP endpoint;
- all three metrics hashes must match the R1 machine-readable evidence;
- head/model/sampling seeds and common rows must match across replicates;
- every numeric input used by R1.5 must be finite;
- official validation and internal test must remain unread.

## Frozen transfer gates

1. **T1 clean replication.** `R1_DSE` must significantly improve clean energy
   and trajectory NLL over `B2_concat`.
2. **T2 cross-degradation.** `R1_DSE` energy over `B2_concat` must be
   significant in at least two of lowlight/dropout/beam and not significantly
   reversed in the third. The same rule must hold on S1/S2 without S3.
3. **T3 against Monte Carlo.** `R1_DSE` must either significantly improve
   clean energy over `B1_MC`, or be non-significantly worse while measured
   prediction time is at least 2x faster.
4. **T4 sharpness.** At empirical 90% final-position coverage, median
   `R1_DSE` ellipse area must be at most 1.10x `B1_MC` on clean, every full
   family aggregate, and every S1/S2-only family aggregate.
5. **T5 risk utility.** Relative to `B1_MC`, `R1_DSE` must significantly
   improve high-FDE AUPRC, risk--coverage, or false-safe rate on clean or an
   S1/S2-only family aggregate. Beam S3 cannot contribute.

All five gates yield `GO TO RISKPROP3D-CORE`. A valid failure yields
`STOP DSE AS THIRD-WORK CORE`.

## Result

**GO TO RISKPROP3D-CORE** (`5/5` transfer gates passed; the audit is valid).

- T1 passed. On clean, `R1_DSE` improved energy over `B2_concat` by 0.01965
  (95% CI 0.00895--0.03100) and trajectory NLL by 0.19392
  (0.10068--0.28678), exactly reproducing the frozen R1 comparison.
- T2 passed strongly. DSE-over-concat energy gains were significant for every
  individual degradation condition. Full-family gains were
  0.01965/0.01721/0.02070 for lowlight/dropout/beam, with CI lower bounds
  0.00945/0.00473/0.00892. S1/S2-only gains were
  0.01928/0.01834/0.01905, with all CI lower bounds above zero.
- T3 passed through the primary energy branch. DSE improved clean energy over
  64-sample MC by 0.05756 (0.04722--0.06851) and was 5.72x faster
  (0.01805 versus 0.10327 ms per target). Diagnostic mixture NLL moved in the
  opposite direction: the MC-minus-DSE difference was -3.39891
  (-3.63046---3.17860). The retained claim is therefore an energy-score and
  quality--runtime advantage, not an NLL advantage over MC.
- T4 passed. Equal-coverage area ratios versus MC were 0.882 on clean and
  0.880/0.897/0.940 on the full lowlight/dropout/beam aggregates. The largest
  individual ratio was 0.968 on beam S3; every full and S1/S2-only ratio was
  below 1.0 and the frozen 1.10 ceiling.
- T5 passed without beam S3. Clean high-FDE AUPRC improved by 0.01299
  (0.00241--0.02408) and false-safe rate improved by 0.00880
  (0.00134--0.01574). AUPRC and false-safe improvements were also significant
  for all three S1/S2-only family aggregates.
- All five validity checks passed. The three metrics hashes, registered seeds,
  common rows, and 88,260,897 numeric metric values were verified. Official
  validation and internal test remained unread.

The result supports a target-centric probabilistic perception-to-forecast
interface: correct distribution representation transfers, while the frozen
pairwise distributional interaction v1 remains not supported and closed.
Complete evidence is `outputs/riskprop3d_r15/result.json`.

## Run

```bash
cd /home/zhousc/multimodal-uq/BEVFusion
source /home/zhousc/miniconda3/bin/activate bev

python tools/riskprop3d_r15.py self-check
python tools/riskprop3d_r15.py audit
```

The audit writes only `outputs/riskprop3d_r15/result.json` and never
overwrites it.
