# RiskProp3D R1 distributional forecasting prototype

## Frozen question and boundary

R1 asks three questions after the positive R0 propagation audit:

1. Does a structured distributional state encoder (DSE) beat simply appending
   M2 uncertainty scalars?
2. Does uncertainty-aware multi-agent interaction improve over encoding only
   the target distribution?
3. Does learned single-pass propagation improve the quality--runtime trade-off
   of 64-sample Monte Carlo propagation?

R1 freezes BEVFusion and the three accepted M2 Student-t/von Mises heads. It
trains only lightweight motion forecasters on clean train. It does not use a
map, a world model, a planning loss, a separate risk head, corruption identity,
official validation, or internal test.

## Frozen data contract

R1 reuses the accepted F1-V3 train/label-cal authority, the ten G0 dev
conditions, nuScenes trainval metadata, and the R0 eligibility rule. Every
target has two to four consecutive matched historical frames, its current
matched detection, and six future GT centers. GT instance tokens are used only
for association and future targets; no current GT residual is a model input.

For each target, the scene input contains the closest eight other matched
agents within 30 m in the current LiDAR frame. A neighbor need not itself be an
eligible forecast target. Agents are stored once per sample and referenced by
integer target/neighbor indices. Missing neighbor slots are masked. Split scene
sets must be disjoint, and condition/family/severity are never stored in model
features.

The deterministic interaction subsets are frozen as follows:

- `crossing`: distance at most 20 m and absolute heading difference 45--135°;
- `close_following`: distance at most 15 m, heading difference at most 30°,
  and lateral separation at most 3.5 m;
- `overtaking`: distance at most 20 m, heading difference at most 30°, lateral
  separation at most 4 m, and longitudinal relative speed at least 2 m/s;
- `high_density`: at least four retained neighbors;
- `interaction_heavy`: the union of the four subsets above.

`high_uncertainty` is reported separately using the clean label-cal 75th
percentile of the target M2 linear-state log determinant. It is not used by a
gate or as a model input.

## Frozen models

All learned groups use the same history/class input and the same 6-mode
diagonal-Gaussian trajectory decoder. The decoder predicts residual future
centers over constant velocity, per-mode per-horizon scale, and mode weight.
Training minimizes mixture trajectory NLL plus `0.1 * best-mode ADE`. A single
global scale temperature per learned group is selected by clean label-cal NLL
and then frozen for every dev condition.

- `B0`: deterministic current-state mean only.
- `B1_MC`: 64 M2 Student-t/von Mises current states propagated through the
  frozen `B0` of the same replicate.
- `B2_concat`: `B0` input plus normalized raw M2 scale, degrees of freedom,
  and yaw concentration.
- `R1_DSE`: a dedicated encoder of Student-t marginal moments, log determinant,
  radial/principal uncertainty, tail index, and von Mises mean resultant.
- `R1_interaction`: `R1_DSE` plus masked attention over probabilistic relative
  position, velocity, heading conflict, overlap, and closest-approach features.

There is no learned collision or high-FDE classifier. Future-risk ranking uses
the predicted final ellipse area, and the ego-proximity proxy is computed from
trajectory samples.

## Seeds and run

The replicates pair M2 head seeds `20260731/20260801/20260802`, model seeds
`20260809/20260810/20260811`, and sampling seeds
`20260812/20260813/20260814`. Bootstrap inference uses 10,000 paired-scene
resamples with seed `20260805`.

Use GPUs 0--3 only:

```bash
cd /home/zhousc/multimodal-uq/BEVFusion
source /home/zhousc/miniconda3/bin/activate bev

python tools/riskprop3d_r1.py prepare
python tools/riskprop3d_r1.py train --replicate 0 --device 0
python tools/riskprop3d_r1.py train --replicate 1 --device 1
python tools/riskprop3d_r1.py train --replicate 2 --device 2
python tools/riskprop3d_r1.py evaluate --replicate 0 --device 0
python tools/riskprop3d_r1.py evaluate --replicate 1 --device 1
python tools/riskprop3d_r1.py evaluate --replicate 2 --device 2
python tools/riskprop3d_r1.py finalize
```

Commands for different replicates may run concurrently. Retained outputs are
never overwritten. `python tools/riskprop3d_r1.py self-check` is a CPU-only
structural check and does not read or write retained evidence.

## Validity checks

Failure makes the run `INVALID R1 RUN`; it is repaired and rerun rather than
interpreted scientifically.

- detector authority, M2 checkpoints, and split metadata remain hash-frozen;
- train, label-cal, and dev scenes are disjoint;
- current GT residual and corruption identity are absent from model inputs;
- all three replicates are present and use their preregistered seeds;
- every retained numeric value is finite;
- official validation and internal test remain unread.

## Scientific gates

Positive improvement always means the named R1 method is better. Primary
confidence intervals use paired-scene bootstrap after averaging replicates.

1. **Distribution encoding.** On clean, `R1_DSE` must improve both energy
   score and trajectory NLL over `B2_concat`, with both 95% CI lower bounds
   above zero.
2. **Interaction.** On clean, `R1_interaction` must improve energy score and
   high-FDE AUPRC over `R1_DSE`; its energy gain on `interaction_heavy` must be
   significant and larger than its all-track point estimate.
3. **Against Monte Carlo.** `R1_interaction` must either significantly improve
   clean energy over `B1_MC`, or be non-significantly worse while reducing
   measured prediction time by at least 5x.
4. **Cross-degradation.** `R1_interaction` over `B2_concat` energy must be
   significant in at least two of lowlight/dropout/beam and not significantly
   reversed in the third. The same rule must hold for S1/S2 without S3.
5. **Accuracy preservation.** Relative to `B0`, the `R1_interaction`
   predictive-mean increase must not exceed 0.05 m ADE or 0.10 m FDE on clean
   or any full family aggregate.
6. **Sharpness.** At empirical 90% final-position coverage, the median
   `R1_interaction` ellipse area must be at most 1.10x `B1_MC` on clean and all
   three full family aggregates.
7. **Risk utility.** Relative to `B1_MC`, `R1_interaction` must significantly
   improve high-FDE AUPRC, risk--coverage, or false-safe rate on clean or an
   S1/S2-only family aggregate.

All seven gates yield `GO TO R2 FULL RISKPROP3D`. A valid failure yields
`STOP RISKPROP3D METHOD ESCALATION`.

## Result

**STOP RISKPROP3D METHOD ESCALATION** (`5/7` gates passed; the run is valid).

- G1 passed. `R1_DSE` improved clean energy over `B2_concat` by 0.01965
  (95% CI 0.00895--0.03100) and trajectory NLL by 0.19392
  (0.10068--0.28678).
- G2 failed. Adding interaction made clean energy significantly worse than
  `R1_DSE` by 0.01367 (the improvement CI was -0.02424---0.00299). The
  interaction-heavy gain was -0.00317 (-0.01424--0.00995), and high-FDE AUPRC
  gain was -0.00366 (-0.01123--0.00287). All three replicates had the same
  adverse clean energy direction.
- G3 passed through quality branch A. `R1_interaction` improved clean energy
  over `B1_MC` by 0.04389 (0.03092--0.05741). Its measured prediction speedup
  was 2.44x, below the alternative 5x runtime threshold but not needed because
  quality improved significantly.
- G4 failed. Full-family `R1_interaction` energy gains over `B2_concat` were
  0.00718/0.00044/0.00768 for lowlight/dropout/beam, but every CI crossed zero;
  the S1/S2-only result had the same non-significant pattern.
- G5--G7 passed. Predictive-mean ADE/FDE improved slightly rather than being
  harmed; equal-coverage area ratios were 0.840--0.893 versus MC; clean
  high-FDE AUPRC improved over MC by 0.00933 (0.00076--0.01765).
- All six validity checks passed. All 343,708,782 audited numeric values were
  finite across 330,564 condition tracks. Official validation and internal
  test remained unread.

The preregistered result supports the dedicated distribution encoder and the
learned quality advantage over brute-force MC, but not the proposed
distributional interaction mechanism or R2 escalation. Complete evidence is
`outputs/riskprop3d_r1/result.json`.

## Retained outputs

- `outputs/riskprop3d_r1/scenes/*.npz`: canonical indexed multi-agent inputs;
- `outputs/riskprop3d_r1/prepare_manifest.json`: split and input hashes;
- `outputs/riskprop3d_r1/checkpoints/replicate_*.pt`: four learned models and
  clean label-cal scale temperatures;
- `outputs/riskprop3d_r1/metrics/replicate_*.npz`: paired per-target metrics,
  subset flags, and measured prediction times;
- `outputs/riskprop3d_r1/result.json`: validity evidence, gates, and endpoint.
