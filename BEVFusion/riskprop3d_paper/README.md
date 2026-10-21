# Perception-to-forecast uncertainty study: paper evidence package

## Frozen project status

- `RISKPROP3D METHOD DEVELOPMENT — CLOSED`
- `R0 STOCHASTIC PROPAGATION PHENOMENON — SUPPORTED`
- `EXPLICIT LEARNED UQ NECESSITY — NOT SUPPORTED`
- `THIRD WORK — CONTROLLED PERCEPTION-TO-FORECAST UNCERTAINTY STUDY`
- `NO FURTHER RESCUE MODEL DEVELOPMENT`

This directory consolidates the accepted and negative evidence from R0,
R1.5, R2-Core, R2-D, and R3-P into a paper-level argument. It does not create
a new RiskProp3D method or reinterpret a failed preregistered endpoint.

## One-sentence argument

In a frozen modular Camera--LiDAR perception-to-forecast pipeline, direct
stochastic pushforward of the full M2 perception distribution consistently
improves future probabilistic quality and risk identification, but controlled
learned audits show that these gains do not make explicit probability
representations necessary for GT-supervised motion forecasting; their
remaining learned value is narrower and appears primarily in risk ranking.

## Evidence boundary

The argument is currently limited to nuScenes-derived train/label-cal/dev
authorities, one frozen BEVFusion detector, accepted M1/M2 uncertainty heads,
lightweight CV/MLP forecasting hosts, three head/model seeds, and clean plus
lowlight/LiDAR-dropout/beam-reduction conditions. It does not establish a
causal heavy-tail or circular-yaw mechanism, universal superiority of an
uncertainty interface, strong-host generalization, planning benefit, or
official-validation generalization.

## Paper-facing terminology ledger

| Canonical term | First-use definition | Internal variants | Decision |
|---|---|---|---|
| perception distribution | Predictive distribution over the current 3D object state | uncertainty head, UQ output | Use the neutral interface term; do not imply causal correctness. |
| stochastic pushforward | Monte Carlo propagation of perception-state samples through a frozen forecaster | MC propagation, P3 propagation | Primary supported phenomenon. |
| Gaussian stochastic pushforward | M1 Gaussian/wrapped-Gaussian samples propagated through the same forecaster | P2 | Main stochastic comparator. |
| M2 stochastic pushforward | M2 Student-t/von Mises samples propagated through the same forecaster | P3, MC teacher | Use distribution-family names only to identify the frozen input law. |
| distributional encoding | A learned encoder that consumes summary statistics of a perception distribution | DSE, C5 | Treat as an evaluated interface, not an established mechanism. |
| pushforward distillation | Single-pass student trained to preserve an MC future distribution | R2-D, D3 | Avoid calling it a necessary method contribution. |
| probability-token conditioning | GT-only forecaster receiving an explicit Gaussian or M2 probability token | R3-P | Use for the final necessity test. |
| trajectory energy score | Sample-based proper score over the full future trajectory | ES, energy | Primary forecasting-quality metric. |
| diagnostic mixture NLL | Exact NLL under the learned/shared-kernel mixture approximation | NLL | Always call diagnostic where frozen protocols do so. |
| equal-coverage area | Median final-position ellipse area at empirical 90% coverage | sharpness area | Report with the comparator and coverage level. |
| future-risk ranking | Ranking by predicted final log ellipse area against future high FDE | risk utility | Does not mean collision-risk prediction. |
| high FDE | Predictive-mean final displacement error above 2 m under the audit-specific frozen baseline | failure event | State the baseline definition for each experiment. |

Internal stage labels such as `P0--P3`, `C0--C5`, and `D0--D3` are retained
only in the reproducibility crosswalk. They are not stable semantic labels
across audits and should not be reused as paper-wide method names.

## Working documents

- [`EVIDENCE_MATRIX.md`](EVIDENCE_MATRIX.md): claim ledger, quantitative
  evidence, negative evidence, and source authority.
- [`PAPER_MAP.md`](PAPER_MAP.md): manuscript argument, section/paragraph map,
  and figure/table plan.
- [`FINAL_CONFIRMATION_PROTOCOL.md`](FINAL_CONFIRMATION_PROTOCOL.md): the only
  permitted confirmation analyses before submission; none has been executed
  by creating this package.

## Source authorities

- [`../riskprop3d_r0/README.md`](../riskprop3d_r0/README.md) and
  [`../outputs/riskprop3d_r0/result.json`](../outputs/riskprop3d_r0/result.json)
- [`../riskprop3d_r15/README.md`](../riskprop3d_r15/README.md) and
  [`../outputs/riskprop3d_r15/result.json`](../outputs/riskprop3d_r15/result.json)
- [`../riskprop3d_r2_core/README.md`](../riskprop3d_r2_core/README.md) and
  [`../outputs/riskprop3d_r2_core/result.json`](../outputs/riskprop3d_r2_core/result.json)
- [`../riskprop3d_r2d/README.md`](../riskprop3d_r2d/README.md) and
  [`../outputs/riskprop3d_r2d/result.json`](../outputs/riskprop3d_r2d/result.json)
- [`../riskprop3d_r3p/README.md`](../riskprop3d_r3p/README.md) and
  [`../outputs/riskprop3d_r3p/result.json`](../outputs/riskprop3d_r3p/result.json)

## Author note

The paper is positioned as a controlled empirical study, not as a new
forecasting block. The title, abstract, and contribution language must lead
with the supported boundary between stochastic propagation and learned
uncertainty necessity. Negative endpoints remain first-class results rather
than ablations to be hidden or repaired.
