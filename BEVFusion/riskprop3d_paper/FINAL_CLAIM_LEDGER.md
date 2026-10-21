# Final claim ledger before official-validation access

This ledger was frozen after the preregistered dev attribute analysis and
external rotation/translation/temporal transfer, and before any official
validation metric was read. Official validation may confirm, contradict, or
narrow these claims, but it cannot create a new model, comparator, metric,
subset, or scientific question.

## Supported

1. **Direct stochastic propagation carries downstream value.** On the frozen
   original dev authority, 64-sample M2 propagation improves trajectory energy
   over both M1 Gaussian propagation and deterministic perception under CV and
   MLP hosts across clean, lowlight, dropout, and beam conditions.
2. **The stochastic result transfers to additional sensor perturbations.** On
   the predeclared external dev conditions, M2-over-M1 energy gains are
   significant for both hosts in every family: rotation `0.02411/0.02260`,
   translation `0.01635/0.01595`, and temporal offset `0.01888/0.01634` for
   CV/MLP. The corresponding M2-over-deterministic gains are also significant.
3. **The stochastic gain is not purchased by wider regions.** External
   equal-coverage M2/M1 area ratios are `0.861--0.896` across the six
   family-host combinations, consistent with the original sharpness result.
4. **The applicability boundary is empirically supported.** A useful M2
   stochastic pushforward does not imply that an explicit M2 representation is
   necessary for ordinary GT-supervised forecasting.

## Bounded or local positive

1. **Learned DSE evidence is local.** R1.5 found DSE gains on its frozen seeds,
   but R2-Core did not establish a stable capacity-independent, M2-family,
   heavy-tail, or circular mechanism. DSE is not the paper's method claim.
2. **Amortization is technically possible but not a necessary objective.**
   R2-D preserved MC distributional fidelity, risk ordering, and sharpness at
   `38.55x` speedup, yet the identical GT-only student was significantly better
   on trajectory energy and diagnostic mixture NLL.
3. **Direct probability conditioning is degradation-dependent.** In external
   transfer, the M2 token improves energy over mean-only on rotation by
   `0.01734` (`0.00484--0.03223`) and temporal offset by `0.01916`
   (`0.00369--0.03599`), but not on translation. It is not significantly
   better than confidence or the Gaussian token on any external family, and
   M2-over-Gaussian seed directions reverse in at least one seed per family.
4. **Residual learned value is narrower in risk ranking.** R3-P clean dev
   supports an M2-token risk--coverage gain over mean and confidence controls.
   External transfer adds family-specific risk evidence: for example,
   rotation risk--coverage improves over mean and confidence, while
   high-FDE AUPRC/false-safe can improve over the Gaussian token even when
   trajectory-energy necessity is absent. These outcomes do not constitute a
   universal proper-score or M2-token claim.
5. **Attribute patterns are descriptive only.** The frozen dev analysis shows
   heterogeneous stochastic effects across range, speed, class, LiDAR-point,
   residual, and history strata. These patterns are not tail/circular or
   sensor-physics mechanisms and were not used to select a new subset or
   model.

## Not supported

1. A stable heavy-tail or von-Mises circular causal mechanism for learned
   forecasting gains.
2. A stable capacity-independent DSE mechanism or strong-host DSE escalation.
3. Distribution-preserving distillation as a necessary or superior training
   objective relative to ordinary GT-only supervision.
4. Explicit M2 probability-token necessity beyond mean, detector
   score/query-density, and M1 Gaussian uncertainty on primary trajectory
   energy.
5. Universal superiority on diagnostic mixture NLL.
6. A new uncertainty-propagation theory, interaction module, map/risk-head
   rescue, R4/R5, or any further RiskProp3D model development.

## Frozen official-validation confirmation questions

- **RQ1:** Do P3 M2 stochastic pushforwards retain positive energy effects over
  P2 Gaussian and P1 deterministic perception for both CV and MLP?
- **RQ2:** Do aggregate capacity/family/tail/circular R2-Core contrasts remain
  insufficient to support the closed mechanism claim?
- **RQ3:** Does the R2-D GT-only student remain at least as strong as the
  risk-distilled student on downstream energy despite successful MC
  compression on dev?
- **RQ4:** Does R3-P continue to lack a stable M2-token energy advantage over
  mean, confidence, and Gaussian conditioning, while any risk-ranking value is
  reported separately?

The primary metric remains trajectory energy. Risk metrics, equal-coverage
area, ADE/FDE, and seed directions are secondary; mixture NLL is diagnostic.
All comparisons use three frozen seeds and paired-scene bootstrap intervals.
Official validation is confirmation only. Internal test remains sealed.
