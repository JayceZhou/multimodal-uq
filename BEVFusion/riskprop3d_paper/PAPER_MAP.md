# Paper map

## Recommended title

**When Does Perception Uncertainty Help Motion Forecasting? A Controlled
Study under Multimodal Sensor Degradation**

This question title is defensible because the study answers it with both a
positive condition (explicit stochastic pushforward) and negative boundaries
(learned representation, distillation necessity, and direct conditioning).

Alternative titles:

1. **Stochastic Propagation Reveals Perception-Uncertainty Benefits That
   Supervised Forecasting Does Not Necessarily Require**
2. **From Perception Distributions to Future Risk: A Controlled Study under
   Camera--LiDAR Degradation**
3. **Beyond Uncertainty Conditioning: Evaluating Perception-to-Forecast Risk
   under Multimodal Sensor Degradation**

Do not use `RiskProp3D` in the title unless it names the evaluation protocol
rather than a claimed new forecasting method.

## Reader and positioning

- **Paper type:** research / controlled empirical study.
- **Primary audience:** autonomous-driving perception, motion forecasting,
  robustness, and reliability researchers.
- **Likely venue position:** IEEE T-IV or T-ITS; the paper does not rely on a
  new-block novelty claim.
- **Lead reader question:** relevance -- does upstream perception uncertainty
  materially affect downstream forecasts under realistic sensor degradation?
- **Trust question:** answered through frozen authorities, paired seeds,
  scene-level inference, explicit negative endpoints, and sealed final data.

## Core argument chain

```text
Modular perception-to-forecast systems lose distributional information
    -> it is unclear whether preserving that information helps downstream
    -> compare four uncertainty interfaces under one controlled pipeline
    -> M2 stochastic pushforward improves future probability and risk
    -> learned encoders/tokens do not show necessary primary-score gains
    -> uncertainty value depends on the computation and supervision regime
```

## Contributions

1. **Controlled evaluation framework.** Establish a frozen Camera--LiDAR
   degradation-to-perception-to-forecast pipeline with scene-disjoint data,
   paired three-seed evaluation, scene-level bootstrap inference, and explicit
   no-read authorities.
2. **Interface comparison.** Compare stochastic pushforward, distributional
   encoding, pushforward distillation, and direct probability-token
   conditioning as distinct ways of connecting perception uncertainty to
   motion forecasting.
3. **Empirical boundary.** Show that full-distribution stochastic pushforward
   can improve probabilistic forecasts and future-error identification, while
   those gains neither establish a heavy-tail/circular learned mechanism nor
   make explicit M2 inputs necessary under ordinary GT supervision; a narrower
   risk-ranking increment remains.

Use `evaluation framework` or `controlled testbed` rather than `benchmark`
unless the data protocol and evaluation code are packaged for public reuse.

## Manuscript architecture

### 1. Introduction -- four paragraphs

1. **Context:** Motion forecasting consumes outputs from imperfect perception,
   and Camera--LiDAR degradation changes not only object-state means but also
   the uncertainty around them.
2. **Gap:** Existing work often improves perception uncertainty, propagates
   samples, or connects latent perception and prediction, but these directions
   do not answer whether explicit uncertainty remains necessary once the
   forecaster receives direct future supervision.
3. **Study design:** Introduce the four interface classes and the controlled
   degradation setting. Emphasize that the study separates stochastic value,
   learned representation, distillation, and input necessity.
4. **Contribution preview:** State the positive stochastic result, the learned
   necessity boundary, and the residual risk-ranking result without listing
   all numbers.

Literature citations are still required for paragraph 2. They must cover:
probabilistic 3D perception, uncertainty propagation, modular versus joint
perception--prediction, robust motion forecasting, and risk/calibration. No
reference has been inserted or verified in this package.

### 2. Controlled study design

#### 2.1 Task and data authority

- Current object-state perception distribution to six-step/3-second future
  trajectory distribution.
- Clean train and label-cal only for training/calibration.
- Clean plus lowlight, LiDAR dropout, and beam reduction for frozen dev
  evaluation.
- Scene-disjoint splits; GT instance tokens only for association and targets.
- Corruption identity and current GT residual excluded from model inputs.

#### 2.2 Frozen perception interfaces

- Deterministic mean state.
- M1 Gaussian/wrapped-Gaussian distribution.
- M2 Student-t/von Mises distribution.
- Detector score and query density as confidence controls.

Describe M1/M2 as frozen predictive laws. Do not re-argue their static
development history or include ProbSet3D/GeoProb3D/DC0 branches.

#### 2.3 Four downstream interface classes

1. Stochastic pushforward through CV and frozen MLP.
2. Target-centric distributional encoding and matched mechanism controls.
3. MC-pushforward distillation into an identical single-pass student.
4. Direct probability-token conditioning under identical GT-only training.

#### 2.4 Metrics and inference

- Primary: trajectory energy score.
- Secondary: high-FDE AUPRC, risk--coverage, false-safe rate, equal-coverage
  final area, predictive-mean ADE/FDE.
- Diagnostic: exact/shared-kernel mixture trajectory NLL, with each audit's
  likelihood construction stated explicitly.
- Three paired replicates; per-track differences reduced within scene;
  10,000 scene bootstrap resamples.

#### 2.5 Sequential audit governance

Show the frozen endpoint chain. Explain that a negative endpoint closed its
claim and motivated a narrower scientific question, not hyperparameter rescue.
The official-validation and internal-test seals remained closed throughout
method selection.

### 3. Results

#### 3.1 Full-distribution stochastic pushforward improves future forecasts

**Paragraph 1 -- main quality result:** Report clean P3-over-P2 energy gains
for CV and MLP with CIs.

**Paragraph 2 -- degradation transfer:** Report full-family and S1/S2-only
ranges, emphasizing agreement across lowlight, dropout, and beam.

**Paragraph 3 -- sharpness and accuracy:** Show equal-coverage area ratios and
bounded ADE/FDE changes.

**Paragraph 4 -- risk and incremental information:** Report clean high-FDE
AUPRC gains and partial-correlation/delta-R2 evidence. Keep the latter
associational.

#### 3.2 Initial encoding gains do not survive the matched mechanism test

**Paragraph 1 -- initial observation:** R1.5 DSE significantly beat raw concat
and MC, motivating a capacity-matched audit.

**Paragraph 2 -- capacity/family control:** R2-Core C5 did not significantly
beat matched mean-only or Gaussianized DSE on clean.

**Paragraph 3 -- proposed mechanism tests:** The overall no-tail comparison
was positive, but preregistered tail-sensitive subsets did not replicate it;
circular concentration was unsupported and reversed for two-wheelers.

**Paragraph 4 -- interpretation boundary:** The evidence supports neither a
stable DSE core nor a heavy-tail/circular causal mechanism. Distributional
Interaction v1 belongs in the supplement as a closed negative control.

#### 3.3 MC pushforward is compressible but preservation is not necessary

**Paragraph 1 -- fidelity:** D3 moved significantly closer than D2 to the
teacher on energy distance, moments, and coverage.

**Paragraph 2 -- efficiency and teacher comparison:** D3 was 38.55x faster,
sharper, retained risk utility, and beat D1 on trajectory energy.

**Paragraph 3 -- decisive control:** The identical GT-only D2 nevertheless
beat D3 on clean and degradation energy, with all three clean seeds adverse to
distillation.

**Paragraph 4 -- interpretation:** A stochastic computation can be amortized,
but matching it is not the optimal objective when direct future labels are
available.

#### 3.4 Explicit probability tokens are not necessary for primary quality

**Paragraph 1 -- necessity comparisons:** Report P3 versus mean-only,
confidence, and Gaussian tokens on clean energy; all CIs cross zero.

**Paragraph 2 -- degradation:** No full or S1/S2 family establishes a stable
P3 energy increment.

**Paragraph 3 -- residual risk value:** Risk--coverage improves over both P0
and P1, whereas high-FDE AUPRC and false-safe do not pass against both.

**Paragraph 4 -- diagnostic divergence:** P3 NLL improves over P0/P1 but not
P2; energy remains the frozen primary endpoint. This is metric-dependent
evidence, not a contradiction to hide.

#### 3.5 Synthesis: interface and supervision determine uncertainty value

One compact synthesis paragraph and matrix should connect the four results:

| Interface | Distribution signal retained? | Primary ES evidence | Risk evidence | Final interpretation |
|---|---:|---:|---:|---|
| Stochastic pushforward | Full samples | Positive | Positive | Supported phenomenon |
| Distributional encoding | Summary features | Unstable after controls | Mixed | Mechanism unsupported |
| Pushforward distillation | Teacher future distribution | Worse than GT-only | Retained | Feasible, unnecessary objective |
| Probability-token conditioning | Explicit parameters | No stable increment | Narrow risk--coverage gain | Not necessary for supervised ES |

### 4. Discussion

1. **Central meaning:** Stochastic propagation tests the consequence of an
   input distribution, whereas supervised conditioning tests whether that
   distribution must remain explicit in a learned predictor. These are not
   equivalent scientific questions.
2. **Plausible explanation, not mechanism claim:** GT supervision and motion
   history may allow the learned model to absorb information correlated with
   explicit uncertainty. This remains an interpretation unless the planned
   descriptive analysis supports it.
3. **Risk/proper-score divergence:** Probability tokens may affect ordering of
   difficult cases without improving average trajectory energy. Discuss why
   reliability evaluation should retain both distributional quality and
   selective-risk metrics.
4. **Relation to modular and joint systems:** Position the study against
   uncertainty propagation and joint perception--prediction literature after
   a verified literature search. Do not claim that latent-query bridges are
   ineffective; they were not tested here.
5. **Limitations:** One dataset, detector family, frozen M1/M2 heads,
   lightweight hosts, object-level 3-second forecasts, proxy risk, and three
   primary degradation families. No map, planning, strong host, interaction
   claim, or official-validation result yet.
6. **Practical implication:** Sampling remains a valid audit tool for whether
   perception distributions carry downstream signal; deploying explicit UQ
   inputs requires an incremental-control test against mean, confidence, and
   Gaussian baselines.

### 5. Conclusion

Use a four-part close:

1. Controlled comparison of four perception-to-forecast uncertainty
   interfaces.
2. Decisive evidence that M2 stochastic pushforward improves future
   probability/risk under the frozen conditions.
3. Decisive boundary that learned explicit M2 representation is not necessary
   for primary trajectory energy under GT supervision.
4. Narrow implication: downstream uncertainty value must be demonstrated for
   the interface, supervision, and risk metric actually used.

## Figure map

| Figure | Claim served | Required panels | Existing evidence | Missing item |
|---|---|---|---|---|
| Fig. 1 | The study separates three uncertainty-use levels under one controlled pipeline. | Camera/LiDAR degradation pipeline; stochastic propagation, representation, and objective branches | Protocols in R0/R2-Core/R2-D/R3-P | Plot generation only |
| Fig. 2 | M2 stochastic pushforward improves future quality without wider regions. | Clean/family ES forest plots for CV/MLP; equal-coverage ratios; high-FDE AUPRC | R0 result and metric archives | Plot generation only |
| Fig. 3 | Upstream trained-law differences transfer to downstream energy, most clearly for XY likelihood. | Fixed population/condition/severity/seed cells; cluster-bootstrap Spearman analysis | Frozen upstream/downstream export | Plot generation only |
| Fig. 4 | Local representation gains and compressibility do not establish learned-interface or objective necessity. | R1.5/R2-Core/R3-P contrasts; teacher fidelity; candidate-oriented downstream energy; 38.55x runtime | R1.5, R2-Core, R2-D, and R3-P | Plot generation only |
| Fig. 5 | Stochastic effects are stable across evaluation populations while learned-interface effects are heterogeneous. | Development, held-out degradation, and final-validation contrasts | R0, R2-Core, R3-P, and final confirmation | Plot generation only |
| Fig. S1 | Direct HTC tokens can retain risk--coverage value when primary ES is null or comparator dependent. | HTC-token energy and risk contrasts; stochastic risk metrics | R3-P and final confirmation | Plot generation only |
| Main conclusion table | The value boundary depends on interface and objective. | Four-row synthesis matrix | All audits | Typesetting only |

Do not create a qualitative trajectory figure unless it answers a specific
claim; cherry-picked examples are not needed for the central argument.

## Table map

| Table | Contents | Placement |
|---|---|---|
| Table 1 | Data splits, conditions, hosts, seeds, samples, and model/input controls for each interface | Main |
| Table 2 | Four-interface evidence matrix with primary ES, risk, sharpness, efficiency, and endpoint | Main |
| Table 3 | Final external-transfer and official-validation confirmation | Main only after execution; currently placeholder |
| Table S1 | Full per-condition R0 metrics and CIs | Supplement |
| Table S2 | R1.5/R2-Core ablations, subsets, parameters, and seed directions | Supplement |
| Table S3 | R2-D fidelity, equivalence, runtime, and diagnostic metrics | Supplement |
| Table S4 | R3-P energy/NLL/risk/accuracy/sharpness by condition and seed | Supplement |
| Table S5 | Authority hashes, gates, validity checks, and sealed-data audit | Supplement |

## Abstract contract -- draft last

The abstract should have five jobs, one sentence or clause block each:

1. Modular motion forecasting depends on uncertain perception under sensor
   degradation.
2. It is unknown whether downstream benefit requires explicit uncertainty in
   a supervised forecaster.
3. We compare four interfaces in a controlled Camera--LiDAR pipeline.
4. M2 stochastic pushforward improves trajectory/risk metrics, but matched
   learned audits do not establish representation necessity; a narrower risk
   increment remains.
5. The value of perception uncertainty depends on how it is propagated and
   supervised.

Do not draft a numeric abstract until final-confirmation results and the target
journal's word limit are frozen.

## Main reviewer risks and planned defenses

| Risk | Defense already available | Remaining action |
|---|---|---|
| “This is a collection of failed method attempts.” | One causal sequence of narrowing questions; preregistered endpoints; negative results answer distinct interface hypotheses. | Show the audit timeline and claim closures in Fig. 1/Table S5. |
| “The result is post hoc.” | Frozen gates, seeds, metrics, source hashes, and no-read constraints exist for every stage. | Preserve all README/result authorities and disclose the sequential design. |
| “Comparisons are unfair across stages.” | Each decisive comparison is paired and controlled within its own audit. | Never compare absolute scores across audits; report only within-audit effects. |
| “Energy was selected because NLL disagreed.” | Energy was preregistered primary throughout; adverse NLL directions are retained. | Put diagnostic NLL in main or prominent supplement, not hidden. |
| “Risk means collision safety.” | Risk ranking uses final-area uncertainty and high-FDE proxies. | Use `future-error risk ranking`, not `safety improvement`. |
| “The conclusion is too broad.” | Frozen boundary is explicit. | Run only the confirmation protocol; keep title/subtitle tied to multimodal degradation. |
| “Benchmark contribution is unclear.” | Complete controlled protocol and evidence archives exist. | Use `controlled study/testbed` unless release packaging is completed. |

## Material excluded from the paper narrative

- ProbSet3D/GeoProb3D development history, health-conditioned artifacts, and
  DC0 partials.
- Interaction v2, maps, world models, planning/collision losses, new risk
  heads, strong-host rescue, new UQ heads, and detector retraining.
- Any claim that Student-t degrees of freedom specifically explain far-range,
  sparse-LiDAR, or high-residual forecasting behavior.
- Any implication that the R3-P STOP invalidates the accepted R0 stochastic
  phenomenon.

## 中文结构说明

- 主文按“正现象 -> 机制边界 -> 蒸馏边界 -> 必要性边界”展开，不按实验时间线
  罗列所有 R 版本。
- R1.5 必须与 R2-Core 放在同一节：前者说明为什么机制问题值得检验，后者给出
  最终受控结论。
- 论文的创新点是区分四种 uncertainty interface 并确定其适用边界，不是重新包装
  DSE 或 distillation。
- 所有跨阶段数字只用于叙事，不进行跨阶段显著性比较；正式比较只来自各自冻结
  audit 内部。
