# Manuscript writing notes

## Draft contract

- **Paper type:** controlled empirical research study.
- **Primary target:** IEEE Transactions on Intelligent Transportation
  Systems. Transportation Research Part C and Engineering Applications of
  Artificial Intelligence remain later alternatives, not co-optimized targets
  for the present draft.
- **Primary reader:** researchers in autonomous-driving perception, motion
  forecasting, robustness, and reliability.
- **Lead question:** when does upstream probabilistic 3D perception produce
  downstream forecasting value under multimodal sensor degradation?
- **One-sentence argument:** Conditional on correct object identity, explicit
  stochastic propagation exposes a predictive-law contrast that remains
  stable across probes, degradations, and evaluation populations, whereas
  learned uncertainty interfaces recover that value only conditionally and
  preserving the induced future distribution is not necessary under direct
  future supervision.
- **Boundary:** no new forecasting method, heavy-tail/circular causal
  mechanism, collision-safety claim, strong-host result, planning result, or
  universal probability-token claim.

## Terminology ledger

| Canonical term | First-use definition | Do not substitute with |
|---|---|---|
| perception distribution | Predictive distribution over the current 3D object state | UQ output, uncertainty head |
| trained predictive law | Complete learned distribution, including its scale, calibration, and family-specific likelihood fit | a pure distribution-family intervention |
| stochastic pushforward | Monte Carlo propagation of perception-state samples through a fixed forecasting probe | generic uncertainty conditioning |
| Gaussian perception distribution (G) | Diagonal-Gaussian linear state and wrapped-Gaussian yaw with the common detector mean | M1 outside the supplementary implementation crosswalk |
| heavy-tailed/circular perception distribution (HTC) | Student-t linear state and von Mises yaw with the common detector mean | M2 or RiskProp3D method in the main text |
| distributional encoding | Learned encoder consuming summaries of a perception distribution | proven mechanism |
| pushforward distillation | Single-pass student trained to preserve the MC future distribution | necessary method contribution |
| probability-token conditioning | GT-only forecaster receiving an explicit probability token | stochastic propagation |
| trajectory energy score | Sample-based proper score for the joint future trajectory | energy loss, ES before first definition |
| diagnostic mixture NLL | Exact or shared-kernel trajectory negative log-likelihood used diagnostically | primary NLL |
| equal-coverage area | Median final-position ellipse area at empirical 90% coverage | confidence area without coverage |
| future-error risk ranking | Ranking by predicted final log ellipse area against future high FDE | collision risk, safety risk |
| held-out degradation transfer | Post-development rotation, translation, and temporal perturbation families within nuScenes | external transfer or external dataset |
| final validation | Downstream interface outcomes read only after downstream models, metrics, and claims were fixed | a claim that nuScenes validation was never used upstream |
| correct-identity condition | GT instance identity used only for association and future-target construction | end-to-end tracking robustness |
| mean-state current interface | Current detector mean propagated through the probe with the same additive future process noise as G/HTC | deterministic future distribution |

## Unified paper framework

1. **Stochastic propagation value:** does changing the complete trained
   perception law improve the downstream predictive distribution under a
   fixed probe and future-noise construction?
2. **Representation utility:** can a GT-supervised predictor exploit explicit
   probability inputs beyond capacity, confidence, and Gaussian controls?
3. **Objective necessity:** is preserving the induced MC future distribution
   necessary when direct future labels are available?

The model-testing chronology belongs in `SUPPLEMENTARY_APPENDIX.md`, not in the
main-paper argument.

## Results-first paragraph map

1. Establish the strongest stochastic-propagation result on the development
   split and original degradation families.
2. Combine the structured encoder and direct token as two representation-level
   tests.
3. Separate pushforward compressibility from preservation-objective necessity.
4. Compare development, held-out degradations, and final validation in one
   cross-population section.
5. Treat risk ranking separately from average proper-score quality.
6. End with the main conclusion table.

## Claim-evidence map

| Claim | Frozen evidence | Status |
|---|---|---|
| The evaluated trained HTC law improves downstream trajectory quality under stochastic propagation relative to the trained Gaussian law and mean-state current interface. | Original dev endpoint, held-out rotation/translation/temporal contrasts, final-validation stochastic table | Supported |
| The stochastic gain is not purchased by wider prediction regions and includes future-error ranking value. | External and official equal-coverage ratios; official risk metrics | Supported |
| A learned DSE explains the gain through heavy tails or circular yaw. | Matched R2-Core contrasts, ablations, and seed directions | Not supported |
| MC behavior can be compressed efficiently. | R2-D teacher fidelity, 38.55x dev speedup, official teacher comparison | Supported, bounded |
| Preserving the teacher distribution is necessary under GT supervision. | GT-only versus distilled dev and official contrasts | Not supported |
| Explicit M2-token conditioning has universal primary-score value. | Dev null, external heterogeneity, official positive aggregate with a Gaussian-seed reversal | Not supported universally |
| Explicit tokens retain some future-error risk-ranking value. | Dev, external, and official risk contrasts versus mean/confidence; no stable Gaussian increment | Supported, narrow |

## Numeric-source rule

Manuscript result numbers may be copied only from:

- `FINAL_CLAIM_LEDGER.md`;
- `EVIDENCE_MATRIX.md`;
- `FINAL_CONFIRMATION_REPORT.md`;
- `OFFICIAL_EVIDENCE_TABLE.md`;
- `UPSTREAM_CHARACTERIZATION_TABLE.md` for pre-forecast G/HTC
  characterization only;
- the machine-readable final-confirmation archives referenced by those files.

Older stage README files are reproducibility records, not manuscript-number
sources.

## Missing author inputs before submission

- Author names, affiliations, corresponding-author details, funding, and
  author contributions.
- Public repository/archival DOI for code and retained evidence.
- Final journal choice and its current abstract, graphical-abstract,
  highlights, and reference-format requirements.
- Final figure files and supplementary-table layout generated only from the
  frozen numeric archives.

## Draft QA record (2026-08-11, v0.6)

- English and Chinese drafts retain matched section, table,
  figure-placeholder, and reference structures. The main text now contains
  five figures and four tables; the upstream-to-downstream transfer analysis
  is in the main text and the detailed risk display remains supplementary.
- Structure: Stochastic Propagation Value--Representation Utility--Objective Necessity, followed by a
  unified cross-population comparison, metric-specific risk result, and main
  conclusion table.
- Main-text terminology: no internal R0--R3, M1/M2, GO/STOP, authority, or
  audit framing outside reference URLs/titles.
- Numeric audit: every result-like decimal appears in the consolidated frozen
  evidence sources; unmatched decimal strings are section numbers, DOI, or
  arXiv identifiers.
- Citation audit: the 2025--2026 update adds the peer-reviewed PU-TSI
  object-state study, ICCV 2025 ForeSight as current joint-interface context,
  and a CVPRW 2026 downstream-decision study. Reference [25] is corrected from
  its earlier arXiv title to the verified 2023 TPAMI journal article. Two 2026
  map-uncertainty preprints were screened but excluded from the main list.
- Boundary audit: correct identity and controlled-probe scope are introduced
  before the method, and causal tail/circular, tracking, collision-safety,
  strong-host, universal token-necessity, and new-method claims remain excluded.
- Predictive-law audit: G/HTC are consistently described as complete trained
  laws whose learned scales and family-specific likelihood fits both differ;
  no pure distribution-family effect is claimed.
- Contribution audit: the three contributions now separate the controlled
  framework, the robust stochastic-propagation result, and the learned-interface
  applicability boundary without repeating the same cross-population claim.
- Interface-stability audit: the abstract, contributions, discussion, and
  conclusion lead with stable stochastic propagation versus conditional
  learned-interface value rather than with HTC-versus-G as the paper identity;
  the asymmetry is explicitly described as observed within the controlled study.
- Upstream-table audit: Main Table 2 now states that G/HTC use identical clean
  development query rows, that NLL/natural coverage/raw scales are
  pre-calibration head outputs, and that equal-coverage volume alone is a
  post-hoc descriptive rescaling.
- Selection-transparency audit: the main text and supplement state that both
  predictive laws, all head seeds, epochs, and checkpoints were fixed before
  downstream outcomes and were not selected by a downstream metric.
- Closest-work audit: Supplementary Table S1 positions [23] and [26] through
  research-design differences without using checkmarks, crosses, or claims of
  methodological superiority.
- Figure-design audit: `FIGURE_4_DESIGN.md` specifies the five-second visual
  conclusion, accessibility encodings, seed-reversal marker, and prohibition
  on comparing magnitudes across unmatched audits.
- Statistical audit: unadjusted multiplicity and the
  confirmatory/secondary/descriptive inference boundary are stated explicitly.

## Chinese terminology crosswalk

| English canonical term | Chinese canonical term |
|---|---|
| perception distribution | 感知分布 |
| stochastic pushforward | 随机推前/随机传播 |
| Gaussian perception distribution (G) | 高斯感知分布（G） |
| heavy-tailed/circular perception distribution (HTC) | 重尾/圆周感知分布（HTC） |
| distributional encoding | 分布编码 |
| pushforward distillation | 推前蒸馏 |
| probability-token conditioning | 概率 token 条件输入 |
| trajectory energy score | 轨迹能量分数 |
| diagnostic mixture NLL | 诊断混合 NLL |
| equal-coverage area | 等覆盖面积 |
| future-error risk ranking | 未来误差风险排序 |
| held-out degradation transfer | 留出退化迁移 |
| final validation population | 最终验证总体 |
| correct-identity condition | 正确目标身份条件 |
| trained predictive law | 经训练预测律 |
| stochastic propagation value | 随机传播价值 |
| mean-state current interface | 均值状态当前接口 |
