# Manuscript figure contracts

These contracts fix the visual claim, evidence hierarchy, source-data role,
and reviewer-risk boundary before plotting. All quantitative panels read the
retained evidence archives without retraining, recalibration, or a new pooled
analysis.

## Figure 1 — Controlled perception-to-forecast interface taxonomy

- **Core conclusion:** the same frozen probabilistic 3D perception output can
  enter forecasting through stochastic propagation, a learned
  representation, or a preservation objective, and evidence at one level does
  not establish utility at another.
- **Archetype:** schematic-led composite.
- **Target/output:** T-ITS two-column width; editable SVG/PDF plus 600-dpi TIFF.
- **Panel map:** a compact common-upstream band shows Camera/LiDAR, the frozen
  detector, shared stop-gradient mean, and the two trained laws. Three parallel
  cards then define stochastic propagation, representation utility, and
  objective necessity through their operations, matched comparison, and
  scientific question. Correct identity is a lightweight scope tag rather than
  a dominant enclosure.
- **Evidence hierarchy:** taxonomy and estimand boundary; no quantitative
  result is encoded.
- **Reviewer risk:** the diagram must not imply an end-to-end tracker, a new
  probability head, a causal heavy-tail/circular mechanism, or that evidence at
  one of the three levels automatically establishes another.

## Figure 2 — Development evidence for stochastic propagation

- **Core conclusion:** under both controlled probes, the evaluated HTC law
  improves trajectory energy over the trained Gaussian law across clean and
  three original degradation families, with sharper equal-coverage regions
  and better clean high-FDE ranking.
- **Archetype:** asymmetric quantitative grid with one hero forest plot.
- **Target/output:** T-ITS two-column width; editable SVG/PDF, 600-dpi TIFF,
  and PNG preview.
- **Panel map:**
  - **a:** HTC-over-Gaussian trajectory-energy gains on clean, lowlight,
    dropout, and beam for CV and MLP;
  - **b:** clean HTC-over-mean-state energy gains;
  - **c:** equal-coverage HTC/Gaussian area ratios;
  - **d:** clean HTC-over-Gaussian high-FDE AUPRC gains.
- **Statistics:** point estimate and 95% paired-scene bootstrap CI; three paired
  head/propagation replicates; 70 development scenes.
- **Reviewer risk:** positive downstream results compare complete trained laws,
  not a pure distribution-family intervention; area ratios must be shown with
  a reference at one.

## Figure 3 — Upstream-to-downstream transfer

- **Core conclusion:** differences between the two trained upstream predictive
  laws are associated with downstream trajectory-energy effects, most clearly
  for XY likelihood; the association is descriptive rather than causal.
- **Archetype:** three-panel scatter plot.
- **Target/output:** T-ITS two-column width; editable SVG/PDF plus 600-dpi TIFF.
- **Panel map:**
  - **a:** upstream XY-NLL gain versus downstream energy gain;
  - **b:** natural 90% coverage-error reduction versus downstream energy gain;
  - **c:** equal-coverage log-area gain versus downstream energy gain.
- **Statistics:** each point is a fixed population--condition--severity--seed
  cell; Spearman intervals use 10,000 cluster bootstrap resamples retaining all
  three seeds within a condition cluster.
- **Reviewer risk:** label beam-reduction S3 as the leverage point; do not infer
  a Student-t, von Mises, calibration, or sharpness causal mechanism from an
  ecological association.

## Figure 4 — Learned-interface and objective controls

- **Core conclusion:** local learned-interface positives do not survive all
  matched necessity controls, while MC behavior is compressible but its
  preservation is not a necessary objective under direct GT supervision.
- **Archetype:** quantitative grid.
- **Target/output:** T-ITS two-column width; editable SVG/PDF plus 600-dpi TIFF.
- **Panel map:**
  - **a:** initial DSE-versus-concatenation result beside capacity-matched and
    Gaussianized controls;
  - **b:** direct HTC token versus mean, confidence, and Gaussian controls with
    aggregate CIs and individual seed directions;
  - **c:** teacher-fidelity reductions on their metric-specific native scales;
  - **d:** preservation-student energy advantage against MC and the identical
    GT-only student, with a common candidate-oriented sign convention;
  - **e:** paired MC and one-pass runtime.
- **Statistics:** 95% paired-scene bootstrap CI and three paired seeds; runtime
  shows each replicate and their mean speedup.
- **Reviewer risk:** teacher-fidelity metrics use different scales and are
  therefore displayed as numerical interval estimates rather than bars on a
  shared axis; the initial and matched audits are visually separated because
  their magnitudes are not paired; speedup does not establish objective
  necessity.

## Figure 5 — Observed cross-population interface-stability asymmetry

- **Core conclusion:** within this controlled study, the trained-law contrast
  remains positive under stochastic propagation across probes, degradations,
  and evaluation populations, whereas learned-representation effects are
  heterogeneous across comparator, seed, and population.
- **Archetype:** asymmetric mixed quantitative figure.
- **Target/output:** T-ITS two-column width; editable SVG/PDF plus 600-dpi TIFF.
- **Panel map:**
  - **a:** common-scale forest plot of HTC-over-Gaussian stochastic energy
    gains across development/original degradations, held-out degradations, and
    final validation for CV and MLP;
  - **b:** categorical matrix for structured-encoder and direct-token
    contrasts, with significance/direction, seed reversal, and not-evaluated
    states encoded separately.
- **Statistics:** 95% paired-scene bootstrap CI; development/held-out analyses
  use 70 scenes and final validation uses 150 scenes; three paired seeds.
- **Reviewer risk:** magnitudes must not be compared across unmatched learned
  audits; “observed within this controlled study” is mandatory wording.

## Supplementary Figure S1 — Proper-score and forecast-error-ranking boundary

- **Core conclusion:** direct probability conditioning can retain
  forecast-error-ranking value when its average energy effect is null or
  comparator dependent; stochastic risk effects are more consistent with the
  primary score.
- **Archetype:** quantitative grid.
- **Target/output:** T-ITS supplementary two-column width; editable SVG/PDF plus
  600-dpi TIFF.
- **Panel map:**
  - **a:** development direct-token energy and risk--coverage effects;
  - **b:** final-validation direct-token energy and risk--coverage effects;
  - **c:** final-validation stochastic high-FDE AUPRC, risk--coverage, and
    false-safe effects for CV and MLP.
- **Statistics:** point estimate and 95% paired-scene bootstrap CI; three paired
  seeds. Development uses 70 scenes; final validation uses 150 scenes, with
  147 evaluable scenes for official AUPRC/false-safe contrasts.
- **Reviewer risk:** use only “forecast-error ranking/triage”; do not infer
  collision reduction, safer planning, or a universal Gaussian-token risk
  advantage.
