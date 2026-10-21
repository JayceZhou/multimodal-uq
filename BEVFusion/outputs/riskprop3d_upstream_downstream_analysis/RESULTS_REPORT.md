# Upstream-to-downstream transfer analysis

## Scope and fixed definitions

This read-only analysis contains 48 condition–seed cells: 16 fixed
`population × condition × severity` clusters and three frozen head seeds per
cluster. Track-level paired differences are averaged within scene. Confidence
intervals for condition effects use 10,000 paired-scene bootstrap resamples.
Spearman intervals use 10,000 cluster bootstrap resamples in which an entire
population–condition–severity cluster, including all three seeds, is sampled
together.

Positive effects mean HTC is better: `G − HTC` for NLL and trajectory energy.
Natural-coverage improvement is the reduction in absolute deviation from 0.90.
Equal-coverage sharpness is `log(median area_G / median area_HTC)`. Correlations
are descriptive ecological associations across evaluation cells, not causal
effects of Student-t tails or von Mises concentration.

## Main findings

1. HTC had a positive scene-weighted XY-NLL gain in all 16 condition cells;
   16/16 paired-scene intervals had a lower bound above zero.
2. Downstream trajectory energy favored HTC in all aggregate cells. The
   paired-scene lower bound was above zero in 16/16 cells for CV
   and 16/16 cells for MLP.
3. No law-ranking reversal occurred in the 45 non-reference
   population/condition/seed comparisons for either host
   (CV=0, MLP=0). This is the strongest direct evidence
   that the trained upstream-law ordering survives the fixed forecasting probes.
4. Relative-to-clean transfer was not uniformly proportional: CV classifications
   were 30 amplified and
   15 attenuated; MLP classifications were
   36 amplified and
   9 attenuated. Hence downstream transformation
   can strengthen or weaken the upstream contrast even when it preserves its sign.

## Cluster-bootstrap rank associations

- Upstream XY NLL gain → CV trajectory ES gain: ρ=0.818, cluster-bootstrap 95% CI [0.590, 0.896].
- Upstream XY NLL gain → MLP trajectory ES gain: ρ=0.814, cluster-bootstrap 95% CI [0.576, 0.910].
- Natural-coverage error reduction → CV trajectory ES gain: ρ=0.132, cluster-bootstrap 95% CI [-0.428, 0.611].
- Natural-coverage error reduction → MLP trajectory ES gain: ρ=0.261, cluster-bootstrap 95% CI [-0.296, 0.681].
- Equal-coverage log-area gain → CV trajectory ES gain: ρ=0.341, cluster-bootstrap 95% CI [0.064, 0.614].
- Equal-coverage log-area gain → MLP trajectory ES gain: ρ=0.382, cluster-bootstrap 95% CI [0.095, 0.642].

The NLL-transfer association is the central analysis. Coverage-error and
equal-coverage-area correlations are complementary diagnostics and should not
be presented as independent confirmatory hypotheses.

## Manuscript-safe interpretation

The evaluated HTC predictive law has better upstream XY likelihood on every
fixed evaluation cell, and this ordering is retained by both downstream probes
without a condition-level sign reversal. However, the magnitude transferred to
trajectory energy is not a fixed function of upstream NLL, calibration error,
or sharpness. This supports the paper's interface-level statement—trained
predictive-law differences can survive stochastic propagation—without implying
that distribution family alone is causal or that better upstream uncertainty
guarantees a proportionally larger downstream gain.

## Figure legend draft

**Fig. X | Upstream probabilistic differences and downstream trajectory-energy
transfer.** a Scene-weighted HTC gain in uncalibrated XY negative log-likelihood
versus the HTC gain in trajectory energy score. b Reduction in absolute natural
90% coverage error versus trajectory-energy gain. c Equal-coverage XY
log-area gain versus trajectory-energy gain. Each point is one fixed
population–condition–severity and head-seed cell; circles denote CV and
triangles denote MLP (48 cells per host; 16 clusters; three seeds per cluster).
Point size denotes severity and colour denotes degradation family or final
validation. Insets report Spearman ρ and 95% intervals from 10,000 cluster
bootstrap resamples that retain all three seeds within each resampled cluster.
Positive values favour HTC. Associations are descriptive and do not identify a
tail or circular causal mechanism. Source data are provided in
`seed_cell_source_data.csv`.
