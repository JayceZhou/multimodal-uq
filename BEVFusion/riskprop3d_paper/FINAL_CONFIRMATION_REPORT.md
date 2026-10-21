# Final confirmation report

## Final status

`FINAL CONFIRMATION COMPLETE — PAPER WRITING AUTHORIZED`

RiskProp3D method development remains closed. No R4/R5, new model, loss,
module, host, metric, subset, or hyperparameter is authorized by these
results. Internal test remains sealed.

Provisional paper title:

> **When Does Perception Uncertainty Help Motion Forecasting? A Controlled
> Study under Multimodal Sensor Degradation**

Final one-sentence argument:

> Explicit stochastic propagation consistently exposes downstream value from
> probabilistic 3D perception, whereas learned uncertainty interfaces show
> objective-, seed-, degradation-, and evaluation-authority-dependent gains
> rather than universal necessity under ordinary GT supervision.

## Protocol completion

| Phase | Outcome | Retained evidence |
|---|---|---|
| F0 | Immutable evidence freeze activated before confirmation reads | `outputs/riskprop3d_final_confirmation/freeze_manifest.json` |
| F1 | Dev attribute analysis completed as descriptive evidence only | `outputs/riskprop3d_final_confirmation/attributes/result.json` |
| F2 | Rotation, translation, and temporal transfer completed for R0 and R3-P only | `outputs/riskprop3d_final_confirmation/external/result.json` |
| F3 | Claim ledger frozen before official-validation metrics were read | `outputs/riskprop3d_final_confirmation/claim_freeze.json` |
| F4 | One-time official-validation confirmation completed | `outputs/riskprop3d_final_confirmation/official/result.json` |

The official authority contains 6,019 samples, 68,436 eligible tracks, and
150 scenes. All three frozen replicates are present. All 60,155,259 audited
official numeric values are finite. No training or method/metric selection was
performed. `official_val_read=true` and `internal_test_read=false`.

## External transfer before official access

The external transfer confirms the stochastic endpoint on every declared
family and both frozen hosts. Positive values mean that M2 propagation is
better than M1 Gaussian propagation.

| Family | CV energy gain (95% CI) | MLP energy gain (95% CI) |
|---|---:|---:|
| Rotation | 0.02411 (0.01750--0.03089) | 0.02260 (0.01765--0.02785) |
| Translation | 0.01635 (0.01085--0.02211) | 0.01595 (0.01156--0.02053) |
| Temporal offset | 0.01888 (0.01337--0.02445) | 0.01634 (0.01213--0.02077) |

The corresponding M2-over-deterministic contrasts are also significant, and
equal-coverage M2/M1 area ratios are 0.861--0.896.

Direct M2-token conditioning is less stable. It improves energy over
mean-only on rotation by 0.01734 (0.00484--0.03223) and temporal offset by
0.01916 (0.00369--0.03599), but not on translation. It is not significantly
better than confidence or the Gaussian token on any external family, and at
least one seed reverses the M2-over-Gaussian direction in every family.

## Official-validation confirmation

Every value below is the frozen baseline-minus-candidate contrast, so a
positive lower-is-better metric favors the named candidate. Confidence
intervals are paired-scene 95% intervals over 150 official scenes.

### RQ1 — Does perception uncertainty propagate downstream?

**Confirmed.** M2 stochastic propagation improves trajectory energy over
both deterministic and M1 Gaussian propagation for CV and MLP, with all seed
directions positive.

| Host | Comparator | M2 energy gain (95% CI) | Equal-coverage M2/comparator area |
|---|---|---:|---:|
| CV | Deterministic P1 | 0.12528 (0.09446--0.15992) | 0.3008 |
| CV | Gaussian P2 | 0.02060 (0.01632--0.02532) | 0.8549 |
| MLP | Deterministic P1 | 0.12382 (0.09352--0.15787) | 0.3194 |
| MLP | Gaussian P2 | 0.02101 (0.01603--0.02688) | 0.8537 |

M2 also significantly improves high-FDE AUPRC, risk--coverage, and false-safe
rate over Gaussian propagation for both hosts. Diagnostic mixture NLL favors
M2 over both comparators. The official result therefore independently
confirms the stochastic-pushforward phenomenon and its sharpness/risk value.

### RQ2 — Can a learned representation explain the gain?

**The mechanism claim remains closed, with local official positives.** Full
DSE improves official aggregate energy over capacity-matched mean-only by
0.03115 (0.01784--0.04570) and Gaussianized DSE by 0.02827
(0.01518--0.04142). It is indistinguishable from no-tail by 0.00046
(-0.01038--0.00928), while it improves over no-circular by 0.01110
(0.00467--0.01837).

These aggregate results do not establish the preregistered mechanism:

- the third seed reverses full DSE against every matched ablation;
- the tail-specific contrast remains null;
- risk gains are not supported against Gaussianized, no-tail, or no-circular
  controls;
- the full-DSE/mean-only equal-coverage area ratio is 1.0728;
- the valid dev mechanism audit already failed capacity, distribution-family,
  tail, circular, sharpness, risk, and seed-stability gates.

The official capacity/Gaussianized positives are reportable local evidence,
not grounds to revive DSE, heavy-tail, or circular mechanism claims.

### RQ3 — Can stochastic propagation be efficiently amortized?

**Compression remains technically supported; superiority of the preservation
objective remains unsupported.** On official energy, GT-only minus distilled
is -0.01740 (-0.04238--0.00968): the point estimate favors GT-only, but the
interval crosses zero. Diagnostic NLL significantly favors GT-only by
0.58845 (0.49018--0.69155 in the GT-only direction). The distilled model is
also 2.715x wider than GT-only at equal 90% coverage.

The distilled model still strongly outperforms the MC teacher on official
energy by 0.09834 (0.08768--0.10803) and has a 0.9427 equal-coverage area
ratio. Together with the frozen dev result—high-fidelity preservation at
38.55x speedup, but significant downstream energy and NLL loss versus the
identical GT-only model—this supports efficient compression as a technical
fact, not distribution-preserving distillation as a necessary method.

### RQ4 — Is explicit probability conditioning necessary?

**Official validation provides a positive aggregate result that contradicts
the dev null, so the final conclusion is evaluation-dependent rather than a
universal negative.** On official energy, the M2 token improves over all three
controls:

| Comparator | M2-token energy gain (95% CI) | Seed directions | Equal-coverage M2/comparator area |
|---|---:|---|---:|
| Mean-only P0 | 0.02702 (0.01588--0.03932) | + / + / + | 0.9874 |
| Confidence P1 | 0.02092 (0.01112--0.03217) | + / + / + | 0.9849 |
| Gaussian token P2 | 0.01682 (0.00456--0.03051) | + / - / + | 0.9723 |

Risk--coverage and false-safe rate improve significantly over mean-only and
confidence, but no risk metric significantly improves over the Gaussian token.
Diagnostic NLL favors M2 over all three controls. Accuracy and sharpness do
not explain the gain: M2 is slightly sharper and improves ADE/FDE.

This result must not be presented as universal M2-token necessity because:

- the preregistered dev necessity audit was null against mean, confidence,
  and Gaussian controls;
- external transfer did not show a significant M2 advantage over confidence
  or Gaussian conditioning in any family;
- one official seed reverses M2 versus the Gaussian token.

The supported interpretation is that direct probability conditioning can
help under some evaluation authorities, but its primary-score value is not
stable enough to be considered a universally necessary forecasting interface.

## Final evidence classification

### Supported

1. Direct M2 stochastic propagation has repeatable downstream trajectory,
   sharpness, and risk value relative to Gaussian and deterministic
   propagation across original dev, external transfer, and official
   validation.
2. The key applicability boundary is real: a useful stochastic pushforward
   does not guarantee that a particular learned uncertainty interface or
   preservation objective is necessary under GT supervision.
3. The four interface paradigms—stochastic propagation, distributional
   encoding, distribution-preserving distillation, and direct probability
   conditioning—are empirically non-equivalent.

### Bounded or local positive

1. DSE has positive results on selected seeds/authorities, including official
   aggregate capacity and Gaussianized contrasts, but no stable mechanism.
2. Amortization preserves MC information efficiently and can outperform the
   MC teacher, but GT-only supervision remains at least as compelling and is
   sharper than the preservation objective.
3. Direct M2-token conditioning is positive on official validation and some
   external degradations, but null on the frozen dev necessity audit and not
   consistently superior to confidence/Gaussian controls in transfer.
4. Probability tokens retain risk-ranking value over mean/confidence in
   several settings; an incremental risk advantage over Gaussian conditioning
   is not stable.

### Not supported

1. A stable heavy-tail, degrees-of-freedom, or von-Mises circular causal
   mechanism for learned forecasting.
2. A capacity-independent universal DSE mechanism or strong-host DSE claim.
3. Distribution-preserving distillation as necessary or superior to ordinary
   GT-only supervision.
4. Universal M2-token necessity or universal probability-conditioning benefit.
5. Universal superiority on diagnostic mixture NLL.
6. A new propagation theory or further RiskProp3D model development.

## Paper map

The Results section is frozen to the following evidence sequence:

1. **RQ1 — Stochastic propagation:** original dev, external transfer, and
   official confirmation.
2. **RQ2 — Learned representation:** R1.5 local positive followed by the
   controlled R2-Core mechanism closure and the mixed official aggregate.
3. **RQ3 — Amortization:** teacher fidelity/38.55x speedup, followed by the
   GT-only necessity comparison and official confirmation.
4. **RQ4 — Direct conditioning:** dev null, external heterogeneity, official
   positive aggregate, and the resulting evaluation-dependent boundary.

The Discussion must explicitly report the official RQ4 contradiction. The
paper must not simplify the evidence into either “uncertainty always helps” or
“uncertainty does not help.” The defensible conclusion is that the downstream
value depends on the interface and supervision regime, and is most consistent
for explicit stochastic pushforward.

## Evidence locations

- Frozen protocol: `riskprop3d_paper/FINAL_CONFIRMATION_PROTOCOL.md`
- Pre-official claim ledger: `riskprop3d_paper/FINAL_CLAIM_LEDGER.md`
- Access log: `outputs/riskprop3d_final_confirmation/access_log.jsonl`
- Descriptive analysis: `outputs/riskprop3d_final_confirmation/attributes/result.json`
- External transfer: `outputs/riskprop3d_final_confirmation/external/result.json`
- Official confirmation: `outputs/riskprop3d_final_confirmation/official/result.json`

