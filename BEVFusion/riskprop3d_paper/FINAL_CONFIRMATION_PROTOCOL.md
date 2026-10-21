# Proposed final-confirmation protocol

## Status

`PROTOCOL READY — ACTIVATION REQUIRES THE F0 MACHINE-READABLE MANIFEST`

Creating this document does not authorize reading official validation,
internal test, or any retained external-result summary. Activation requires a
separate immutable manifest that hashes this protocol, all source results,
all checkpoints, evaluation code, condition availability, and the
pre-confirmation claim ledger. The final claim ledger is frozen later in F3,
after the preregistered external transfer has been reported.

## Purpose

Final confirmation tests whether the already established paper-level boundary
generalizes. It does not select a method, change an endpoint, or reopen
RiskProp3D development.

```text
Supported on frozen dev:
  M2 stochastic pushforward > Gaussian stochastic pushforward

Not supported on frozen dev:
  explicit M2 token is necessary for GT-supervised trajectory energy

Narrow retained result:
  explicit M2 token improves risk--coverage over mean/confidence controls
```

There is no GO/STOP gate and no result can authorize new training.

## Permanent prohibitions

- No new model, loss, module, interaction, map, risk head, forecasting host,
  detector, or UQ head.
- No checkpoint selection, epoch selection, temperature refit, threshold
  tuning, feature redesign, or seed replacement.
- No new primary metric or change from trajectory energy score.
- No mechanism subset designed after seeing confirmation outcomes.
- No internal-test read.
- No use of corruption identity or current GT residual as model input.
- No R4 rescue, regardless of the confirmation direction.

## Phase F0 — immutable evidence freeze

Before any new metric is read, write one machine-readable freeze manifest with:

1. SHA-256 hashes for the R0, R1.5, R2-Core, R2-D, and R3-P result authorities.
2. SHA-256 hashes for every checkpoint and evaluation script used.
3. The pre-confirmation claim ledger and this protocol hash.
4. Frozen head/model/sampling/bootstrap seeds.
5. Frozen input fields, metrics, risk definitions, aggregation units, and
   confidence-interval procedure.
6. A file-access log initialized before official-validation access.
7. `official_val_read=false` and `internal_test_read=false` at activation.

Any source-hash mismatch invalidates confirmation until the mismatch is
explained; it does not permit regenerating a preferred source result.

## Phase F1 — descriptive attribute analysis on existing dev archives

### Question

How do the stochastic and direct-token M2-vs-Gaussian effects vary with target
properties, without converting those patterns into mechanism claims?

### Frozen effects

- `delta_ES_pushforward = ES(M1 Gaussian MC) - ES(M2 MC)` from R0.
- `delta_ES_token = ES(Gaussian token) - ES(M2 token)` from R3-P.
- The corresponding risk--coverage differences are secondary.

Positive values mean the M2 interface is better. The two effects are plotted
side by side but never subtracted across audits because their models and seeds
differ.

### Frozen descriptive strata

| Attribute | Bins/groups | Authority |
|---|---|---|
| Detector range | `<20 m`, `20--<40 m`, `>=40 m` | Current detector mean |
| Target speed | `<2 m/s`, `2--<5 m/s`, `>=5 m/s` | Frozen current motion state |
| History length | `2`, `3`, `4` matched frames | R0 track archive |
| LiDAR point-count proxy | `<=5`, `6--20`, `>20` | GT annotation, evaluation only |
| Current XY residual | Clean label-cal quartiles, then frozen | GT association, evaluation only |
| Class group | vehicle, pedestrian, two-wheeler | Frozen class mapping |

The analysis reports scene-aggregated point estimates and 95% bootstrap
intervals without significance stars, subset gates, or “mechanism discovered”
language. Sparse groups report counts and are suppressed if fewer than ten
scenes contribute.

### Allowed interpretation

This phase may identify where the observed interface contrast is concentrated.
It cannot attribute an effect to Student-t degrees of freedom, von Mises
concentration, sensor physics, or causal error mechanisms.

## Phase F2 — external degradation transfer

### Scope

Only three previously named transfer families are eligible:

- camera extrinsic rotation;
- camera extrinsic translation;
- camera temporal offset.

The previously retained external feature shards were found during source-only
inventory to use official-validation metadata and are therefore ineligible for
this pre-official phase. F2 instead performs fresh, frozen-detector exports on
the existing 70-scene dev authority. Before any external metric is read, F0
fixes the five conditions to rotation S2/S3, translation S3, and temporal
offset S2/S3. No severity is added after inspection.

Missing-modality conditions are excluded from this confirmation phase.

### Mandatory comparisons

1. **Stochastic endpoint:** R0 M2 MC versus M1 Gaussian MC and deterministic
   mean under both frozen CV and MLP hosts.
2. **Conditioning endpoint:** R3-P full M2 token versus mean-only, confidence,
   and Gaussian-token baselines.

R2-Core and R2-D are not rerun for external transfer. Their negative claims
already rely on controlled within-audit comparisons, and repeating every
ablation would multiply comparisons without testing the headline boundary.

### Frozen evaluation

- Reuse the exact eligibility rule, history/future transforms, head seeds,
  model checkpoints, sampling seeds, and clean label-cal temperatures.
- Use trajectory energy as primary, with diagnostic NLL, risk, sharpness, and
  ADE/FDE reported exactly as in the source audit.
- Aggregate each external family over all predeclared available severities and
  report per-severity values descriptively.
- Use the same paired-scene bootstrap and report all adverse directions.
- Do not require significance for publication and do not create a new
  scientific endpoint.

### Interpretation table fixed before results

| Outcome | Permitted interpretation |
|---|---|
| R0 positive; R3-P null/limited | The central interface boundary transfers to additional calibration/temporal corruptions. |
| R0 null or reversed | The supported stochastic phenomenon is limited to the original clean/lowlight/dropout/beam scope. |
| R3-P positive | Direct conditioning may depend on degradation type; report heterogeneity without reopening method development. |
| Both positive | Stochastic and learned value coexist under those external conditions; narrow the dev-based necessity statement. |
| Both null/reversed | External corruptions do not confirm the original boundary; retain the original audits as bounded evidence. |

## Phase F3 — final paper-claim freeze

After F1/F2, update only the scope and wording of the claim ledger. Then freeze:

- final title and one-sentence argument;
- main model/comparator set;
- primary/secondary/diagnostic metrics;
- all table and figure specifications;
- all checkpoint and code hashes;
- exact official-validation hypotheses and reporting template.

No scientific code or model may change after this point. Plotting and table
formatting code may be corrected only when it is proven to leave numeric
content unchanged.

## Phase F4 — one-time official-validation confirmation

### Data boundary

Read official validation once after F3. Keep internal test sealed. Do not reuse
unrelated official-validation summaries from other projects as evidence for
this study; build the study's row authority and access log explicitly.

### Frozen comparisons

Evaluate the decisive within-audit contrasts for all four interface layers:

1. R0: deterministic, M1 Gaussian MC, and M2 MC for CV and MLP.
2. R2-Core: matched mean-only, Gaussianized DSE, no-tail, no-circular, and full
   DSE. Because the mechanism claim is already closed, official confirmation
   reports aggregate within-audit contrasts and does not create new mechanism
   subsets.
3. R2-D: MC teacher, GT-only student, and distilled student.
4. R3-P: mean-only, confidence, Gaussian-token, and M2-token GT-only models.

All models, temperatures, distributions, and random seeds remain unchanged.
Official-validation labels are used only for association, future targets, and
metrics under the existing contracts.

### Reporting

- Use a prewritten table schema so no comparator can be omitted after results.
- Report effect sizes, paired-scene intervals, seed directions, sharpness,
  accuracy, risk metrics, and diagnostic NLL.
- Label official-validation results `confirmation`, not model selection.
- Report contradictions in the abstract/discussion if they alter the scope of
  the central boundary.

### Post-read rule

After official-validation access, only these actions are allowed:

- correct a proven implementation defect and rerun the entire affected table
  with a public amendment log;
- update manuscript wording to match the observed evidence;
- generate tables/figures from the frozen numeric archive;
- perform reproducibility verification that does not select results.

No new training, comparator, metric, threshold, subset, or hypothesis is
permitted.

## Completion condition

Final confirmation is complete only when:

- the access log shows no internal-test read;
- every frozen comparison is present for all three seeds;
- every numeric retained value is finite;
- row and source hashes match;
- all outcomes, including adverse ones, appear in the paper evidence table;
- the paper claim ledger has been narrowed where necessary without adding a
  new method claim.

## Not executed by this package

No external-result summary, official-validation artifact, or internal-test
artifact was opened to draft this protocol. This document plans the final
confirmation; it is not its result.
