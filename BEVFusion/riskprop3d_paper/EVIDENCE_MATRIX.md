# Evidence matrix

## Reading rule

Every quantitative statement below is copied from a retained result authority.
Positive improvement follows each audit's frozen sign convention. Results from
different audits must not be combined as if they were produced by one common
model or seed set. Confidence intervals are paired-scene bootstrap intervals
unless the source explicitly states otherwise.

## Four-interface evidence matrix

| Layer | Scientific question | Frozen comparison | Decisive evidence | Endpoint | Paper-level interpretation |
|---|---|---|---|---|---|
| Stochastic pushforward (R0) | Does the full M2 perception distribution carry downstream information beyond deterministic and Gaussian perception? | M2 Student-t/von Mises MC (`P3`) vs M1 Gaussian/wrapped-Gaussian MC (`P2`) and deterministic mean (`P1`), using 64 samples through both CV and frozen MLP hosts | Clean P3-over-P2 energy gain: 0.01614 (0.01102--0.02150) for CV and 0.01230 (0.00850--0.01619) for MLP. Full-family gains were 0.01411--0.04902; S1/S2-only gains were 0.01222--0.02963. Equal-coverage area ratios were 0.866--0.929. Clean high-FDE AUPRC gains were 0.03745/0.03251 for CV/MLP. | `GO`, 7/7 | Direct propagation of the full M2 law is a supported phenomenon under both hosts and all three frozen degradation families. |
| Distributional encoding, initial transfer (R1.5) | Can a learned target-centric encoder outperform raw uncertainty concatenation and brute-force MC? | Frozen R1 DSE vs raw concat and 64-sample MC | DSE-over-concat energy gains were significant in all full families (0.01721--0.02070) and all S1/S2 aggregates (0.01834--0.01928). Clean DSE-over-MC gain was 0.05756 (0.04722--0.06851), with 5.72x speedup and clean equal-coverage area ratio 0.8817. | `GO`, 5/5 | A positive learned result existed, but it did not yet isolate distribution information from capacity or seed effects. It motivates, but cannot replace, R2-Core. |
| Distributional encoding, controlled mechanism audit (R2-Core) | Is the DSE gain independent of capacity, specific to M2, and attributable to tail/circular inputs? | Full M2 DSE (`C5`) vs matched mean-only (`C0`), Gaussianized DSE (`C2`), no-tail (`C3`), and no-circular (`C4`) | C5-over-C0 clean energy was 0.00690 (-0.00150--0.01605); C5-over-C2 was 0.00508 (-0.00775--0.01753). C5-over-C3 was significant overall (0.00796; 0.00080--0.01570), but no preregistered tail-sensitive subset was significant. C5 was non-significantly worse than C4 on clean, and two-wheelers significantly favored removing circular concentration. Sharpness and seed-stability gates also failed. | `STOP`, 1/8 | The initial DSE result does not support a stable capacity-independent M2, heavy-tail, or circular-yaw mechanism. |
| Pushforward distillation (R2-D) | Can the accepted MC future distribution be amortized without losing downstream utility, and is preserving it necessary? | Identical single-pass students trained with GT only (`D2`) or GT plus transport/calibration/risk preservation (`D3`), compared with the 64-sample MC teacher (`D1`) | D3 was significantly closer to the teacher than D2 on all four fidelity metrics. D3 beat D1 energy by 0.10957 on clean and 0.09929--0.11080 across full families, retained risk utility, used smaller equal-coverage areas, and was 38.55x faster. However, D2 significantly beat D3 energy by 0.05941 (0.02657--0.08835 in D2's favor); all three clean seeds and the S1/S2 families favored D2. | `STOP`, 5/6 | MC behavior can be compressed efficiently, but teacher-distribution preservation is not a necessary or beneficial training contribution once direct GT supervision is available. |
| Probability-token conditioning (R3-P) | Is explicit M2 probability information necessary under ordinary GT supervision? | Identical GT-only architecture, parameters, initialization, data, and loss: mean only (`P0`), mean plus confidence (`P1`), Gaussian token (`P2`), and full M2 token (`P3`) | Clean P3-over-P0 energy was 0.00205 (-0.01378--0.01559); P3-over-P1 was 0.00069 (-0.01531--0.01321); P3-over-P2 was -0.00140 (-0.01454--0.00943). All full-family CIs crossed zero. P3 did improve clean risk--coverage over P0 by 0.00699 (0.00065--0.01369) and P1 by 0.00672 (0.00027--0.01381), while satisfying sharpness and accuracy limits. | `STOP`, 2/6 | Explicit M2 probability tokens are not necessary for primary trajectory energy under ordinary supervision. A narrower risk-ranking increment remains supported. |

## Claim ledger

| ID | Candidate manuscript claim | Status | Evidence that permits the claim | Boundary or disallowed extension |
|---|---|---|---|---|
| C1 | M2 stochastic pushforward improves trajectory energy over Gaussian stochastic pushforward in the frozen modular pipeline. | **Supported** | R0: significant clean and family gains for both CV and MLP; all three clean replicate directions positive. | Do not generalize to arbitrary detectors, predictors, datasets, or corruptions before final confirmation. |
| C2 | The stochastic gains are not purchased by wider final prediction regions and include future-error identification benefits. | **Supported** | R0: equal-coverage P3/P2 area ratios 0.866--0.929; high-FDE AUPRC and false-safe evidence; ADE/FDE harm within frozen limits. | “Risk” means the frozen high-FDE/final-area proxies, not demonstrated collision or planning safety. |
| C3 | A dedicated learned encoder can show gains over raw concatenation and MC in one frozen experiment. | **Supported but superseded for mechanism** | R1.5: 5/5 transfer gates, significant energy and risk results. | Must immediately disclose that matched R2-Core controls did not establish capacity independence or seed stability. Do not use R1.5 alone as the method claim. |
| C4 | DSE gains are caused by Student-t tails or von Mises concentration. | **Not supported** | R2-Core G3/G4 failed; no tail-sensitive subset replicated the overall tail ablation; no core yaw subset improved and two-wheelers reversed. | No causal or feature-specific mechanism language. |
| C5 | The full M2 learned representation is necessary beyond capacity, confidence, or Gaussian uncertainty. | **Not supported** | R2-Core G1/G2/G8 and R3-P G1/G2/G3/G6 failed. | Do not claim that learned forecasting “understands” heavy tails/circular uncertainty. |
| C6 | A single-pass model can approximate important aspects of the MC future distribution at much lower inference cost. | **Supported** | R2-D fidelity G1, risk-retention G4, sharpness G5, and 38.55x efficiency G6 passed. | Approximation fidelity is not evidence that the preservation loss improves downstream forecasting. |
| C7 | Distribution-preserving distillation is preferable to ordinary GT training. | **Contradicted** | R2-D: identical GT-only D2 significantly beat D3 on clean and degradation energy; diagnostic NLL also favored D2. | Present as a negative result, not as a proposed training method. |
| C8 | Explicit M2 probability conditioning improves primary forecasting quality. | **Not supported** | R3-P primary energy CIs crossed zero against P0/P1/P2; no degradation family was significant. | Diagnostic NLL improvements over P0/P1 do not repair the frozen primary result. |
| C9 | Explicit probability conditioning retains incremental future-risk ranking information. | **Supported, narrow** | R3-P: clean risk--coverage improved significantly over both mean-only and confidence baselines. | High-FDE AUPRC and false-safe did not pass against both baselines; no universal risk claim. |
| C10 | Explicit uncertainty-aware pairwise interaction improves forecasting. | **Contradicted in the evaluated form** | R1/R1.5 negative control: clean energy effect -0.01367 (-0.02424---0.00299), adverse in all three seeds. | State only that Interaction v1 was unsupported; do not infer that uncertainty can never affect interaction. |
| C11 | One uncertainty interface is superior on every probabilistic metric. | **Not supported** | R0 favored M2 MC on energy and diagnostic NLL, whereas later learned audits showed metric-dependent directions; R1.5 DSE NLL was worse than MC, R2-D favored GT-only over distillation, and R3-P NLL/energy diverged. | Keep trajectory energy primary and report diagnostic NLL honestly. |

## Central synthesis

The evidence supports a distinction between **information revealed by an
explicit stochastic computation** and **information required as an explicit
input to a supervised learner**:

```text
M2 perception distribution
        |
        +-- stochastic pushforward --> supported trajectory/risk gain
        |
        +-- handcrafted learned encoding --> mechanism not established
        |
        +-- distribution-preserving distillation --> feasible, not necessary
        |
        `-- direct probability token + GT supervision --> no primary ES gain
                                                       `-- narrow risk gain
```

The paper must not collapse these branches into the generic statement
“uncertainty helps” or “uncertainty does not help.” The finding is conditional
on the interface and supervision regime.

## Validity and evidence scale

| Audit | Scientific endpoint | Retained numeric values | Condition tracks | Key governance |
|---|---:|---:|---:|---|
| R0 | 7/7 GO | 126,936,582 | 330,564 | Three head/propagation seeds, two hosts, paired-scene bootstrap |
| R1.5 | 5/5 GO | Read-only audit of three retained R1 metric archives | 330,564 | No retraining or new metric archive |
| R2-Core | 1/8 STOP | 107,342,589 | 330,564 | Capacity/initialization matching, frozen mechanism subsets |
| R2-D | 5/6 STOP | 75,515,193 | 330,564 | Exact R0 teacher replay, paired student initialization |
| R3-P | 2/6 STOP | 67,729,089 | 330,564 | Identical architecture/loss/data/init across four tokens |

All listed audits retained the official-validation and internal-test seals.
The differing numeric-value counts reflect different retained arrays and must
not be summed and presented as independent observations.

## Internal-to-paper crosswalk

| Paper-facing comparison | Internal source |
|---|---|
| Deterministic vs Gaussian vs M2 stochastic pushforward | R0 `P1/P2/P3` under CV and MLP |
| Raw uncertainty concatenation vs distributional encoding | R1/R1.5 `B2_concat/R1_DSE` |
| Capacity and family controls | R2-Core `C0_mean_matched/C2_gaussian_dse/C5_full_dse` |
| Tail and circular feature controls | R2-Core `C3_no_tail/C4_no_circular/C5_full_dse` |
| MC teacher vs GT-only vs distilled student | R2-D `D1_mc_teacher/D2_gt_only/D3_risk_distilled` |
| Mean/confidence/Gaussian/M2 token necessity | R3-P `P0_mean_only/P1_confidence/P2_gaussian/P3_m2_probability` |

## Machine-readable authorities

- R0: [`../outputs/riskprop3d_r0/result.json`](../outputs/riskprop3d_r0/result.json)
- R1.5: [`../outputs/riskprop3d_r15/result.json`](../outputs/riskprop3d_r15/result.json)
- R2-Core: [`../outputs/riskprop3d_r2_core/result.json`](../outputs/riskprop3d_r2_core/result.json)
- R2-D: [`../outputs/riskprop3d_r2d/result.json`](../outputs/riskprop3d_r2d/result.json)
- R3-P: [`../outputs/riskprop3d_r3p/result.json`](../outputs/riskprop3d_r3p/result.json)
