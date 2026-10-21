# Level-I Moment-Matched Law Control — Results Report

**RETROSPECTIVE OUTCOME A — AGGREGATE HIGHER-ORDER LAW-SHAPE EFFECT SUPPORTED**

This is a retrospective matched-law audit on already used development, held-out-degradation, and final-validation populations. It does not open the internal test and cannot establish a new confirmatory component mechanism.

## Primary matched-law contrast

Positive values are `ES(HTC-MMG) - ES(HTC)` and favor the complete HTC law.

| Population | Probe | Effect | 95% CI | Seed effects |
|---|---|---:|---:|---:|
| Development clean | CV | 0.01663 | [0.01301, 0.02067] | 0.01512, 0.01958, 0.01520 |
| Development clean | MLP | 0.01695 | [0.01325, 0.02112] | 0.01609, 0.01935, 0.01540 |
| Final validation | CV | 0.01760 | [0.01437, 0.02095] | 0.01557, 0.01907, 0.01817 |
| Final validation | MLP | 0.01763 | [0.01441, 0.02097] | 0.01574, 0.01922, 0.01793 |

The clean (M=256) sensitivity run reproduced the direction and magnitude:
CV `0.01710` (95% CI `0.01353--0.02109`) and MLP `0.01737`
(`0.01383--0.02141`).

## Degradation-family transfer

All six prespecified family aggregates favored full HTC for both probes. No
family had a negative point estimate or a confidence interval supporting a
reversal.

| Family | CV effect (95% CI) | MLP effect (95% CI) |
|---|---:|---:|
| Camera lowlight | 0.02085 [0.01664, 0.02557] | 0.02068 [0.01650, 0.02547] |
| LiDAR point dropout | 0.02352 [0.01910, 0.02831] | 0.02246 [0.01812, 0.02722] |
| LiDAR beam reduction | 0.04776 [0.03920, 0.05676] | 0.04573 [0.03763, 0.05429] |
| Camera extrinsic rotation | 0.02340 [0.01888, 0.02838] | 0.02220 [0.01754, 0.02735] |
| Camera extrinsic translation | 0.01570 [0.01193, 0.01969] | 0.01486 [0.01100, 0.01890] |
| Camera temporal offset | 0.01769 [0.01402, 0.02165] | 0.01717 [0.01352, 0.02110] |

## Upstream matched-law quality

On development clean, positive `NLL(HTC-MMG) - NLL(HTC)` contrasts jointly
favored HTC: full eight-dimensional linear NLL `1.19309`
(`0.98029--1.42246`), XY NLL `0.21478` (`0.17267--0.26307`), and yaw NLL
`0.45148` (`0.23413--0.73091`). Under the frozen wording rule, the joint
upstream and downstream results permit the bounded statement that
higher-order upstream predictive-law quality survives the evaluated
downstream transform. This is not a causal Student-t-tail or von-Mises-yaw
claim.

## Factorial diagnostics

The clean aggregate was carried almost entirely by the linear matched-law
contrast. CV, which does not read yaw, returned exact zero for the yaw and
interaction checks. For MLP, the linear contrast was `0.01696`
(`0.01327--0.02114`), whereas the yaw contrast was `-0.00002`
(`-0.00019--0.00014`) and the interaction was `-0.00000`
(`-0.00005--0.00004`). These are secondary diagnostics only and do not revive
the closed component-mechanism claims.

## Sharpness and mean-accuracy diagnostics

The energy improvement was not accompanied by a material equal-coverage
region change. HTC/HTC-MMG final-position 90%-equal-coverage area ratios were
`0.9984/1.0024` for CV/MLP on development clean and `1.0096/1.0109` on final
validation. Corresponding MMG-minus-HTC mean ADE/FDE differences were below
`0.0015 m` in every clean population/probe combination. Natural coverage and
all per-law, per-condition, and per-seed values are retained in
`condition_seed_summary.csv`.

## Interpretation checks

- Joint clean-probe criterion: **PASS**.
- Clean M=256 direction check: **PASS**.
- Six-family directional stability: **PASS**.
- Clean upstream full-linear/XY/yaw NLL joint criterion: **PASS**.
- Higher-order upstream-quality transport wording permitted by the frozen rule: **YES**.

## Integrity

All integrity gates: **PASS**.
Minimum df: `2.100000`; maximum analytic covariance relative error: `5.417e-16`; maximum resultant error: `1.110e-16`; maximum invalid rate: `0.000e+00`; maximum R0 energy replay error: `0.000e+00`.

No clipping or rejection is used by the frozen linear sampler; yaw is wrapped identically. CV yaw contrasts are exact zero checks and are not circular-mechanism evidence.

## Files

- `result.json`: machine-readable endpoint and all paired effects.
- `bootstrap_summary.csv`: condition/family/upstream/MC intervals.
- `condition_seed_summary.csv`: per-condition, seed, and law summaries.
- `paired_track_metrics.h5`: paired per-target metrics.
- `scene_level_metrics.h5`: traceable scene-level energy arrays.
- `figures/`: primary and factorial manuscript/supplement figures.

## Packaging note

The frozen finalizer initially reached its HDF5 serialization step but h5py
rejected NumPy Unicode token arrays. The failed partial files are retained in
`packaging_failures/`. `packaging_shim.json` documents a serialization-only
conversion to fixed-width UTF-8 bytes; no estimator, bootstrap, gate, or
endpoint code changed.
