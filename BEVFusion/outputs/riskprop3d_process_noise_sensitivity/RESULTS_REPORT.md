# Critical Issue 6 — Future process-noise sensitivity audit

## Frozen question

Does the Level-I `G − HTC` trajectory-energy advantage depend on the single
additive future process-noise magnitude fitted on clean label-calibration
residuals?

The audit changes only the covariance multiplier:

```text
0×, 0.5×, 1×, 2× fitted covariance
```

These correspond to standard-deviation multipliers `0`, `0.707`, `1`, and
`1.414`. G and HTC use the same base Gaussian draws at every multiplier. The
probability heads, 64 perception samples, two perception draw sets, CV/MLP
probes, propagation seeds, common random numbers, tracks, and future targets
are unchanged. No model was retrained.

## Result

**PROCESS-NOISE SENSITIVITY SUPPORTED.**

- All `16/16` population × probe × covariance-multiplier aggregate effects
  favored HTC, with every paired scene-bootstrap 95% CI lower bound above
  zero.
- All `48/48` individual seed directions were positive.
- The `1×` sensitivity implementation reproduced every retained G/HTC energy
  value for development clean and final validation exactly
  (`max_abs_difference = 0.0`).
- Increasing process noise attenuated the law contrast, as expected, but did
  not reverse or eliminate it at twice the fitted covariance.

Positive values below are `ES_G − ES_HTC`, so larger values favor HTC.

| Population | Probe | Covariance | Effect | Paired scene-bootstrap 95% CI |
|---|---|---:|---:|---:|
| Development clean | CV | 0× | 0.03050 | 0.02449–0.03660 |
| Development clean | CV | 0.5× | 0.02059 | 0.01507–0.02629 |
| Development clean | CV | 1× | 0.01614 | 0.01102–0.02150 |
| Development clean | CV | 2× | 0.01108 | 0.00636–0.01601 |
| Development clean | MLP | 0× | 0.02290 | 0.01834–0.02759 |
| Development clean | MLP | 0.5× | 0.01614 | 0.01206–0.02032 |
| Development clean | MLP | 1× | 0.01230 | 0.00850–0.01619 |
| Development clean | MLP | 2× | 0.00782 | 0.00439–0.01138 |
| Final validation | CV | 0× | 0.03372 | 0.02862–0.03943 |
| Final validation | CV | 0.5× | 0.02522 | 0.02061–0.03038 |
| Final validation | CV | 1× | 0.02060 | 0.01632–0.02532 |
| Final validation | CV | 2× | 0.01502 | 0.01118–0.01920 |
| Final validation | MLP | 0× | 0.03144 | 0.02552–0.03845 |
| Final validation | MLP | 0.5× | 0.02519 | 0.01981–0.03151 |
| Final validation | MLP | 1× | 0.02101 | 0.01603–0.02688 |
| Final validation | MLP | 2× | 0.01579 | 0.01136–0.02105 |

From `0×` to `2×` covariance, the point effect decreased by 63.7%/65.9% for
development CV/MLP and 55.5%/49.8% for final-validation CV/MLP. This confirms
attenuation by downstream aleatoric noise while preserving the HTC-over-G
direction and interval support.

## Recommended English manuscript text

### Methods

> We assessed whether the Level-I contrast depended on the magnitude of the
> additive future process-noise model. Without retraining any model, we repeated
> the clean development and final-validation G-versus-HTC propagation
> comparisons using covariance multipliers of 0, 0.5, 1, and 2. These values
> correspond to standard-deviation multipliers of 0, 0.707, 1, and 1.414. The
> two predictive laws shared the same base Gaussian process-noise draws at each
> multiplier; all perception samples, propagation seeds, common random numbers,
> forecasting probes, tracks, and targets were otherwise unchanged. Effects
> were defined as ES_G − ES_HTC, averaged within scene and replicate and then
> across three paired replicates. Uncertainty was quantified using 10,000
> paired scene-bootstrap resamples.

### Results

> The HTC-over-G trajectory-energy advantage was robust to the fitted
> process-noise magnitude. All 16 population-by-probe-by-scale contrasts were
> positive with 95% confidence intervals excluding zero, and all 48 individual
> seed directions agreed. On development clean, the CV contrast decreased from
> 0.03050 (95% CI 0.02449–0.03660) without added process noise to 0.01108
> (0.00636–0.01601) at twice the fitted covariance; the corresponding MLP
> effects were 0.02290 (0.01834–0.02759) and 0.00782
> (0.00439–0.01138). Final validation showed the same pattern: the CV effect
> decreased from 0.03372 (0.02862–0.03943) to 0.01502
> (0.01118–0.01920), and the MLP effect from 0.03144
> (0.02552–0.03845) to 0.01579 (0.01136–0.02105). Thus, increasing downstream
> process uncertainty attenuated but did not reverse or eliminate the
> predictive-law contrast.

### Discussion

> The sensitivity analysis separates the existence of the stochastic
> predictive-law contrast from one particular fitted process-noise magnitude.
> Larger additive noise progressively reduced the G-versus-HTC energy
> difference, consistent with downstream aleatoric variation masking upstream
> distributional distinctions. Nevertheless, the contrast remained supported
> for both controlled probes and both evaluation populations from no added
> noise to twice the fitted covariance. The Level-I result is therefore not an
> artifact of selecting the original covariance magnitude. This audit varies
> magnitude only and does not establish robustness to a misspecified
> non-Gaussian or temporally correlated future-noise family.

### Reviewer response

> We thank the reviewer for identifying the possible dependence on the fitted
> future process-noise model. We added a preregistered magnitude-sensitivity
> analysis on clean development and final validation, using 0×, 0.5×, 1×, and
> 2× the original horizon-wise covariance. G and HTC shared identical base
> process-noise draws, and no model was retrained. The original 1× computation
> was reproduced exactly at the per-target level. Across both CV and MLP, all
> 16 aggregate contrasts favored HTC with paired scene-bootstrap 95% confidence
> intervals above zero, and all 48 seed-level directions were positive. The
> effect decreased monotonically as process noise increased but remained
> supported at 2× covariance. We have added the curve and clarified that the
> conclusion concerns robustness to process-noise magnitude, not to arbitrary
> alternative noise families.

## 建议的中文论文表述

### 方法

> 为检验 Level-I 结论是否依赖加性未来过程噪声的特定幅度，我们在不重新训练任何模型的情况下，将 clean label-calibration 拟合所得逐预测时刻协方差分别乘以 0、0.5、1 和 2，并重新执行 development clean 与 final validation 上的 G–HTC 传播比较。上述设置对应的标准差倍数为 0、0.707、1 和 1.414。在每个倍数下，G 与 HTC 共享相同的基础高斯过程噪声随机数；感知样本、传播 seed、common random numbers、预测探针、目标行和未来真值均保持不变。效应定义为 `ES_G − ES_HTC`，先在 scene 和 replicate 内配对汇总，再对三个 replicate 求平均，并采用 10,000 次 paired scene bootstrap 计算95%置信区间。

### 结果

> HTC 相对 G 的轨迹 energy 优势对过程噪声幅度保持稳健。全部16个 population × probe × noise-scale 对比均为正，且95%置信区间下界高于0；48个单seed方向也全部一致。在 development clean 上，CV效应由无附加过程噪声时的0.03050（95% CI 0.02449–0.03660）下降至2倍协方差时的0.01108（0.00636–0.01601），MLP对应效应由0.02290（0.01834–0.02759）下降至0.00782（0.00439–0.01138）。Final validation呈现相同规律：CV效应由0.03372（0.02862–0.03943）下降至0.01502（0.01118–0.01920），MLP效应由0.03144（0.02552–0.03845）下降至0.01579（0.01136–0.02105）。因此，更大的下游过程不确定性会削弱上游概率律差异，但在预设范围内不会使其反转或消失。

### 讨论边界

> 该分析证明 Level-I 正向对比并非由单一 fitted covariance magnitude 人为产生，但只检验了同一高斯过程噪声族内的幅度敏感性，不能外推为对非高斯、时间相关或其他错误指定过程噪声模型的稳健性。

## Artifacts

- `result.json`: complete machine-readable effects and audit endpoint.
- `figure_source_data.csv`: source data for every plotted point and interval.
- `figure_process_noise_sensitivity.pdf/.svg/.png/.tiff`: submission-ready
  figure exports.
- `metrics/replicate_*.npz`: paired per-target energy values.
- `protocol.json`: frozen inputs, multipliers, seeds, and source hashes.
- `FIGURE_CONTRACT.md`: figure logic and QA contract.
