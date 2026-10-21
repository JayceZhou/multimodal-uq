# Level-I 矩匹配预测律对照：结果报告

## 冻结终点

**RETROSPECTIVE OUTCOME A — AGGREGATE HIGHER-ORDER LAW-SHAPE EFFECT SUPPORTED**

这是在已经使用过的 development、held-out degradation 和 final-validation
population 上完成的 retrospective matched-law audit。internal test 没有打开，
因此结果不能包装成新的独立 confirmatory mechanism discovery。

## 主结论

主估计量为：

```text
ES(HTC-MMG) - ES(HTC)
```

正值表示：在均值、八维线性协方差和 yaw first trigonometric moment 已匹配后，
完整 HTC law 仍具有更好的下游 trajectory energy。

| Population | Probe | Effect | 95% CI | 三个 seed |
|---|---|---:|---:|---:|
| Development clean | CV | 0.01663 | [0.01301, 0.02067] | 0.01512, 0.01958, 0.01520 |
| Development clean | MLP | 0.01695 | [0.01325, 0.02112] | 0.01609, 0.01935, 0.01540 |
| Final validation | CV | 0.01760 | [0.01437, 0.02095] | 0.01557, 0.01907, 0.01817 |
| Final validation | MLP | 0.01763 | [0.01441, 0.02097] | 0.01574, 0.01922, 0.01793 |

clean 的 M=256 敏感性复核方向与效应量一致：CV 为 `0.01710`
（`0.01353–0.02109`），MLP 为 `0.01737`（`0.01383–0.02141`）。

## 六类退化

六个预设 family aggregate 在两个探针上全部为正，且没有一个 family 的 CI
显著支持反向：

| Family | CV effect (95% CI) | MLP effect (95% CI) |
|---|---:|---:|
| Camera lowlight | 0.02085 [0.01664, 0.02557] | 0.02068 [0.01650, 0.02547] |
| LiDAR point dropout | 0.02352 [0.01910, 0.02831] | 0.02246 [0.01812, 0.02722] |
| LiDAR beam reduction | 0.04776 [0.03920, 0.05676] | 0.04573 [0.03763, 0.05429] |
| Camera extrinsic rotation | 0.02340 [0.01888, 0.02838] | 0.02220 [0.01754, 0.02735] |
| Camera extrinsic translation | 0.01570 [0.01193, 0.01969] | 0.01486 [0.01100, 0.01890] |
| Camera temporal offset | 0.01769 [0.01402, 0.02165] | 0.01717 [0.01352, 0.02110] |

## 上游与下游联合解释

development clean 上，`NLL(HTC-MMG) - NLL(HTC)` 同时支持 HTC：

- full eight-dimensional linear NLL：`1.19309`（`0.98029–1.42246`）；
- XY NLL：`0.21478`（`0.17267–0.26307`）；
- yaw NLL：`0.45148`（`0.23413–0.73091`）。

因此，按冻结的文字规则，可以使用以下**有边界的**表述：

> Higher-order upstream predictive-law quality survives the evaluated
> downstream transform.

不能把它扩写为一般性的 Student-t 优于 Gaussian，或单独的 tail/yaw 因果机制。

## 析因诊断

clean 的 aggregate effect 几乎全部来自 linear matched-law contrast：MLP 的
linear effect 为 `0.01696`（`0.01327–0.02114`）。MLP yaw effect 为
`-0.00002`（`-0.00019–0.00014`），interaction 为近似零；CV 因为不读取 yaw，
yaw 与 interaction 均为逐位精确零。这些只属于 secondary diagnostics，不能
恢复此前已经关闭的 component-mechanism claim。

## Sharpness、accuracy 与完整性

- HTC/HTC-MMG 的 90% 等覆盖面积比：development clean 为 CV `0.9984`、MLP
  `1.0024`；final validation 为 `1.0096/1.0109`。能量优势不依赖明显区域膨胀。
- clean population/probe 上的 MMG-minus-HTC mean ADE/FDE 差异均小于
  `0.0015 m`。
- 12 项完整性 gate 全部通过；最小自由度 `2.0999999 > 2`。
- 解析协方差最大相对误差 `5.42×10⁻¹⁶`；resultant 最大误差
  `1.11×10⁻¹⁶`。
- invalid/clipping/rejection rate 均为 `0`。
- trained G 和完整 HTC 对原 R0 trajectory-energy archive 的最大 replay 误差为
  `0.0`。
- internal test 未读取；没有重训练模型或依据结果选择条件。

## 论文应如何使用

可以补入论文的主结果是：

> In a retrospective moment-matched control, the complete HTC predictive law
> retained a significant trajectory-energy advantage over an analytically
> derived Gaussian/wrapped-Gaussian surrogate with the same mean, linear
> covariance, and yaw first trigonometric moment across both frozen probes.
> The contrast remained directionally stable across all six evaluated
> degradation families and on final validation.

必须紧接限制：

> Because this control was motivated by previously observed results and used
> existing evaluation populations, it bounds the interpretation of the
> original Level-I finding rather than establishing a new confirmatory causal
> mechanism. Factorial diagnostics localized the aggregate contrast mainly to
> the linear law replacement but do not justify an independent tail-mechanism
> claim.

## 主要文件

- `result.json`：完整机器可读结果；
- `bootstrap_summary.csv`：condition/family/upstream/M=256 的 bootstrap；
- `condition_seed_summary.csv`：逐 condition、seed、law 的全指标；
- `paired_track_metrics.h5`：逐目标配对数据；
- `scene_level_metrics.h5`：可追溯 scene-level ES；
- `figures/matched_law_primary.*`：主图；
- `figures/matched_law_factorial.*`：补充析因图；
- `integrity_report.json` 与 `sampler_validation.json`：QA 证据。
