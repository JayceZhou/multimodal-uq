# RiskProp3D 前置研究补充方案：HiVT 代表性预测宿主扩展

## 0. 材料身份与当前状态

- 来源：作者提供的《代表性预测宿主补实验方案》，`strong_host_extension_plan_v1`，原日期 2026-10-12。
- 本版：`hivt_host_extension_server_plan_v1`；整理日期 2026-10-13（北京时间）。
- 原方案整理时状态：方案与只读盘点完成，尚未执行。作者随后授权启动并在缺失时重跑输入；当前进度见 [EXECUTION_STATUS.md](EXECUTION_STATUS.md)，恢复约束更新见 [RECOVERY_PROTOCOL.md](RECOVERY_PROTOCOL.md)。本方案的历史盘点不作为当前完成状态。
- 对应研究：RiskProp3D／HTC 感知到预测接口研究。当前 IDE 中的 Study3 FV1/QD1 属于另一个检测研究序列，不作为本方案的实验父项。
- 本轮交付是方案文档。新宿主使用独立目录和前向冻结记录；旧 MTR、AutoBots、CV/MLP 与原确认实验的协议、判据和终态保持各自权威。

推荐先完成 Level I：三次配对宿主训练，以及 mean-state、G、HTC、HTC-MMG 四种传播评估。若论文还需要直接概率输入的宿主外推证据，则在开始新结果读出前同时登记 Level II 四组；Level III 沿用原有结果并保留其宿主边界。

## 1. 补实验回答什么

给定共同的冻结感知均值、正确目标身份、固定地图，以及由感知构建的固定历史与邻居背景，检验：

1. HTC 相对独立训练的 G，是否仍改善具有地图与主体交互的预测宿主上的轨迹概率质量？
2. HTC 相对同均值、同协方差的 HTC-MMG，是否仍有高阶律形状差异？
3. 若完成 Level II，HTC 概率输入相对既有 confidence/density 控制，是否有额外学习价值？

主宿主固定为附件指定的 HiVT nuScenes 改编版。使用其 `base` 地图编码分支和原生多模态 Laplace decoder；HD 地图由当前位置对应的固定地图几何提供。源码与依赖在适配阶段锁定 commit 和文件哈希，不能只记录会移动的 `main`。这是代表性架构扩展，不承担当前 SOTA 比较。[实现入口](https://github.com/alfredgu001324/MapUncertaintyPrediction/blob/main/docs/trj.md)、[宿主源码](https://github.com/alfredgu001324/MapUncertaintyPrediction/blob/main/HiVT_modified/models/hivt.py)。

**输入范围按 HiVT 真正读取的坐标限定。** 最低实现保留原生 XY 历史、由 XY 位移导出的 heading、主体相对几何和地图相对几何；当前状态随机干预为 HTC/G 的 **XY 边缘律**。不为补实验额外加入 z、尺寸、速度或检测 yaw 通道。由位移导出的 heading 不等于直接读取 von Mises yaw；因此本扩展不能声称验证了完整 3D box 或 circular-yaw 机制。

## 2. 已有研究状态与本版对附件的修订

| 已有材料 | 服务器核查到的状态 | 对新方案的影响 |
| --- | --- | --- |
| CV/MLP R0 与匹配律控制 | 轨迹、预测器和匹配律构造代码仍在；匹配律控制已完成 | 复用输入、构造与统计；跨宿主比较在本轮相同车辆目标上重算或重聚合 |
| UniTraj-MTR | MH1 三重复评估无效；根因审计定位到极端但有限的 HTC log-size 输入导致 context encoder 非有限 | 保存支持域失败；新 HiVT 的原生 XY 输入需独立做支持检查；不裁剪旧样本或重开旧门槛 |
| AutoBots MH-A1 | 最新资格记录为 `STOP_MH_A1_MEAN_STATE_QUALIFICATION` | 不把训练完成写成宿主通过，不继续 MH-A2 |
| AutoBots 邻居构造器 | 当前及过去邻居位置来自 nuScenes annotation translation | 可以参考其过去时间访问和地图变换检查，不能作为本方案“感知邻居”的现成构造器 |
| 原 official validation | 原简单宿主结果已经被观察 | 新研究属于观察旧结果后设计的扩展，不能标记为整项研究从未访问的盲测 |

AutoBots 最新记录中，相同 34,856 条 clean-dev 目标、70 个场景上的场景平均 ADE 为 0.876821 m，旧 residual MLP 为 0.597188 m；故其原资格门槛未通过。这些是已有资格结果，不是本次 HiVT 的结果。[资格终态](../outputs/riskprop3d_modern_host_autobots/mh_a1_mean_qualification_attempt3/mh_a1_mean_qualification_attestation.json)。MTR 的无效终态与根因见[旧宿主报告](../riskprop3d_modern_host/README.md)、[根因决策](../riskprop3d_modern_host/MH1_ROOT_CAUSE_DECISION.md)。

本版主要修订：将附件的“需要恢复全部输入”细化为已存在输入和缺失输入；将完整律的表述收窄到原生 XY 边缘律；明确已有 GT 邻居构造器不符合新契约；补入缺失权重、两套 ES 抽样和执行前冻结清单。所有修订在新 HiVT 结果产生前提出。

## 3. 服务器可复用输入与真正缺口

### 3.1 已核查的轨迹数量

车辆类别事前固定为本工程 `probset3d_f0_score.CLASSES` 的索引 0–4：car、truck、construction_vehicle、bus、trailer。不套用其他适配器的类别编号，不按结果选择类别。[类别权威](../tools/probset3d_f0_score.py)。这与参考 dataset 对车辆的筛选范围相容，但目标必须使用本研究 retained targets。[参考 dataset](https://github.com/alfredgu001324/MapUncertaintyPrediction/blob/main/HiVT_modified/datasets/nuscenes_dataset.py)。

下表仅统计已有身份与类别字段，未计算新模型指标；车辆数是原档案中该固定子集的数量。最终上下文构建若失败，记录失败并解决输入问题，不能按候选律的成功情况删行。

| 角色／条件 | 相对 BEVFusion 的现存轨迹路径 | 原目标数／场景数 | 车辆目标数／场景数 |
| --- | --- | ---: | ---: |
| clean training | `outputs/riskprop3d_r0/tracks/train.npz` | 295,718 / 560 | 162,919 / 559 |
| label-calibration | `outputs/riskprop3d_r0/tracks/label_cal.npz` | 39,192 / 70 | 22,188 / 69 |
| development / clean_s0 | `outputs/riskprop3d_r0/tracks/clean_s0.npz` | 34,856 / 70 | 18,230 / 70 |
| development / dropout_s3 | `outputs/riskprop3d_r0/tracks/dropout_s3.npz` | 33,675 / 70 | 17,946 / 70 |
| held_out_degradation / rotation_s3 | `outputs/riskprop3d_final_confirmation/tracks/external/rotation_s3.npz` | 34,047 / 70 | 17,702 / 70 |
| final_validation / official_clean | `outputs/riskprop3d_final_confirmation/tracks/official/clean.npz` | 68,436 / 150 | 38,644 / 150 |

轨迹 `input` 为 `[N,64]`：四个历史槽与当前槽各 10 个状态坐标、四个历史 mask、十类编码；`future` 为 `[N,6,2]`。实际历史长度为 2–4，间隔 0.5 s。历史左填充首个有效状态并由 mask 排除；不能将填充值当新观测。状态表示含 XY/Z、log 尺寸、XY 速度、sin/cos yaw。[原构造代码](../tools/riskprop3d_r0.py)。

原构造在当前条件的同一 observation map 中寻找连续历史，所以已有目标历史与当前帧来自**同一退化条件**，并非固定 clean 历史。新方案沿用此规则；组间只改变当前目标 XY 样本。

### 3.2 输入恢复清单

| 输入 | 可复用材料 | 还需完成的工作 |
| --- | --- | --- |
| 身份、共同历史、mask、未来标签 | 上述轨迹 NPZ | 恢复原始时间戳和 LiDAR/global 变换；验证分割互斥与原行顺序；模型仅收到历史／当前特征 |
| 开发／外部／official 的概率头输入 | 对应轨迹含 `feature[128]`、`logits[10]`、`box[9]`、`seed` | 恢复冻结头后重新前向导出完整参数；逐行核对共同均值 |
| train／label-cal 的概率头输入 | `outputs/riskprop3d_r1/scenes/{train,label_cal}.npz` 含 agent 特征表及 `target_agent_index` | 通过目标索引恢复头输入并核对 scene/sample/instance；R0 train NPZ 本身没有这些 query 字段 |
| 邻居当前感知输出 | R1 scene graph 有 `agent_box`、`neighbor_agent_index` | 复用既有 30 m／最多 8 个邻居的选择规则；从同条件过去／当前感知帧补齐邻居历史、身份和 mask；外部与 official 也建立相同契约 |
| 地图 | nuScenes HD map；既有代码有坐标变换参考 | 生成 HiVT 需要的固定 lane 几何；不引入地图预测误差或未来信息 |
| 指标／配对索引 | `outputs/riskprop3d_upstream_downstream_export/paired_rows.h5` | 仅用于行对齐和旧结果复核；文件不含完整均值／scale／df 等头参数，不能直接训练 HiVT |

R1 邻居是已关联的感知输出；它不是包括漏检、背景和身份错误的完整部署邻居集。新结果保持“给定正确关联和保留目标”的条件解释。邻居当前集合由共同感知均值选择后固定；不得用未来存续筛邻居，不为缺失邻居历史补 GT 位置。

**原方案整理时的执行阻塞：** 三组冻结头在原权威路径 `outputs/probset3d_v3/f1_formal/checkpoints/seed_{head_seed}.pt` 均不存在，按这些文件名在工作区搜索也未找到。作者之后明确授权重跑；本轮新头单独存储，不再要求其等于旧头哈希。原先只允许字节恢复的约束由 [RECOVERY_PROTOCOL.md](RECOVERY_PROTOCOL.md) 更新；旧冻结清单仍提供原头 SHA-256 供区分版本。[哈希权威](../outputs/riskprop3d_level1_matched_law/frozen_config.json)。

| head_seed | 原权重 SHA-256 |
| --- | --- |
| 20260731 | `087b3c4c63d40ab3314e1a6730c220384ea0212c83c4f3a895b083145679ffbc` |
| 20260801 | `1781113c58cd5fb11d81befcd7f28be346ead4d431f18b71d2c1b74ca05f2f90` |
| 20260802 | `ddf350e5dc5ea02aa87831b0cfb26b63b3b4ba555081048b3c0fd625cab60a5f` |

## 4. HiVT 数据适配与先行资格检查

时间窗改为五个历史／当前槽和六个未来槽，当前索引 4。参考实现的 dataset 和可视化存在 20 历史、30 未来、当前索引 19 的硬编码；须检查 dataset、mask、local/global encoder、decoder、指标、可视化与 batch 索引全部涉及的位置。参考 validation 默认读 `av_index`，新评估改为带完整 token 的 retained target index。[时间窗与索引依据](https://github.com/alfredgu001324/MapUncertaintyPrediction/blob/main/HiVT_modified/datasets/nuscenes_dataset.py)、[评估依据](https://github.com/alfredgu001324/MapUncertaintyPrediction/blob/main/HiVT_modified/models/hivt.py)。

模型坐标变换由共同均值构建并冻结，标签独立存于评分端。当前 XY 在概率头的原当前 LiDAR 坐标系采样，再转换到模型坐标系。地图和邻居的物理坐标保持固定，随目标当前 XY 改变而重新计算位移、heading、agent-agent 与 lane-agent 相对特征；不能把地图随目标平移。邻居／lane 身份集合预先固定；状态相关的相对边及原生半径 mask 按冻结算法重新计算。

如对角概率参数保留在 LiDAR 坐标中，token 必须声明该坐标系；旋转后的 XY 样本直接变换，不将旋转后的协方差继续假装为对角。预测的整个轨迹样本先转回原固定 LiDAR 坐标，再评分。

先行检查应在新 ES 读出前完成：

1. 固定身份小样本核对时间戳、历史 mask、目标／邻居身份、标签隔离、地图与坐标 round trip；不得将 `gt_input`、`ego_future`、`collision_event` 或目标未来位置交给 inference features。
2. Monte Carlo 输入中 `data.y=None`；每次使用独立 clone。参考 forward 会旋转并修改 `data.y`，重复推理需验证原对象、标签与坐标没有累计变更。[forward 依据](https://github.com/alfredgu001324/MapUncertaintyPrediction/blob/main/HiVT_modified/models/hivt.py)。
3. clean training 小样本过拟合与正常规模训练检查；三次宿主均须有限更新、损失相对 epoch-0 改善、预测参数有限。地图和邻居 mask 消融应能改变至少一个固定检查样本的输出；报告差异量。
4. 报告 clean-dev mean-state 的绝对 ES/ADE/FDE、训练／开发曲线、模式概率与 scale，并与同车辆目标的旧 MLP 比较。旧 AutoBots 的 ADE 门槛不迁移为新 HiVT 的判据；本次“代表性”指架构及合理收敛，不预设必须胜过 MLP。
5. 冻结 checkpoint 后，对 G/HTC/MMG 的实际当前 XY 抽样检查输入、关键 encoder 激活、模式 logits、均值和 scales 的有限性。除固定开发检查外，全量正式推理也持续检查并保留失败行。

实现缺陷可在没有读取新律比较结果前修正并记录版本；支持域失败不能通过裁剪／截断当前状态、重抽样或删行改造成成功。任一正式比较出现非有限输入或输出时，该受影响比较标记无效并保存定位证据，不当作负收益或零效应。有效的零／负收益则完整报告。

## 5. 实验 A：固定宿主的四律传播（Level I，最低版）

### 5.1 宿主训练与校准

冻结 detector 与概率头。只用原 clean training 的车辆感知轨迹训练 HiVT；原 clean development 可用于宿主设置与 checkpoint 选择，故其结果是开发描述。label-calibration 只用于冻结后输出校准；两个退化条件和 official_clean 不用于新宿主选择。

三次独立宿主训练与三组 head_seed 预先一一配对。同一重复内四种律共享完全相同 checkpoint。建议新增 host seeds 为 `202610130/202610131/202610132`，state-sampling seeds 为 `202610140/202610141/202610142`，future-sampling seeds 为 `202610150/202610151/202610152`；登记后固定。三对重复用于稳定性描述，不视为 host × head 的全因子估计。

在正式训练前，使用源码默认训练配方作为起点，经开发阶段确定 epoch 数、batch、优化器、checkpoint 规则及 AMP；冻结后四组共同执行，不按概率律结果补 epochs 或换 seed。当前方案不把这些待实测项包装为已冻结配置。

如实施 Level II，为 A 的宿主从一开始保留固定宽度 token adapter，概率坐标置常数，使用下节规定的 XY token 投影；这三次模型即 B0。只做 Level I 时可省略 adapter，之后增加 B 必须另记训练预算，不能声称自动复用了同容量 B0。

输出校准只在 clean label-cal 拟合：每个宿主使用一个正的 Laplace scale 乘子，按预先冻结的混合分布 NLL 拟合并记录参数；四种 A 接口共享该乘子。不得再叠加旧 MLP 的未来 Gaussian 残差核。

### 5.2 比较组与分布定义

| 组 | 当前目标 XY | 未来分布 |
| --- | --- | --- |
| A0 | 共同感知均值 | HiVT 原生多模态 Laplace mixture |
| AG | 冻结 G 的 XY 边缘采样 | 对状态与宿主原生未来概率同时积分 |
| AH | 冻结 HTC 的 XY 边缘采样 | 同上 |
| AM | HTC-MMG 的 XY 边缘采样 | 同上；匹配 HTC 的均值／协方差 |

固定历史为 H，固定物理地图和邻居背景为 Z，当前位置为 x：

\[
p_L(Y\mid H,Z)=\int q_\theta(Y\mid H,x,Z)\,p_L^{XY}(x)\,\mathrm dx.
\]

复用[现有匹配律构造](../tools/riskprop3d_level1_matched_law.py)：HTC 线性协方差为 `df/(df-2) × diag(scale²)`，MMG 的 XY scale 为 `scale_XY × sqrt(df/(df-2))`。每行验证 df>2、有限协方差；Student-t 的 XY 两坐标共享该状态样本的径向随机因子，不分别独立抽两个单变量 t。完整旧构造的 yaw 第一矩核对可保留为参数审计，但不是本 HiVT 的传播干预。

对每个状态样本，根据 `softmax(pi)` 抽一个模式，再按其 Laplace 均值／scale 抽整条六步轨迹；同一模式贯穿整条轨迹。未来轴／时刻残差按 decoder 所定义的条件分布抽样，抽样后旋转整个向量。推理不以 GT 选 best mode。[decoder 分布依据](https://github.com/alfredgu001324/MapUncertaintyPrediction/blob/main/HiVT_modified/models/decoder.py)。

### 5.3 抽样与成本口径

复用旧 R0 的 ES 估计量，但直接输入新宿主轨迹样本，不复用其 Gaussian 核：

\[
\widehat{ES}=\frac1{64}\sum_m\|Y_m^{a}-y\|_2
-\frac1{2\cdot64}\sum_m\|Y_m^{a}-Y_m^{b}\|_2.
\]

a、b 是彼此独立的两套状态／未来随机流，各套 64 条轨迹；组间在同一套内共享兼容的状态基础正态、模式 uniform 与 Laplace 基础随机数。**64 是每套数量，ES 总共需要两套；AG/AH/AM 每条目标涉及 128 次条件宿主预测。** A0 只需一次均值条件前向并从其分布抽两套样本。随机流按重复、条件、row_id、draw-set 定义，batch/chunk 调整不能改变抽样。

四个预定条件合计 92,522 条车辆目标；三重复、三种随机律、两套各 64 次，合计约 1.066 亿个条件目标输入，需先实测批处理成本。这个数是目标输入数，不能写成独立场景数或 Python forward 调用数。除状态无关的物理地图几何，可缓存部分必须经完整重算等价核对；状态相关 attention／相对几何不能缓存成常数。

开发阶段在固定 scene/token 哈希子集比较 64 与 256 的 Monte Carlo 精度，记录差异和成本，不按显著性选择样本数。默认正式评估全量车辆目标；若成本要求子集，应在正式结果前冻结哈希规则、各场景覆盖与目标表，所有宿主／方法共用，并明确估计目标已改变。

## 6. 实验 B：匹配的概率输入（Level II，推荐版）

若登记 B，登记应早于 A 的新律结果读出。B 的执行依据是宿主与数据／数值资格，不要求 A 获得正收益；这样零传播收益时仍能回答直接学习输入的问题。实现工作与正式预算在运行前一并冻结。

在 HiVT 编码表示与 decoder 之间使用一个小 token adapter；四组维度、结构、参数量相同。保留既有 R3-P 的 20 槽布局与变换，但**投影到本宿主 XY 范围**：公共均值仅填 XY，其余公共状态槽置常数；G/HTC 仅填 XY log scales，HTC 另填原 df 槽；z、尺寸、速度、yaw 的均值及概率槽均置常数。概率参数保持在原 LiDAR 坐标并明确记载。这个投影是新接口契约，不能写成旧“完整 3D HTC token”的复现。

| 组 | 公共输入之外的 token | 对照意义 |
| --- | --- | --- |
| B0 | 概率槽为常数 | 同容量 mean-only；预先保留 adapter 时同时承担 A 宿主 |
| BC | 既有 score 与 density 变换 | 复用原 confidence 控制；明确包含 density，而非纯 detector-score 控制 |
| BG | G 的 XY log scales | Gaussian 概率坐标 |
| BH | HTC 的 XY log scales 与 `log(df-2)` | HTC 边缘律信息 |

槽号与数值变换以[现有 `make_tokens`](../tools/riskprop3d_r3p.py)和[现有 raw token](../tools/riskprop3d_r2d.py)为依据，写出投影映射并做核对；没有统计语义的槽保持同一常数。A 的采样还须同步写回 adapter 的公共 XY 与所有读取当前 XY 的几何，避免重复均值通道旁路采样。

同重复四组从相同 backbone/adapter 参数开始，使用相同 batches、训练数据、GT 监督、训练预算与 checkpoint 规则。每组归一化只在 clean training 拟合；输出 scale 校准采用同一程序，只使用 label-cal。若 warm start，B0 也获得与其他组相同的继续训练预算。

推荐版合计 **4 × 3 = 12 次正式训练**，其中三次 B0 承担 A，另九次为 BC/BG/BH；先行工程试跑另计。B 不再积分当前状态采样，直接从一次 mean-state + token 条件下的原生预测律评分，避免混合 Level I 与 II。

主对比是 `ES(BC)-ES(BH)`；`ES(B0)-ES(BH)` 和 `ES(BG)-ES(BH)` 是预定次要对比。独立报告 A/B 的推断；若声称多个对比共同成立，在读出前写清多重性处理。

## 7. 预定人群、指标与统计

| 条件 | 作用 | 调整边界 |
| --- | --- | --- |
| development / clean_s0 | 宿主开发、同条件描述 | 宿主选择只限此开发范围；其结果不称独立确认 |
| development / dropout_s3 | LiDAR 强退化扩展，75% 点 dropout | 不用其新律收益选择 checkpoint／token |
| held_out_degradation / rotation_s3 | 相机外参退化扩展，5° rotation | 同上；“held_out”描述旧条件角色，不声称本轮此前全未观察 |
| final_validation / official_clean | **固定主要评估人群** | 新宿主配置与结果表先冻结，再计算；旧宿主结果已被观察 |

主要终点为六步 XY 拼接成 12 维向量的轨迹 ES，使用米，不另除以步数或改为逐步平均 ES。A 主对比 `AG-AH`，关键解释性对比 `AM-AH`；B 主对比 `BC-BH`。正值表示 HTC 更好。其他三个条件完整报告，不根据区间是否显著替换主要人群。

所有同条件比较使用相同 row_id、目标顺序与标签。不同条件的人群记录保留比例；对前三个复用 development 场景的条件另做共同目标敏感性。official_clean 的独立场景人群单独分析，不与前三个条件求 token 交集。

逐目标配对差异先在场景内平均，再在三对运行间平均。使用 10,000 次 paired-scene bootstrap、seed `20260805`；同一场景的所有方法与重复一起重采样。分别保留每次运行的绝对值和方向；场景条件化 CI 与三次训练稳定性分开解释，不将帧／目标或 GPU 当独立重复。

除 ES 外，预定报告：

- 绝对 ES、配对差值、场景估计量的相对降幅、95% CI、每次运行方向与目标／场景数。
- **预测混合均值** ADE/FDE：原生模式均值按概率加权，再对当前状态抽样求平均；不使用 Laplace 噪声样本均值代替可计算的条件均值。minADE/minFDE 仅描述。
- clean 校准后的自然未来 coverage／面积。采用包含模式间、状态间和 Laplace 条件方差的全协方差；二维 Laplace 条件方差为 `2b²`，旋转时变换整个矩阵。矩椭圆的 coverage 是实测诊断，不称多模态分布的真实 HDR 或概率保证；同覆盖面积事后读出另列。
- shared-label AUPRC、risk-coverage、false-safe rate。参考事件使用同重复 **B0/A0 mean-state HiVT 的 mean FDE>2 m**，所有候选共用；风险排序沿用最终 log ellipse area，其他阈值／比例继承旧评价契约并冻结。报告 prevalence 与基线绝对值。
- 实际硬件、batch/chunk、峰值显存、单次宿主前向及两套 64 样本总成本；说明是否含上游 detector／head 和缓存预处理。

旧 CV/MLP 在固定车辆子集及必要的共同目标上重算／重聚合，沿用其原始预测分布，不把 HiVT 的 Laplace 核强加给旧宿主。绝对性能与组内律差异分开解释；不能将 HiVT 车辆均值直接与旧十类均值比较。

## 8. 执行顺序、冻结与资源

| 阶段 | 工作 | 进入下一步所需产物 |
| --- | --- | --- |
| H0 输入恢复 | 恢复三组头、核验哈希、生成车辆 row manifest、补齐过去感知邻居与地图 | 真实输入清单与缺失／失败记录；模型特征与未来标签分离 |
| H1 适配与工程检查 | 锁定源码环境，适配五槽／六步／target index，核对坐标和 MC 输入 | 自检、模型读取坐标表、重复推理无状态残留、成本试测 |
| H2 开发与执行配置冻结 | clean 数据先行训练检查；确定公共训练／校准规则和是否实施 B | 训练 seeds、配置、checkpoint 规则、token 映射、样本数、人群、主对比全部锁定 |
| H3 正式训练 | 最低版三次，推荐版十二次；记录每 epoch 与失败尝试 | checkpoint/config/environment hashes、train/dev 记录、输出校准参数 |
| H4 支持与传播／输入评估 | 同条件完整执行 A，若登记 B 则完整执行 B；official 新结果前再冻结模型和报告表 | 所有组的逐目标指标与数值失败日志，不按方向筛选 |
| H5 统计与交付 | paired-scene CI、重复方向、共同目标与 MC 敏感性、旧宿主同子集比较 | 主表、补充指标、审计清单和符合实际结果的结论 |

GPU 执行遵循本机约定：**实际启动前先查当前使用，优先 GPU 0–3；仅容量需要或首选不可用时使用 4–7；明确 `CUDA_VISIBLE_DEVICES` 并在长任务前报告分配。** GPU 权限受限时走环境提升流程，不自动转 CPU。本次仅文档与只读输入盘点，没有启动 GPU 工作，也没有提前占卡。

不预估未经测量的训练小时数。先测单卡训练 batch、条件样本批处理、地图／邻居缓存大小与磁盘空间；三重复可以在空闲首选卡上并行，每个重复显式固定物理卡。新增缓存按真实必要字段构建，不复制既有 AutoBots 的大型地图缓存。当前仓库没有可核实的 HiVT 新实验入口；本方案不提供伪装成现成脚本的训练命令。

正式执行前清单至少包括：参考 commit／补丁、环境、输入及 split hashes、头 hashes、host/head/MC seeds、读取坐标、上下文来源、所有 token 槽、训练与选择规则、输出校准、row_id 子集规则、数值失败策略、抽样／ES 实现、bootstrap 单位、主要终点与完整表 schema。当前文档是这个冻结动作的输入，不能替代完整执行配置。

## 9. 交付与结论规则

主文表在四个人群／条件中列出 `AG-AH`、`AM-AH`；若执行 B，再列 `BC-BH`、`B0-BH`、`BG-BH`。每项含 ES 差值、CI、逐运行方向和目标数。补充材料给绝对 ES/ADE/FDE、coverage／面积、risk、训练与支持域诊断及成本；同子集 CV/MLP 结果并列。仅做 A 时 B 的栏目标为未执行，不填写推测结果。

| 观察结果 | 允许的解释 |
| --- | --- |
| AG-AH 与 AM-AH 均稳定为正 | 原生 XY 边缘律的传播及同矩高阶差异在另一类宿主中保留 |
| AG-AH 为正，AM-AH 未明确区分 | 相对独立训练 G 有收益，不能将其全部归为高阶律形状 |
| A 保留收益，BH 未稳定胜过 BC/BG | 传播收益不自动建立直接概率输入的必要性 |
| BH 稳定胜过预定控制 | 承认本宿主上的学习输入增量；修订原结论的宿主范围 |
| 有效的零／负效应 | 传播或输入价值存在宿主依赖；不删除不利模型 |
| CI 宽或运行方向不同 | 结果未明确区分；不能据此证明等效或无作用 |
| 正式比较出现非有限输出 | 该比较无效，报告支持域／数值边界；不当成科学零效应 |

验收以数据契约正确、宿主合理收敛、正式比较完整且可复核为准，不要求达到人为指定收益百分比。即使推荐版完成，范围仍是 nuScenes、一个 detector、正确关联的车辆目标、固定地图／背景条件与当前 XY 边缘接口；不外推全部类别、六个退化族、完整 box/yaw 机制、部署身份错误、规划安全或普遍宿主收益。

## 10. 本次方案整理的检查边界

已核查上述 NPZ 的字段／shape、身份／类别数量和历史长度，HDF5 指标字段，现有概率／匹配律／邻居构造源码，MTR 与 AutoBots 的终态文件，以及公开 HiVT dataset/model/decoder 的适配风险。原附件中的 `/Users/jayce/...` 路径已替换为服务器实际材料或明确待恢复项。

尚未恢复权重、构建新感知邻居历史、安装／锁定 HiVT 环境、训练或评估新宿主；现有数据盘点不等于全量数值与关联审计。最先需要解决的是冻结头缺失，随后是邻居历史和五槽适配。所有新效应仍待执行，不能把本方案作为已获得强宿主证据引用。
