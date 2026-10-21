# 概率3D感知何时有助于运动预测？多模态传感器退化下的受控接口研究

**作者一**，**作者二**，**作者三**

*作者单位待补充*

*通讯作者待补充*

> 中文对应初稿 v0.6，目标期刊为 IEEE Transactions on Intelligent Transportation Systems。科学框架与文中实验依据均已固定；作者信息、最终图件、代码仓库 DOI 与期刊格式尚待补充。

## 摘要

标准运动预测评价通常依赖经过整理或由真值构建的目标历史，而部署中的预测器实际接收由不完美感知系统产生的状态。因此，两个问题仍未得到清楚回答：上游预测律之间的差异能否在下游传播中保留，以及监督式预测器是否必须显式接收这些预测律。本文研究目标级接口的三个层面：**随机传播价值**通过固定预测探针传播完整预测律检验；**表示效用**通过容量、置信度和分布形式相匹配的输入检验；**目标必要性**通过蒸馏感知分布诱导的未来分布检验。实验采用固定的 Camera–LiDAR 检测器、共享检测均值的两个经训练3D预测律、两个预测探针、六类退化和配对场景级推断。随机预测律的对比方向在开发集、留出退化和最终验证集上保持一致。最终验证中，两个探针的轨迹能量分数增益为 0.02060–0.02101，等覆盖面积比约为 0.854。相比之下，学习式不确定性表示的稳定性更弱：匹配编码器和直接概率 token 的收益随随机种子、比较对象和评价总体变化，且未形成组件层机制证据。随机未来分布虽可被压缩，但保持该分布并不优于结构相同的纯真值训练。这揭示了观测到的接口稳定性不对称：预测律差异可以穿过随机变换，却未必成为普遍有效的学习表示或训练目标。在本受控设定中，随机传播可作为下游概率价值的一致性诊断，而学习式不确定性接口必须另行证明其增量效用。

**关键词：** 自动驾驶；运动预测；概率3D感知；传感器退化；不确定性传播；校准；可靠性

## 1. 引言

运动预测连接环境感知与下游规划。预测器需要根据摄像头、LiDAR 等传感器重建的目标状态，推断车辆、行人和骑行者的多种可能未来。现代预测架构已经能够有效建模目标历史、地图、交互和多模态未来 [7]–[10]。但在实际部署中，预测器的目标状态输入是估计量，而非直接观测；其质量会随光照、点云密度、标定、时间同步、遮挡和硬件状态而变化。Camera–LiDAR 融合提高了常规条件下的3D感知质量 [2]，但公共退化研究表明，融合与单模态检测器仍会受到环境和传感器扰动的显著影响 [3], [4]。因此，只使用真值轨迹评价预测器，可能掩盖真实感知—预测接口上的不确定性。

概率3D检测器为此提供了自然接口：将当前目标状态表示为预测分布，而不是单一框 [5], [6]。这一分布至少可以通过三种不同方式影响运动预测。第一，对其采样并通过固定动力学传播，可能诱导不同的未来分布；第二，其参数或统计摘要可能成为监督式预测器的有效输入特征；第三，重复传播产生的未来分布本身可能成为有用的训练目标。这三个命题常被统一称为“面向不确定性的运动预测”，但它们在逻辑上并不等价。一个预测律可能具有可测的随机传播价值，而学习 token 无法稳定利用它；学生模型也可能高保真复现随机教师，却不意味着在可获得未来真值时，该教师就是最优训练目标。因此，本文第一个问题不是某一预测律是否创造了上游不存在的新信息，而是上游已有的概率质量差异能否穿过固定预测计算、在传播后仍保持 sharpness，并迁移到不同传感器扰动。

这一差别对模块化系统和联合式自动驾驶系统都很重要。联合感知—预测方法通过特征、轨迹或 query 连接多个任务 [13]–[17]，能够减少人工接口带来的信息损失。然而，更丰富接口的收益可能来自额外容量、普通置信度、不同监督，或概率分布本身。反过来，显式不确定性 token 的零结果也不能否定随机推前分布的正结果。与本文最接近的目标状态研究考察了如何把感知不确定性纳入轨迹预测，并报告了不确定性感知学习目标或特征编码的收益 [23], [26]。本文的问题不同：随机传播价值、显式表示效用和保持目标必要性是否在经验上等价。为此，本文设置匹配的均值、置信度、高斯、容量与同架构纯 GT 对照，以 Camera–LiDAR 3D 感知输出和六类传感器退化为上游，并同时保留正结果与零结果以界定适用边界。相关研究还处理了检测结果中的不确定关联 [24], [25] 和在线地图不确定性 [18]。

本文通过一个**以正确目标身份为条件的目标级感知—预测接口**研究这一空缺。该身份条件是主动设定的：真值实例标识仅用于关联观测和构建未来目标，从而将状态不确定性与身份切换、轨迹出生/消失、虚假轨迹和漏检轨迹连续性分离。预测 host 包括常速度模型和轻量残差多层感知机（MLP）。它们不是当前最强运动预测基线，而是两个结构不同的固定预测探针，用于在不引入地图和多智能体交互混杂的条件下，隔离相同概率感知输入的下游后果。

全文围绕一个统一科学问题展开：

> 当上游3D感知输出一个预测分布时，其中的概率信息究竟在何处影响下游运动预测？

本文从三个层面回答。**随机传播层**将两个经训练的高斯和重尾/圆周预测律通过固定探针传播；**表示层**让监督式预测器接收结构化分布编码或直接概率 token，并控制模型容量、检测置信度和高斯不确定性；**学习目标层**比较单次前向学生是保持 Monte Carlo 未来分布，还是只使用未来真值训练。所有学习式预测器仅使用干净训练数据；低照度、LiDAR 点丢弃、LiDAR 线束减少、相机外参旋转/平移和时间偏移仅用于评价。

本文贡献如下：

1. **受控评价框架。** 本文将随机计算、学习表示和学习目标明确分离，并对检测均值、预测架构、初始化、数据与监督进行相应匹配，在 Camera–LiDAR 退化下同时评价平均概率预测质量和未来误差排序。
2. **稳健的随机传播发现。** 两个经训练上游预测律之间的差异能够穿过两个结构不同的预测探针，并在六类退化以及开发与最终验证总体中保留；这一收益在等覆盖条件下仍然成立，而非依靠扩大预测区域获得。
3. **学习式接口的适用边界。** 稳定的随机价值不等于学习式接口具有普遍效用：结构化表示对随机种子与匹配对照敏感，直接概率条件输入依赖评价总体，而在存在直接未来监督时，保持 Monte Carlo 推前分布也不是必要目标。

因此，本文的结论既不是“不确定性总是有用”，也不是“不确定性可以被忽略”，而是区分了显式随机传播的稳定性质与学习式概率接口更具条件性的效用。

## 2. 相关工作

### 2.1 概率3D感知

多传感器3D感知结合摄像头的语义信息与 LiDAR 的几何信息。BEVFusion 在统一鸟瞰表示中对齐两种模态，是本文使用的检测骨干 [2]。常规精度不足以描述部署可靠性：Dong 等人针对多类3D检测器建立了摄像头和 LiDAR 公共退化基准 [3]，Robo3D 则研究了环境退化与传感器故障下的 LiDAR 感知 [4]。本文固定检测器，并继续追踪这些扰动对下游预测的影响。

概率检测器显式表示定位不确定性。LaserNet 预测多模态3D框分布 [5]；Feng 等人进一步指出，概率检测的校准性必须经过评价，而不能被默认成立 [6]。本文使用两个预先训练的概率侧头作为上游输入：高斯/包裹高斯分布和 Student-t/von Mises 分布。它们不是本文提出的算法。本文关心的是：在控制当前状态均值后，这些预测分布是否会影响未来轨迹。

### 2.2 不完美感知下的运动预测

现代运动预测方法广泛采用结构化场景表示和多模态解码器。VectorNet 编码地图与目标折线 [7]，LaneGCN 显式建模车道拓扑及目标—地图交互 [8]，Motion Transformer 和 MotionLM 则以 query 或自回归方式建模多模态、联合未来 [9], [10]。但多数基准仍提供经过整理的轨迹，或默认上游跟踪已经生成完整目标历史。

已有若干研究更接近真实感知接口。Ivanovic 等人通过统计距离目标将感知状态不确定性显式纳入轨迹预测，并获得更好的校准结果 [23]；Cao 等人则在交互式车辆轨迹预测器中编码感知信息不确定性 [26]。这些工作的核心问题是如何利用上游不确定性改善预测器。本文则检验通常由此类正结果推导出的三项主张是否等价：随机传播是否有价值、显式表示是否具有增量效用，以及保持诱导未来分布是否必要。相应地，本文比较完整随机预测律，匹配表示容量与初始化，控制普通置信度和高斯不确定性，并把保持训练与同架构纯 GT 学习器直接比较。上游预测律来自 Camera–LiDAR 3D 感知，并在六类退化下评价；目标是给出适用边界，而非提出一个始终为正的不确定性感知方法。补充表 S1 以研究设计而非优劣排序的方式明确列出这些差异。Weng 等人以亲和关系替代硬关联，提高预测对跟踪错误的鲁棒性 [24]；Zhang 等人则使用不确定性感知的亲和编码直接从检测结果预测轨迹 [25]。本文以正确身份为条件，不主张对跟踪错误具有鲁棒性。

### 2.3 感知—预测接口

联合感知与预测方法试图减少模块化系统中的误差累积和信息损失。PnPNet 联合优化检测、跟踪和轨迹预测 [13]；FIERY 从环视相机视频预测概率未来实例 [14]；BEVerse 在感知与未来预测之间共享时空鸟瞰表示 [15]；ViP3D 将3D目标 query 传递到轨迹预测 [16]；UniAD 通过 query 接口连接感知、预测和规划 [17]。多视角特征也被用于联合检测与预测 [22]，ForeSight 则通过共享流式 query memory 联合完成检测与轨迹预测 [27]。

这些方法证明了丰富可微接口的价值，但其收益本身不能隔离“不确定性特异性”。特征或 query 可能携带外观、上下文、置信度或额外容量。在线地图不确定性提供了一个互补例子：Gu 等人显式表示地图不确定性并改善轨迹预测 [18]。本文针对目标状态接口，分别控制均值、检测置信度、高斯不确定性、重尾/圆周不确定性、重复随机计算和未来监督。

### 2.4 概率传播与评价

采样是将非高斯输入分布通过非线性动力学传播的直接方法，但重复前向计算代价较高。分布蒸馏可将集成模型或预测分布压缩为单个学生；例如 Ensemble Distribution Distillation 不只拟合集成均值，还拟合集成预测的分布 [20]。感知不确定性也已被传递至机会约束运动规划 [19]。本文检验由重复感知采样得到的经验未来分布是否既可压缩，又适合作为训练目标，并以结构相同的纯真值学生作为决定性对照。

近期一项代理驾驶任务还研究了显式告知自动驾驶智能体感知不确定性是否会改变其行为 [28]。该工作关注下游决策策略，而不是目标轨迹预测；它进一步说明，应把“下游是否出现响应”与“某个学习式预测接口是否具有增量效用”区分开来。

分布敏感的评价同样关键。最小样本位移指标可能奖励多样性，却不一定正确评价完整预测分布。Shahroudi 等人提出使用轨迹能量分数 [11]，其基础来自 proper scoring rule 理论 [12]；概率多智能体动力学研究也采用能量分数评价预测分布 [21]。因此，本文以轨迹能量分数为主指标，并将负对数似然、均值位移、等覆盖 sharpness 和未来误差排序分开报告。

## 3. 问题定义与接口分类

### 3.1 目标级感知—预测不确定性

令 \(\mathcal{S}_{\le t}\) 表示时刻 \(t\) 可用的 Camera–LiDAR 观测和已匹配目标历史。概率检测器定义当前状态分布

\[
p(X_t\mid\mathcal{S}_{\le t}),
\]

其中 \(X_t=(S_t,\Theta_t)\)，\(S_t\) 表示线性3D框与运动坐标，\(\Theta_t\) 表示 yaw。运动预测的目标是得到六步未来中心轨迹分布：

\[
p(Y_{t+1:t+H}\mid\mathcal{S}_{\le t}),\qquad H=6.
\]

本文研究 \(p(X_t\mid\mathcal{S}_{\le t})\) 中的信息如何进入未来分布。目标身份始终作为条件，而非预测对象。真值实例标识仅用于关联跨帧检测和获得未来目标。当前真值残差、未来状态、退化类别和严重度均不进入模型。因此，本文估计的是**正确身份条件下的状态不确定性价值**，而不是端到端跟踪性能。

### 3.2 不确定性使用的三个层面

本文依据概率信息进入预测计算的位置组织实验（图1）。

#### 层面 I：随机传播价值

从完整当前状态分布采样，并通过固定预测探针 \(f\) 传播：

\[
X_t^{(m)}\sim p(X_t\mid\mathcal{S}_{\le t}),\qquad
Y^{(m)}=f(H_t,X_t^{(m)},C_t)+\varepsilon^{(m)},
\quad m=1,\ldots,64.
\]

其中，\(H_t\) 为观测历史，\(C_t\) 为目标类别，\(\varepsilon^{(m)}\) 为共享的校准未来噪声。如果在固定探针和未来噪声构造时，更换经训练的当前状态预测律会改变下游 proper score，则该预测律在这一探针下具有可测的下游分布价值。这是一个基于 proper score 的操作性结论，而非信息论主张，也不要求学习式不确定性表示。

#### 层面 II：表示效用

监督式预测器接收固定维度的分布表示 \(r[p(X_t)]\)：

\[
q_\theta(Y\mid H_t,\mu_t,r[p(X_t)],C_t).
\]

本文同时检验结构化分布编码器和最小化的直接 token，并通过均值状态、模型容量、检测 score/query density 和高斯不确定性对照，判断增益能否被这些因素解释。该层面评价未来监督下的增量效用，不能由层面 I 自动推出。

#### 层面 III：目标必要性

层面 I 产生的经验未来分布也可用于监督学生：

\[
\widehat p_{\mathrm{MC}}(Y\mid X_t)
\longrightarrow q_\theta(Y\mid H_t,p(X_t),C_t).
\]

核心比较是两个完全相同的学生：一个在未来真值之外还保持 Monte Carlo 推前分布，另一个只使用未来真值训练。由此可以区分随机计算的**可压缩性**与将其作为学习目标加以保持的**必要性**。

**表1 感知—预测接口的统一分类。**

| 科学层面 | 待检验接口 | 匹配对照 | 推断问题 |
|---|---|---|---|
| 随机传播价值 | 64样本随机传播 | 均值状态当前接口；高斯随机传播 | 在固定探针下，更换经训练的上游预测律能否改善下游预测分布？ |
| 表示效用 | 结构化编码器与直接概率 token | 容量匹配均值；score/query density；高斯 token；组成消融 | 在 GT 监督下，预测器能否增量利用显式概率信息？ |
| 目标必要性 | 推前分布保持学生 | MC 教师；完全相同的纯 GT 学生 | 保持诱导的未来分布是否必要，而不只是可行？ |

![图1 感知到预测接口的受控分类框架。](figures/figure_1_interface_taxonomy.png)

**图1 | 感知到预测接口的受控分类框架。** 同一个冻结 Camera–LiDAR 检测器向两个此前训练完成的预测律提供 stop-gradient 状态均值。在正确身份条件下，概率信息通过三个不同层面进入运动预测：（I）64样本随机传播经过固定 CV/MLP 探针；（II）在未来 GT 监督下使用结构化或直接概率表示；（III）保持 Monte Carlo 未来预测律，并与结构相同的纯 GT 学习器比较。本文主动排除目标身份关联不确定性。该图定义实验分类框架，不表示重尾/圆周机制已得到因果支持。

## 4. 实验协议

### 4.1 数据集、划分与轨迹构建

实验使用 nuScenes 元数据 [1]，干净训练、标签校准、开发和最终验证总体在场景层面互不重叠。每个合格目标包含连续的2–4个匹配历史帧（1–2 s）、当前匹配检测，以及3 s内的6个未来真值中心。全部状态和未来位置均转换到当前 LiDAR 坐标系。若未来标注缺失，或连续匹配历史少于两帧，则机械排除该目标。

真值实例身份仅用于关联与目标构建。由此，身份切换、轨迹出生/消失、虚假轨迹连续性和关联歧义均不属于当前比较。当前真值残差与退化元数据仅用于评价。所有学习式预测器只在干净场景上训练；退化场景从不用于参数学习。

### 4.2 上游感知接口

上游检测器是固定的 Camera–LiDAR BEVFusion/TransFusion 模型 [2]。概率头不会改变检测器解码框、类别顺序、score 和所选 query。在概率头训练全过程中，检测器均保持冻结：解码均值是 stop-gradient 检测器输出，两个概率头都不能更新检测器的均值分支。对每个保留的 query，两个概率头接收相同的128维解码 query 特征、10维分离梯度的类别 logits、8维分离梯度的线性状态均值，以及1维检测 score；query density 不进入概率头。两个头采用相同的 18,443 参数架构，仅似然解释和训练目标不同。

**高斯感知分布**（实现标签 **G**）以对角高斯建模8个线性坐标，并以包裹高斯建模 yaw：

\[
p_{\mathrm{G}}(X_t)
=\mathcal N(S_t;\mu_t,\Sigma_t^{\mathrm G})
\,p_{\mathrm{WG}}(\Theta_t;\mu_t^\theta,\sigma_{t,\theta}^2).
\]

**重尾/圆周感知分布**（实现标签 **HTC**）采用对角多元 Student-t 线性分布和 von Mises yaw 分布：

\[
p_{\mathrm{HTC}}(X_t)
=t_{\nu_t}(S_t;\mu_t,\Sigma_t^{\mathrm{HTC}})
\,\mathrm{VM}(\Theta_t;\mu_t^\theta,\kappa_t).
\]

关键在于，G 与 HTC 共享检测器解码均值 \((\mu_t,\mu_t^\theta)\)，但其学习得到的 scale 和预测律形式不同。两个头均在相同的560个干净场景、22,508个样本上使用各自似然训练20个 epoch，采用相同优化配置和三个 head seed。互斥的标签校准划分含70个场景、2,814个样本，仅提供后文所述的下游校准量，不作用于表2的上游指标。在读取任何下游运动预测结果之前，两个预测律、三个 head seed 与全部上游 checkpoint 均已固定；没有使用任何下游指标选择预测律、checkpoint、epoch 或 seed。两个概率头在本文研究之前已完成开发，属于上游输入而非本文提出的方法。由于它们使用不同的似然目标并学到不同的 scale，G 与 HTC 的比较是两个完整的经训练预测律之间的比较，而不是只干预分布族的因果实验。因此，HTC 的优势不能被专门归因于 \(\nu_t\)、\(\kappa_t\)、Student-t 尾部或 von Mises 几何。

表2在每个 head seed 的干净开发评价中，使用完全相同的匹配固定 query 行，在进入运动预测前刻画两个预测律。NLL、自然覆盖和原始 scale 均直接使用概率头未校准输出，不施加标签校准温度或事后 scale 修正；等覆盖体积是唯一的事后描述量，通过在同一批行上把每个预测律重标定到经验90%覆盖得到。HTC 的完整几何、平面位置和 yaw NLL 更低，线性覆盖也更接近90%；但 G 的 yaw 覆盖略接近标称值，且 HTC 虽有更小的原始线性 scale 中位数，等覆盖静态体积却更大。这种混合的校准—sharpness 特征排除了“HTC 在所有方面都校准更好”或“HTC 只是输出更宽区域”这两种简单解释。

随机传播比较并不旨在证明 HTC 创造了上游不存在的新下游信息。它检验的是：两个经训练预测律之间已经存在的差异，能否穿过固定预测算子、在传播后保持 sharpness，并跨扰动与评价总体泛化。上游优势并不保证下游优势，因为非线性动力学、加性过程不确定性、时间聚合和分布敏感型评价都可能削弱或重排预测律质量。本文真正检验的是，这种上游差异能否一致地穿过下游变换。

**表2 两个经训练上游预测律在干净开发集相同匹配 query 行上的固定表征。NLL、自然覆盖和原始 scale 使用未校准概率头输出；只有等覆盖体积采用事后经验重标定。NLL 对比为 G-minus-HTC，正值支持 HTC；覆盖差为经验覆盖减去标称覆盖；比值范围覆盖三个 head seed。**

| 预测前指标 | 固定表征 |
|---|---:|
| 完整几何 NLL 对比 | 4.10243（95% CI 1.97022, 7.42856） |
| XY NLL 对比 | 0.12799（0.11494, 0.14169） |
| Yaw NLL 对比 | 2.71034（0.62140, 6.01430） |
| 线性状态90%覆盖差 | G: -0.03827；HTC: +0.01330 |
| Yaw 90%覆盖差 | G: +0.02549；HTC: +0.03107 |
| 线性 scale 中位数比，HTC/G | 0.758–0.791 |
| 等覆盖90%椭球体积比，HTC/G | 1.095–1.249 |

### 4.3 受控预测探针与学习式接口

随机传播层使用两个固定的无地图探针：常速度（CV）和一个在干净轨迹上训练一次的 26,380 参数残差 MLP。CV 提供解析动力学对照，MLP 提供学习式非线性对照。二者均不用于竞争地图感知、多智能体预测模型，而是检验感知预测律效应能否在预测计算发生变化时保持。每个随机结果使用64个传播样本；兼容的 G/HTC 线性采样使用共同随机数，两个圆周分布采用各自原生采样器。

对每个预测探针，干净标签校准残差分别拟合六个零均值二维高斯协方差，即每个未来时刻一个 \(2\times2\) 协方差。均值状态当前接口、G 传播和 HTC 传播使用完全相同的探针特定加性噪声律及共同过程噪声采样；拟合过程不读取任何退化或最终验证结果。因此，在同一探针内，比较只更换当前状态预测律，并固定未来过程噪声构造。均值状态接口加入共享过程噪声后仍输出概率未来，故本文不再将其简称为“确定性未来分布”。

在表示层，所有学习组共享相同的历史/类别输入和六模态对角高斯残差轨迹解码器。结构化编码比较包括完整 HTC 编码器、参数匹配的均值对照、架构相同的高斯化编码器，以及 no-tail/no-circular 变体。直接条件比较使用同一个 24,118 参数网络与四种固定长度输入：仅均值；均值加检测 score/query density；均值加 G 不确定性；均值加 HTC 不确定性。同一重复内，匹配组共享可训练初始化、batch 顺序、epoch、解码器和真值损失。

在学习目标层，教师为 HTC 的64样本通过固定 MLP 得到的推前分布。两个学生具有相同输入、架构、初始化和解码器。纯 GT 学生仅最小化未来混合似然和 best-mode 位移项；推前分布保持学生在此基础上加入样本分布、矩、校准—sharpness 和风险顺序匹配。完整特征与目标定义见补充材料 S2。

### 4.4 传感器退化与留出退化迁移

开发评价包括干净数据和三类退化：相机低照度、LiDAR 点丢弃和 LiDAR 线束减少。每类退化包含预先设定的严重度，并在适用时同时报告完整 family 与 S1/S2 聚合。开发分析结束后，在不改变模型和指标的条件下评价三类新增退化：相机外参旋转、相机外参平移和相机时间偏移。本文将其称为**留出退化迁移**；它表示 nuScenes 内不同扰动 family 之间的迁移，而非外部数据集测试。

### 4.5 指标与统计推断

轨迹能量分数是主指标，因为它以 proper scoring rule 评价联合样本分布 [11], [12]。给定预测样本 \(Y^{(1)},\ldots,Y^{(M)}\) 和真值 \(y\)：

\[
\operatorname{ES}
=\frac{1}{M}\sum_{m=1}^{M}\lVert Y^{(m)}-y\rVert_2
-\frac{1}{2M(M-1)}\sum_{m\ne n}\lVert Y^{(m)}-Y^{(n)}\rVert_2.
\]

数值越低越好。由于经验 MC 分布与学习式混合分布使用的似然构造不同，精确或共享核轨迹负对数似然保留为诊断指标。预测均值 ADE/FDE 评价均值精度。Sharpness 使用每种方法缩放到经验90%最终位置覆盖后，其最终位置椭圆面积中位数衡量；等覆盖面积越小，预测越 sharp。

未来误差排序使用预测最终 log 椭圆面积作为风险分数。High FDE 表示在比较特定参考下，预测均值 FDE 超过 2 m。本文报告 high-FDE AUPRC、risk–coverage 和 false-safe rate；false-safe rate 在场景内最低风险80%的轨迹上计算。它们是预测误差筛查指标，不是碰撞或规划安全指标。

所有决定性比较均包含三次配对重复。首先在场景内平均轨迹差值，再使用10,000次配对场景 bootstrap 计算95%置信区间，并单独报告逐 seed 方向。确认性主张仅来自预先指定的干净集与 family 级主能量对比及其联合判据。置信区间未做多重性校正；组件、风险和属性分层区间属于次要或描述性分析，本文不依据单个次要或分层区间提出 family-wise 结论。

### 4.6 预先规定与最终确认

每次留出评价之前均已明确假设、比较对象、主指标和统计标准。详细 seed、完整性检查、特征契约和决策标准见补充材料。

最终确认使用 6,019 个 nuScenes validation 元数据样本，从150个场景产生 68,436 条合格轨迹。需要严格说明的是：**该划分上的下游运动预测结果和接口比较，直到下游模型、指标、比较对象和主张固定后才被读取**。这并不意味着 nuScenes validation 在任何上游检测器开发中都从未被使用。最终验证结果读取后，不进行下游重训练、再校准、比较对象筛选或指标筛选。内部 test 划分始终未使用。

## 5. 结果

### 5.1 完整分布随机传播表现出稳定的下游价值

首先在不要求学习模型解释概率参数的条件下检验随机传播价值。在干净开发集上，相比传播经训练的高斯预测律，传播经训练的 HTC 预测律使 CV 的轨迹能量分数降低 0.01614（95% CI 0.01102, 0.02150），使 MLP 的能量分数降低 0.01230（0.00850, 0.01619）。HTC 相比均值状态当前接口也显著改善两个探针。所有重复的干净 HTC-over-Gaussian 方向均为正。

这一效应在原始低照度、LiDAR 点丢弃和线束减少评价中保持。六个探针—退化组合的完整 family HTC-over-Gaussian 能量增益范围为 0.01411–0.04902，去除最严重 S3 后的增益范围为 0.01222–0.02963。两个探针在每个 family 上的点估计均为正。由于同一比较中的预测 host、当前状态均值、过程噪声模型和兼容随机数均相同，这些效应隔离了被传播的经训练预测律，而不是不同的均值轨迹或未来噪声构造。

更好的能量分数并非来自无差别扩大预测区域。开发集等覆盖 HTC/Gaussian 最终面积比为 0.866–0.929；HTC 相比均值状态接口的预测均值 ADE/FDE 最大增加仅为 0.01125/0.01692 m。相比高斯传播，HTC 使 CV 和 MLP 的 high-FDE AUPRC 分别提高 0.03745 和 0.03251。因此，完整分布效应同时体现在 proper-score 质量、校准后的 sharpness 和未来误差排序中。

该结果构成全文第一个锚点：在固定目标身份和受控预测探针下，本文所评价的 HTC 预测律相对对应高斯预测律具有可重复的下游分布价值。但它没有建立纯分布族效应，没有说明具体哪个参数造成对比，也没有说明监督式模型必须显式接收任何参数。

![图2 随机传播的开发集证据。](figures/figure_2_stochastic_propagation.png)

**图2 | 随机传播的开发集证据。** **a，** 在 CV 与 MLP 探针下，HTC 相对高斯传播在干净数据及低照度、LiDAR 点丢弃和线束减少完整 family 聚合上的轨迹能量改善；正值支持 HTC。**b，** 干净条件下 HTC 相对均值状态当前接口的能量效应。**c，** 各预测律缩放到经验90%覆盖后的终点椭圆面积比；小于1表示 HTC 更 sharp。**d，** 干净条件下 HTC 相对高斯的 high-FDE AUPRC 效应。点为三个配对 head/传播重复的平均值，误差线为10,000次配对场景 bootstrap 得到的95%区间（开发集 \(n=70\) 个场景）。同一探针内，各当前状态接口共用同一个由标签校准集拟合的未来过程噪声律。源数据见 `figures/source_data/figure_2.csv`。

在固定的评价总体—条件—严重度—head seed 单元上，上游 XY NLL 增益与两个探针的下游轨迹能量增益均呈较强关联；自然覆盖误差与等覆盖 sharpness 的关联更弱。图3对这一传递关系作描述性展示，并直接标注具有高 leverage 的线束减少 S3 簇。这些生态层面的关联不能识别分布族或具体组件机制。

![图3 上游概率差异与下游轨迹能量传递。](figures/figure_3_upstream_downstream_transfer.png)

**图3 | 上游概率差异与下游轨迹能量传递。** **a，** 未校准上游 XY 负对数似然中 HTC 相对高斯的场景加权增益，与下游轨迹能量增益的关系。**b，** 自然90%覆盖绝对误差的减少量与下游能量增益。**c，** 等覆盖 XY log-area 增益与下游能量增益。每个点对应一个固定的评价总体—条件—严重度—head seed 单元；圆形表示 CV，三角形表示 MLP（每个探针48个单元、16个条件簇、每簇3个 seed）。点大小表示严重度，颜色表示退化 family 或最终验证；图中直接标注线束减少 S3 leverage 簇。插图给出 Spearman \(\rho\) 及其95%区间，区间来自10,000次 cluster bootstrap，每次重采样均保留条件簇内全部三个 seed。正值支持 HTC。该关联仅为描述性证据，不识别 tail 或 circular 因果机制。源数据见 `figures/source_data/figure_3.csv`。

### 5.2 学习式表示不会普遍继承随机传播价值

下一步检验显式概率表示在普通未来监督下是否具有增量收益。本文采用两种互补设计：结构化编码器允许模型充分变换分布统计量；直接 token 则尽量减少架构解释空间。

#### 5.2.1 结构化分布编码

初始比较表明，专用分布编码器可以优于简单拼接：在干净开发集上，其能量增益为 0.01965（0.00895, 0.03100），诊断 NLL 增益为 0.19392（0.10068, 0.28678）。这是一个局部正结果，但 raw concatenation 没有控制表示容量。

参数和初始化匹配的比较没有支持更强解释。完整 HTC 编码相比仅均值对照的干净能量增益为 0.00690，但区间跨过零（-0.00150, 0.01605）；相比架构相同的高斯化编码器，增益为 0.00508（-0.00775, 0.01753）。Tail 和 circular 组件检验也没有同时通过预先规定的总体、机制分层、风险、sharpness 与 seed 判据。关键组件和分层结果见补充材料 S4。

因此，最初的编码器收益没有通过机制主张所需的匹配控制。专门编码器可以在某次训练中有效，但现有证据既不支持与容量无关的 HTC 信息利用，也不足以把收益归因于 Student-t tail 或 von Mises concentration。组件和属性分层结果仅保留在补充材料 S4。

#### 5.2.2 直接概率条件输入

直接 token 实验进一步消除了编码器容量解释。在干净开发集上，HTC token 相比仅均值的能量增益仅为 0.00205（-0.01378, 0.01559），相比检测 score/query density 为 0.00069（-0.01531, 0.01321）。相对高斯 token，HTC 还呈现非显著劣势：Gaussian-minus-HTC 为 -0.00140（-0.01454, 0.00943）。原始退化 family 的效应也均不显著，且一个重复反转了 HTC-versus-Gaussian 的干净方向。

不过，直接 token 保留了更窄的信号。在不显著损害均值精度和等覆盖 sharpness 的前提下，HTC 使干净 risk–coverage 相比仅均值提高 0.00699（0.00065, 0.01369），相比置信度对照提高 0.00672（0.00027, 0.01381）。因此，即使平均轨迹能量增益尚不明确，显式概率条件仍可能影响失败排序。第5.5节集中讨论这一指标边界。

结构化和直接表示共同回答了表示层问题：完整随机传播的价值，不会自动转化为概率信息作为监督特征时的稳定增量价值。

### 5.3 保持随机推前分布不是必要学习目标

Monte Carlo 未来分布具有很强的可压缩性。相比结构相同的纯 GT 学生，推前分布保持学生在干净开发集上更接近教师：样本能量距离减少 0.53689（0.50198, 0.57474），一阶矩误差减少 0.10111（0.07879, 0.12572），相对二阶矩误差减少 0.43758（0.41599, 0.45856），90%覆盖差绝对值减少 0.05178（0.04484, 0.05903）。它还比 MC 教师改善下游能量 0.10957（0.09758, 0.12076），保持了教师风险排序和 sharpness，并实现 38.55 倍预测加速。

但是，可压缩性不等于目标必要性。相对推前分布保持学生，纯 GT 学生使干净轨迹能量降低 0.05941（0.02657, 0.08835），三个干净重复均支持纯 GT 训练。相同的不利保持方向也出现在原始退化 family 中。诊断混合 NLL 同样支持纯 GT，增益为 0.60502（0.49776, 0.70776）。

这一对照消除了分布蒸馏研究中的一个常见歧义。教师保真度说明学生可以保持昂贵的随机计算，却不能证明在存在直接未来标签时，该计算是最好的学习目标。本文学生能够高效学习教师矩、覆盖和风险行为，但额外保持目标相对同架构纯真值训练降低了预先指定的下游 proper score。

![图4 学习式接口与学习目标对照。](figures/figure_4_learned_interfaces.png)

**图4 | 学习式接口与学习目标对照。** **a，** 初始 DSE 相对 raw concatenation 的能量结果，与参数和初始化匹配的完整编码器相对仅均值、高斯化对照并列展示；菱形为总体效应，空心圆为三个 seed 方向。初始审计与匹配审计采用不同的配对训练重复，因此不能把二者的效应大小视为直接配对比较。**b，** 在完全相同的纯 GT 训练下，直接 HTC token 相对均值、置信度和高斯对照的能量效应。**c，** 保持学生相对纯 GT 学生的四项教师保真度误差减少量；各行使用各自指标的原始尺度，不能跨行比较数值大小。**d，** 以保持学生为固定 candidate 的轨迹能量优势：相对 MC 教师为正，而相对纯 GT 学生为负。**e，** 64次前向 MC 与单次前向保持学生的三次配对预测时间。总体误差线为10,000次配对场景 bootstrap 的95%区间（开发集 \(n=70\) 个场景）。组件与属性分层结果保留在补充材料。源数据见 `figures/source_data/figure_4.csv`。

### 5.4 随机接口与学习式接口具有不同的泛化特性

留出退化迁移与最终验证揭示了全文最重要的跨总体对比：完整随机传播效应稳定，而学习式接口效应具有异质性。

在相机外参旋转、平移和时间偏移下，HTC 随机传播在两个探针上均显著优于高斯传播，六个探针—退化组合的增益范围为 0.01595–0.02411。对应的 HTC-over-mean-state 对比也全部显著，等覆盖 HTC/Gaussian 面积比为 0.861–0.896。这些扰动不属于原始低照度、点丢弃和线束减少评价，因此观测到的随机传播价值扩展到了六类退化，但不应被解释为新数据集迁移。各 family 的精确效应保留在补充表 S2。

最终验证在一个留出评价总体上确认了随机传播效应。HTC 相比高斯传播的能量增益在 CV 上为 0.02060（0.01632, 0.02532），在 MLP 上为 0.02101（0.01603, 0.02688）；相比均值状态接口的增益分别为 0.12528（0.09446, 0.15992）和 0.12382（0.09352, 0.15787）。最终验证的每个重复均支持 HTC 优于两个探针上的两个对照。HTC/Gaussian 等覆盖面积比分别为 0.8549 和 0.8537。

学习式表示没有表现出同等不变性。在留出退化中，直接 HTC token 相比仅均值在旋转下改善 0.01734（0.00484, 0.03223），在时间偏移下改善 0.01916（0.00369, 0.03599），但在平移下不显著。它在任何留出 family 上都未显著优于置信度或高斯条件，且每个 family 至少有一个重复反转 HTC-versus-Gaussian 方向。

最终验证比开发集更支持学习式表示。完整结构化编码相比均值匹配和高斯化对照显著改善能量，但第三个重复反转了所有匹配比较，no-tail 对比为零，且完整编码器/均值对照的等覆盖面积比为 1.0728。直接 HTC 条件相对均值、置信度和高斯 token 都改善能量，面积比也略有优势；但一个重复仍反转了 HTC-versus-Gaussian。

学习目标层结论更为一致。最终验证中，推前分布保持学生相对 MC 教师改善能量 0.09834（0.08768, 0.10803），但没有优于纯 GT 学生：GT-only-minus-preservation 为 -0.01740（-0.04238, 0.00968），三个点估计方向均支持纯 GT。诊断 NLL 支持纯 GT 0.58845，且保持学生的等覆盖面积是纯 GT 的 2.7151 倍。表3集中列出最终验证对比；完整学习式接口表保留在补充表 S3。

**表3 最终验证结果汇总。数值为 baseline-minus-candidate，正能量效应支持 candidate；面积比为 candidate/baseline 在经验90%覆盖下的比值。**

| 层面 | Candidate | Baseline | 能量效应（95% CI） | 面积比 | Seed 方向 |
|---|---|---|---:|---:|---|
| 随机传播，CV | HTC | 高斯 | 0.02060 (0.01632, 0.02532) | 0.8549 | + / + / + |
| 随机传播，MLP | HTC | 高斯 | 0.02101 (0.01603, 0.02688) | 0.8537 | + / + / + |
| 结构化表示 | 完整 HTC 编码器 | 均值匹配 | 0.03115 (0.01784, 0.04570) | 1.0728 | + / + / - |
| 结构化表示 | 完整 HTC 编码器 | 高斯化编码器 | 0.02827 (0.01518, 0.04142) | 0.9648 | + / + / - |
| 直接条件 | HTC token | 仅均值 | 0.02702 (0.01588, 0.03932) | 0.9874 | + / + / + |
| 直接条件 | HTC token | 置信度 | 0.02092 (0.01112, 0.03217) | 0.9849 | + / + / + |
| 直接条件 | HTC token | 高斯 token | 0.01682 (0.00456, 0.03051) | 0.9723 | + / - / + |
| 学习目标 | 保持学生 | MC 教师 | 0.09834 (0.08768, 0.10803) | 0.9427 | + / + / + |
| 学习目标 | 保持学生 | 纯 GT 学生 | -0.01740 (-0.04238, 0.00968) | 2.7151 | - / - / - |

描述性属性分析与这种分离一致，但不能提供因果解释。HTC-over-Gaussian 随机增益在距离、速度、历史长度、LiDAR 点数代理、当前残差四分位数和类别分层上普遍为正，并常在退化下扩大；直接 token 效应则异质且经常收缩或反转。完整分层结果置于补充材料，任何分层均未被用于选择模型或提出新机制。

![图5 跨评价总体观测到的接口稳定性不对称。](figures/figure_5_interface_stability.png)

**图5 | 跨评价总体观测到的接口稳定性不对称。** **a，** CV 与 MLP 在干净/原始退化（包括 LiDAR 点丢弃）、留出旋转/平移/时间扰动和最终验证上的 HTC-over-Gaussian 随机轨迹能量效应。点为配对重复均值，误差线为95%配对场景 bootstrap 区间。**b，** 结构化编码器和直接 token 表示在实际评价总体上的效应方向与区间。蓝色实心上三角表示正效应且95%区间高于零；蓝色空心上三角表示正点估计但区间跨零；橙色空心下三角表示负点估计但区间跨零；斜线单元表示未评价。蓝橙双色分环表示至少一个 seed 反转总体方向。开发与留出分析使用 \(n=70\) 个场景，最终验证使用 \(n=150\) 个场景，每项对比包含三个配对 seed。不得跨非匹配实验比较学习式表示的效应大小。源数据见 `figures/source_data/figure_5.csv`。

### 5.5 平均概率质量之外仍可能保留风险排序效用

未来误差排序揭示了比平均轨迹能量更窄的效用。在随机传播层，风险结果与主指标一致。最终验证中，HTC-over-Gaussian 的 high-FDE AUPRC 增益在 CV 上为 0.03099（0.02033, 0.04185），在 MLP 上为 0.01988（0.01034, 0.02946）；risk–coverage 分别提高 0.00889（0.00087, 0.01726）和 0.00855（0.00142, 0.01768）；false-safe rate 分别改善 0.01945（0.01191, 0.02702）和 0.01346（0.00596, 0.02116）。这些结果将稳定的随机传播价值与更好的预测失败识别联系起来。

在表示层，风险证据更具选择性。直接 HTC token 在开发集能量为零时，仍相对均值和置信度改善 risk–coverage。最终验证中，相对仅均值和置信度的 risk–coverage 增益分别为 0.02308（0.01334, 0.03384）和 0.01313（0.00325, 0.02338），false-safe rate 也相对二者改善；但没有任何风险指标显著优于高斯 token。留出退化结果同样依赖 family 和比较对象。

推前分布保持学生保留了教师风险行为，但这一成功不能建立目标必要性，因为完全相同的纯 GT 对照在开发集能量和 NLL 上仍然更优。风险保持和平均分布质量是不同终点。

因此，即使平均 proper score 优势不稳定，显式概率表示仍可能服务于选择性使用，例如识别需要额外审查或更保守处理的预测。该解释仅适用于基于预测椭圆面积的 high-FDE 筛查，不能被视为碰撞风险降低或规划更安全的证据。补充图 S1 并列展示 proper-score 与风险排序效应。

### 5.6 证据综合

表4总结了最终适用性边界。完整预测分布作为固定计算的随机输入时最可靠；一旦引入未来监督和学习式接口，增量价值就取决于不确定性的表示方式、训练目标和评价总体。

**表4 主要结论汇总。**

| 科学层面 | 概率输入或目标 | 主能量证据 | 未来误差风险证据 | 支持的解释 |
|---|---|---|---|---|
| 随机传播价值 | 完整当前状态预测律的样本 | 跨探针、六类退化、开发集和最终验证均为正 | 相比高斯和均值状态传播稳定改善 | 在受控探针下，所评价 HTC 预测律相对经训练高斯预测律具有可重复的下游分布价值 |
| 表示：结构化编码器 | 人工设计的分布摘要 | 局部正；在容量、组件和 seed 对照下不稳定 | 超出均值对照的证据混合 | 学习机制受限；不支持重尾/圆周因果归因 |
| 表示：直接 token | GT-only 训练下的显式参数 | 开发集为零、留出异质、最终验证总体为正 | 狭窄且依赖比较对象 | 显式条件输入可以有效，但并非普遍必要 |
| 目标：推前分布保持 | 训练时的经验 MC 未来分布 | 优于 MC 教师；不优于相同的纯 GT 学生 | 保留教师行为 | 随机计算可压缩，但保持目标非必要 |

## 6. 讨论

### 6.1 为什么随机价值不等于学习式必要性

本文在受控研究范围内观测到一种**接口稳定性不对称**。随机传播询问：当前目标状态不确定性通过固定动力学后，会产生怎样的未来分布；学习式条件输入询问：监督模型是否需要显式接收该分布的参数。前一类对比能够跨探针、退化和评价总体保持，后一类对比却随表示、比较对象、随机种子和总体变化。严格而言，前者是预测律与计算共同具有的性质，后者则是在特定表示、假设空间、优化路径和监督条件下的增量效用主张。

第一条主张在本文得到强支持。本文所评价的 HTC 预测律在 CV 与 MLP、六类退化、开发与最终验证总体上均优于对应的经训练高斯预测律，同时保持或改善等覆盖 sharpness 与失败排序。比较内共享的检测均值和过程噪声排除了“改变点预测”或“扩大未来噪声模型”这两种简单解释，但不能把对比变成纯分布族干预，因为两个预测律还具有各自似然训练得到的 scale。

更强的学习式主张并不普遍成立。GT 监督预测器已经观察历史、均值状态和类别，并可从未来标签学习相关关系；检测置信度或高斯 scale 也可能吸收与 HTC token 相关的信息；优化噪声还可能改变模型使用哪个相关输入。这些因素可以解释从随机传播到学习表示的收益收缩，但并非本文已证明的机制。匹配组件实验和描述性分层尤其不支持将该效应归因于自由度或 yaw 浓度。

### 6.2 对感知—预测接口设计的启示

对模块化系统而言，即使64次重复推理不适合部署，随机传播仍是有价值的诊断工具。它首先回答上游预测分布在固定 host 下是否具有可证明的下游影响。正结果可以支持进一步研究接口，但不能预先决定某个编码器，也不能单独证明显式参数是必要的。

对 PnPNet、ViP3D 和 UniAD 等联合系统 [13], [16], [17] 而言，本文结果说明仍需设置匹配的不确定性对照。下游传递的丰富特征或 query 可能同时保留外观、场景上下文、关联证据和置信度。因此，要证明不确定性特异性收益，仍需与均值状态、普通置信度、高斯不确定性、匹配容量和直接未来监督比较。本文受控探针不评价端到端 latent feature 是否整体更优，而是说明何种证据才足以把收益称为“不确定性收益”。

最终验证的直接 token 正结果也说明，评价依赖性不能通过简单平均消除。开发集、留出退化和最终验证对学习式接口给出不同结论，却对随机传播保持一致。场景构成、轨迹合格规则、退化后的特征分布和优化方差都可能参与其中。由于进一步事后建模不符合固定分析原则，本文将结论限定在直接证据支持的层面：学习式概率条件输入可以有效，但其增量依赖评价总体。

基于上述结果，可以得到一条简洁的评价顺序：

1. 先将完整的经训练上游预测律通过固定探针传播，确认随机传播价值；
2. 再用均值、置信度、高斯、容量和初始化匹配对照评价学习表示；
3. 任何教师保持目标都必须与结构相同的纯 GT 学习器比较；
4. 在多个评价总体上同时报告 proper score、等覆盖 sharpness、均值精度、失败排序与逐 seed 方向。

这些是评价原则，并非主张以本文探针取代地图感知或联合预测架构。

### 6.3 指标依赖的效用

轨迹能量、诊断 NLL、等覆盖面积、均值 ADE/FDE 和未来误差排序评价的是不同性质，因此不必一致。能量分数评价联合样本分布；NLL 强烈依赖似然参数化；等覆盖面积评价校准后的 sharpness；ADE/FDE 描述预测均值；风险指标则检验不确定性能否排序未来失败。

直接 token 结果具体展示了这一差别：开发集能量为零，但相对均值和置信度的 risk–coverage 改善；最终验证中，能量与风险均优于二者，但风险没有优于高斯条件。一个接口可能有助于预测筛查，却不普遍改善平均预测分布。若在结果可见后选择有利指标，会抹去这一边界，因此本文始终以能量分数为主指标，并保留所有不利诊断方向。

### 6.4 研究范围与局限

本文主动隔离一个接口，所有结论均以正确目标身份为条件。真值实例标识消除了关联不确定性，因此不覆盖身份切换、虚假或漏检轨迹、轨迹出生/消失和在线轨迹管理。专门处理不确定关联的方法 [24], [25] 研究的是互补问题。

预测 host 是无地图的 CV 和残差 MLP 探针，而非现代地图感知、交互丰富的预测器。二者的结构差异有助于隔离感知分布对比，但限制了架构外推。因此，本文主张的是目标级受控接口的下游价值，而不是最先进运动预测或规划系统中的普遍行为。

所有证据来自 nuScenes [1]、一个固定 BEVFusion/TransFusion 检测器 family [2] 和两个特定的经训练上游预测律。留出退化扩展了扰动类型，但没有扩展数据集或检测器。随机 HTC-over-G 对比没有把分布族与不同似然下学到的 scale 和校准性质分离，因此不能推出 Student-t/von Mises 普遍优于所有高斯或非高斯替代；匹配编码器结果也没有识别 tail 或 circular 参数的因果作用。

未来范围仅为3 s内的6个中心。地图、显式社会交互、场景占据、模态缺失、长期 world prediction 和闭环规划均不在范围内。低照度、点丢弃、线束减少、外参旋转/平移和时间偏移也不能穷尽天气、标定、硬件与域偏移。

最后，high-FDE AUPRC、risk–coverage 和 false-safe rate 评价未来误差识别，而非碰撞规避、乘坐舒适性或规划安全。本文采用分阶段的预先规定分析，后续比较在科学上由早期结果推动。完整治理历史保留在补充材料中，以便读者区分论文的统一科学框架与模型检验的实际时间顺序。

## 7. 结论

本文的受控研究在概率感知—预测耦合中观测到接口稳定性不对称。显式随机传播提供了最稳定的证据：两个经训练上游预测律之间的差异能够穿过两个预测探针、六类退化和两个评价总体，并同时改善轨迹能量、等覆盖 sharpness 与未来误差排序。结构化编码器和直接概率 token 却不能稳定恢复同等价值；学生虽可高效压缩 Monte Carlo 未来分布，但保持该分布也不优于结构相同的纯真值训练。这一预测律对比不代表纯分布族效应，也不支持稳定的重尾/圆周机制。因此，在本文所评价的接口范围内，随机传播可以作为下游概率价值的稳健诊断，而学习式不确定性接口必须在具体监督、指标和评价总体下，以匹配对照另行证明其增量效用。

## 数据可用性

本文使用公开 nuScenes 数据集，并遵守其许可。派生轨迹元数据、退化定义、指标档案和机器可读证据账本均保留完整性元数据。发布包完成后将补充公开归档地址及再分发限制：**[待补充 repository/DOI]**。内部 test 划分未被访问。

## 代码可用性

随机传播、学习式接口比较、留出退化评价、bootstrap 推断和最终确认代码均保存在项目仓库。投稿前将补充公开仓库地址、环境说明和 checkpoint 获取规则：**[待补充 repository URL]**。

## 作者贡献

**[按目标期刊 CRediT taxonomy 补充。]**

## 利益冲突

作者声明：**[无利益冲突/待确认。]**

## 致谢

**[待补充项目资助、计算资源与其他致谢。]**

## 参考文献

[1] H. Caesar, V. Bankiti, A. H. Lang, *et al.*, “nuScenes: A multimodal dataset for autonomous driving,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 11621--11631. [Online](https://arxiv.org/abs/1903.11027)

[2] Z. Liu, H. Tang, A. Amini, *et al.*, “BEVFusion: Multi-task multi-sensor fusion with unified bird's-eye view representation,” in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2023, pp. 2774--2781. [Online](https://doi.org/10.1109/ICRA48891.2023.10160968)

[3] Y. Dong, C. Kang, J. Zhang, *et al.*, “Benchmarking robustness of 3D object detection to common corruptions,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 1022--1032. [Online](https://openaccess.thecvf.com/content/CVPR2023/html/Dong_Benchmarking_Robustness_of_3D_Object_Detection_to_Common_Corruptions_CVPR_2023_paper.html)

[4] L. Kong, Y. Liu, X. Li, *et al.*, “Robo3D: Towards robust and reliable 3D perception against corruptions,” in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2023, pp. 19994--20006. [Online](https://openaccess.thecvf.com/content/ICCV2023/html/Kong_Robo3D_Towards_Robust_and_Reliable_3D_Perception_against_Corruptions_ICCV_2023_paper.html)

[5] G. P. Meyer, A. Laddha, E. Kee, C. Vallespi-Gonzalez, and C. K. Wellington, “LaserNet: An efficient probabilistic 3D object detector for autonomous driving,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 12677--12686. [Online](https://openaccess.thecvf.com/content_CVPR_2019/html/Meyer_LaserNet_An_Efficient_Probabilistic_3D_Object_Detector_for_Autonomous_Driving_CVPR_2019_paper.html)

[6] D. Feng, L. Rosenbaum, C. Glaeser, F. Timm, and K. Dietmayer, “Can we trust you? On calibration of a probabilistic object detector for autonomous driving,” arXiv:1909.12358, 2019. [Online](https://arxiv.org/abs/1909.12358)

[7] J. Gao, C. Sun, H. Zhao, *et al.*, “VectorNet: Encoding HD maps and agent dynamics from vectorized representation,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 11525--11533. [Online](https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_VectorNet_Encoding_HD_Maps_and_Agent_Dynamics_From_Vectorized_Representation_CVPR_2020_paper.html)

[8] M. Liang, B. Yang, R. Hu, *et al.*, “Learning lane graph representations for motion forecasting,” in *Proc. European Conf. Computer Vision (ECCV)*, 2020. [Online](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3622_ECCV_2020_paper.php)

[9] S. Shi, L. Jiang, D. Dai, and B. Schiele, “Motion Transformer with global intention localization and local movement refinement,” in *Advances in Neural Information Processing Systems*, vol. 35, 2022. [Online](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html)

[10] A. Seff, B. Cera, D. Chen, *et al.*, “MotionLM: Multi-agent motion forecasting as language modeling,” in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2023, pp. 8579--8590. [Online](https://openaccess.thecvf.com/content/ICCV2023/html/Seff_MotionLM_Multi-Agent_Motion_Forecasting_as_Language_Modeling_ICCV_2023_paper.html)

[11] N. Shahroudi, M. Lepson, and M. Kull, “Evaluation of trajectory distribution predictions with energy score,” in *Proc. 41st Int. Conf. Machine Learning (ICML)*, PMLR 235, 2024, pp. 44322--44341. [Online](https://proceedings.mlr.press/v235/shahroudi24a.html)

[12] T. Gneiting and A. E. Raftery, “Strictly proper scoring rules, prediction, and estimation,” *Journal of the American Statistical Association*, vol. 102, no. 477, pp. 359--378, 2007. [Online](https://doi.org/10.1198/016214506000001437)

[13] M. Liang, B. Yang, W. Zeng, *et al.*, “PnPNet: End-to-end perception and prediction with tracking in the loop,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 11553--11562. [Online](https://openaccess.thecvf.com/content_CVPR_2020/html/Liang_PnPNet_End-to-End_Perception_and_Prediction_With_Tracking_in_the_Loop_CVPR_2020_paper.html)

[14] A. Hu, Z. Murez, N. Mohan, *et al.*, “FIERY: Future instance prediction in bird's-eye view from surround monocular cameras,” in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2021, pp. 15273--15282. [Online](https://openaccess.thecvf.com/content/ICCV2021/html/Hu_FIERY_Future_Instance_Prediction_in_Birds-Eye_View_From_Surround_Monocular_ICCV_2021_paper.html)

[15] Y. Zhang, Z. Zhu, W. Zheng, *et al.*, “BEVerse: Unified perception and prediction in bird's-eye-view for vision-centric autonomous driving,” arXiv:2205.09743, 2022. [Online](https://arxiv.org/abs/2205.09743)

[16] J. Gu, C. Hu, T. Zhang, *et al.*, “ViP3D: End-to-end visual trajectory prediction via 3D agent queries,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 5496--5506. [Online](https://openaccess.thecvf.com/content/CVPR2023/html/Gu_ViP3D_End-to-End_Visual_Trajectory_Prediction_via_3D_Agent_Queries_CVPR_2023_paper.html)

[17] Y. Hu, J. Yang, L. Chen, *et al.*, “Planning-oriented autonomous driving,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 17853--17862. [Online](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html)

[18] X. Gu, G. Song, I. Gilitschenski, M. Pavone, and B. Ivanovic, “Producing and leveraging online map uncertainty in trajectory prediction,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2024, pp. 14521--14530. [Online](https://openaccess.thecvf.com/content/CVPR2024/html/Gu_Producing_and_Leveraging_Online_Map_Uncertainty_in_Trajectory_Prediction_CVPR_2024_paper.html)

[19] L. Ding, D. Li, B. Liu, *et al.*, “Safe motion planning for autonomous vehicles by quantifying uncertainties of deep learning-enabled environment perception,” *IEEE Transactions on Intelligent Vehicles*, vol. 9, no. 1, pp. 2318--2332, 2024. [Online](https://doi.org/10.1109/TIV.2023.3297735)

[20] A. Malinin, B. Mlodozeniec, and M. Gales, “Ensemble distribution distillation,” in *Proc. 8th Int. Conf. Learning Representations (ICLR)*, 2020. [Online](https://openreview.net/forum?id=BygSP6Vtvr)

[21] S. H. Sun, R. Walters, J. Li, and R. Yu, “Probabilistic symmetry for multi-agent dynamics,” in *Proc. 5th Annual Learning for Dynamics and Control Conf.*, PMLR 211, 2023, pp. 1231--1244. [Online](https://proceedings.mlr.press/v211/sun23a.html)

[22] S. Fadadu, S. Pandey, D. Hegde, *et al.*, “Multi-view fusion of sensor data for improved perception and prediction in autonomous driving,” in *Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV)*, 2022, pp. 2349--2357. [Online](https://openaccess.thecvf.com/content/WACV2022/html/Fadadu_Multi-View_Fusion_of_Sensor_Data_for_Improved_Perception_and_Prediction_WACV_2022_paper.html)

[23] B. Ivanovic, Y. Lin, S. Shrivastava, P. Chakravarty, and M. Pavone, “Propagating state uncertainty through trajectory forecasting,” in *Proc. IEEE Int. Conf. Robotics and Automation (ICRA)*, 2022, pp. 2351--2358. [Online](https://doi.org/10.1109/ICRA46639.2022.9811776)

[24] X. Weng, B. Ivanovic, K. Kitani, and M. Pavone, “Whose track is it anyway? Improving robustness to tracking errors with affinity-based trajectory prediction,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 6563--6572. [Online](https://openaccess.thecvf.com/content/CVPR2022/html/Weng_Whose_Track_Is_It_Anyway_Improving_Robustness_to_Tracking_Errors_CVPR_2022_paper.html)

[25] P. Zhang, L. Bai, Y. Wang, J. Fang, J. Xue, N. Zheng, and W. Ouyang, “Towards trajectory forecasting from detection,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 45, no. 10, pp. 12550--12561, 2023. [Online](https://doi.org/10.1109/TPAMI.2023.3274686)

[26] M. Cao, C. Wang, W. Zhao, and Z. Zhang, “PU-TSI: Interactive vehicle trajectory prediction considering perception information uncertainty,” *IEEE Internet of Things Journal*, vol. 12, no. 15, pp. 30518--30532, 2025. [Online](https://doi.org/10.1109/JIOT.2025.3570741)

[27] S. Papais, L. Wang, B. Cheong, and S. L. Waslander, “ForeSight: Multi-view streaming joint object detection and trajectory forecasting,” in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2025, pp. 25474--25484. [Online](https://openaccess.thecvf.com/content/ICCV2025/html/Papais_ForeSight_Multi-View_Streaming_Joint_Object_Detection_and_Trajectory_Forecasting_ICCV_2025_paper.html)

[28] N. Grabowsky, A. Mütze, and M. Rottmann, “Does knowledge about perceptual uncertainty help an agent in automated driving?” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2026, pp. 4219--4229. [Online](https://openaccess.thecvf.com/content/CVPR2026W/SAIAD/html/Grabowsky_Does_Knowledge_About_Perceptual_Uncertainty_Help_an_Agent_in_Automated_CVPRW_2026_paper.html)
