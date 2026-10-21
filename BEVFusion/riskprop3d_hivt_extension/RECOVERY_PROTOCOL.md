# HiVT 输入恢复：独立重导出与概率头重跑

2026-10-13 北京时间。作者新授权：“你找一下哪个原始文件能用，不行的话就重跑一下之前的吧”。此授权允许重跑缺失输入，更新先前方案中只允许字节恢复的约束；旧冻结文件、结果与终态仍只读。

## 恢复结论与范围

原 ProbSet3D V2/V3 authority、baseline 和三组 F1-V3 头均未找到。Stage3D-B 的 `work/fusion_dev.pt` 仍含完整 200 queries，但抽查 32 个既有 R0 目标时，框、特征和 logits 均未逐值复现冻结输入，不能标作原冻结导出的恢复。正式恢复从原 nuScenes 传感器、相同 BEVFusion checkpoint/config 和原 P0 train/label-cal/dev 划分重新导出。

原 detector checkpoint SHA-256 为 `ee0a389213922343508db40dff0ec04767b36d8f512da34b093939c37aaf2122`，配置 SHA-256 为 `cb7bbbc25e8fdf67c2e914b11661828b684312300deb2afc410793192713e04b`；当前文件已核实相同。保留 P0 22,508／2,814／2,808 帧、560／70／70 场景的原始来源；不读取新的 official 指标或 sealed test。

## 固定重跑契约

- 执行目录为本实验 `recovery_v1/`；重跑与旧输出分别保存，新权重不能填写成旧 SHA-256 相同的恢复品。
- 复用 `probset3d_v2_sweeps.canonicalize_infos/bind_pipeline`：当前帧加最近九个过去 sweeps，固定选择，test_mode=True；不复用早期随机 sweep 缓存。
- 冻结原 BEVFusion，原生 FP16 测试前向、200 queries、128D decoder features、10D class logits、decoded 9D boxes、dense seed scores。启用确定性执行、关闭 TF32，保留模型参数与 buffer 前后身份检查。
- 复用原 native cost、确定性 Hungarian assignment、2 m 保留门槛、原 duplicate membership 构造。保留全部 200 queries 及背景，而不是只拿 R1 的 matched agents 重训。
- 原不允许科学读取的 `query_density` 历史列不恢复；M1/M2/M3 均沿用原训练的零 density 输入。新的普通 Gaussian feature density 使用 clean train 拟合，只供原 baseline/context 接口；这不同于已删除的历史 RealNVP density。
- clean train 和 label-cal 的重新导出按 sample_token 全局排序、每 part 32 帧，维持原 batch 边界；每 batch 8 帧，末 batch 4 帧。重新拟合原 class temperature 和 Gaussian feature density；不以新 dev 科学结果选模型或改预算。
- 复用原概率头和 `f1_loss`，三 seeds 为 20260731／20260801／20260802，M1 Gaussian、M2 Student-t/von Mises、M3 duplicate control 同原初始化顺序；各 20 clean epochs，AdamW lr=1e-3、wd=1e-4、clip=5、FP32。保存最后 epoch，不做 checkpoint 选择。M3 一并训练以保留原 RNG/训练顺序，但本补实验只使用 M1/M2。
- 记录每 epoch 的覆盖、各组 loss、更新次数、有限性、真实环境、代码及输入哈希。失败保存，不能删行或用 GT 感知状态替代。

## 执行与校验

先在 GPU 0 做真实帧 smoke：相同输入重复前向、query/seed 身份一致、assignment 合法、概率头训练梯度与共同均值检查；确认内存与吞吐后使用 GPU 0、2、3 执行三份独立 split 导出，并行设备均明确设置 `CUDA_VISIBLE_DEVICES`。GPU 1 的已有任务保留。启动时再次检查使用情况。

完整导出预计约 3–4 GiB（不保存 dense heatmaps）；canonical train/cal 可采用无重复磁盘数据的索引式读取，或在空间允许时另存最小训练 parts。新增磁盘需求实測核对，不删除既有研究缓存。三次头训练在导出与归一化/校准完成后自动进入，正式结果未完成时不得标作可用头。

旧宿主报告、概率头旧哈希清单及 HiVT 原先 H0 失败记录保留。重导出将核对可比较的旧 row_id、共同均值和 feature 差异；如不能逐值复现，明确新输入版本，并在 HiVT 执行前冻结目标与历史/邻居来源。不能把新头结果混入旧 CV/MLP 指标后称为原冻结三组复现；所需旧宿主对照在新版本上重算。

恢复完成后生成新的输入 authority、三个 checkpoint、baseline 与 recovery manifest。随后继续 HiVT 的真实上下文适配和三次 Level I 宿主训练；恢复工程检查本身不构成 HiVT 训练或概率传播结论。
