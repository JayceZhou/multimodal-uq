# 原始材料搜索与 clean 输入重跑完成记录

2026-10-14 北京时间更新。**clean 恢复阶段已完成：28,130 帧 authority、baseline 和三组 20-epoch 概率头齐备；HiVT 正式训练尚未启动。** 完成身份见 [recovery_manifest.json](recovery_v1/recovery_manifest.json)，后续阶段见 [EXECUTION_STATUS.md](EXECUTION_STATUS.md)。

## 哪些文件可以用

| 原始文件／目录 | 核查结论 | 本次用途 |
| --- | --- | --- |
| `BEVFusion/pretrained/bevfusion-det.pth` | SHA-256 与旧权威一致 | 冻结原检测器，重新导出完整 200 queries |
| 原 `convfuser.yaml` | SHA-256 与旧权威一致 | 原模型结构与测试路径 |
| `BEVFusion/probset3d/P0_DATA_SPLIT_AUTHORITY.json` 与 `outputs/stage3/splits/nuscenes_infos_stage3_{train,label_cal,dev}.pkl` 及 token/scene 清单 | 三组原分割文件哈希、样本集合和数量核对通过 | 原 22,508／2,814／2,808 帧的 clean 重导出 |
| `outputs/riskprop3d_r0/tracks/` 和 `outputs/riskprop3d_r1/scenes/` | 既有轨迹、目标 query 输入和索引仍在；此前 R1 文件哈希核对通过 | 保留目标、历史、标签和旧输入复核 |
| `outputs/stage3d/stage3d_b/work/fusion_dev.pt` | 保留 200-query 原始缓存，但抽查 32 个 R0 目标没有逐值一致的框 | 不用它替代后来的确定性冻结 authority |
| 原 ProbSet3D V2/V3 authority、F1 baseline、三组头 | 对应目录为空或文件缺失，未找到可用备份 | 单独重跑，使用新产物与哈希 |

早期 dev 缓存的这 32 条抽查中，框最大绝对差 0.6796875，特征差 0.7729533315，logits 差 0.5546875。它来自同一个 detector checkpoint，但不能据此视作相同感知输入；其他早期模型、指标导出和不同概率头也不能替代原 G/HTC 权重。

## 已通过的真实工程检查

最初两帧包含的 12 条原目标逐值复现；随后完整核对 train／label-cal／clean-dev 的 **162,919／22,188／18,230 个车辆目标，box、128D feature、10D logits、seed score 均逐值相同，无缺失原目标**。这是保留车辆输入的全量核对，不是原 authority 文件或旧头权重的字节恢复。[train 全量比较](query_alignment_clean_train_attempt1.json)、[cal/dev 全量比较](query_alignment_clean_cal_dev_attempt1.json)、[最初抽查](recovery_v1/smoke/old_input_comparison.json)。

同一真实帧重复前向逐值一致；200-query seed/类别身份、native assignment、全部数值有限性、detector 参数/buffer 不变检查通过。Gaussian、Student-t 和 duplicate control 的真实损失梯度有限、非零，三者共同均值检查通过；工程检查没有执行任何 head optimizer 更新。[工程记录](recovery_v1/smoke/engineering_check.json)。

## 已完成的恢复

- 显式使用物理 GPU **0、2、3**；最初 worker PID 分别为 1920510、1920511、1920512，已通过 GPU 进程查询核对。GPU 1 的已有任务保留。
- 八个 clean 导出任务、canonical 数据与 baseline、三次原 seeds 的 20 epoch 头训练均已完成。每组头模型更新 56,280 次，各 epoch 覆盖 22,508 帧；使用最后 epoch，不根据新 dev 收益选择。
- 原始导出暂存在 `/tmp/hivt_recovery_v1_20261013/raw`；canonical 训练输入、baseline、checkpoint、哈希与训练日志保留在本目录 `recovery_v1/`。最终 HiVT 使用新权重前还需核对新输入版本及上下文契约。
- 此次是作者授权的**重跑恢复**，不是旧权重的字节备份恢复。新 checkpoint 不冒称旧 SHA-256；旧实验终态和指标均不覆盖。

[恢复协议](RECOVERY_PROTOCOL.md)、[运行前冻结](recovery_v1/pre_recovery_freeze.json)、[控制器日志](recovery_v1/controller.log)、[启动信息](recovery_v1/launcher.json)、[存储说明](recovery_v1/storage_manifest.json)。实际入口为 `recover.py`。

三个 `training/seed_*_complete.json` 的有限性和 checkpoint 哈希已核对；阶段无 failure.json。新权重如下，均区别于旧清单：

| head seed | 新 checkpoint SHA-256 |
|---|---|
| 20260731 | `9173260dd82c72cedc3f17ba1560d9b3071221efaebcac1d1bb8f70ae2041e1e` |
| 20260801 | `10d6bd9ee7dddb7540d05821097fea7d56d887204d825c35d97db2170336e5be` |
| 20260802 | `58611bdf35c0c3ebc7016193d1576fa8efb01c02af4fc5a0910b4e2db61c37db` |

本完成记录仅覆盖 clean 输入和概率头；HiVT 训练、其他条件背景与传播指标仍待完成。canonical authority 已在项目目录保存，临时 raw 保留作复核；`/tmp` 不作为长期归档位置。
