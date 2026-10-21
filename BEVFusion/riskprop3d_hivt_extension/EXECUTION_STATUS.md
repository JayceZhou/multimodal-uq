# HiVT 补实验执行状态：已完成

作者授权：执行已启动的最低版 Level I，直到完整做完实验；缺失输入允许重跑。Level II 未登记、未启动。

最终更新：2026-10-14 10:24 北京时间。最低版 Level I 已全部完成，训练与控制进程已退出，无待执行阶段。

三次 HiVT 各完成 64 epochs、325,888 次 optimizer 更新及 clean 校准；四种律 × 四个条件 × 三次配对运行全部完成，每次覆盖 92,522 个评估目标。64/256 Monte Carlo 敏感性、17,381 个共同目标敏感性、CV/MLP 同车辆重评及配对场景统计均已交付。

最终复核通过：78,217 个文件哈希（含 78,123 个结果分片）、冻结源码／输入、训练曲线、校准及支持检查均一致。全部行的身份／有限性已由冻结汇总程序核验，末次审计再次核验全部分片哈希。见 [最终审计](results_v1/final_delivery_audit.json)、[完成清单](results_v1/completion.json) 及 [最终报告](REPORT.md)。

主要 official_clean 对比 AG−AH 的 ES 差异为 −0.000634 m，95% 场景 CI [−0.002659, +0.001347]，未能明确区分；同矩对比 AM−AH 为 +0.023108 m，CI [+0.016992, +0.029842]。正差值表示 HTC 更好。未将主要对比写成稳定提升。

本次结案范围为已授权的最低版 Level I；Level II 和新的独立确认实验未执行。

## 启动阶段历史快照

以下保留当时进度，当前状态以上述最终更新及完成清单为准。

截至 2026-10-14 北京时间本次更新：28,130 帧 clean 输入、三组 20-epoch 概率头及 baseline 完成；11,635 帧退化／official 背景和六个角色的上下文完成。全部 203,337 个 clean 车辆目标的 box/feature/logits/seed score 与原档案逐值相同。数据审计、真实小样本过拟合、地图／邻居消融与成本测量均通过；三次 64-epoch HiVT 训练已在 GPU 0、2、3 启动。校准、传播评估及统计仍待完成，实验未结案。

首个完整 epoch 中，三次开发 minFDE 从随机初始化的 4.936／4.768／4.959 m 降至 0.737／0.716／0.762 m；每次覆盖 162,919 个训练目标并更新 5,092 次。固定小样本过拟合损失 3.892 → 1.141；地图／邻居消融最大均值变化 0.286／1.235 m，重复前向最大差异 3.81e-6。实测 128 个图的单次均值前向约 0.0241 s，峰值张量分配约 1.02 GiB；这是工程测量，不是完整传播运行时间。

rotation_s3／official_clean 的新背景分别未匹配 34／66 个原目标；原目标全部保留且四种律共用同一新背景。版本差异保存在各 data manifest，不能声称这些背景逐值恢复了旧导出。[数据审计](data_v1/audit.json)、[工程资格](host_v1/engineering.json)、[宿主配置冻结](host_v1/freeze.json)。

已完成检查：原 detector/config 与 P0 输入哈希；真实 detector query/seed 重复一致与有限梯度；原目标 12 行抽查逐值复现；HiVT 原生前向／有限梯度；PyG batch 索引兼容；当前抽样同步写回；固定历史／邻居／地图；空地图／邻居；标签排除于 forward；两套独立 ES 随机流与 64/256 固定前缀；旋转后的完整协方差；场景内平均与三次配对场景 bootstrap。

持续队列分三条，动态 JSON 和日志优先于本文的静态快照：

- [recovery_v1/status.json](recovery_v1/status.json)：clean 查询导出 → canonical authority → baseline → 三组原配方概率头，旧输出不覆盖。
- [pipeline_v1/cpu_status.json](pipeline_v1/cpu_status.json)：clean train / cal / dev 感知上下文构建。
- [pipeline_v1/gpu_status.json](pipeline_v1/gpu_status.json)：待恢复完成，使用空闲 GPU 0、2、3；额外 dropout / rotation / official 感知背景 → 上下文 → 时间／坐标／标签审计 → 真实小样本过拟合／消融／成本试测 → 配方冻结 → 三次 HiVT 训练 → clean 校准。
- [pipeline_v1/finish_status.json](pipeline_v1/finish_status.json)：待三次校准完成，依次运行实际随机律支持检查 → 结果前冻结 → 四律／四人群全量评估 → 64/256 MC 敏感性 → 原 CV/MLP 同车辆重评 → 配对场景统计与最终 REPORT.md。

控制器初始启动身份见 [launchers.json](pipeline_v1/launchers.json) 和 [finish_launcher.json](pipeline_v1/finish_launcher.json)，当前两个后续控制器的替换身份见 [queue_restart_attempt2.json](pipeline_v1/queue_restart_attempt2.json)。此次只重启等待控制器以同时检查显存和利用率，未停止任何训练任务。所有 GPU worker 显式设置 CUDA_VISIBLE_DEVICES，阶段启动前再查 nvidia-smi；不回退 CPU。错误写入 failure JSON 并阻止后续阶段，保留日志后处理。

新输入版本见 [DATA_CONTRACT.md](DATA_CONTRACT.md)：原目标身份、历史、均值、头输入和标签保留，邻居背景由同条件重新导出的感知框构建，GT 坐标只用于关联核对／评分。重训头不声称等于已删除原头的哈希；旧宿主终态不改写。输入缺失与版本差异会在最终报告完整披露。

完成要求：三次宿主通过真实数据资格、四律／四人群完整且数值可解释、三次校准、所有预定统计／敏感性／旧宿主对照完成并可复核。有效负结果照常交付；支持域失败须记录受影响比较，不裁剪／重抽样／删行来制造成功。
