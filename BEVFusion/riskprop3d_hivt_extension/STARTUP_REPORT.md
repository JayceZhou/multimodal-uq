# HiVT 补实验启动记录

2026-10-13 22:01:59（北京时间，最新检查记录时间）。

**已执行准备与 GPU 工程检查；正式训练未启动。** 当前状态为 `INPUTS_MISSING`，不是宿主训练失败或科学负结果。最低版 Level I 先行，Level II 尚未登记为本次正式执行范围。

## 已完成

- 下载方案指定的公开参考实现，锁定 commit `0a11ee4b925e4967f2f889079d4086b663d396dc`，保留许可证与源码；局部 checkout 约 4.1 MiB。
- 复用现有 `bev` 的 PyTorch 2.4.1／CUDA 12.4，在本实验 `python/` 目录安装必要依赖，约 113 MiB。没有修改共享 Python 环境。实际版本见[机器记录](H0_attempt3_preflight.json)。
- 修复独立参考副本的两类初始化／前向兼容问题：torchmetrics 已移除的 `compute_on_step` 参数，以及 PyTorch temporal encoder 的 `is_causal` 参数。计算指标的 update/compute 公式、HiVT 网络容量与原有 causal mask 保留；补丁见[runtime_compatibility.patch](runtime_compatibility.patch)。
- GPU 0 上执行五个历史／当前槽、六步未来的原生 HiVT 合成输入检查：六模式输出有限、Laplace scales 为正、重复推理逐值相同、原输入没有累计修改、独立损失的梯度有限且非零。没有 optimizer 更新，没有真实数据训练。
- 检查冻结输入：已有 train／label-cal／clean-dev 轨迹、R1 scene graph 与对应清单保留；R1 三份 scene graph 的内容哈希与旧清单一致。车辆行数分别为 162,919、22,188、18,230。
- `preflight.py --self-check` 验证缺失、匹配及被改写的文件会正确通过或阻止启动。

GPU 最初盘点中 0、2、3 空闲，1 有高负载任务；本次工程检查明确使用 `CUDA_VISIBLE_DEVICES=0`。其他任务未被停止。三次正式训练预计优先使用物理卡 0、2、3，但实际启动前仍须重新检查并报告分配。

## 当前必须恢复的材料

路径相对 `/home/zhousc/multimodal-uq/BEVFusion`：

| 材料 | 当前状态 | 恢复要求 |
| --- | --- | --- |
| `outputs/probset3d_v3/f1_formal/checkpoints/seed_20260731.pt` | 缺失 | 旧 SHA-256 一致 |
| `outputs/probset3d_v3/f1_formal/checkpoints/seed_20260801.pt` | 缺失 | 同上 |
| `outputs/probset3d_v3/f1_formal/checkpoints/seed_20260802.pt` | 缺失 | 同上 |
| `outputs/probset3d_v3/f1_formal/baseline.json` | 缺失 | 恢复原 density／关联构造依赖；不得重新拟合替代 |
| `outputs/probset3d_v3/f1_authority/run_a/manifest.json` 及原始导出 | 清单缺失，train／label_cal／dev 子目录均为空 | 恢复原清单与其引用的所有 parts，核验清单及各 part 哈希 |

原 authority 清单 SHA-256 为 `af91451de4d16524941da82a1caa7329067564dcf0967bb5b1f9f2563013b672`，来自仍保留的 R1 清单。三个头的 SHA-256 见[方案](SUPPLEMENTARY_PLAN.md)。

按原文件名和有关备份文件名搜索了当前工作区、可访问的 `/home/zhousc` 与 `/data2/zhousc` 常规目录，未找到这些原文件。搜索排除了 `.git` 内部、环境／依赖及大数据集目录，不代表所有外部归档不存在；当前 Git refs 中也没有查到这几个原路径的可恢复版本。需要作者提供原目录备份或挂载位置。

缺失原始导出意味着不能从源材料补齐与目标同条件的感知邻居历史。现存 AutoBots context cache 的邻居位置来自 GT，不能用它替代本方案要求的感知邻居。

## 保留证据与下一步

[attempt 1](H0_attempt1_preflight.json) 保留 torchmetrics 初始化兼容失败；[attempt 2](H0_attempt2_preflight.json) 保留 `is_causal` 兼容失败；[attempt 3](H0_attempt3_preflight.json) 的 GPU 检查通过，剩余五个文件级阻塞全部是缺失输入。各记录保存当时的源码／runner／方案哈希、输入检查与环境版本，未覆盖早期记录。

恢复材料后先运行新编号的启动检查，再实现感知邻居、固定 HD 地图及 retained-target 数据适配，完成真实样本坐标／mask／批处理自检和成本试测，冻结训练配置，最后启动三次正式宿主训练。`preflight.py` 是实际可运行的启动检查入口；当前没有可以直接启动正式训练的适配数据或训练入口。

未执行真实数据训练、宿主收敛资格、当前律传播、64／256 样本比较或新的 official 指标；单图合成工程检查也不等于真实上下文及批处理验证。项目磁盘在收尾检查时约剩 19 GiB，之后需按实际缓存估算空间，不能直接复制旧大型地图缓存。原冻结概率头和源导出的恢复是当前首要阻塞。
