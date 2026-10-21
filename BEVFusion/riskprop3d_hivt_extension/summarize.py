"""Verify paired outputs, summarize scene inference, and produce the final report."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from evaluate import HERE, OUT as EVAL, CONDITIONS, LAWS, check_freeze
from run_host import OUT as HOST, sha, write

OUT = HERE / "results_v1"


def read_parts(directory):
    manifest = json.loads((directory / "complete.json").read_text())
    fields = {}
    for record in manifest["parts"]:
        path = directory / record["file"]
        assert sha(path) == record["sha256"], path
        with np.load(path, allow_pickle=False) as part:
            for key in part.files:
                fields.setdefault(key, []).append(part[key])
    result = {key: np.concatenate(value) for key, value in fields.items()}
    assert len(result["row"]) == manifest["rows"] and len(np.unique(result["row"])) == len(result["row"])
    assert all(np.isfinite(v).all() for v in result.values())
    return result, manifest


def scene_vector(values, scene):
    return np.asarray([values[scene == token].mean() for token in np.unique(scene)])


def bootstrap(values):
    rng = np.random.default_rng(20260805)
    means = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    return [float(v) for v in np.quantile(means, [.025, .975])]


def risk_coverage(risk, error):
    order = np.argsort(risk, kind="stable")
    fractions = np.linspace(.1, 1, 91)
    means = [error[order[:max(1, int(len(error) * fraction))]].mean() for fraction in fractions]
    return float(np.trapz(means, fractions))


def risks(event, error, score):
    event = event.astype(int)
    safe = score <= np.quantile(score, .8)
    return {"prevalence": float(event.mean()),
            "auprc": float(average_precision_score(event, score)) if len(np.unique(event)) == 2 else None,
            "auroc": float(roc_auc_score(event, score)) if len(np.unique(event)) == 2 else None,
            "risk_coverage": risk_coverage(score, error),
            "false_safe": float((event * safe).sum() / event.sum()) if event.sum() else None}


def paired(runs, baseline, scene, mask=None):
    if mask is None:
        mask = np.ones(len(scene), dtype=bool)
    differences, bases, seed_directions = [], [], []
    for run in runs:
        difference = run[baseline + "_energy"] - run["AH_energy"]
        vector = scene_vector(difference[mask], scene[mask])
        differences.append(vector)
        bases.append(scene_vector(run[baseline + "_energy"][mask], scene[mask]))
        seed_directions.append(float(vector.mean()))
    average = np.asarray(differences).mean(0)
    base = float(np.asarray(bases).mean())
    return {"contrast": baseline + "-AH", "difference": float(average.mean()), "ci95": bootstrap(average),
            "relative_reduction_percent": float(100 * average.mean() / base),
            "per_run_difference": seed_directions, "rows": int(mask.sum()), "scenes": len(average)}


def paired_risk(runs, baseline, scene):
    result = {}
    for metric in ("auprc", "risk_coverage", "false_safe"):
        maps = []
        for run in runs:
            current = {}
            for token in np.unique(scene):
                select = scene == token
                event = run["A0_mean_fde"][select] > 2
                error = run["A0_mean_fde"][select]
                base = risks(event, error, run[baseline + "_log_area90"][select])[metric]
                htc = risks(event, error, run["AH_log_area90"][select])[metric]
                if base is not None and htc is not None:
                    current[str(token)] = (htc - base) if metric == "auprc" else (base - htc)
            maps.append(current)
        shared = sorted(set(maps[0]) & set(maps[1]) & set(maps[2]))
        if shared:
            values = np.asarray([[m[token] for token in shared] for m in maps])
            result[metric] = {"improvement": float(values.mean()), "ci95": bootstrap(values.mean(0)),
                              "per_run": values.mean(1).astype(float).tolist(), "eligible_scenes": len(shared)}
        else:
            result[metric] = {"improvement": None, "ci95": None, "eligible_scenes": 0}
    return result


def finalize():
    config = check_freeze()
    OUT.mkdir(exist_ok=True)
    main, absolute, comparisons, diagnostics, controls, costs, risk_comparisons = {}, [], [], {}, [], [], []
    absolute_estimates, control_contrasts = [], []
    for condition in CONDITIONS:
        datasets = []
        for replicate in range(3):
            data, manifest = read_parts(EVAL / "main" / ("run_" + str(replicate)) / condition)
            np.testing.assert_array_equal(data["row"], config["rows"][condition])
            datasets.append(data)
            costs.append(dict(manifest, parts=None))
        with np.load(HERE / "data_v1" / condition / "geometry.npz", allow_pickle=False) as geometry:
            row = datasets[0]["row"]
            scene = geometry["scene_token"][row]
            identities = np.char.add(np.char.add(geometry["sample_token"][row], "/"), geometry["instance_token"][row])
        main[condition] = {"runs": datasets, "scene": scene, "identities": identities}
        for baseline in ("AG", "AM"):
            comparisons.append(dict(condition=condition, **paired(datasets, baseline, scene)))
            risk_comparisons.append({"condition": condition, "baseline": baseline, "candidate": "AH",
                                     "metrics": paired_risk(datasets, baseline, scene)})
        for law in LAWS:
            vectors = np.asarray([scene_vector(data[law + "_energy"], scene) for data in datasets])
            absolute_estimates.append({"condition": condition, "law": law, "energy": float(vectors.mean()),
                "ci95": bootstrap(vectors.mean(0)), "per_run_energy": vectors.mean(1).astype(float).tolist(),
                "rows": len(row), "scenes": len(np.unique(scene))})
        diagnostics[condition] = []
        for replicate, data in enumerate(datasets):
            event = data["A0_mean_fde"] > 2
            for law in LAWS:
                fields = {k[len(law) + 1:]: v for k, v in data.items() if k.startswith(law + "_")}
                means = {name: float(scene_vector(v, scene).mean()) for name, v in fields.items()}
                absolute.append(dict(condition=condition, replicate=replicate, law=law, rows=len(row), scenes=len(np.unique(scene)), **means))
                calibration = {"same_coverage_area90": float(np.pi * np.quantile(fields["final_maha2"], .9) * np.sqrt(fields["final_det"]).mean()),
                               "same_coverage_quantile90": float(np.quantile(fields["final_maha2"], .9)),
                               "area_readout_is_posthoc": True}
                diagnostics[condition].append(dict(replicate=replicate, law=law,
                    global_risk=risks(event, data["A0_mean_fde"], fields["log_area90"]), calibration=calibration))
        control_runs = []
        for replicate in range(3):
            data, _ = read_parts(HERE / "controls_v1" / ("run_" + str(replicate)) / condition)
            np.testing.assert_array_equal(data["row"], datasets[replicate]["row"])
            control_runs.append(data)
            for host in ("CV", "MLP"):
                for law in LAWS:
                    values = {key[len(host) + len(law) + 2:]: value for key, value in data.items() if key.startswith(host + "_" + law + "_")}
                    means = {name: float(scene_vector(v, scene).mean()) for name, v in values.items()}
                    controls.append(dict(condition=condition, replicate=replicate, host=host, law=law, **means))
        for host in ("CV", "MLP"):
            runs = [{key[len(host) + 1:]: value for key, value in data.items() if key.startswith(host + "_")}
                    for data in control_runs]
            for baseline in ("AG", "AM"):
                control_contrasts.append(dict(condition=condition, host=host, **paired(runs, baseline, scene)))
    common = set(main[CONDITIONS[0]]["identities"].tolist())
    for condition in CONDITIONS[1:3]:
        common &= set(main[condition]["identities"].tolist())
    sensitivity = []
    for condition in CONDITIONS[:3]:
        current = main[condition]
        mask = np.asarray([key in common for key in current["identities"]])
        for baseline in ("AG", "AM"):
            sensitivity.append(dict(condition=condition, **paired(current["runs"], baseline, current["scene"], mask)))
    monte_carlo = []
    for replicate in range(3):
        data, manifest = read_parts(EVAL / "sensitivity" / ("run_" + str(replicate)) / "clean_s0")
        full = main["clean_s0"]["runs"][replicate]
        index = {int(row): i for i, row in enumerate(full["row"])}
        selected = [index[int(row)] for row in data["row"]]
        for law in LAWS:
            diff = data[law + "_energy"] - full[law + "_energy"][selected]
            monte_carlo.append(dict(replicate=replicate, law=law, rows=len(diff), mean_difference_256_minus_64=float(diff.mean()),
                                    mean_absolute_difference=float(np.abs(diff).mean()), max_absolute_difference=float(np.abs(diff).max()),
                                    seconds_256=manifest["seconds"]))
    write(OUT / "summary.json", {"status": "RESULTS_COMPUTED", "contrasts": comparisons, "absolute": absolute,
        "absolute_scene_estimates": absolute_estimates, "control_contrasts": control_contrasts,
        "common_target_sensitivity": sensitivity, "common_target_count": len(common), "MC_sensitivity": monte_carlo,
        "diagnostics": diagnostics, "controls": controls, "costs": costs,
        "paired_scene_risk": risk_comparisons,
        "evaluation_freeze_sha256": sha(EVAL / "freeze.json"), "bootstrap_unit": "paired scene after averaging three paired runs",
        "Level_II": "not registered, not executed", "original_results_rewritten": False})
    tables = (("contrasts", comparisons), ("absolute", absolute), ("absolute_estimates", absolute_estimates),
              ("controls", controls), ("control_contrasts", control_contrasts),
              ("common_targets", sensitivity), ("MC_sensitivity", monte_carlo))
    for name, table in tables:
        with (OUT / (name + ".csv")).open("x", newline="") as stream:
            keys = list(table[0])
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(table)
    report(comparisons, absolute, absolute_estimates, controls, control_contrasts, sensitivity, monte_carlo, costs)
    artifacts = [OUT / "summary.json", HERE / "REPORT.md"] + [OUT / (name + ".csv") for name, _ in tables]
    write(OUT / "completion.json", {"status": "LEVEL_I_COMPLETE", "main_run_condition_outputs": 12,
        "main_target_rows_per_run": sum(len(config["rows"][c]) for c in CONDITIONS),
        "artifacts_sha256": {str(p.relative_to(HERE)): sha(p) for p in artifacts},
        "evaluation_freeze_sha256": sha(EVAL / "freeze.json"), "no_required_statistical_tables_pending": True})


def report(comparisons, absolute, absolute_estimates, controls, control_contrasts, sensitivity, monte_carlo, costs):
    official = {r["contrast"]: r for r in comparisons if r["condition"] == "official_clean"}
    lines = ["# HiVT Level I 补实验完成报告", "", "状态：三次宿主、四种律、四个人群条件完成，三次 clean 输出校准、64/256 Monte Carlo 敏感性及旧 CV/MLP 同车辆重评完成。Level II 未登记、未执行。", ""]
    for contrast, title in (("AG-AH", "主要对比：训练 G 相对 HTC"), ("AM-AH", "同矩对比：HTC-MMG 相对 HTC")):
        r = official[contrast]
        direction = "HTC 改善" if r["ci95"][0] > 0 else ("HTC 变差" if r["ci95"][1] < 0 else "区间跨零，未明确区分")
        lines.append("official_clean {}：ES 差值 {:.6f} m，95% 场景 CI [{:.6f}, {:.6f}]，{}。三次配对运行差值为 {}。".format(title, r["difference"], *r["ci95"], direction, ", ".join("{:.6f}".format(v) for v in r["per_run_difference"])))
        lines.append("")
    lines += ["正差值表示 HTC 更好。CI 条件化于这三次配对模型，三次训练稳定性由各运行方向单独描述；目标和帧不是独立重复。唯一主要对比为 official_clean AG-AH，其余为预定解释性／扩展对比；区间逐项计算，不作跨终点／人群的同时成立声明。", "", "## 四个人群的主要对比", "", "| 条件 | 对比 | ES 差值 (m) | 95% CI | 相对降幅 | 目标／场景 |", "|---|---|---:|---|---:|---:|"]
    for r in comparisons:
        lines.append("| {} | {} | {:.6f} | [{:.6f}, {:.6f}] | {:.3f}% | {} / {} |".format(r["condition"], r["contrast"], r["difference"], *r["ci95"], r["relative_reduction_percent"], r["rows"], r["scenes"]))
    lines += ["", "## 绝对性能（三次运行平均，场景等权）", "", "| 条件 | 律 | ES (m), 95% CI | mean ADE (m) | mean FDE (m) | coverage 90 | final area 90 (m²) |", "|---|---|---|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        for law in LAWS:
            rows = [r for r in absolute if r["condition"] == condition and r["law"] == law]
            mean = {k: np.mean([r[k] for r in rows]) for k in ("energy", "mean_ade", "mean_fde", "coverage_90", "area90")}
            estimate = next(r for r in absolute_estimates if r["condition"] == condition and r["law"] == law)
            lines.append("| {} | {} | {:.6f} [{:.6f}, {:.6f}] | {:.6f} | {:.6f} | {:.4f} | {:.4f} |".format(
                condition, law, mean["energy"], *estimate["ci95"], mean["mean_ade"], mean["mean_fde"], mean["coverage_90"], mean["area90"]))
    lines += ["", "coverage 与面积来自包括模式间、状态间及旋转后 Laplace 条件方差的矩椭圆；它不是多模态 HDR。相同覆盖面积是事后诊断，单列于 summary.json。风险事件固定为同运行 A0 混合均值 FDE>2 m；所有组共用标签、A0 误差和面积排序口径。", "", "## 宿主资格、输入和范围", ""]
    engineering = json.loads((HOST / "engineering.json").read_text())
    lines.append("固定小样本过拟合损失 {:.6f} → {:.6f}；地图／邻居消融最大均值变化 {:.6f} / {:.6f} m。重复前向最大差异 {:.6g}，参考 GPU scatter 使用原生归约，不承诺逐 bit 一致。".format(engineering["initial_loss"], engineering["final_loss"], engineering["map_ablation_max_mean_difference"], engineering["neighbor_ablation_max_mean_difference"], engineering["repeat_max_difference"]))
    for replicate in range(3):
        directory = HOST / ("run_" + str(replicate))
        complete = json.loads((directory / "training_complete.json").read_text())
        calibration = json.loads((directory / "calibration.json").read_text())
        probe = json.loads((directory / "support_probe.json").read_text())
        lines.append("- 运行 {}：host_seed {}，{} epochs，{} updates；epoch-0 minFDE {:.6f} → 选择的 {:.6f} m；clean scale 乘子 {:.6f}，混合 NLL {:.6f} → {:.6f}。".format(replicate, complete["host_seed"], complete["epochs"], complete["updates"], complete["initial_development"]["min_fde"], complete["selected_min_fde"], calibration["scale_factor"], calibration["initial_mixture_nll"], calibration["calibrated_mixture_nll"]))
        lines.append("  正式 checkpoint 的地图／邻居消融最大均值变化 {:.6f} / {:.6f} m；固定实际随机律支持检查 {}。".format(probe["trained_host_map_ablation_max_mean_difference"], probe["trained_host_neighbor_ablation_max_mean_difference"], probe["status"]))
    lines += ["", "| 条件 | 原目标在新背景中未匹配 | 与旧当前框逐值相同 | 当前框最大差异 | 无地图目标 |", "|---|---:|---:|---:|---:|"]
    for condition in ("train", "label_cal", *CONDITIONS):
        m = json.loads((HERE / "data_v1" / condition / "manifest.json").read_text())
        lines.append("| {} | {} | {} / {} | {:.6g} | {} |".format(condition, m["background_current_missing_retained_target"], m["current_box_exact_retained_targets"], m["targets"], m["current_box_max_absolute_difference"], m["empty_map_targets"]))
    lines += ["", "原目标历史、均值、头输入、标签和身份始终保留；邻居历史由同条件感知重导出，GT 仅用于关联／审计。新概率头是独立恢复版本，不声称与删除的原头哈希相同。旧宿主终态未更改。0.5 s / 3 s 是原关键帧的名义时间窗；完整训练历史最大间隔偏差 0.155802 s，未因此删行，实际跨度见 [时间说明](TIME_AUDIT_NOTE.md) 及 [审计](data_v1/audit.json)。", "", "## 同目标的旧宿主对照", "", "| 条件 | 宿主 | 律 | ES (m) | mean ADE (m) | mean FDE (m) |", "|---|---|---|---:|---:|---:|"]
    for condition in CONDITIONS:
        for host in ("CV", "MLP"):
            for law in LAWS:
                rows = [r for r in controls if r["condition"] == condition and r["host"] == host and r["law"] == law]
                values = [np.mean([r[key] for r in rows]) for key in ("energy", "mean_ade", "mean_fde")]
                lines.append("| {} | {} | {} | {:.6f} | {:.6f} | {:.6f} |".format(condition, host, law, *values))
    lines += ["", "| 条件 | 宿主 | 组内对比 | ES 差值 (m) | 95% 场景 CI | 三次方向 |", "|---|---|---|---:|---|---|"]
    for r in control_contrasts:
        lines.append("| {} | {} | {} | {:.6f} | [{:.6f}, {:.6f}] | {} |".format(r["condition"], r["host"], r["contrast"],
            r["difference"], *r["ci95"], ", ".join("{:.6f}".format(v) for v in r["per_run_difference"])))
    lines += ["", "CV/MLP 沿用原冻结预测器、原 Gaussian 残差核与完整 state/yaw 读取范围；重评使用新三组头和同车辆目标。HiVT 只读取 XY，所以跨宿主差异不构成完整 3D/yaw 机制的同接口因果比较。", "", "## 敏感性与成本", ""]
    lines.append("前三个 dev 场景条件的共同目标对比见 [common_targets.csv](results_v1/common_targets.csv)，固定哈希子集的 64/256 抽样差异见 [MC_sensitivity.csv](results_v1/MC_sensitivity.csv)。official_clean 单独分析。")
    lines.append("正式传播累计 GPU 运行时间 {:.3f} 小时（各运行／条件 wall time 求和；不等于整个实验历时），峰值张量分配 {:.3f} GiB。计入条件宿主推理、构图、抽样与指标，不含 detector/head 恢复和宿主训练。完整成本明细与上游日志分别保留。".format(sum(r["seconds"] for r in costs) / 3600, max(r["peak_allocation_bytes"] for r in costs) / 2 ** 30))
    lines.append("硬件 {}，物理 GPU 0／2／3；row batch={}、state draw chunk={}。正式矩阵总计 {:,} 个随机条件目标输入、{:,} 次均值条件目标输入、{:,} 次批量 native forward 调用；这些计数均不作为独立统计样本。".format(
        costs[0]["gpu_name"], costs[0]["row_batch"], costs[0]["draw_chunk"],
        sum(r["stochastic_conditional_target_inputs"] for r in costs), sum(r["mean_state_conditional_target_inputs"] for r in costs),
        sum(r["native_forward_calls"] for r in costs)))
    lines += ["", "研究范围仍限于 nuScenes、一个 detector、正确关联车辆、固定地图／邻居背景和当前 XY 边缘律。clean-dev 属开发描述；旧 official 结果已被观察，本轮是后设计扩展。未访问 sealed internal test，未将有效负结果改写为数值无效。", "", "完整逐目标结果、冻结清单、训练曲线与校准参数见各独立目录；主表和可复核汇总见 [summary.json](results_v1/summary.json)、[contrasts.csv](results_v1/contrasts.csv)、[absolute.csv](results_v1/absolute.csv)。"]
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_check():
    scene = np.asarray(["a", "a", "b"])
    runs = [{"AG_energy": np.asarray([3., 5., 6.]), "AH_energy": np.asarray([1., 1., 2.])} for _ in range(3)]
    r = paired(runs, "AG", scene)
    assert r["difference"] == 3.5 and r["rows"] == 3 and r["scenes"] == 2
    assert r["per_run_difference"] == [3.5] * 3
    assert bootstrap(np.ones(10)) == [1., 1.]
    print("PASS: paired scene weighting, three-run average and bootstrap unit")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        self_check()
    else:
        finalize()
