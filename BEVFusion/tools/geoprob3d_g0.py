#!/usr/bin/env python3
"""Evaluate and decide the frozen GeoProb3D G0 probability-geometry audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.special import gammaln, i0e, logsumexp
from scipy.stats import chi2, f, norm, spearmanr, vonmises

from probset3d.f1.implementation.probabilistic_query import ProbabilisticQueryHead
from tools import probset3d_f0_score as f0
from tools import probset3d_v3_formal as formal
from tools import probset3d_v3_pilot as pilot
from tools.stage3d_b_native_assignment import detection_gt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/geoprob3d_g0"
SEEDS = formal.SEEDS
MODELS = ("M0", "M1", "M2", "M5")
LEVELS = (0.50, 0.80, 0.90, 0.95)
MC_SAMPLES = 32
CONDITIONS = {
    "clean_s0": ROOT / "outputs/probset3d_v2/f0/exports/fusion-dev-clean-s0",
    "lowlight_s1": OUT / "exports/fusion-dev-camera-lowlight-s1",
    "lowlight_s2": OUT / "exports/fusion-dev-camera-lowlight-s2",
    "lowlight_s3": ROOT / "outputs/probset3d_v2/f0/exports/fusion-dev-camera-lowlight-s3",
    "dropout_s1": OUT / "exports/fusion-dev-lidar-dropout-s1",
    "dropout_s2": OUT / "exports/fusion-dev-lidar-dropout-s2",
    "dropout_s3": ROOT / "outputs/probset3d_v2/f0/exports/fusion-dev-lidar-dropout-s3",
    "beam_s1": OUT / "exports/fusion-dev-lidar-beam-reduction-s1",
    "beam_s2": OUT / "exports/fusion-dev-lidar-beam-reduction-s2",
    "beam_s3": ROOT / "outputs/probset3d_v2/f0/exports/fusion-dev-lidar-beam-reduction-s3",
}
FAMILIES = {
    family: tuple(f"{family}_s{severity}" for severity in (1, 2, 3))
    for family in ("lowlight", "dropout", "beam")
}
COMMON = (
    "scene_index", "class", "range", "lidar_points", "detector_score",
    "query_density", "box_log_volume", "velocity_magnitude", "residual_norm",
    "yaw_boundary",
)
METRICS = (
    "full_linear_nll", "full_geometry_nll", "xy_nll", "xyz_nll", "size_nll",
    "velocity_nll", "yaw_nll", "energy_score", "mahalanobis2",
    "normalized_abs_mean", "log_scale_product", "log_volume90", "yaw_width90",
    "coverage_50", "coverage_80", "coverage_90", "coverage_95",
    "yaw_coverage_50", "yaw_coverage_80", "yaw_coverage_90", "yaw_coverage_95",
)
YAW_INTERVAL_METRICS = tuple(name for name in METRICS if name.startswith("yaw_coverage_")) + ("yaw_width90",)
BASE_METRICS = tuple(name for name in METRICS if name not in YAW_INTERVAL_METRICS)


def variable(part, name: str, row: int):
    offsets = part[name + "_offsets"]
    return np.asarray(part[name][offsets[row] : offsets[row + 1]])


def write_npz(path: Path, arrays) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_info():
    import pickle

    authority = pilot.strict_load(ROOT / "probset3d/P0_DATA_SPLIT_AUTHORITY.json")["splits"]["dev"]
    with (ROOT / authority["info_path"]).open("rb") as stream:
        value = pickle.load(stream)
    infos = value["infos"] if isinstance(value, dict) else value
    return {item["token"]: item for item in infos}


def validate_inputs() -> None:
    if not pilot.strict_load(formal.PILOT)["passed"]:
        raise ValueError("formal G0 inputs are not authorized")
    for condition, root in CONDITIONS.items():
        manifests = list(f0.iter_export_manifests(root))
        if len(manifests) != 1 or manifests[0][1]["sample_count"] != 2808:
            raise ValueError(f"incomplete condition authority: {condition}")
        if condition.endswith(("s1", "s2")) and condition != "clean_s0":
            if not manifests[0][1].get("deterministic_sweep_contract_id"):
                raise ValueError(f"missing G0 sweep binding: {condition}")


def full_student_nll(target, mean, scale, degrees, indices):
    return formal.student_nll(target, mean, scale, degrees, indices)


def yaw_width(kind, parameter, level):
    parameter = np.asarray(parameter, dtype=np.float64)
    if kind == "wrapped_gaussian":
        return np.minimum(2.0 * math.pi, 2.0 * norm.ppf((1.0 + level) / 2.0) * parameter)
    lower = np.zeros_like(parameter)
    upper = np.full_like(parameter, math.pi)
    target = (1.0 + level) / 2.0
    for _ in range(42):
        middle = (lower + upper) / 2.0
        below = vonmises.cdf(middle, parameter) < target
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    return lower + upper


def energy_score(target, mean, scale, kind, degrees, draws):
    z1, z2, u1, u2 = draws
    if kind == "student_t":
        factor1 = np.sqrt(np.asarray(degrees)[:, None] / chi2.ppf(u1[None, :], np.asarray(degrees)[:, None]))
        factor2 = np.sqrt(np.asarray(degrees)[:, None] / chi2.ppf(u2[None, :], np.asarray(degrees)[:, None]))
    else:
        factor1 = factor2 = 1.0
    first = mean[:, None, :] + scale[:, None, :] * z1[None, :, :] * np.asarray(factor1)[..., None]
    second = mean[:, None, :] + scale[:, None, :] * z2[None, :, :] * np.asarray(factor2)[..., None]
    return np.linalg.norm(first - target[:, None, :], axis=-1).mean(-1) - 0.5 * np.linalg.norm(first - second, axis=-1).mean(-1)


def distribution_metrics(target, mean, scale, yaw_target, yaw_mean, kind, yaw_kind, yaw_parameter, degrees, draws):
    scale = np.asarray(scale, dtype=np.float64)
    delta = target - mean
    component = formal.gaussian_parts(target, mean, scale)
    if kind == "student_t":
        full_linear = full_student_nll(target, mean, scale, degrees, tuple(range(8)))
        xy = full_student_nll(target, mean, scale, degrees, (0, 1))
        xyz = full_student_nll(target, mean, scale, degrees, (0, 1, 2))
        size = full_student_nll(target, mean, scale, degrees, (3, 4, 5))
        velocity = full_student_nll(target, mean, scale, degrees, (6, 7))
    else:
        full_linear = component.sum(-1)
        xy = component[:, :2].sum(-1)
        xyz = component[:, :3].sum(-1)
        size = component[:, 3:6].sum(-1)
        velocity = component[:, 6:8].sum(-1)
    if yaw_kind == "wrapped_gaussian":
        yaw_nll = formal.wrapped_gaussian_nll(yaw_target, yaw_mean, yaw_parameter)
    else:
        yaw_nll = formal.von_mises_nll(yaw_target, yaw_mean, yaw_parameter)
    mahalanobis = ((delta / scale) ** 2).sum(-1)
    log_product = np.log(scale).sum(-1)
    result = {
        "full_linear_nll": full_linear,
        "full_geometry_nll": full_linear + yaw_nll,
        "xy_nll": xy,
        "xyz_nll": xyz,
        "size_nll": size,
        "velocity_nll": velocity,
        "yaw_nll": yaw_nll,
        "energy_score": energy_score(target, mean, scale, kind, degrees, draws),
        "mahalanobis2": mahalanobis,
        "normalized_abs_mean": np.abs(delta / scale).mean(-1),
        "log_scale_product": log_product,
    }
    unit_log_volume = 4.0 * math.log(math.pi) - gammaln(5.0)
    for level in LEVELS:
        suffix = str(int(level * 100))
        if kind == "student_t":
            threshold = 8.0 * f.ppf(level, 8, degrees)
        else:
            threshold = np.full(len(target), chi2.ppf(level, 8))
        result[f"coverage_{suffix}"] = (mahalanobis <= threshold).astype(np.float32)
        if level == 0.90:
            result["log_volume90"] = unit_log_volume + 4.0 * np.log(threshold) + log_product
    result["_yaw_residual"] = np.abs(f0.wrap_angle(yaw_target - yaw_mean))
    result["_yaw_parameter"] = np.asarray(yaw_parameter, dtype=np.float64)
    return result


def fixed_parameters(logits, boxes, reference):
    buckets = f0.bucket_id(logits.argmax(-1), boxes[:, :2])
    scale = np.empty((len(boxes), 8), dtype=np.float64)
    kappa = np.empty(len(boxes), dtype=np.float64)
    for query, bucket in enumerate(buckets):
        value = reference["buckets"][str(int(bucket))]
        scale[query] = np.sqrt(np.asarray(value["gaussian_variance"], dtype=np.float64))
        kappa[query] = float(value["yaw_kappa"])
    return scale, kappa


def evaluate(seed: int, device_index: int) -> None:
    validate_inputs()
    baseline = pilot.strict_load(formal.BASELINE)
    reference = pilot.strict_load(formal.REFERENCE)
    checkpoint = torch.load(formal.OUT / "checkpoints" / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    heads = {name: ProbabilisticQueryHead().to(device).eval() for name in ("M1", "M2")}
    for name in heads:
        heads[name].load_state_dict(checkpoint["models"][name])
    infos = load_info()
    rng = np.random.default_rng(seed)
    draws = (
        rng.standard_normal((MC_SAMPLES, 8)),
        rng.standard_normal((MC_SAMPLES, 8)),
        rng.uniform(1e-6, 1.0 - 1e-6, MC_SAMPLES),
        rng.uniform(1e-6, 1.0 - 1e-6, MC_SAMPLES),
    )
    arrays = {"meta__seed": np.asarray(seed, dtype=np.int64)}
    for condition, root in CONDITIONS.items():
        common = {name: [] for name in COMMON}
        storage = {model: {name: [] for name in BASE_METRICS} for model in MODELS}
        yaw_storage = {model: {name: [] for name in ("residual", "parameter")} for model in MODELS}
        tokens = []
        scene_index = 0
        for _, part in f0.iter_parts(root):
            for begin in range(0, len(part["sample_tokens"]), formal.BATCH_SIZE):
                stop = min(begin + formal.BATCH_SIZE, len(part["sample_tokens"]))
                features = torch.from_numpy(np.asarray(part["query_features"][begin:stop])).to(device)
                logits_tensor = torch.from_numpy(np.asarray(part["query_logits"][begin:stop])).to(device)
                boxes_tensor = torch.from_numpy(np.asarray(part["query_boxes"][begin:stop])).to(device)
                seed_score = torch.from_numpy(np.asarray(part["seed_probabilities"][begin:stop])).to(device)
                zero = torch.zeros(stop - begin, 200, device=device)
                outputs = {}
                for name, head in heads.items():
                    with torch.inference_mode():
                        value = head(features, logits_tensor, boxes_tensor, seed_score, zero)
                    outputs[name] = {key: tensor.detach().cpu().numpy() for key, tensor in value.items()}
                for local, row in enumerate(range(begin, stop)):
                    token = str(part["sample_tokens"][row])
                    tokens.append(token)
                    logits = np.asarray(part["query_logits"][row], dtype=np.float64)
                    boxes = np.asarray(part["query_boxes"][row], dtype=np.float64)
                    gt_boxes = variable(part, "gt_boxes", row).astype(np.float64)
                    gt_labels = variable(part, "gt_labels", row).astype(np.int64)
                    matched = np.asarray(part["native_matched_gt"][row], dtype=np.int64)
                    queries = np.flatnonzero(matched >= 0)
                    gt_index = matched[queries]
                    if len(queries):
                        source_indices, source_boxes, source_labels = detection_gt(infos[token])
                        if len(source_boxes) != len(gt_boxes) or not np.array_equal(source_labels, gt_labels) or np.max(np.abs(source_boxes - gt_boxes)) > 1e-4:
                            raise ValueError(f"GT/point-count alignment failed: {token}")
                        target = f0.state_from_boxes(gt_boxes)[gt_index]
                        mean = f0.state_from_boxes(boxes)[queries]
                        yaw_target = gt_boxes[gt_index, 6]
                        yaw_mean = boxes[queries, 6]
                        fixed_scale, fixed_kappa = fixed_parameters(logits, boxes, reference)
                        fixed = distribution_metrics(
                            target, mean, fixed_scale[queries], yaw_target, yaw_mean,
                            "gaussian", "von_mises", fixed_kappa[queries],
                            np.full(len(queries), np.nan), draws,
                        )
                        learned = {}
                        for name in ("M1", "M2"):
                            output = {key: value[local] for key, value in outputs[name].items()}
                            learned[name] = distribution_metrics(
                                target, mean, output["scale"][queries], yaw_target, yaw_mean,
                                "gaussian" if name == "M1" else "student_t",
                                "wrapped_gaussian" if name == "M1" else "von_mises",
                                output["yaw_scale"][queries] if name == "M1" else output["yaw_kappa"][queries],
                                np.full(len(queries), np.nan) if name == "M1" else output["degrees_of_freedom"][queries],
                                draws,
                            )
                        for model, value in (("M0", fixed), ("M5", fixed), ("M1", learned["M1"]), ("M2", learned["M2"])):
                            for metric in BASE_METRICS:
                                storage[model][metric].append(np.asarray(value[metric], dtype=np.float32))
                            yaw_storage[model]["residual"].append(np.asarray(value["_yaw_residual"], dtype=np.float64))
                            yaw_storage[model]["parameter"].append(np.asarray(value["_yaw_parameter"], dtype=np.float64))
                        predicted = logits.argmax(-1)
                        density = formal.density_score(np.asarray(part["query_features"][row]), predicted, baseline["density"])
                        common_values = {
                            "scene_index": np.full(len(queries), scene_index, dtype=np.int32),
                            "class": gt_labels[gt_index].astype(np.int16),
                            "range": np.linalg.norm(gt_boxes[gt_index, :2], axis=-1),
                            "lidar_points": np.asarray(infos[token]["num_lidar_pts"])[source_indices][gt_index],
                            "detector_score": logits[queries].max(-1),
                            "query_density": density[queries],
                            "box_log_volume": np.log(np.clip(boxes[queries, 3:6], 1e-8, None)).sum(-1),
                            "velocity_magnitude": np.linalg.norm(boxes[queries, 7:9], axis=-1),
                            "residual_norm": np.linalg.norm(target - mean, axis=-1),
                            "yaw_boundary": ((np.abs(f0.wrap_angle(yaw_target)) >= math.pi - 0.25) | (np.abs(f0.wrap_angle(yaw_mean)) >= math.pi - 0.25)).astype(np.float32),
                        }
                        for name, value in common_values.items():
                            common[name].append(np.asarray(value))
                    scene_index += 1
        if scene_index != 2808 or len(tokens) != 2808 or len(set(tokens)) != 2808:
            raise ValueError(f"scene conservation failed: {condition}")
        arrays[f"{condition}__sample_tokens"] = np.asarray(tokens)
        for name in COMMON:
            arrays[f"{condition}__common__{name}"] = np.concatenate(common[name])
        for model in MODELS:
            for metric in BASE_METRICS:
                value = np.concatenate(storage[model][metric])
                if not np.isfinite(value).all():
                    raise ValueError(f"non-finite metric: {condition}/{model}/{metric}")
                arrays[f"{condition}__{model}__{metric}"] = value
            residual = np.concatenate(yaw_storage[model]["residual"])
            parameter = np.concatenate(yaw_storage[model]["parameter"])
            yaw_kind = "wrapped_gaussian" if model == "M1" else "von_mises"
            for level in LEVELS:
                suffix = str(int(level * 100))
                width = yaw_width(yaw_kind, parameter, level)
                coverage = (residual <= width / 2.0).astype(np.float32)
                if not np.isfinite(width).all():
                    raise ValueError(f"non-finite yaw interval: {condition}/{model}/{suffix}")
                arrays[f"{condition}__{model}__yaw_coverage_{suffix}"] = coverage
                if level == 0.90:
                    arrays[f"{condition}__{model}__yaw_width90"] = width.astype(np.float32)
    path = OUT / "metrics" / f"seed_{seed}.npz"
    write_npz(path, arrays)
    print(json.dumps({"evaluated": True, "seed": seed, "output": str(path.relative_to(ROOT))}))


def scene_mean(value, scene, count=2808):
    total = np.bincount(scene, weights=np.asarray(value, dtype=np.float64), minlength=count)
    number = np.bincount(scene, minlength=count)
    result = np.full(count, np.nan, dtype=np.float64)
    result[number > 0] = total[number > 0] / number[number > 0]
    return result


def bootstrap(value, seed=20260730, repeats=10000):
    value = np.asarray(value, dtype=np.float64)
    value = value[np.isfinite(value)]
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        means[index] = value[rng.integers(0, len(value), len(value))].mean()
    return {"mean": float(value.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}


def risk_metrics(uncertainty, error):
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    order = np.argsort(uncertainty, kind="stable")
    oracle = np.argsort(error, kind="stable")
    fractions = np.linspace(0.05, 1.0, 96)

    def curve(indices):
        return np.asarray([error[indices[: max(1, int(len(error) * fraction))]].mean() for fraction in fractions])

    value = curve(order)
    optimal = curve(oracle)
    return {
        "spearman": float(spearmanr(uncertainty, error).statistic),
        "aurc": float(np.trapz(value, fractions)),
        "ause": float(np.trapz(value - optimal, fractions)),
        "risk_at_50": float(error[order[: max(1, len(error) // 2)]].mean()),
        "risk_at_80": float(error[order[: max(1, int(len(error) * 0.8))]].mean()),
    }


def nested_comparison(run, conditions):
    rows = []
    for condition in conditions:
        common = lambda name: np.asarray(run[f"{condition}__common__{name}"], dtype=np.float64)
        cls = common("class").astype(np.int64)
        numeric = np.column_stack(
            (
                common("detector_score"), common("query_density"), common("range"),
                np.log1p(common("lidar_points")), common("box_log_volume"),
                common("velocity_magnitude"),
            )
        )
        numeric = (numeric - numeric.mean(0)) / np.maximum(numeric.std(0), 1e-8)
        one_hot = np.eye(10)[cls][:, 1:]
        target = np.log1p(common("residual_norm"))
        uncertainty = np.asarray(run[f"{condition}__M2__log_volume90"], dtype=np.float64)
        rows.append((numeric, one_hot, target, uncertainty))
    numeric = np.concatenate([row[0] for row in rows])
    one_hot = np.concatenate([row[1] for row in rows])
    target = np.concatenate([row[2] for row in rows])
    uncertainty = np.concatenate([row[3] for row in rows])
    base = np.column_stack((np.ones(len(target)), numeric, one_hot))
    beta = np.linalg.lstsq(base, target, rcond=None)[0]
    residual_target = target - base @ beta
    gamma = np.linalg.lstsq(base, uncertainty, rcond=None)[0]
    residual_uncertainty = uncertainty - base @ gamma
    extended = np.column_stack((base, uncertainty))
    prediction_base = base @ beta
    prediction_extended = extended @ np.linalg.lstsq(extended, target, rcond=None)[0]
    total = ((target - target.mean()) ** 2).sum()
    return {
        "delta_r2": float((((target - prediction_base) ** 2).sum() - ((target - prediction_extended) ** 2).sum()) / total),
        "partial_correlation": float(np.corrcoef(residual_target, residual_uncertainty)[0, 1]),
        "target_count": len(target),
    }


def finalize() -> None:
    validate_inputs()
    runs = {seed: np.load(OUT / "metrics" / f"seed_{seed}.npz", allow_pickle=False) for seed in SEEDS}

    canonical_tokens = np.asarray(runs[SEEDS[0]]["clean_s0__sample_tokens"])
    for condition in CONDITIONS:
        for seed in SEEDS:
            if not np.array_equal(runs[seed][f"{condition}__sample_tokens"], canonical_tokens):
                raise ValueError(f"sample order mismatch: {condition}/{seed}")
            for name in COMMON:
                if not np.array_equal(
                    runs[seed][f"{condition}__common__{name}"],
                    runs[SEEDS[0]][f"{condition}__common__{name}"],
                    equal_nan=True,
                ):
                    raise ValueError(f"source metric mismatch across seeds: {condition}/{name}/{seed}")

    def target(seed, condition, model, metric):
        return np.asarray(runs[seed][f"{condition}__{model}__{metric}"], dtype=np.float64)

    def common(seed, condition, name):
        return np.asarray(runs[seed][f"{condition}__common__{name}"])

    def paired(condition, metric, baseline):
        values = []
        for seed in SEEDS:
            scene = common(seed, condition, "scene_index").astype(np.int64)
            values.append(scene_mean(target(seed, condition, baseline, metric) - target(seed, condition, "M2", metric), scene))
        return bootstrap(np.nanmean(np.stack(values), axis=0))

    proper = {
        condition: {
            metric: {baseline: paired(condition, metric, baseline) for baseline in ("M0", "M1", "M5")}
            for metric in ("full_linear_nll", "full_geometry_nll", "xy_nll", "xyz_nll", "size_nll", "velocity_nll", "yaw_nll", "energy_score")
        }
        for condition in CONDITIONS
    }

    def family_summary(family, metric, baseline, severities=(1, 2, 3)):
        seed_values = []
        for seed in SEEDS:
            per_severity = []
            for severity in severities:
                condition = f"{family}_s{severity}"
                scene = common(seed, condition, "scene_index").astype(np.int64)
                per_severity.append(scene_mean(target(seed, condition, baseline, metric) - target(seed, condition, "M2", metric), scene))
            seed_values.append(np.nanmean(np.stack(per_severity), axis=0))
        return bootstrap(np.nanmean(np.stack(seed_values), axis=0))

    family_full = {family: family_summary(family, "full_geometry_nll", "M1") for family in FAMILIES}
    family_no_s3 = {family: family_summary(family, "full_geometry_nll", "M1", (1, 2)) for family in FAMILIES}
    family_yaw = {family: family_summary(family, "yaw_nll", "M1") for family in FAMILIES}

    combined = []
    for seed in SEEDS:
        rows = []
        for condition in CONDITIONS:
            scene = common(seed, condition, "scene_index").astype(np.int64) + list(CONDITIONS).index(condition) * 2808
            improvement = target(seed, condition, "M1", "full_linear_nll") - target(seed, condition, "M2", "full_linear_nll")
            rows.append(
                {
                    "scene": scene,
                    "improvement": improvement,
                    "residual": common(seed, condition, "residual_norm"),
                    "range": common(seed, condition, "range"),
                    "points": common(seed, condition, "lidar_points"),
                    "degraded": np.full(len(improvement), condition != "clean_s0"),
                    "boundary": common(seed, condition, "yaw_boundary") > 0,
                    "class": common(seed, condition, "class").astype(np.int64),
                    "yaw_improvement": target(seed, condition, "M1", "yaw_nll") - target(seed, condition, "M2", "yaw_nll"),
                }
            )
        combined.append({name: np.concatenate([row[name] for row in rows]) for name in rows[0]})
    residual = combined[0]["residual"]
    q80, q99 = np.quantile(residual, (0.8, 0.99))

    def group_summary(field, mask):
        seed_scene = []
        for value in combined:
            selected = mask(value)
            seed_scene.append(scene_mean(value[field][selected], value["scene"][selected], 2808 * len(CONDITIONS)))
        return bootstrap(np.nanmean(np.stack(seed_scene), axis=0))

    tail = {
        "top_residual_quintile": group_summary("improvement", lambda value: value["residual"] >= q80),
        "central_80_percent": group_summary("improvement", lambda value: value["residual"] < q80),
        "far_over_40m": group_summary("improvement", lambda value: value["range"] > 40.0),
        "sparse_at_most_5_points": group_summary("improvement", lambda value: value["points"] <= 5),
        "degraded": group_summary("improvement", lambda value: value["degraded"]),
    }
    mean_improvement = np.mean(np.stack([value["improvement"] for value in combined]), axis=0)
    positive_total = np.maximum(mean_improvement, 0.0).sum()
    tail["top_1_percent_positive_gain_fraction"] = float(np.maximum(mean_improvement[residual >= q99], 0.0).sum() / max(positive_total, 1e-12))

    yaw_boundary = group_summary("yaw_improvement", lambda value: value["boundary"])
    class_yaw = {}
    for class_index in range(10):
        class_yaw[str(class_index)] = float(np.mean([value["yaw_improvement"][value["class"] == class_index].mean() for value in combined]))

    calibration = {}
    sharpness = {model: [] for model in ("M1", "M2", "M5")}
    risk = {}
    for condition in CONDITIONS:
        calibration[condition] = {}
        risk[condition] = {}
        for model in MODELS:
            model_rows = []
            risk_rows = []
            for seed in SEEDS:
                row = {
                    "coverage_gap": {
                        str(int(level * 100)): float(target(seed, condition, model, f"coverage_{int(level * 100)}").mean() - level)
                        for level in LEVELS
                    },
                    "yaw_coverage_gap": {
                        str(int(level * 100)): float(target(seed, condition, model, f"yaw_coverage_{int(level * 100)}").mean() - level)
                        for level in LEVELS
                    },
                    "median_log_volume90": float(np.median(target(seed, condition, model, "log_volume90"))),
                    "median_yaw_width90": float(np.median(target(seed, condition, model, "yaw_width90"))),
                    "normalized_abs_mean": float(target(seed, condition, model, "normalized_abs_mean").mean()),
                }
                model_rows.append(row)
                risk_rows.append(risk_metrics(target(seed, condition, model, "log_volume90"), common(seed, condition, "residual_norm")))
            calibration[condition][model] = {
                "coverage_gap": {key: float(np.mean([row["coverage_gap"][key] for row in model_rows])) for key in model_rows[0]["coverage_gap"]},
                "yaw_coverage_gap": {key: float(np.mean([row["yaw_coverage_gap"][key] for row in model_rows])) for key in model_rows[0]["yaw_coverage_gap"]},
                "median_log_volume90": float(np.mean([row["median_log_volume90"] for row in model_rows])),
                "median_yaw_width90": float(np.mean([row["median_yaw_width90"] for row in model_rows])),
                "normalized_abs_mean": float(np.mean([row["normalized_abs_mean"] for row in model_rows])),
            }
            if model == "M0":
                risk_rows = [
                    risk_metrics(-common(seed, condition, "detector_score"), common(seed, condition, "residual_norm"))
                    for seed in SEEDS
                ]
            elif model == "M5":
                risk_rows = [
                    risk_metrics(common(seed, condition, "query_density"), common(seed, condition, "residual_norm"))
                    for seed in SEEDS
                ]
            risk[condition][model] = {key: float(np.mean([row[key] for row in risk_rows])) for key in risk_rows[0]}

    equal_volume = {}
    scale_ratio = []
    for seed in SEEDS:
        equal_volume[str(seed)] = {}
        for model in ("M1", "M2", "M5"):
            radius = np.sqrt(target(seed, "clean_s0", model, "mahalanobis2"))
            q90 = np.quantile(radius, 0.90)
            log_product = target(seed, "clean_s0", model, "log_scale_product")
            equal_volume[str(seed)][model] = float(4.0 * math.log(math.pi) - gammaln(5.0) + 8.0 * math.log(q90) + np.median(log_product))
        m2_scale = np.exp(target(seed, "clean_s0", "M2", "log_scale_product") / 8.0)
        m1_scale = np.exp(target(seed, "clean_s0", "M1", "log_scale_product") / 8.0)
        scale_ratio.append(float(np.median(m2_scale) / np.median(m1_scale)))

    seed_pass = {}
    seed_direction = {}
    for seed in SEEDS:
        signs = []
        for metric in ("full_geometry_nll", "xy_nll", "yaw_nll"):
            for baseline in ("M1", "M5"):
                signs.append(float((target(seed, "clean_s0", baseline, metric) - target(seed, "clean_s0", "M2", metric)).mean()))
        seed_pass[str(seed)] = all(value > 0 for value in signs)
        seed_direction[str(seed)] = signs

    nontriviality = {str(seed): nested_comparison(runs[seed], CONDITIONS) for seed in SEEDS}
    gate1 = all(proper["clean_s0"][metric][baseline]["ci95"][0] > 0 for metric in ("full_geometry_nll", "xy_nll", "yaw_nll") for baseline in ("M1", "M5"))
    gate2 = sum(value["ci95"][0] > 0 for value in family_full.values()) >= 2 and all(value["ci95"][1] >= 0 for value in family_full.values()) and sum(value["ci95"][0] > 0 for value in family_no_s3.values()) >= 2 and all(value["ci95"][1] >= 0 for value in family_no_s3.values())
    gate3 = tail["top_residual_quintile"]["ci95"][0] > 0 and tail["far_over_40m"]["ci95"][0] > 0 and tail["sparse_at_most_5_points"]["ci95"][0] > 0 and tail["degraded"]["ci95"][0] > 0 and tail["top_residual_quintile"]["mean"] > tail["central_80_percent"]["mean"] and tail["top_1_percent_positive_gain_fraction"] <= 0.5
    gate4 = proper["clean_s0"]["yaw_nll"]["M1"]["ci95"][0] > 0 and sum(value["ci95"][0] > 0 for value in family_yaw.values()) >= 2 and yaw_boundary["ci95"][0] > 0 and sum(value > 0 for value in class_yaw.values()) >= 5
    gate5 = all(equal_volume[str(seed)]["M2"] <= min(equal_volume[str(seed)]["M1"], equal_volume[str(seed)]["M5"]) for seed in SEEDS) and max(scale_ratio) <= 1.10
    gate6 = all(all(value > 0 for value in signs) for signs in seed_direction.values()) and sum(seed_pass.values()) >= 2
    gates = {
        "G1_clean_geometry_quality": gate1,
        "G2_cross_degradation_stability": gate2,
        "G3_heavy_tail_value": gate3,
        "G4_periodic_yaw_value": gate4,
        "G5_no_sharpness_cheating": gate5,
        "G6_three_seed_stability": gate6,
    }
    passed = all(gates.values())
    result = {
        "calibration": calibration,
        "endpoint": "GO TO GEOMETRY-CONSISTENT PROBABILISTIC 3D DETECTION" if passed else "STOP STATIC PROBABILISTIC 3D GEOMETRY DIRECTION",
        "equal_coverage_log_volume90": equal_volume,
        "family_full_geometry": family_full,
        "family_full_geometry_without_s3": family_no_s3,
        "family_yaw": family_yaw,
        "gates": gates,
        "input_sha256": {
            "baseline": pilot.sha256(formal.BASELINE),
            "checkpoints": {
                str(seed): pilot.sha256(formal.OUT / "checkpoints" / f"seed_{seed}.pt") for seed in SEEDS
            },
            "condition_manifests": {
                condition: pilot.sha256(list(f0.iter_export_manifests(root))[0][0])
                for condition, root in CONDITIONS.items()
            },
            "evaluator": pilot.sha256(Path(__file__)),
            "reference_parameters": pilot.sha256(formal.REFERENCE),
        },
        "models": list(MODELS),
        "nontriviality": nontriviality,
        "official_val_read": False,
        "passed": passed,
        "proper_scores": proper,
        "risk_ranking": risk,
        "scale_ratio_M2_over_M1": scale_ratio,
        "seed_clean_directions": seed_direction,
        "seed_independent_pass": seed_pass,
        "tail_analysis": tail,
        "yaw_boundary": yaw_boundary,
        "yaw_positive_class_count": sum(value > 0 for value in class_yaw.values()),
        "yaw_per_class_improvement": class_yaw,
    }
    write_json(OUT / "result.json", result)
    for run in runs.values():
        run.close()
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    evaluator = sub.add_parser("evaluate")
    evaluator.add_argument("--seed", type=int, choices=SEEDS, required=True)
    evaluator.add_argument("--device", type=int, choices=(0, 1, 2, 3), required=True)
    sub.add_parser("finalize")
    args = parser.parse_args()
    evaluate(args.seed, args.device) if args.command == "evaluate" else finalize()


if __name__ == "__main__":
    main()
