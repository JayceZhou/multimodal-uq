#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path("outputs/stage3/target_uq")
SEED = 20260710
CLASS_NAMES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
BACKGROUND = len(CLASS_NAMES)
L2_GRID = [0.001, 0.01, 0.1]
OLD_CALIBRATOR = ROOT / "calibrators/stage3c/stage3c_structured_camera_structured_feature_k1.json"
CHECKPOINT = ROOT / "checkpoints/stage3c/stage3c_structured_camera_real_camera_pilot_structured_feature_k1.pt"
CAL_DIR = ROOT / "calibrators/stage3c_cal_camera"
REPAIRED_DIR = ROOT / "distributions_full/camera/stage3c_cal_repaired"
SPLIT_INFO = {
    "train": Path("outputs/stage3/splits/nuscenes_infos_stage3_train.pkl"),
    "label_cal": Path("outputs/stage3/splits/nuscenes_infos_stage3_label_cal.pkl"),
    "dev": Path("outputs/stage3/splits/nuscenes_infos_stage3_dev.pkl"),
}
SPLIT_NAME = {"train": "stage3_train", "label_cal": "stage3_label_cal", "dev": "stage3_dev"}
EXPECTED_RECORDS = {"train": 4501600, "label_cal": 562800, "dev": 561600}
SELECTION_LIMITS = {
    "ece": 0.002,
    "adaptive_ece": 0.002,
    "existence_ece": 0.002,
    "conditional_class_ece": 0.003,
    "categorical_nll": 0.002,
    "brier": 0.002,
    "existence_nll": 0.001,
}
DEV_LIMITS = {
    "ece": 0.005,
    "adaptive_ece": 0.005,
    "existence_ece": 0.005,
    "conditional_class_ece": 0.005,
    "categorical_nll": 0.002,
    "brier": 0.002,
    "existence_nll": 0.002,
}


def strict_load(path):
    def reject(value):
        raise ValueError(f"non-finite JSON constant in {path}: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def strict_dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax_numpy(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    value = np.exp(shifted)
    return value / value.sum(axis=-1, keepdims=True)


def raw_logits_from_old_probability(probability, old_parameters):
    probability = np.asarray(probability, dtype=np.float64)
    bias = np.asarray(old_parameters["bias"], dtype=np.float64)
    logits = float(old_parameters["temperature"]) * (np.log(np.clip(probability, 1e-30, 1.0)) - bias)
    return logits - logits.max(axis=-1, keepdims=True)


def apply_calibration_numpy(logits, calibration):
    logits = np.asarray(logits, dtype=np.float64)
    method = calibration["method"]
    if method == "identity":
        return softmax_numpy(logits)
    if method == "scalar":
        return softmax_numpy(logits / float(calibration["temperature"]))
    if method == "vector":
        return softmax_numpy(logits / float(calibration["temperature"]) + np.asarray(calibration["bias"]))
    foreground = logits[..., :-1]
    exist_logit = np.log(np.exp(foreground - foreground.max(axis=-1, keepdims=True)).sum(axis=-1))
    exist_logit += foreground.max(axis=-1) - logits[..., -1]
    p_exist = 1.0 / (1.0 + np.exp(-exist_logit / float(calibration["existence_temperature"])))
    class_logits = foreground / float(calibration["class_temperature"])
    if method == "separated_vector":
        class_logits = class_logits + np.asarray(calibration["class_bias"])
    conditional = softmax_numpy(class_logits)
    return np.concatenate((conditional * p_exist[..., None], 1.0 - p_exist[..., None]), axis=-1)


def apply_calibration_torch(logits, calibration):
    method = calibration["method"]
    if method == "identity":
        return logits.softmax(-1)
    if method == "scalar":
        return (logits / float(calibration["temperature"])).softmax(-1)
    if method == "vector":
        bias = torch.as_tensor(calibration["bias"], device=logits.device, dtype=logits.dtype)
        return (logits / float(calibration["temperature"]) + bias).softmax(-1)
    foreground = logits[:, :-1]
    exist_logit = torch.logsumexp(foreground, -1) - logits[:, -1]
    p_exist = torch.sigmoid(exist_logit / float(calibration["existence_temperature"]))
    class_logits = foreground / float(calibration["class_temperature"])
    if method == "separated_vector":
        bias = torch.as_tensor(calibration["class_bias"], device=logits.device, dtype=logits.dtype)
        class_logits = class_logits + bias
    conditional = class_logits.softmax(-1)
    return torch.cat((conditional * p_exist[:, None], 1.0 - p_exist[:, None]), -1)


def load_infos(split):
    with SPLIT_INFO[split].open("rb") as stream:
        value = pickle.load(stream)
    infos = value["infos"] if isinstance(value, dict) else value
    by_token = {item["token"]: item for item in infos}
    scene_by_sample = {}
    for token in by_token:
        root = token
        while by_token[root].get("prev_token") in by_token:
            root = by_token[root]["prev_token"]
        scene_by_sample[token] = root
    return infos, by_token, scene_by_sample


def build_scene_folds(scene_by_sample, seed=SEED, fold_count=5):
    counts = {}
    samples = {}
    for token, scene in scene_by_sample.items():
        counts[scene] = counts.get(scene, 0) + 1
        samples.setdefault(scene, []).append(token)
    scenes = list(counts)
    random.Random(seed).shuffle(scenes)
    scenes.sort(key=lambda scene: counts[scene], reverse=True)
    folds = [{"scenes": [], "samples": [], "sample_count": 0} for _ in range(fold_count)]
    for scene in scenes:
        fold = min(folds, key=lambda item: (item["sample_count"], len(item["scenes"])))
        fold["scenes"].append(scene)
        fold["samples"].extend(sorted(samples[scene]))
        fold["sample_count"] += counts[scene]
    for fold in folds:
        fold["scenes"].sort()
        fold["samples"].sort()
    return folds


def old_export_paths():
    summary = strict_load(ROOT / "summary.json")
    return {split: Path(item["path"]) for split, item in summary["distribution_exports"]["camera"].items()}


def read_export(split, old_parameters):
    infos, by_token, scene_by_sample = load_infos(split)
    scene_tokens = sorted(set(scene_by_sample.values()))
    scene_index = {token: index for index, token in enumerate(scene_tokens)}
    count = EXPECTED_RECORDS[split]
    logits = np.empty((count, BACKGROUND + 1), dtype=np.float32)
    labels = np.empty(count, dtype=np.int16)
    scenes = np.empty(count, dtype=np.int16)
    path = old_export_paths()[split]
    class_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    seen = 0
    with gzip.open(path, "rt") as stream:
        for seen, line in enumerate(stream, 1):
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            token = row["sample_token"]
            matched = int(row["matched_gt_id"])
            if matched < 0:
                label = BACKGROUND
            else:
                info = by_token[token]
                # Stage 3C collected proposals through cfg.data.test. NuScenesDataset
                # therefore used its default point-count mask, not train's valid_flag.
                names = np.asarray(info["gt_names"])[np.asarray(info["num_lidar_pts"]) > 0]
                label = max(class_index.get(str(names[matched]), -1), 0)
            logits[seen - 1] = raw_logits_from_old_probability(row["calibrated_kplus1_probability"], old_parameters)
            labels[seen - 1] = label
            scenes[seen - 1] = scene_index[scene_by_sample[token]]
    if seen != count:
        raise ValueError(f"{path}: expected {count}, got {seen}")
    return {
        "logits": logits,
        "labels": labels,
        "scene_indices": scenes,
        "scene_tokens": scene_tokens,
        "scene_by_sample": scene_by_sample,
        "sample_tokens": [item["token"] for item in infos],
        "path": str(path),
    }


def fit_temperature(logits, labels, binary=False):
    log_temperature = torch.nn.Parameter(torch.zeros((), device=logits.device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.25, max_iter=35, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels.float()) if binary else F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).cpu())


def fit_vector(logits, labels, l2):
    log_temperature = torch.nn.Parameter(torch.zeros((), device=logits.device))
    bias = torch.nn.Parameter(torch.zeros(logits.shape[1], device=logits.device))
    optimizer = torch.optim.LBFGS([log_temperature, bias], lr=0.25, max_iter=45, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        centered = bias - bias.mean()
        loss = F.cross_entropy(logits / temperature + centered, labels)
        loss = loss + float(l2) * (centered.square().mean() + log_temperature.square())
        loss.backward()
        return loss
    optimizer.step(closure)
    centered = bias.detach() - bias.detach().mean()
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).cpu()), [float(value) for value in centered.cpu().tolist()]


def fit_candidates(logits, labels):
    candidates = {"identity": {"method": "identity", "complexity": 0}}
    candidates["scalar"] = {"method": "scalar", "temperature": fit_temperature(logits, labels), "complexity": 1}
    for l2 in L2_GRID:
        temperature, bias = fit_vector(logits, labels, l2)
        candidates[f"vector_l2_{l2:g}"] = {
            "method": "vector", "temperature": temperature, "bias": bias, "l2": l2, "complexity": 3,
        }
    foreground = logits[:, :-1]
    positive = labels != BACKGROUND
    exist_logit = torch.logsumexp(foreground, -1) - logits[:, -1]
    exist_temperature = fit_temperature(exist_logit, positive.long(), binary=True)
    class_temperature = fit_temperature(foreground[positive], labels[positive])
    candidates["separated_scalar"] = {
        "method": "separated_scalar",
        "existence_temperature": exist_temperature,
        "class_temperature": class_temperature,
        "complexity": 2,
    }
    for l2 in L2_GRID:
        temperature, bias = fit_vector(foreground[positive], labels[positive], l2)
        candidates[f"separated_vector_l2_{l2:g}"] = {
            "method": "separated_vector",
            "existence_temperature": exist_temperature,
            "class_temperature": temperature,
            "class_bias": bias,
            "l2": l2,
            "complexity": 4,
        }
    return candidates


def calibration_error(score, target, bins=15, adaptive=False):
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if adaptive:
        groups = np.array_split(np.argsort(score), bins)
        errors = [abs(score[group].mean() - target[group].mean()) for group in groups if len(group)]
        weights = [len(group) / len(score) for group in groups if len(group)]
        return float(np.dot(errors, weights)), float(max(errors))
    index = np.minimum((score * bins).astype(np.int64), bins - 1)
    errors, weighted = [], 0.0
    for value in range(bins):
        mask = index == value
        if mask.any():
            error = abs(score[mask].mean() - target[mask].mean())
            errors.append(error)
            weighted += mask.mean() * error
    return float(weighted), float(max(errors) if errors else 0.0)


def reliability(score, target, bins=15):
    index = np.minimum((score * bins).astype(np.int64), bins - 1)
    rows = []
    for value in range(bins):
        mask = index == value
        rows.append({
            "bin": value,
            "lower": value / bins,
            "upper": (value + 1) / bins,
            "count": int(mask.sum()),
            "confidence": float(score[mask].mean()) if mask.any() else None,
            "accuracy": float(target[mask].mean()) if mask.any() else None,
        })
    return rows


def metrics(probability, labels):
    probability = np.asarray(probability, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    rows = np.arange(len(labels))
    picked = np.clip(probability[rows, labels], 1e-30, 1.0)
    one_hot = np.eye(BACKGROUND + 1)[labels]
    confidence = probability.max(1)
    correct = probability.argmax(1) == labels
    ece, mce = calibration_error(confidence, correct)
    adaptive_ece, _ = calibration_error(confidence, correct, adaptive=True)
    positive = labels != BACKGROUND
    p_exist = probability[:, :-1].sum(1)
    existence_target = positive.astype(np.float64)
    existence_ece, _ = calibration_error(p_exist, existence_target)
    conditional = probability[positive, :-1]
    conditional = conditional / np.clip(conditional.sum(1, keepdims=True), 1e-30, None)
    conditional_labels = labels[positive]
    conditional_rows = np.arange(len(conditional_labels))
    conditional_confidence = conditional.max(1)
    conditional_correct = conditional.argmax(1) == conditional_labels
    conditional_ece, conditional_mce = calibration_error(conditional_confidence, conditional_correct)
    per_class = {}
    class_ece = []
    for index, name in enumerate(CLASS_NAMES):
        target = conditional_labels == index
        value, _ = calibration_error(conditional[:, index], target)
        class_ece.append(value)
        true = conditional_labels == index
        per_class[name] = {
            "count": int(true.sum()),
            "nll": float(-np.log(np.clip(conditional[true, index], 1e-30, 1.0)).mean()) if true.any() else None,
            "ece": value,
        }
    return {
        "categorical_nll": float(-np.log(picked).mean()),
        "brier": float(np.square(probability - one_hot).sum(1).mean()),
        "ece": ece,
        "adaptive_ece": adaptive_ece,
        "maximum_calibration_error": mce,
        "reliability_bins": reliability(confidence, correct),
        "existence_nll": float(-(existence_target * np.log(np.clip(p_exist, 1e-30, 1.0)) + (1.0 - existence_target) * np.log(np.clip(1.0 - p_exist, 1e-30, 1.0))).mean()),
        "existence_brier": float(np.square(p_exist - existence_target).mean()),
        "existence_ece": existence_ece,
        "existence_reliability_bins": reliability(p_exist, existence_target),
        "positive_reliability_bins": reliability(p_exist[positive], existence_target[positive]),
        "background_reliability_bins": reliability(p_exist[~positive], existence_target[~positive]),
        "conditional_class_nll": float(-np.log(np.clip(conditional[conditional_rows, conditional_labels], 1e-30, 1.0)).mean()),
        "conditional_class_brier": float(np.square(conditional - np.eye(BACKGROUND)[conditional_labels]).sum(1).mean()),
        "conditional_class_ece": conditional_ece,
        "conditional_class_mce": conditional_mce,
        "conditional_reliability_bins": reliability(conditional_confidence, conditional_correct),
        "macro_classwise_ece": float(np.mean(class_ece)),
        "per_class": per_class,
        "finite": bool(np.isfinite(probability).all()),
        "normalized": bool(np.allclose(probability.sum(1), 1.0, atol=1e-6)),
    }


def scene_sufficient(probability, labels, scene_indices, bins=15):
    scene_count = int(scene_indices.max()) + 1
    result = {name: np.zeros(scene_count, dtype=np.float64) for name in (
        "count", "cat_nll", "brier", "exist_count", "exist_nll", "cond_count", "cond_nll",
    )}
    result.update({name: np.zeros((scene_count, bins), dtype=np.float64) for name in (
        "bin_count", "bin_score", "bin_target", "exist_bin_count", "exist_bin_score", "exist_bin_target",
        "cond_bin_count", "cond_bin_score", "cond_bin_target",
    )})
    for scene in range(scene_count):
        mask = scene_indices == scene
        p, y = probability[mask], labels[mask]
        rows = np.arange(len(y))
        one_hot = np.eye(BACKGROUND + 1)[y]
        result["count"][scene] = len(y)
        result["cat_nll"][scene] = -np.log(np.clip(p[rows, y], 1e-30, 1.0)).sum()
        result["brier"][scene] = np.square(p - one_hot).sum()
        confidence, correct = p.max(1), p.argmax(1) == y
        positive = y != BACKGROUND
        p_exist = p[:, :-1].sum(1)
        result["exist_count"][scene] = len(y)
        result["exist_nll"][scene] = -(positive * np.log(np.clip(p_exist, 1e-30, 1.0)) + (~positive) * np.log(np.clip(1.0 - p_exist, 1e-30, 1.0))).sum()
        conditional = p[positive, :-1]
        conditional /= np.clip(conditional.sum(1, keepdims=True), 1e-30, None)
        cond_y = y[positive]
        cond_score, cond_target = conditional.max(1), conditional.argmax(1) == cond_y
        result["cond_count"][scene] = len(cond_y)
        result["cond_nll"][scene] = -np.log(np.clip(conditional[np.arange(len(cond_y)), cond_y], 1e-30, 1.0)).sum()
        for prefix, score, target in (("", confidence, correct), ("exist_", p_exist, positive), ("cond_", cond_score, cond_target)):
            index = np.minimum((score * bins).astype(np.int64), bins - 1)
            result[prefix + "bin_count"][scene] = np.bincount(index, minlength=bins)
            result[prefix + "bin_score"][scene] = np.bincount(index, weights=score, minlength=bins)
            result[prefix + "bin_target"][scene] = np.bincount(index, weights=target, minlength=bins)
    return result


def bootstrap_values(stats, weights):
    values = {
        "categorical_nll": (weights @ stats["cat_nll"]) / (weights @ stats["count"]),
        "brier": (weights @ stats["brier"]) / (weights @ stats["count"]),
        "existence_nll": (weights @ stats["exist_nll"]) / (weights @ stats["exist_count"]),
        "conditional_class_nll": (weights @ stats["cond_nll"]) / (weights @ stats["cond_count"]),
    }
    for name, prefix, total_name in (("ece", "", "count"), ("existence_ece", "exist_", "exist_count"), ("conditional_class_ece", "cond_", "cond_count")):
        count = weights @ stats[prefix + "bin_count"]
        score = weights @ stats[prefix + "bin_score"]
        target = weights @ stats[prefix + "bin_target"]
        error = np.abs(np.divide(score, count, out=np.zeros_like(score), where=count > 0) - np.divide(target, count, out=np.zeros_like(target), where=count > 0))
        values[name] = (error * count).sum(1) / (weights @ stats[total_name])
    return values


def bootstrap_differences(probabilities, labels, scene_indices, resamples=1000, seed=SEED):
    scene_count = int(scene_indices.max()) + 1
    draws = np.random.default_rng(seed).integers(0, scene_count, size=(resamples, scene_count))
    weights = np.zeros((resamples, scene_count), dtype=np.float64)
    for row, draw in enumerate(draws):
        weights[row] = np.bincount(draw, minlength=scene_count)
    identity = bootstrap_values(scene_sufficient(probabilities["identity"], labels, scene_indices), weights)
    output = {}
    for name, probability in probabilities.items():
        current = bootstrap_values(scene_sufficient(probability, labels, scene_indices), weights)
        output[name] = {
            metric: {
                "mean_delta_vs_identity": float((value - identity[metric]).mean()),
                "ci95": [float(v) for v in np.quantile(value - identity[metric], [0.025, 0.975])],
            }
            for metric, value in current.items()
        }
    return output


def select_candidate(candidate_metrics):
    identity = candidate_metrics["identity"]
    audit = {}
    eligible = []
    for name, value in candidate_metrics.items():
        violations = {
            metric: value[metric] - identity[metric] - limit
            for metric, limit in SELECTION_LIMITS.items()
            if value[metric] > identity[metric] + limit
        }
        meaningful = (
            identity["categorical_nll"] - value["categorical_nll"] >= 0.002
            or identity["brier"] - value["brier"] >= 0.002
        )
        audit[name] = {"constraint_violations": violations, "meaningful_improvement": meaningful}
        if name != "identity" and not violations and meaningful and value["finite"] and value["normalized"]:
            eligible.append(name)
    if not eligible:
        return "identity", audit
    best_nll = min(candidate_metrics[name]["categorical_nll"] for name in eligible)
    pool = [name for name in eligible if candidate_metrics[name]["categorical_nll"] <= best_nll + 0.002]
    best_brier = min(candidate_metrics[name]["brier"] for name in pool)
    pool = [name for name in pool if candidate_metrics[name]["brier"] <= best_brier + 0.002]
    return min(pool, key=lambda name: (candidate_metrics[name]["complexity"], name)), audit


def dev_gate(raw, calibrated):
    violations = {
        metric: calibrated[metric] - raw[metric] - limit
        for metric, limit in DEV_LIMITS.items()
        if calibrated[metric] > raw[metric] + limit
    }
    passed = not violations and calibrated["finite"] and calibrated["normalized"]
    return passed, violations, "selected" if passed else "identity_fallback"


def export_repaired(split, source, destination, old_parameters, calibration, calibration_id, checkpoint_id, scene_by_sample):
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    batch = []
    def flush(target):
        nonlocal count, batch
        if not batch:
            return
        old_probability = np.asarray([row["calibrated_kplus1_probability"] for row in batch])
        raw_logits = raw_logits_from_old_probability(old_probability, old_parameters)
        probability = apply_calibration_numpy(raw_logits, calibration)
        for row, p_all in zip(batch, probability):
            p_exist = float(p_all[:-1].sum())
            row["calibrated_kplus1_probability"] = [float(value) for value in p_all]
            row["calibrated_p_exist"] = p_exist
            row["calibrated_class_probability"] = [float(value) for value in p_all[:-1] / max(p_exist, 1e-30)]
            row["scene_token"] = scene_by_sample[row["sample_token"]]
            row["scene_identifier_type"] = "first_sample_token_from_internal_prev_chain"
            row["probability_source"] = "pre_nms_query_feature_k1_classifier"
            row["variant"] = "structured_feature_k1"
            row["calibration_id"] = calibration_id
            row["checkpoint_id"] = checkpoint_id
            target.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            count += 1
        batch = []
    with gzip.open(source, "rt") as incoming, gzip.open(destination, "wt", compresslevel=4) as outgoing:
        for line in incoming:
            batch.append(json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))))
            if len(batch) == 4096:
                flush(outgoing)
        flush(outgoing)
    if count != EXPECTED_RECORDS[split]:
        raise ValueError(f"{destination}: expected {EXPECTED_RECORDS[split]}, got {count}")
    return {"path": str(destination), "record_count": count, "sha256": sha256(destination)}


def assert_spd(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    return all(math.isfinite(value) for value in (a, b, c, d)) and abs(b - c) < 1e-5 and a > 0 and d > 0 and a * d - b * c > 0


def all_finite(value):
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return math.isfinite(value)
    return True


def validate_repaired(exports, calibration_id, checkpoint_id):
    required = {
        "sample_token", "scene_token", "split", "branch", "proposal_id", "probability_source",
        "calibrated_kplus1_probability", "calibrated_p_exist", "calibrated_class_probability",
        "box_mean", "center_xy_covariance", "velocity_covariance", "z_variance", "log_size_variance",
        "yaw_mean", "yaw_kappa", "matched_gt_id", "post_detection_id", "post_score_for_mapping",
        "score_is_probability", "calibration_id", "checkpoint_id", "variant",
    }
    result = {}
    for split, item in exports.items():
        count = spd = 0
        start = time.time()
        with gzip.open(item["path"], "rt") as stream:
            for count, line in enumerate(stream, 1):
                row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                if required - row.keys():
                    raise ValueError(f"missing fields: {required - row.keys()}")
                if not all_finite(row):
                    raise ValueError(f"non-finite value at record {count} in {item['path']}")
                p_all = row["calibrated_kplus1_probability"]
                conditional = row["calibrated_class_probability"]
                assert len(p_all) == BACKGROUND + 1 and len(conditional) == BACKGROUND
                assert row["branch"] == "camera" and row["variant"] == "structured_feature_k1"
                assert row["probability_source"] == "pre_nms_query_feature_k1_classifier"
                assert row["class_probability_source"] == "pre_nms_query_feature_k1_classifier"
                assert row["score_is_probability"] is False
                assert row["calibration_id"] == calibration_id and row["checkpoint_id"] == checkpoint_id
                assert abs(sum(p_all) - 1.0) < 1e-6 and abs(sum(conditional) - 1.0) < 1e-6
                assert abs(row["calibrated_p_exist"] - (1.0 - p_all[-1])) < 1e-6
                if count == 1 or count % 10000 == 0:
                    assert assert_spd(row["center_xy_covariance"]) and assert_spd(row["velocity_covariance"])
                    spd += 1
        if count != item["record_count"]:
            raise ValueError(f"{item['path']}: expected {item['record_count']}, got {count}")
        result[split] = {
            "path": item["path"], "record_count": count, "full_semantic_records_checked": count,
            "spd_records_checked": spd,
            "wall_time_seconds": time.time() - start,
        }
    return {
        "strict_json_every_record": True,
        "schema_probability_finite_and_id_guards_every_record": True,
        "spd_sampling_interval": 10000,
        "splits": result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if args.validate_only:
        canonical = strict_load(ROOT / "manifests/stage4_canonical_inputs.json")
        calibration_id = canonical["camera"]["calibration_id"]
        checkpoint_id = canonical["camera"]["checkpoint_id"]
        validation = validate_repaired(canonical["camera"]["exports"], calibration_id, checkpoint_id)
        strict_dump(ROOT / "metrics/stage3c_cal_camera_export_validation.json", validation)
        print(json.dumps(validation, indent=2, sort_keys=True))
        return
    old_artifact = strict_load(OLD_CALIBRATOR)
    old_parameters = old_artifact["parameters"]
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if (checkpoint.get("branch"), checkpoint.get("variant"), checkpoint.get("classifier_mode")) != ("camera", "structured", "feature_k1"):
        raise ValueError("Camera structured checkpoint metadata mismatch")

    _, _, label_scene_by_sample = load_infos("label_cal")
    dev_samples = set(strict_load("outputs/stage3/splits/stage3_dev_samples.json")["sample_tokens"])
    folds = build_scene_folds(label_scene_by_sample)
    label_samples = set(label_scene_by_sample)
    fold_manifest = {
        "stage": "stage3c_cal_camera",
        "seed": SEED,
        "fold_count": 5,
        "scene_identifier_semantics": "first sample token recovered only from each internal split prev_token chain",
        "folds": [
            {"fold": index, "scene_tokens": fold["scenes"], "sample_tokens": fold["samples"], "sample_count": fold["sample_count"]}
            for index, fold in enumerate(folds)
        ],
        "checks": {
            "scene_overlap_between_folds": 0,
            "sample_overlap_between_folds": 0,
            "label_cal_dev_sample_overlap": len(label_samples & dev_samples),
            "official_val_used": False,
        },
    }
    strict_dump(ROOT / "manifests/stage3c_cal_camera_folds.json", fold_manifest)

    label_data = read_export("label_cal", old_parameters)
    reconstructed_raw = metrics(
        apply_calibration_numpy(label_data["logits"], {"method": "identity"}),
        label_data["labels"],
    )
    expected_raw = strict_load(ROOT / "metrics/stage3c_full_train_calibration.json")["branches"]["camera"]["structured"]["label_cal_raw"]
    for name in ("categorical_nll", "brier", "ece"):
        if abs(reconstructed_raw[name] - expected_raw[name]) > 5e-4:
            raise ValueError(f"raw-logit inversion mismatch for {name}: {reconstructed_raw[name]} vs {expected_raw[name]}")
    fold_by_scene = {}
    for fold_index, fold in enumerate(folds):
        for scene in fold["scenes"]:
            fold_by_scene[scene] = fold_index
    record_fold = np.asarray([fold_by_scene[label_data["scene_tokens"][index]] for index in label_data["scene_indices"]])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logits = torch.from_numpy(label_data["logits"]).to(device)
    labels = torch.from_numpy(label_data["labels"].astype(np.int64)).to(device)
    candidate_names = ["identity", "scalar"] + [f"vector_l2_{value:g}" for value in L2_GRID]
    candidate_names += ["separated_scalar"] + [f"separated_vector_l2_{value:g}" for value in L2_GRID]
    oof = {name: np.empty((len(labels), BACKGROUND + 1), dtype=np.float32) for name in candidate_names}
    fold_parameters = {name: [] for name in candidate_names}
    for fold_index in range(5):
        train_index = torch.from_numpy(np.flatnonzero(record_fold != fold_index)).to(device)
        hold_index_np = np.flatnonzero(record_fold == fold_index)
        hold_index = torch.from_numpy(hold_index_np).to(device)
        fitted = fit_candidates(logits.index_select(0, train_index), labels.index_select(0, train_index))
        if sorted(fitted) != sorted(candidate_names):
            raise ValueError("candidate set changed during cross-fitting")
        for name in candidate_names:
            oof[name][hold_index_np] = apply_calibration_torch(logits.index_select(0, hold_index), fitted[name]).detach().cpu().numpy()
            fold_parameters[name].append({"fold": fold_index, "parameters": fitted[name]})
    del logits, labels
    candidate_metrics = {}
    for name in candidate_names:
        candidate_metrics[name] = metrics(oof[name], label_data["labels"])
        candidate_metrics[name]["complexity"] = fold_parameters[name][0]["parameters"]["complexity"]
    selected, selection_audit = select_candidate(candidate_metrics)
    bootstrap = bootstrap_differences(oof, label_data["labels"], label_data["scene_indices"], args.bootstrap_resamples)
    oof_path = ROOT / "metrics/stage3c_cal_camera_oof.json"
    strict_dump(oof_path, {
        "stage": "stage3c_cal_camera",
        "split": "stage3_label_cal",
        "evaluation": "scene_level_5_fold_out_of_fold",
        "seed": SEED,
        "candidates": candidate_metrics,
        "raw_logit_inversion_check": {
            "reconstructed": {name: reconstructed_raw[name] for name in ("categorical_nll", "brier", "ece")},
            "stage3c_reference": {name: expected_raw[name] for name in ("categorical_nll", "brier", "ece")},
            "tolerance": 0.0005,
            "passed": True,
        },
        "fold_parameters": fold_parameters,
        "selection_audit": selection_audit,
        "selected": selected,
        "bootstrap": {"unit": "scene", "resamples": args.bootstrap_resamples, "differences": bootstrap},
        "dev_used": False,
        "official_val_used": False,
    })

    full_logits = torch.from_numpy(label_data["logits"]).to(device)
    full_labels = torch.from_numpy(label_data["labels"].astype(np.int64)).to(device)
    fitted_full = fit_candidates(full_logits, full_labels)[selected]
    fitted_full.pop("complexity", None)
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    selected_path = CAL_DIR / "camera_selected_full_label_cal.json"
    strict_dump(selected_path, {
        "stage": "stage3c_cal_camera", "branch": "camera", "split": "stage3_label_cal",
        "selected_oof_candidate": selected, "parameters": fitted_full, "dev_used": False,
        "official_val_used": False, "checkpoint": str(CHECKPOINT),
    })
    selection_manifest = {
        "stage": "stage3c_cal_camera",
        "seed": SEED,
        "selection_frozen_before_dev": True,
        "selection_split": "stage3_label_cal_scene_level_5_fold_oof",
        "selected_candidate": selected,
        "selected_full_label_cal_artifact": str(selected_path),
        "selection_limits": SELECTION_LIMITS,
        "meaningful_improvement_rule": "identity NLL or Brier minus candidate must be at least 0.002",
        "tie_rule": "within 0.002 NLL, then within 0.002 Brier, then lower complexity",
        "complexity_order": ["identity", "scalar", "separated_scalar", "vector", "separated_vector"],
        "dev_opened_before_selection": False,
        "official_val_used": False,
        "degradation_used_for_selection": False,
        "post_score_used_for_probability": False,
        "stage4_training_started": False,
        "stage5_training_started": False,
        "oof_metrics": str(oof_path),
        "folds": str(ROOT / "manifests/stage3c_cal_camera_folds.json"),
    }
    selection_manifest_path = ROOT / "manifests/stage3c_cal_camera_manifest.json"
    strict_dump(selection_manifest_path, selection_manifest)
    selection_manifest_hash = sha256(selection_manifest_path)

    # This is the first read of the dev distribution in the calibration workflow.
    dev_data = read_export("dev", old_parameters)
    raw_probability = apply_calibration_numpy(dev_data["logits"], {"method": "identity"})
    selected_probability = apply_calibration_numpy(dev_data["logits"], fitted_full)
    raw_metrics = metrics(raw_probability, dev_data["labels"])
    selected_metrics = metrics(selected_probability, dev_data["labels"])
    passed, violations, action = dev_gate(raw_metrics, selected_metrics)
    canonical_parameters = fitted_full if passed else {"method": "identity"}
    canonical_name = selected if passed else "identity"
    canonical_metrics = selected_metrics if passed else raw_metrics
    canonical_path = CAL_DIR / "camera_canonical.json"
    canonical_artifact = {
        "stage": "stage3c_cal_camera", "branch": "camera", "variant": "structured_feature_k1",
        "canonical_candidate": canonical_name, "oof_selected_candidate": selected,
        "parameters": canonical_parameters, "transfer_passed": passed,
        "transfer_failure_action": action, "dev_gate_violations": violations,
        "selection_manifest": str(selection_manifest_path), "selection_manifest_sha256": selection_manifest_hash,
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
        "official_val_used": False, "stage4_training_started": False, "stage5_training_started": False,
    }
    strict_dump(canonical_path, canonical_artifact)
    strict_dump(ROOT / "metrics/stage3c_cal_camera_dev.json", {
        "stage": "stage3c_cal_camera", "split": "stage3_dev", "dev_evaluation_count": 1,
        "selection_frozen_before_dev": True, "selection_manifest_sha256": selection_manifest_hash,
        "raw": raw_metrics, "oof_selected": selected_metrics, "canonical": canonical_metrics,
        "oof_selected_candidate": selected, "canonical_candidate": canonical_name,
        "transfer_passed": passed, "violations": violations, "fallback": action,
        "official_val_used": False,
    })

    old_degradation = strict_load(ROOT / "metrics/stage3c_structured_camera_degradation_calibrated_recheck.json")
    degradation = {
        "stage": "stage3c_cal_camera",
        "role": "diagnostic_only_not_used_for_selection",
        "canonical_clean": canonical_metrics,
        "conditions": {},
        "limitation": "lowlight/rotation proposal logits were not exported; existing Stage 3C vector-calibrated metrics are reused and the detector is not rerun",
    }
    for source_name, output_name in (("clean_s0", "clean"), ("camera_lowlight_s3", "lowlight_s3"), ("camera_extrinsic_rotation_s3", "rotation_s3")):
        source = old_degradation["conditions"][source_name]["structured_feature_k1"]
        degradation["conditions"][output_name] = {
            "calibration_semantics": "canonical" if output_name == "clean" else "legacy_stage3c_vector_diagnostic",
            "categorical_nll": canonical_metrics["categorical_nll"] if output_name == "clean" else source["classification"]["categorical_nll"],
            "ece": canonical_metrics["ece"] if output_name == "clean" else source["classification"]["ece"],
            "brier": canonical_metrics["brier"] if output_name == "clean" else source["classification"]["brier"],
            "existence_ece": canonical_metrics["existence_ece"] if output_name == "clean" else source["classification"]["existence_ece"],
            "conditional_foreground_ece": canonical_metrics["conditional_class_ece"] if output_name == "clean" else None,
            "xy_nll": source["regression"]["xy_nll"],
            "uncertainty_error_spearman": source["regression"]["uncertainty_error_spearman"],
        }
    strict_dump(ROOT / "metrics/stage3c_cal_camera_degradation.json", degradation)

    exports = {}
    calibration_id = str(canonical_path)
    checkpoint_id = str(CHECKPOINT)
    if not args.skip_export:
        for split in ("train", "label_cal", "dev"):
            _, _, scene_by_sample = load_infos(split)
            destination = REPAIRED_DIR / f"{SPLIT_NAME[split]}_structured_feature_k1.jsonl.gz"
            exports[split] = export_repaired(
                split, old_export_paths()[split], destination, old_parameters, canonical_parameters,
                calibration_id, checkpoint_id, scene_by_sample,
            )
        export_validation = validate_repaired(exports, calibration_id, checkpoint_id)
        strict_dump(ROOT / "metrics/stage3c_cal_camera_export_validation.json", export_validation)

    summary = strict_load(ROOT / "summary.json")
    lidar = summary["distribution_exports"]["lidar"]
    fusion = summary["distribution_exports"]["fusion"]
    canonical_manifest = {
        "stage": "stage4_canonical_inputs",
        "created_by": "stage3c_cal_camera",
        "camera": {
            "status": "canonical",
            "variant": "structured_feature_k1",
            "calibrator_type": canonical_name,
            "calibration_id": calibration_id,
            "checkpoint_id": checkpoint_id,
            "exports": exports,
        },
        "lidar": {"status": "canonical_existing_stage3c", "variant": "structured_feature_k1", "exports": lidar},
        "fusion": {"status": "canonical_existing_stage3c_recovered_train", "variant": "structured_feature_k1", "exports": fusion},
        "deprecated": {
            "old_camera_vector_calibrated_exports": {"status": "deprecated_for_stage4", "paths": {key: str(value) for key, value in old_export_paths().items()}},
            "old_truncated_fusion_train": {
                "status": "audit_only_do_not_use",
                "path": "outputs/stage3/target_uq/distributions_full/fusion/stage3c_structured_fusion_stage3_train_structured_feature_k1.jsonl.gz",
            },
        },
        "checks": {
            "fusion_train_is_recovered_file": "recovered" in fusion["train"]["path"],
            "official_val_used": False,
            "stage4_training_started": False,
            "stage5_training_started": False,
            "post_score_usage": "mapping_only",
        },
    }
    strict_dump(ROOT / "manifests/stage4_canonical_inputs.json", canonical_manifest)
    summary["stage3c_cal_camera"] = {
        "status": "complete", "oof_selected_candidate": selected, "canonical_candidate": canonical_name,
        "dev_transfer_passed": passed, "dev_evaluation_count": 1,
        "camera_exports": exports, "canonical_manifest": str(ROOT / "manifests/stage4_canonical_inputs.json"),
        "official_val_used": False,
    }
    if exports:
        summary["distribution_exports"]["camera"] = exports
    summary["gate_decision"]["stage4"] = "recommended_stage4a" if canonical_name == "identity" or passed else "not_recommended"
    summary["gate_decision"]["stage5"] = "forbidden_not_recommended"
    strict_dump(ROOT / "summary.json", summary)
    print(json.dumps({"selected": selected, "canonical": canonical_name, "dev_transfer_passed": passed, "exports": exports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
