#!/usr/bin/env python3
"""Minimal M0/M1/M2/M3/M5 F1-V3 train, evaluate, and decide workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit, gammaln, i0e, logsumexp

from probset3d.f1.implementation.probabilistic_query import (
    CLASS_COUNT,
    ProbabilisticQueryHead,
    f1_loss,
)
from tools import probset3d_f0_score as f0
from tools import probset3d_v3_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/probset3d_v3/f1_formal"
BASELINE = OUT / "baseline.json"
PILOT = ROOT / "outputs/probset3d_v3/f1_pilot/pilot.json"
REFERENCE = ROOT / "outputs/probset3d_v2/f0/reference_parameters.json"
DETECTOR_CHECKPOINT = ROOT / "pretrained/bevfusion-det.pth"
MODELS = ("M0", "M1", "M2", "M3", "M5")
LEARNED = ("M1", "M2", "M3")
SEEDS = (20260731, 20260801, 20260802)
EPOCHS = 20
BATCH_SIZE = 8
K_BEST = 10
EPS = 1e-9
CONDITIONS = {
    "clean": ROOT / "outputs/probset3d_v2/f0/exports/fusion-dev-clean-s0",
    "lowlight": ROOT / "outputs/probset3d_v2/f0/exports/fusion-dev-camera-lowlight-s3",
    "dropout": ROOT / "outputs/probset3d_v2/f0/exports/fusion-dev-lidar-dropout-s3",
    "beam": ROOT / "outputs/probset3d_v2/f0/exports/fusion-dev-lidar-beam-reduction-s3",
}
METRICS = (
    "mb_nll",
    "existence_nll",
    "existence_brier",
    "xy_nll",
    "size_nll",
    "velocity_nll",
    "yaw_nll",
    "duplicate_spread",
    "median_scale",
    "matched_count",
    "duplicate_valid",
)


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


def write_torch(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def write_npz(path: Path, arrays) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def ensure_authorized() -> None:
    value = pilot.strict_load(PILOT)
    if not value["passed"] or not value["formal_f1_scientific_test_authorized"]:
        raise ValueError("F1-V3 legality pilot did not authorize formal science")


def variable(part, name: str, row: int):
    offsets = part[name + "_offsets"]
    return np.asarray(part[name][offsets[row] : offsets[row + 1]])


def softmax_temperature(logits, temperature):
    scaled = np.asarray(logits, dtype=np.float64) / np.asarray(temperature)[None, :]
    scaled -= scaled.max(axis=-1, keepdims=True)
    value = np.exp(scaled)
    return value / value.sum(axis=-1, keepdims=True)


def logistic_fit(design, target):
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    parameter = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(100):
        probability = expit(design @ parameter)
        weight = np.clip(probability * (1.0 - probability), 1e-8, None)
        gradient = design.T @ (probability - target) + 1e-6 * parameter
        hessian = (design.T * weight) @ design + 1e-6 * np.eye(design.shape[1])
        step = np.linalg.solve(hessian, gradient)
        parameter -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return parameter


def fit_class_temperature(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    def objective(log_temperature):
        probability = softmax_temperature(logits, np.exp(log_temperature))
        return float(-np.log(np.clip(probability[np.arange(len(labels)), labels], EPS, 1.0)).mean())

    result = minimize(objective, np.zeros(CLASS_COUNT), method="L-BFGS-B", bounds=[(-3.0, 3.0)] * CLASS_COUNT)
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("per-class temperature fit failed")
    return np.exp(result.x)


def density_raw(features, labels, density):
    means = np.asarray(density["means"], dtype=np.float64)[labels]
    variances = np.asarray(density["variances"], dtype=np.float64)[labels]
    return 0.5 * (
        ((np.asarray(features, dtype=np.float64) - means) ** 2 / variances).sum(-1)
        + np.log(variances).sum(-1)
        + features.shape[-1] * math.log(2.0 * math.pi)
    )


def density_score(features, labels, density):
    raw = density_raw(features, labels, density)
    lower = np.asarray(density["lower"], dtype=np.float64)[labels]
    upper = np.asarray(density["upper"], dtype=np.float64)[labels]
    return np.clip((raw - lower) / np.maximum(upper - lower, 1e-6), 0.0, 1.0)


def prepare() -> None:
    ensure_authorized()
    counts = np.zeros(CLASS_COUNT, dtype=np.int64)
    sums = np.zeros((CLASS_COUNT, 128), dtype=np.float64)
    squares = np.zeros((CLASS_COUNT, 128), dtype=np.float64)
    for path in pilot.authority_parts("train"):
        with np.load(path, allow_pickle=False) as part:
            for row in range(len(part["sample_tokens"])):
                matched = np.asarray(part["native_matched_gt"][row], dtype=np.int64)
                positive = np.flatnonzero(matched >= 0)
                if not len(positive):
                    continue
                labels = variable(part, "gt_labels", row)[matched[positive]].astype(np.int64)
                features = np.asarray(part["query_features"][row, positive], dtype=np.float64)
                for class_index in range(CLASS_COUNT):
                    selected = features[labels == class_index]
                    if len(selected):
                        counts[class_index] += len(selected)
                        sums[class_index] += selected.sum(0)
                        squares[class_index] += (selected * selected).sum(0)
    if np.any(counts < 2):
        raise ValueError("density baseline has an empty class")
    means = sums / counts[:, None]
    variances = np.maximum(squares / counts[:, None] - means * means, 1e-4)
    density = {"means": means.tolist(), "variances": variances.tolist()}
    raw_by_class = [[] for _ in range(CLASS_COUNT)]
    for path in pilot.authority_parts("train"):
        with np.load(path, allow_pickle=False) as part:
            for row in range(len(part["sample_tokens"])):
                matched = np.asarray(part["native_matched_gt"][row], dtype=np.int64)
                positive = np.flatnonzero(matched >= 0)
                if not len(positive):
                    continue
                labels = variable(part, "gt_labels", row)[matched[positive]].astype(np.int64)
                raw = density_raw(np.asarray(part["query_features"][row, positive]), labels, density)
                for class_index in range(CLASS_COUNT):
                    selected = raw[labels == class_index]
                    if len(selected):
                        raw_by_class[class_index].append(selected)
    density["lower"] = [float(np.quantile(np.concatenate(value), 0.001)) for value in raw_by_class]
    density["upper"] = [float(np.quantile(np.concatenate(value), 0.999)) for value in raw_by_class]
    density["counts"] = counts.tolist()

    scores, targets, all_density = [], [], []
    class_logits, class_labels = [], []
    for path in pilot.authority_parts("label_cal"):
        with np.load(path, allow_pickle=False) as part:
            for row in range(len(part["sample_tokens"])):
                logits = np.asarray(part["class_logits"][row])
                matched = np.asarray(part["native_matched_gt"][row], dtype=np.int64)
                target = matched >= 0
                predicted = logits.argmax(-1)
                scores.append(logits.max(-1))
                targets.append(target.astype(np.float64))
                all_density.append(density_score(np.asarray(part["query_features"][row]), predicted, density))
                positive = np.flatnonzero(target)
                if len(positive):
                    class_logits.append(logits[positive])
                    class_labels.append(variable(part, "gt_labels", row)[matched[positive]])
    score = np.concatenate(scores)
    target = np.concatenate(targets)
    query_density = np.concatenate(all_density)
    temperature = fit_class_temperature(np.concatenate(class_logits), np.concatenate(class_labels))
    m0 = logistic_fit(np.column_stack((score, np.ones_like(score))), target)
    m5 = logistic_fit(np.column_stack((score, query_density, np.ones_like(score))), target)

    checkpoint = torch.load(DETECTOR_CHECKPOINT, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    detector_parameters = int(sum(value.numel() for value in state.values() if torch.is_tensor(value)))
    del checkpoint, state
    clean_manifest = next(f0.iter_export_manifests(CONDITIONS["clean"]))[1]
    result = {
        "class_temperature": temperature.tolist(),
        "density": density,
        "detector_average_seconds_per_scene": float(clean_manifest["wall_time_seconds"] / clean_manifest["sample_count"]),
        "detector_parameter_count": detector_parameters,
        "m0_existence": m0.tolist(),
        "m5_existence": m5.tolist(),
        "models": list(MODELS),
        "pilot_sha256": pilot.sha256(PILOT),
        "reference_parameters": str(REFERENCE.relative_to(ROOT)),
        "reference_parameters_sha256": pilot.sha256(REFERENCE),
        "schema_version": "probset3d_f1_v3_formal_baseline_v1",
    }
    write_json(BASELINE, result)
    print(json.dumps({"prepared": True, "baseline": str(BASELINE.relative_to(ROOT))}))


def training_batches(device):
    for path in pilot.authority_parts("train"):
        with np.load(path, allow_pickle=False) as part:
            total = len(part["sample_tokens"])
            for begin in range(0, total, BATCH_SIZE):
                rows = []
                for row in range(begin, min(begin + BATCH_SIZE, total)):
                    rows.append(
                        {
                            "query_features": np.asarray(part["query_features"][row]),
                            "class_logits": np.asarray(part["class_logits"][row]),
                            "decoded_box_mean": np.asarray(part["decoded_box_mean"][row]),
                            "seed_score": np.asarray(part["seed_score"][row]),
                            "native_matched_gt": np.asarray(part["native_matched_gt"][row]),
                            "gt_boxes": variable(part, "gt_boxes", row),
                            "gt_labels": variable(part, "gt_labels", row),
                            "duplicate_edge_query_id": variable(part, "duplicate_edge_query_id", row).astype(np.int64),
                            "duplicate_edge_gt_index": variable(part, "duplicate_edge_gt_index", row).astype(np.int64),
                        }
                    )
                yield pilot.make_batch(rows, device)


def train(seed: int, device_index: int) -> None:
    ensure_authorized()
    if seed not in SEEDS:
        raise ValueError(seed)
    baseline = pilot.strict_load(BASELINE)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    heads = {name: ProbabilisticQueryHead().to(device) for name in LEARNED}
    optimizers = {
        name: torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        for name, head in heads.items()
    }
    temperature = torch.tensor(baseline["class_temperature"], dtype=torch.float32, device=device)
    checkpoint_before = pilot.sha256(DETECTOR_CHECKPOINT)
    seen = 0
    started = time.time()
    for _epoch in range(EPOCHS):
        epoch_seen = 0
        for batch in training_batches(device):
            inputs = (
                batch["query_feature"], batch["class_logits"], batch["decoded_boxes"],
                batch["seed_score"], batch["query_density"],
            )
            for name, family, neighborhood in (
                ("M1", "gaussian", None),
                ("M2", "student_t", None),
                ("M3", "student_t", batch["neighborhood"]),
            ):
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                output = heads[name](*inputs)
                losses = f1_loss(
                    output, batch["matched_gt_index"], batch["gt_state"],
                    batch["gt_yaw"], batch["gt_labels"], temperature,
                    family=family, neighborhood=neighborhood,
                )
                if not all(torch.isfinite(value) for value in losses.values()):
                    raise ValueError(f"non-finite formal loss: {name}")
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(heads[name].parameters(), 5.0)
                optimizer.step()
            batch_count = len(batch["query_feature"])
            epoch_seen += batch_count
            seen += batch_count
        if epoch_seen != 22508:
            raise ValueError(f"incomplete epoch: {epoch_seen}")
    if pilot.sha256(DETECTOR_CHECKPOINT) != checkpoint_before:
        raise ValueError("detector checkpoint changed during formal head training")
    path = OUT / "checkpoints" / f"seed_{seed}.pt"
    write_torch(
        path,
        {
            "epochs": EPOCHS,
            "models": {name: head.state_dict() for name, head in heads.items()},
            "probability_head_sha256": pilot.sha256(ROOT / "probset3d/f1/implementation/probabilistic_query.py"),
            "seed": seed,
            "train_scene_visits": seen,
            "wall_time_seconds": time.time() - started,
        },
    )
    print(json.dumps({"trained": True, "seed": seed, "checkpoint": str(path.relative_to(ROOT))}))


def state_from_boxes(boxes):
    return f0.state_from_boxes(np.asarray(boxes, dtype=np.float64))


def gaussian_parts(target, mean, scale):
    delta = (target - mean) / scale
    return 0.5 * (math.log(2.0 * math.pi) + 2.0 * np.log(scale) + delta * delta)


def student_nll(target, mean, scale, degrees, indices):
    delta = (target[..., indices] - mean[..., indices]) / scale[..., indices]
    dimension = len(indices)
    nu = np.asarray(degrees)
    return -(
        gammaln((nu + dimension) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * dimension * np.log(nu * math.pi)
        - np.log(scale[..., indices]).sum(-1)
        - 0.5 * (nu + dimension) * np.log1p((delta * delta).sum(-1) / nu)
    )


def von_mises_nll(target, mean, kappa):
    return -(
        kappa * np.cos(f0.wrap_angle(target - mean))
        - math.log(2.0 * math.pi)
        - (np.log(i0e(kappa)) + np.abs(kappa))
    )


def wrapped_gaussian_nll(target, mean, scale):
    residual = target - mean
    offsets = np.arange(-3, 4, dtype=np.float64) * (2.0 * math.pi)
    component = (
        -0.5 * ((residual[..., None] + offsets) / scale[..., None]) ** 2
        - np.log(scale[..., None])
        - 0.5 * math.log(2.0 * math.pi)
    )
    return -logsumexp(component, axis=-1)


def learned_geometry(output, gt_boxes, family):
    mean = np.asarray(output["mean_state"], dtype=np.float64)
    scale = np.asarray(output["scale"], dtype=np.float64)
    gt_state = state_from_boxes(gt_boxes)
    target = gt_state[None, :, :]
    center = mean[:, None, :]
    query_scale = scale[:, None, :]
    if family == "gaussian":
        linear = -gaussian_parts(target, center, query_scale).sum(-1)
        yaw = -wrapped_gaussian_nll(
            np.asarray(gt_boxes[:, 6])[None, :],
            np.asarray(output["yaw_mean"])[:, None],
            np.asarray(output["yaw_scale"])[:, None],
        )
    else:
        nu = np.asarray(output["degrees_of_freedom"], dtype=np.float64)[:, None]
        linear = -student_nll(target, center, query_scale, nu, tuple(range(8)))
        yaw = -von_mises_nll(
            np.asarray(gt_boxes[:, 6])[None, :],
            np.asarray(output["yaw_mean"])[:, None],
            np.asarray(output["yaw_kappa"])[:, None],
        )
    return linear + yaw


def fixed_distribution(logits, boxes, gt_boxes, gt_labels, reference, temperature):
    probability = softmax_temperature(logits, temperature)
    class_log = np.log(np.clip(probability[:, gt_labels], EPS, 1.0))
    mean = state_from_boxes(boxes)
    target = state_from_boxes(gt_boxes)
    buckets = f0.bucket_id(logits.argmax(-1), boxes[:, :2])
    geometry = np.empty((len(boxes), len(gt_boxes)), dtype=np.float64)
    scale = np.empty((len(boxes), 8), dtype=np.float64)
    kappa = np.empty(len(boxes), dtype=np.float64)
    for query in range(len(boxes)):
        values = reference["buckets"][str(int(buckets[query]))]
        scale[query] = np.sqrt(np.asarray(values["gaussian_variance"], dtype=np.float64))
        kappa[query] = float(values["yaw_kappa"])
        linear = -gaussian_parts(target, mean[query], scale[query]).sum(-1)
        yaw = -von_mises_nll(gt_boxes[:, 6], boxes[query, 6], kappa[query])
        geometry[query] = linear + yaw
    return class_log, geometry, scale, kappa


def mb_score(existence, class_log, geometry_log):
    existence = np.clip(np.asarray(existence, dtype=np.float64), EPS, 1.0 - EPS)
    query_count = len(existence)
    gt_count = class_log.shape[1]
    log_absent = np.log1p(-existence)
    base = float(-log_absent.sum())
    edge = -(np.log(existence)[:, None] + class_log + geometry_log).T + log_absent[None, :]
    solutions = f0.murty_k_best(edge, K_BEST)
    if not solutions:
        raise ValueError("MB assignment has no finite solution")
    values = np.asarray([base + item[0] for item in solutions], dtype=np.float64)
    result = float(-logsumexp(-values))
    if not np.isfinite(result):
        raise ValueError("non-finite learned MB score")
    return result


def duplicate_spread(existence, class_log, geometry_log, logits, boxes, gt_boxes, gt_labels):
    if not len(gt_boxes):
        return 0.0, 0.0
    same = logits.argmax(-1)[None, :] == gt_labels[:, None]
    distance = np.linalg.norm(gt_boxes[:, None, :2] - boxes[None, :, :2], axis=-1)
    neighborhood = same & (distance <= 2.0)
    mass = np.log(np.clip(existence, EPS, 1.0))[:, None] + class_log + geometry_log
    values = []
    for gt_index in range(len(gt_boxes)):
        selected = np.flatnonzero(neighborhood[gt_index])
        if len(selected) > 1:
            current = mass[selected, gt_index]
            values.append(float(logsumexp(current) - current.max()))
    return (float(np.mean(values)), 1.0) if values else (0.0, 0.0)


def matched_metrics(model, output, fixed, boxes, gt_boxes, matched):
    positive = np.flatnonzero(matched >= 0)
    if not len(positive):
        return {name: 0.0 for name in ("xy_nll", "size_nll", "velocity_nll", "yaw_nll", "median_scale")}
    gt_index = matched[positive]
    target = state_from_boxes(gt_boxes)[gt_index]
    mean = state_from_boxes(boxes)[positive]
    yaw_target = gt_boxes[gt_index, 6]
    yaw_mean = boxes[positive, 6]
    if model in {"M0", "M5"}:
        scale, kappa = fixed
        selected_scale = scale[positive]
        parts = gaussian_parts(target, mean, selected_scale)
        yaw = von_mises_nll(yaw_target, yaw_mean, kappa[positive])
        xy, size, velocity = parts[:, :2].sum(-1), parts[:, 3:6].sum(-1), parts[:, 6:8].sum(-1)
    elif model == "M1":
        selected_scale = np.asarray(output["scale"])[positive]
        parts = gaussian_parts(target, mean, selected_scale)
        yaw = wrapped_gaussian_nll(yaw_target, yaw_mean, np.asarray(output["yaw_scale"])[positive])
        xy, size, velocity = parts[:, :2].sum(-1), parts[:, 3:6].sum(-1), parts[:, 6:8].sum(-1)
    else:
        selected_scale = np.asarray(output["scale"])[positive]
        nu = np.asarray(output["degrees_of_freedom"])[positive]
        xy = student_nll(target, mean, selected_scale, nu, (0, 1))
        size = student_nll(target, mean, selected_scale, nu, (3, 4, 5))
        velocity = student_nll(target, mean, selected_scale, nu, (6, 7))
        yaw = von_mises_nll(yaw_target, yaw_mean, np.asarray(output["yaw_kappa"])[positive])
    return {
        "xy_nll": float(np.mean(xy)),
        "size_nll": float(np.mean(size)),
        "velocity_nll": float(np.mean(velocity)),
        "yaw_nll": float(np.mean(yaw)),
        "median_scale": float(np.median(selected_scale)),
    }


def evaluate(seed: int, device_index: int) -> None:
    ensure_authorized()
    baseline = pilot.strict_load(BASELINE)
    reference = pilot.strict_load(REFERENCE)
    checkpoint_path = OUT / "checkpoints" / f"seed_{seed}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    heads = {name: ProbabilisticQueryHead().to(device).eval() for name in LEARNED}
    for name in LEARNED:
        heads[name].load_state_dict(checkpoint["models"][name])
    temperature = np.asarray(baseline["class_temperature"], dtype=np.float64)
    arrays = {}
    forward_seconds = {name: 0.0 for name in LEARNED}
    forward_scenes = 0
    for condition, root in CONDITIONS.items():
        storage = {model: {metric: [] for metric in METRICS} for model in MODELS}
        tokens = []
        for _, part in f0.iter_parts(root):
            total = len(part["sample_tokens"])
            for begin in range(0, total, BATCH_SIZE):
                stop = min(begin + BATCH_SIZE, total)
                features = torch.from_numpy(np.asarray(part["query_features"][begin:stop])).to(device)
                logits_tensor = torch.from_numpy(np.asarray(part["query_logits"][begin:stop])).to(device)
                boxes_tensor = torch.from_numpy(np.asarray(part["query_boxes"][begin:stop])).to(device)
                seed_tensor = torch.from_numpy(np.asarray(part["seed_probabilities"][begin:stop])).to(device)
                density_zero = torch.zeros(stop - begin, 200, device=device)
                outputs = {}
                for name in LEARNED:
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        value = heads[name](features, logits_tensor, boxes_tensor, seed_tensor, density_zero)
                    torch.cuda.synchronize(device)
                    forward_seconds[name] += time.perf_counter() - started
                    outputs[name] = {key: tensor.detach().cpu().numpy() for key, tensor in value.items()}
                forward_scenes += stop - begin
                for local, row in enumerate(range(begin, stop)):
                    token = str(part["sample_tokens"][row])
                    tokens.append(token)
                    logits = np.asarray(part["query_logits"][row], dtype=np.float64)
                    boxes = np.asarray(part["query_boxes"][row], dtype=np.float64)
                    gt_boxes = variable(part, "gt_boxes", row).astype(np.float64)
                    gt_labels = variable(part, "gt_labels", row).astype(np.int64)
                    matched = np.asarray(part["native_matched_gt"][row], dtype=np.int64)
                    target = matched >= 0
                    class_log, fixed_geometry, fixed_scale, fixed_kappa = fixed_distribution(
                        logits, boxes, gt_boxes, gt_labels, reference, temperature
                    )
                    score = logits.max(-1)
                    m0_r = expit(np.column_stack((score, np.ones_like(score))) @ np.asarray(baseline["m0_existence"]))
                    predicted = logits.argmax(-1)
                    density = density_score(np.asarray(part["query_features"][row]), predicted, baseline["density"])
                    m5_r = expit(np.column_stack((score, density, np.ones_like(score))) @ np.asarray(baseline["m5_existence"]))
                    distributions = {
                        "M0": (m0_r, class_log, fixed_geometry, None),
                        "M5": (m5_r, class_log, fixed_geometry, None),
                    }
                    class_probability = softmax_temperature(logits, temperature)
                    learned_class_log = np.log(np.clip(class_probability[:, gt_labels], EPS, 1.0))
                    for name in LEARNED:
                        output = {key: value[local] for key, value in outputs[name].items()}
                        existence = expit(np.asarray(output["existence_logit"], dtype=np.float64))
                        geometry = learned_geometry(output, gt_boxes, "gaussian" if name == "M1" else "student_t")
                        distributions[name] = (existence, learned_class_log, geometry, output)
                    for model in MODELS:
                        existence, current_class, geometry, output = distributions[model]
                        values = storage[model]
                        values["mb_nll"].append(mb_score(existence, current_class, geometry))
                        values["existence_nll"].append(float(-np.mean(target * np.log(np.clip(existence, EPS, 1.0)) + (~target) * np.log(np.clip(1.0 - existence, EPS, 1.0)))))
                        values["existence_brier"].append(float(np.mean((existence - target.astype(float)) ** 2)))
                        spread, valid = duplicate_spread(existence, current_class, geometry, logits, boxes, gt_boxes, gt_labels)
                        values["duplicate_spread"].append(spread)
                        values["duplicate_valid"].append(valid)
                        matched_value = matched_metrics(model, output, (fixed_scale, fixed_kappa), boxes, gt_boxes, matched)
                        for metric, value in matched_value.items():
                            values[metric].append(value)
                        values["matched_count"].append(float(target.sum()))
        if len(tokens) != 2808 or len(set(tokens)) != 2808:
            raise ValueError(f"condition token conservation failed: {condition}")
        arrays[f"{condition}__sample_tokens"] = np.asarray(tokens)
        for model in MODELS:
            for metric in METRICS:
                value = np.asarray(storage[model][metric], dtype=np.float64)
                if not np.isfinite(value).all():
                    raise ValueError(f"non-finite formal metric: {condition}/{model}/{metric}")
                arrays[f"{condition}__{model}__{metric}"] = value
    detector_seconds = float(baseline["detector_average_seconds_per_scene"])
    for name in LEARNED:
        arrays[f"meta__{name}__head_seconds_per_scene"] = np.asarray(forward_seconds[name] / forward_scenes)
        arrays[f"meta__{name}__overhead_fraction"] = np.asarray((forward_seconds[name] / forward_scenes) / detector_seconds)
    arrays["meta__seed"] = np.asarray(seed, dtype=np.int64)
    path = OUT / "metrics" / f"seed_{seed}.npz"
    write_npz(path, arrays)
    print(json.dumps({"evaluated": True, "seed": seed, "metrics": str(path.relative_to(ROOT))}))


def bootstrap_ci(value, seed=20260730, repeats=10000):
    value = np.asarray(value, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        means[index] = value[rng.integers(0, len(value), len(value))].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def finalize() -> None:
    ensure_authorized()
    baseline = pilot.strict_load(BASELINE)
    runs = {
        seed: np.load(OUT / "metrics" / f"seed_{seed}.npz", allow_pickle=False)
        for seed in SEEDS
    }

    def values(seed, condition, model, metric):
        return np.asarray(runs[seed][f"{condition}__{model}__{metric}"], dtype=np.float64)

    def strongest_differences(condition, metric, valid_metric=None):
        differences, selected = [], []
        for seed in SEEDS:
            candidates = {}
            for model in ("M0", "M1", "M2", "M5"):
                current = values(seed, condition, model, metric)
                mask = np.ones(len(current), dtype=bool)
                if valid_metric:
                    mask = values(seed, condition, model, valid_metric) > 0
                candidates[model] = float(current[mask].mean())
            model = min(candidates, key=candidates.get)
            mask = np.ones(2808, dtype=bool)
            if valid_metric:
                mask = (values(seed, condition, model, valid_metric) > 0) & (values(seed, condition, "M3", valid_metric) > 0)
            differences.append(values(seed, condition, model, metric)[mask] - values(seed, condition, "M3", metric)[mask])
            selected.append(model)
        if len({len(value) for value in differences}) != 1:
            raise ValueError("paired valid-scene masks differ across seeds")
        per_scene = np.stack(differences).mean(0)
        return differences, selected, {
            "mean_improvement": float(per_scene.mean()),
            "paired_scene_ci95": bootstrap_ci(per_scene),
        }

    comparisons = {}
    mb_seed = {}
    for condition in CONDITIONS:
        differences, selected, summary = strongest_differences(condition, "mb_nll")
        comparisons[f"{condition}_mb"] = {"strongest_baseline_by_seed": selected, **summary}
        mb_seed[condition] = [float(value.mean()) for value in differences]
    for metric, valid in (
        ("existence_nll", None),
        ("existence_brier", None),
        ("xy_nll", "matched_count"),
        ("yaw_nll", "matched_count"),
    ):
        _, selected, summary = strongest_differences("clean", metric, valid)
        comparisons[f"clean_{metric}"] = {"strongest_baseline_by_seed": selected, **summary}

    duplicate_differences = []
    scale_ratios = []
    seed_core = []
    for seed in SEEDS:
        duplicate_mask = values(seed, "clean", "M2", "duplicate_valid") > 0
        duplicate = values(seed, "clean", "M2", "duplicate_spread")[duplicate_mask] - values(seed, "clean", "M3", "duplicate_spread")[duplicate_mask]
        duplicate_differences.append(duplicate)
        matched = values(seed, "clean", "M3", "matched_count") > 0
        ratio = np.median(values(seed, "clean", "M3", "median_scale")[matched]) / np.median(values(seed, "clean", "M2", "median_scale")[matched])
        scale_ratios.append(float(ratio))
        existence_diff = values(seed, "clean", "M0", "existence_nll") - values(seed, "clean", "M3", "existence_nll")
        xy_diff = values(seed, "clean", "M1", "xy_nll")[matched] - values(seed, "clean", "M3", "xy_nll")[matched]
        yaw_diff = values(seed, "clean", "M1", "yaw_nll")[matched] - values(seed, "clean", "M3", "yaw_nll")[matched]
        seed_core.append(bool(mb_seed["clean"][SEEDS.index(seed)] > 0 and existence_diff.mean() > 0 and xy_diff.mean() > 0 and yaw_diff.mean() > 0 and duplicate.mean() > 0))
    duplicate_scene = np.stack(duplicate_differences).mean(0)
    comparisons["competition_duplicate_spread"] = {
        "mean_improvement": float(duplicate_scene.mean()),
        "paired_scene_ci95": bootstrap_ci(duplicate_scene),
    }
    comparisons["scale_ratio_M3_over_M2"] = {
        "by_seed": scale_ratios,
        "mean": float(np.mean(scale_ratios)),
    }
    overhead = [float(runs[seed]["meta__M3__overhead_fraction"]) for seed in SEEDS]
    parameter_fraction = 18443 / float(baseline["detector_parameter_count"])

    clean_mb = comparisons["clean_mb"]
    degradation = [comparisons[f"{name}_mb"]["paired_scene_ci95"] for name in ("lowlight", "dropout", "beam")]
    gates = {
        "detection_output_unchanged": bool(pilot.strict_load(PILOT)["gates"]["P02_host_byte_identity"]),
        "clean_mb_set_nll_improved": clean_mb["paired_scene_ci95"][0] > 0,
        "degradation_two_positive_none_reverse": sum(interval[0] > 0 for interval in degradation) >= 2 and all(interval[1] >= 0 for interval in degradation),
        "existence_nll_and_brier_improved": comparisons["clean_existence_nll"]["paired_scene_ci95"][0] > 0 and comparisons["clean_existence_brier"]["paired_scene_ci95"][0] > 0,
        "xy_and_yaw_nll_improved": comparisons["clean_xy_nll"]["paired_scene_ci95"][0] > 0 and comparisons["clean_yaw_nll"]["paired_scene_ci95"][0] > 0,
        "competition_reduces_duplicates": comparisons["competition_duplicate_spread"]["paired_scene_ci95"][0] > 0,
        "no_scale_inflation_explanation": max(scale_ratios) <= 1.10,
        "inference_overhead_acceptable": max(overhead) <= 0.05 and parameter_fraction <= 0.03,
        "three_seed_direction_consistent": all(seed_core),
    }
    passed = all(gates.values())
    result = {
        "comparisons": comparisons,
        "conditions": {name: "S0" if name == "clean" else "S3" for name in CONDITIONS},
        "endpoint": "GO TO F2 GEOMETRY AND SET LEARNING" if passed else "STOP LEARNED PROBABILISTIC QUERY",
        "gates": gates,
        "inference_overhead_fraction_by_seed": overhead,
        "models": list(MODELS),
        "parameter_fraction": parameter_fraction,
        "passed": passed,
        "scientific_scope": "internal clean dev plus deterministic lowlight/dropout/beam S3; official val sealed",
        "seed_core_direction": dict(zip(map(str, SEEDS), seed_core)),
        "seeds": list(SEEDS),
    }
    write_json(OUT / "result.json", result)
    for value in runs.values():
        value.close()
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    trainer = sub.add_parser("train")
    trainer.add_argument("--seed", type=int, choices=SEEDS, required=True)
    trainer.add_argument("--device", type=int, choices=(0, 1, 2, 3), required=True)
    evaluator = sub.add_parser("evaluate")
    evaluator.add_argument("--seed", type=int, choices=SEEDS, required=True)
    evaluator.add_argument("--device", type=int, choices=(0, 1, 2, 3), required=True)
    sub.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "train":
        train(args.seed, args.device)
    elif args.command == "evaluate":
        evaluate(args.seed, args.device)
    else:
        finalize()


if __name__ == "__main__":
    main()
