#!/usr/bin/env python3
"""Fit reference distributions and score 3D MB/PMB sets for ProbSet3D F0."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/probset3d-matplotlib")

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar
from scipy.special import expit, gammaln, i0e, logsumexp
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "probset3d/P0_DATA_SPLIT_AUTHORITY.json"
EXPORT_ROOT = ROOT / "outputs/probset3d/f0/exports"
OUT = ROOT / "outputs/probset3d/f0"
CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)
RANGE_EDGES = (20.0, 40.0)
STATE_DIM = 8
PPP_STATE_DIM = 6
EPS = 1e-9
K_BEST = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(arrays: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def strict_load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def strict_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def wrap_angle(value):
    return np.arctan2(np.sin(value), np.cos(value))


def logit(value):
    value = np.clip(value, EPS, 1.0 - EPS)
    return np.log(value) - np.log1p(-value)


def softmax(value, temperature):
    scaled = np.asarray(value, dtype=np.float64) / float(temperature)
    scaled -= scaled.max(axis=-1, keepdims=True)
    probability = np.exp(scaled)
    return probability / probability.sum(axis=-1, keepdims=True)


def state_from_boxes(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, STATE_DIM), dtype=np.float64)
    if np.any(boxes[:, 3:6] <= 0):
        raise ValueError("non-positive box size")
    return np.column_stack(
        (
            boxes[:, :3],
            np.log(boxes[:, 3:6]),
            boxes[:, 7:9],
        )
    )


def ppp_state_from_boxes(boxes):
    state = state_from_boxes(boxes)
    return state[:, 2:]


def range_bucket(xy):
    radius = np.linalg.norm(np.asarray(xy, dtype=np.float64), axis=-1)
    return np.digitize(radius, RANGE_EDGES).astype(np.int64)


def bucket_id(labels, xy):
    return np.asarray(labels, dtype=np.int64) * 3 + range_bucket(xy)


def gaussian_logpdf(delta, covariance):
    delta = np.asarray(delta, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("covariance is not SPD")
    solved = np.linalg.solve(covariance, delta.T).T
    quadratic = np.sum(delta * solved, axis=-1)
    dimension = covariance.shape[0]
    return -0.5 * (dimension * math.log(2.0 * math.pi) + logdet + quadratic)


def student_logpdf(delta, scale, degrees_of_freedom):
    delta = np.asarray(delta, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    nu = float(degrees_of_freedom)
    sign, logdet = np.linalg.slogdet(scale)
    if sign <= 0:
        raise ValueError("Student-t scale is not SPD")
    solved = np.linalg.solve(scale, delta.T).T
    quadratic = np.sum(delta * solved, axis=-1)
    dimension = scale.shape[0]
    normalizer = (
        gammaln((nu + dimension) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * (dimension * math.log(nu * math.pi) + logdet)
    )
    return normalizer - 0.5 * (nu + dimension) * np.log1p(quadratic / nu)


def von_mises_logpdf(delta, kappa):
    kappa = float(kappa)
    log_i0 = math.log(float(i0e(kappa))) + abs(kappa)
    return kappa * np.cos(delta) - math.log(2.0 * math.pi) - log_i0


def kappa_from_residual(residual):
    residual = np.asarray(residual, dtype=np.float64)
    if not len(residual):
        return 0.0
    resultant = float(
        np.hypot(np.mean(np.cos(residual)), np.mean(np.sin(residual)))
    )
    if resultant < 1e-8:
        return 0.0
    if resultant < 0.53:
        value = 2 * resultant + resultant**3 + 5 * resultant**5 / 6
    elif resultant < 0.85:
        value = -0.4 + 1.39 * resultant + 0.43 / (1.0 - resultant)
    else:
        value = 1.0 / (resultant**3 - 4 * resultant**2 + 3 * resultant)
    return float(np.clip(value, 0.0, 1e4))


def logistic_fit(score, target):
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    design = np.column_stack((score, np.ones_like(score)))
    parameter = np.zeros(2, dtype=np.float64)
    for _ in range(50):
        probability = expit(design @ parameter)
        weight = np.clip(probability * (1.0 - probability), 1e-8, None)
        gradient = design.T @ (probability - target) + 1e-6 * parameter
        hessian = (design.T * weight) @ design + 1e-6 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        parameter -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return parameter


def fit_temperature(logits, target):
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)

    def objective(log_temperature):
        probability = softmax(logits, math.exp(float(log_temperature)))
        return float(-np.mean(np.log(np.clip(probability[np.arange(len(target)), target], EPS, 1.0))))

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    if not result.success:
        raise RuntimeError("class temperature fit failed")
    return float(math.exp(result.x))


def low_rank_covariance(residual, rank=2, minimum_variance=1e-4):
    residual = np.asarray(residual, dtype=np.float64)
    covariance = np.cov(residual, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    covariance = 0.95 * covariance + 0.05 * np.diag(np.diag(covariance))
    diagonal = np.maximum(0.25 * np.diag(covariance), minimum_variance)
    remainder = 0.5 * ((covariance - np.diag(diagonal)) + (covariance - np.diag(diagonal)).T)
    eigenvalue, eigenvector = np.linalg.eigh(remainder)
    keep = np.argsort(eigenvalue)[::-1][:rank]
    positive = np.maximum(eigenvalue[keep], 0.0)
    factor = eigenvector[:, keep] * np.sqrt(positive)[None, :]
    scale = np.diag(diagonal) + factor @ factor.T
    return np.diag(covariance), diagonal, factor, scale


def iter_export_manifests(run_root: Path):
    manifests = sorted(run_root.glob("shard-*/accepted.json"))
    if not manifests:
        raise FileNotFoundError(f"no accepted export shards under {run_root}")
    for path in manifests:
        value = strict_load(path)
        if not value.get("accepted"):
            raise ValueError(f"unaccepted export: {path}")
        yield path, value


def iter_parts(run_root: Path):
    for manifest_path, manifest in iter_export_manifests(run_root):
        for part in manifest["parts"]:
            path = Path(part["path"])
            if not path.is_absolute():
                path = ROOT / path
            if sha256(path) != part["sha256"]:
                raise ValueError(f"export part hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as value:
                yield manifest, {name: value[name] for name in value.files}


def scene_from_part(part, index):
    def variable(name):
        offsets = part[name + "_offsets"]
        return part[name][offsets[index] : offsets[index + 1]]

    scene = {
        "sample_token": str(part["sample_tokens"][index]),
        "scene_token": str(part["scene_tokens"][index]),
        "query_indices": part["query_indices"][index],
        "query_labels": part["query_labels"][index],
        "query_logits": part["query_logits"][index],
        "query_boxes": part["query_boxes"][index],
        "query_scores": part["query_scores"][index],
        "query_kept": part["query_kept"][index],
        "native_matched_gt": part["native_matched_gt"][index],
        "native_ignore": part["native_ignore"][index],
        "gt_boxes": variable("gt_boxes"),
        "gt_labels": variable("gt_labels"),
        "final_boxes": variable("final_boxes"),
        "final_scores": variable("final_scores"),
        "final_labels": variable("final_labels"),
    }
    if "query_features" in part:
        scene["query_features"] = part["query_features"][index]
    if "dense_logits" in part:
        scene["dense_logits"] = part["dense_logits"][index]
        scene["local_max_packbits"] = part["local_max_packbits"][index]
        scene["seed_probabilities"] = part["seed_probabilities"][index]
    return scene


def iter_scenes(run_root: Path):
    for manifest, part in iter_parts(run_root):
        for index in range(len(part["sample_tokens"])):
            yield manifest, scene_from_part(part, index)


def fit_reference(args):
    run_root = Path(args.export_root).resolve() / "fusion-train-clean-s0"
    authority = strict_load(P0)["splits"]["train"]
    allowed = set(strict_load(ROOT / authority["sample_tokens_path"])["sample_tokens"])
    scores, existence_target = [], []
    class_logits, class_target = [], []
    residual_by_bucket = {index: [] for index in range(30)}
    yaw_by_bucket = {index: [] for index in range(30)}
    gt_state_by_class = {index: [] for index in range(10)}
    missed_counts = []
    seen = []
    seen_set = set()
    started = time.time()
    for _, scene in iter_scenes(run_root):
        token = scene["sample_token"]
        if token not in allowed or token in seen_set:
            raise ValueError("train export token authority/uniqueness failure")
        seen.append(token)
        seen_set.add(token)
        logits = np.asarray(scene["query_logits"], dtype=np.float64)
        matched = np.asarray(scene["native_matched_gt"], dtype=np.int64)
        valid = ~np.asarray(scene["native_ignore"], dtype=bool)
        scores.append(logits.max(axis=1)[valid])
        existence_target.append((matched[valid] >= 0).astype(np.float64))
        query_state = state_from_boxes(scene["query_boxes"])
        gt_state = state_from_boxes(scene["gt_boxes"])
        query_bucket = bucket_id(scene["query_labels"], scene["query_boxes"][:, :2])
        for query_index in np.flatnonzero(matched >= 0):
            gt_index = int(matched[query_index])
            class_logits.append(logits[query_index])
            class_target.append(int(scene["gt_labels"][gt_index]))
            bucket = int(query_bucket[query_index])
            residual_by_bucket[bucket].append(gt_state[gt_index] - query_state[query_index])
            yaw_by_bucket[bucket].append(
                float(wrap_angle(scene["gt_boxes"][gt_index, 6] - scene["query_boxes"][query_index, 6]))
            )
        ppp_state = ppp_state_from_boxes(scene["gt_boxes"])
        for gt_index, label in enumerate(scene["gt_labels"]):
            gt_state_by_class[int(label)].append(ppp_state[gt_index])
        missed_counts.append(len(scene["gt_boxes"]) - len(set(matched[matched >= 0].tolist())))
        if len(seen) % 1000 == 0:
            print(json.dumps({"fit_samples": len(seen), "elapsed_seconds": time.time() - started}), flush=True)
    if len(seen) != authority["sample_count"] or set(seen) != allowed:
        raise ValueError("train reference fit does not cover exactly 22,508 authorized samples")

    score = np.concatenate(scores)
    target = np.concatenate(existence_target)
    existence_parameter = logistic_fit(score, target)
    temperature = fit_temperature(np.asarray(class_logits), np.asarray(class_target))
    global_residual = np.concatenate(
        [np.asarray(value) for value in residual_by_bucket.values() if value], axis=0
    )
    global_yaw = np.concatenate(
        [np.asarray(value) for value in yaw_by_bucket.values() if value], axis=0
    )
    bucket_parameters = {}
    for index in range(30):
        residual = np.asarray(residual_by_bucket[index], dtype=np.float64)
        yaw = np.asarray(yaw_by_bucket[index], dtype=np.float64)
        fallback = len(residual) < 50
        if fallback:
            residual = global_residual
            yaw = global_yaw
        diagonal_variance, diagonal_base, factor, scale = low_rank_covariance(residual)
        bucket_parameters[str(index)] = {
            "sample_count": len(residual_by_bucket[index]),
            "fallback_to_global": fallback,
            "gaussian_variance": np.maximum(diagonal_variance, 1e-4).tolist(),
            "student_diagonal": diagonal_base.tolist(),
            "student_factor": factor.tolist(),
            "student_scale": scale.tolist(),
            "yaw_kappa": kappa_from_residual(yaw),
        }
    ppp_prior = {}
    global_gt_state = np.concatenate(
        [np.asarray(value) for value in gt_state_by_class.values() if value], axis=0
    )
    for class_index in range(10):
        state = np.asarray(gt_state_by_class[class_index], dtype=np.float64)
        fallback = len(state) < 50
        if fallback:
            state = global_gt_state
        mean = state.mean(axis=0)
        _, diagonal, factor, scale = low_rank_covariance(
            state - mean, rank=2, minimum_variance=1e-3
        )
        ppp_prior[str(class_index)] = {
            "sample_count": len(gt_state_by_class[class_index]),
            "fallback_to_global": fallback,
            "mean": mean.tolist(),
            "diagonal": diagonal.tolist(),
            "factor": factor.tolist(),
            "scale": scale.tolist(),
        }
    parameters = {
        "schema_version": "probset3d_f0_reference_parameters_v1",
        "source_run": str(run_root.relative_to(ROOT)),
        "source_sample_count": len(seen),
        "source_token_set_sha256": hashlib.sha256(
            "\n".join(sorted(seen)).encode("ascii")
        ).hexdigest(),
        "existence": {
            "input": "maximum_pre_nms_query_class_logit",
            "slope": float(existence_parameter[0]),
            "intercept": float(existence_parameter[1]),
            "positive_rate": float(target.mean()),
        },
        "class_temperature": temperature,
        "range_edges_m": list(RANGE_EDGES),
        "student_degrees_of_freedom": 5.0,
        "low_rank": 2,
        "buckets": bucket_parameters,
        "ppp": {
            "mean_missed_count": float(np.mean(missed_counts)),
            "missed_count_std": float(np.std(missed_counts)),
            "state_priors": ppp_prior,
            "spatial_floor_mixture": 1e-6,
            "explained_kernel_radius_cells": 3,
            "explained_kernel_sigma_cells": 1.5,
        },
        "fit_checks": {
            "all_finite": bool(
                np.isfinite(score).all()
                and np.isfinite(global_residual).all()
                and np.isfinite(global_gt_state).all()
            ),
            "train_only": True,
            "official_val_used": False,
            "internal_test_used": False,
            "training_performed": False,
        },
    }
    strict_write(Path(args.output), parameters)
    print(json.dumps(parameters, indent=2, sort_keys=True), flush=True)


def solve_assignment(cost, prefix=(), banned=frozenset()):
    cost = np.asarray(cost, dtype=np.float64)
    rows, columns = cost.shape
    prefix = tuple(int(value) for value in prefix)
    if len(prefix) > rows or len(set(prefix)) != len(prefix):
        return None
    assignment = np.full(rows, -1, dtype=np.int64)
    if prefix:
        assignment[: len(prefix)] = prefix
    used = set(prefix)
    if any(not np.isfinite(cost[row, column]) for row, column in enumerate(prefix)):
        return None
    remaining_rows = list(range(len(prefix), rows))
    remaining_columns = [column for column in range(columns) if column not in used]
    if len(remaining_columns) < len(remaining_rows):
        return None
    if remaining_rows:
        sub = cost[np.ix_(remaining_rows, remaining_columns)].copy()
        column_lookup = {column: index for index, column in enumerate(remaining_columns)}
        row_lookup = {row: index for index, row in enumerate(remaining_rows)}
        for row, column in banned:
            if row in row_lookup and column in column_lookup:
                sub[row_lookup[row], column_lookup[column]] = np.inf
        if np.any(~np.isfinite(sub).any(axis=1)):
            return None
        selected_rows, selected_columns = linear_sum_assignment(sub)
        if len(selected_rows) != len(remaining_rows):
            return None
        selected_cost = sub[selected_rows, selected_columns]
        if not np.isfinite(selected_cost).all():
            return None
        for row_index, column_index in zip(selected_rows, selected_columns):
            assignment[remaining_rows[row_index]] = remaining_columns[column_index]
    total = float(cost[np.arange(rows), assignment].sum()) if rows else 0.0
    return total, assignment


def murty_k_best(cost, k):
    first = solve_assignment(cost)
    if first is None:
        return []
    queue = []
    counter = 0
    heapq.heappush(queue, (first[0], counter, (), frozenset(), first[1]))
    result = []
    while queue and len(result) < k:
        total, _, prefix, banned, assignment = heapq.heappop(queue)
        result.append((float(total), assignment.copy()))
        start = len(prefix)
        for row in range(start, len(assignment)):
            child_prefix = tuple(int(value) for value in assignment[:row])
            child_banned = frozenset(set(banned) | {(row, int(assignment[row]))})
            solved = solve_assignment(cost, child_prefix, child_banned)
            if solved is None:
                continue
            counter += 1
            heapq.heappush(
                queue,
                (solved[0], counter, child_prefix, child_banned, solved[1]),
            )
    return result


def prepared_queries(scene, parameters):
    logits = np.asarray(scene["query_logits"], dtype=np.float64)
    maximum_logit = logits.max(axis=1)
    existence = expit(
        parameters["existence"]["slope"] * maximum_logit
        + parameters["existence"]["intercept"]
    )
    if "existence_override" in scene:
        existence = np.asarray(scene["existence_override"], dtype=np.float64)
    class_probability = softmax(logits, parameters["class_temperature"])
    if "class_probability_override" in scene:
        class_probability = np.asarray(scene["class_probability_override"], dtype=np.float64)
    return (
        np.clip(existence, EPS, 1.0 - EPS),
        np.clip(class_probability, EPS, 1.0),
        state_from_boxes(scene["query_boxes"]),
        np.asarray(scene["query_boxes"][:, 6], dtype=np.float64),
    )


def query_log_density(scene, parameters, model):
    existence, class_probability, mean, yaw_mean = prepared_queries(scene, parameters)
    gt_state = state_from_boxes(scene["gt_boxes"])
    gt_yaw = np.asarray(scene["gt_boxes"][:, 6], dtype=np.float64)
    gt_labels = np.asarray(scene["gt_labels"], dtype=np.int64)
    query_bucket = bucket_id(scene["query_labels"], scene["query_boxes"][:, :2])
    query_count, gt_count = len(mean), len(gt_state)
    class_log = (
        np.log(class_probability[:, gt_labels])
        if gt_count
        else np.empty((query_count, 0), dtype=np.float64)
    )
    geometry_log = np.empty((query_count, gt_count), dtype=np.float64)
    multiplier = float(scene.get("covariance_multiplier", 1.0))
    for query_index in range(query_count):
        delta = gt_state - mean[query_index]
        bucket = parameters["buckets"][str(int(query_bucket[query_index]))]
        if model == "E0":
            covariance = np.diag(
                np.asarray([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 4e-4, 4e-4])
                * multiplier
            )
            linear_log = gaussian_logpdf(delta, covariance)
            yaw_log = von_mises_logpdf(wrap_angle(gt_yaw - yaw_mean[query_index]), 1e4)
        elif model == "E1":
            covariance = np.diag(np.asarray(bucket["gaussian_variance"]) * multiplier)
            linear_log = gaussian_logpdf(delta, covariance)
            yaw_log = von_mises_logpdf(
                wrap_angle(gt_yaw - yaw_mean[query_index]), bucket["yaw_kappa"]
            )
        else:
            scale = np.asarray(bucket["student_scale"]) * multiplier
            linear_log = student_logpdf(
                delta, scale, parameters["student_degrees_of_freedom"]
            )
            yaw_log = von_mises_logpdf(
                wrap_angle(gt_yaw - yaw_mean[query_index]), bucket["yaw_kappa"]
            )
        geometry_log[query_index] = linear_log + yaw_log
    return existence, class_log, geometry_log


def ppp_log_intensity(scene, parameters, existence, grid):
    gt_boxes = np.asarray(scene["gt_boxes"], dtype=np.float64)
    gt_labels = np.asarray(scene["gt_labels"], dtype=np.int64)
    dense = expit(np.asarray(scene["dense_logits"], dtype=np.float64))
    classes, height, width = dense.shape
    if classes != len(CLASSES):
        raise ValueError("dense heatmap class count mismatch")
    explained = np.zeros_like(dense)
    radius = int(parameters["ppp"]["explained_kernel_radius_cells"])
    sigma = float(parameters["ppp"]["explained_kernel_sigma_cells"])
    for query_index, (flat_index, label) in enumerate(
        zip(scene["query_indices"], scene["query_labels"])
    ):
        row, column = divmod(int(flat_index), width)
        row0, row1 = max(0, row - radius), min(height, row + radius + 1)
        col0, col1 = max(0, column - radius), min(width, column + radius + 1)
        yy = np.arange(row0, row1) - row
        xx = np.arange(col0, col1) - column
        kernel = existence[query_index] * np.exp(
            -(yy[:, None] ** 2 + xx[None, :] ** 2) / (2.0 * sigma**2)
        )
        view = explained[int(label), row0:row1, col0:col1]
        np.maximum(view, kernel, out=view)
    residual = dense * (1.0 - np.clip(explained, 0.0, 1.0))
    total = float(residual.sum())
    floor = float(parameters["ppp"]["spatial_floor_mixture"])
    probability_mass = (1.0 - floor) * residual / max(total, EPS) + floor / residual.size
    normalization_error = abs(float(probability_mass.sum()) - 1.0)
    cell_size_x = float(grid["voxel_size"][0]) * int(grid["out_size_factor"])
    cell_size_y = float(grid["voxel_size"][1]) * int(grid["out_size_factor"])
    cell_area = cell_size_x * cell_size_y
    x_index = np.floor((gt_boxes[:, 0] - float(grid["point_cloud_range"][0])) / cell_size_x).astype(int)
    y_index = np.floor((gt_boxes[:, 1] - float(grid["point_cloud_range"][1])) / cell_size_y).astype(int)
    inside = (x_index >= 0) & (x_index < width) & (y_index >= 0) & (y_index < height)
    spatial = np.full(len(gt_boxes), floor / residual.size / cell_area, dtype=np.float64)
    spatial[inside] = probability_mass[
        gt_labels[inside], y_index[inside], x_index[inside]
    ] / cell_area
    remaining = ppp_state_from_boxes(gt_boxes)
    state_log = np.empty(len(gt_boxes), dtype=np.float64)
    for class_index in range(len(CLASSES)):
        mask = gt_labels == class_index
        if not mask.any():
            continue
        prior = parameters["ppp"]["state_priors"][str(class_index)]
        delta = remaining[mask] - np.asarray(prior["mean"])
        state_log[mask] = student_logpdf(
            delta,
            np.asarray(prior["scale"]),
            parameters["student_degrees_of_freedom"],
        ) - math.log(2.0 * math.pi)
    rate = float(scene.get("ppp_rate_override", parameters["ppp"]["mean_missed_count"]))
    rate = max(rate, 1e-12)
    return (
        np.log(rate) + np.log(np.clip(spatial, 1e-300, None)) + state_log,
        rate,
        normalization_error,
    )


def score_set(scene, parameters, model, grid=None, k=K_BEST):
    if model not in {"E0", "E1", "E2", "E3"}:
        raise ValueError(model)
    existence, class_log, geometry_log = query_log_density(scene, parameters, model)
    query_count = len(existence)
    gt_count = len(scene["gt_boxes"])
    log_absent = np.log1p(-existence)
    base = float(-log_absent.sum())
    query_edge = -(
        np.log(existence)[:, None] + class_log + geometry_log
    ).T + log_absent[None, :]
    ppp_log = None
    rate = 0.0
    ppp_normalization_error = 0.0
    if model == "E3":
        if grid is None or "dense_logits" not in scene:
            raise ValueError("E3 requires dense pre-topK authority and grid metadata")
        ppp_log, rate, ppp_normalization_error = ppp_log_intensity(
            scene, parameters, existence, grid
        )
        edge = np.full((gt_count, query_count + gt_count), np.inf, dtype=np.float64)
        edge[:, :query_count] = query_edge
        if gt_count:
            edge[np.arange(gt_count), query_count + np.arange(gt_count)] = -ppp_log
        base += rate
    else:
        edge = query_edge
    solutions = murty_k_best(edge, k)
    if not solutions:
        return {
            "nll": float("inf"),
            "best_nll": float("inf"),
            "decomposition": {},
            "solution_count": 0,
        }
    assignment_nll = np.asarray([base + value[0] for value in solutions], dtype=np.float64)
    marginal = float(-logsumexp(-assignment_nll))
    best_assignment = solutions[0][1]
    query_matched = best_assignment < query_count
    matched_queries = best_assignment[query_matched]
    unmatched_queries = np.ones(query_count, dtype=bool)
    unmatched_queries[matched_queries] = False
    existence_nll = float(
        -np.log(existence[matched_queries]).sum()
        - np.log1p(-existence[unmatched_queries]).sum()
    )
    matched_gt = np.flatnonzero(query_matched)
    classification_nll = float(
        -class_log[matched_queries, matched_gt].sum()
    )
    geometry_nll = float(-geometry_log[matched_queries, matched_gt].sum())
    miss_nll = 0.0
    if model == "E3":
        missed_gt = np.flatnonzero(~query_matched)
        miss_nll = float(rate - ppp_log[missed_gt].sum())
    best_nll = existence_nll + classification_nll + geometry_nll + miss_nll
    association_nll = marginal - best_nll
    decomposition = {
        "existence": existence_nll,
        "classification": classification_nll,
        "geometry": geometry_nll,
        "miss": miss_nll,
        "association": association_nll,
    }
    reconstruction = sum(decomposition.values())
    return {
        "nll": marginal,
        "best_nll": best_nll,
        "nll_k1": float(assignment_nll[0]),
        "nll_k5": float(-logsumexp(-assignment_nll[: min(5, len(assignment_nll))])),
        "nll_k10": marginal,
        "decomposition": decomposition,
        "reconstruction_error": abs(reconstruction - marginal),
        "solution_count": len(solutions),
        "best_assignment": best_assignment,
        "ppp_rate": rate,
        "ppp_spatial_normalization_error": ppp_normalization_error,
    }


def detection_error(scene):
    keep = np.asarray(scene["final_scores"], dtype=np.float64) >= 0.1
    boxes = np.asarray(scene["final_boxes"], dtype=np.float64)[keep]
    labels = np.asarray(scene["final_labels"], dtype=np.int64)[keep]
    gt = np.asarray(scene["gt_boxes"], dtype=np.float64)
    gt_labels = np.asarray(scene["gt_labels"], dtype=np.int64)
    matched_distance = []
    if len(boxes) and len(gt):
        distance = np.linalg.norm(gt[:, None, :2] - boxes[None, :, :2], axis=2)
        cost = distance + (gt_labels[:, None] != labels[None, :]) * 1e6
        rows, columns = linear_sum_assignment(cost)
        matched_distance = [
            float(distance[row, column])
            for row, column in zip(rows, columns)
            if gt_labels[row] == labels[column] and distance[row, column] <= 2.0
        ]
    true_positive = len(matched_distance)
    false_negative = len(gt) - true_positive
    false_positive = len(boxes) - true_positive
    composite = false_positive + false_negative + sum(min(value, 4.0) for value in matched_distance) / 4.0
    return {
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
        "matched_xy_sum": float(sum(matched_distance)),
        "composite": float(composite),
    }


def score_export(args):
    run_root = Path(args.export_root).resolve() / args.run_id
    parameters = strict_load(Path(args.parameters))
    models = args.models
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite score run: {output}")
    tokens, scene_tokens, gt_counts, detection_errors = [], [], [], []
    score_rows = {model: [] for model in models}
    best_rows = {model: [] for model in models}
    k1_rows = {model: [] for model in models}
    k5_rows = {model: [] for model in models}
    reconstruction_rows = {model: [] for model in models}
    ppp_rate_rows = {model: [] for model in models}
    ppp_normalization_rows = {model: [] for model in models}
    decomposition_rows = {
        model: {name: [] for name in ("existence", "classification", "geometry", "miss", "association")}
        for model in models
    }
    runtimes = []
    grid = None
    started = time.time()
    for manifest, scene in iter_scenes(run_root):
        if grid is None:
            grid = manifest["grid"]
        scene_started = time.perf_counter()
        error = detection_error(scene)
        for model in models:
            value = score_set(scene, parameters, model, grid, args.k)
            score_rows[model].append(value["nll"])
            best_rows[model].append(value["best_nll"])
            k1_rows[model].append(value["nll_k1"])
            k5_rows[model].append(value["nll_k5"])
            reconstruction_rows[model].append(value["reconstruction_error"])
            ppp_rate_rows[model].append(value["ppp_rate"])
            ppp_normalization_rows[model].append(
                value["ppp_spatial_normalization_error"]
            )
            for name in decomposition_rows[model]:
                decomposition_rows[model][name].append(value["decomposition"][name])
        runtimes.append(time.perf_counter() - scene_started)
        tokens.append(scene["sample_token"])
        scene_tokens.append(scene["scene_token"])
        gt_counts.append(len(scene["gt_boxes"]))
        detection_errors.append(error["composite"])
        if args.max_scenes and len(tokens) >= args.max_scenes:
            break
        if len(tokens) % 100 == 0:
            print(
                json.dumps(
                    {
                        "run_id": args.run_id,
                        "scored": len(tokens),
                        "elapsed_seconds": time.time() - started,
                    }
                ),
                flush=True,
            )
    arrays = {
        "sample_tokens": np.asarray(tokens, dtype="U32"),
        "scene_tokens": np.asarray(scene_tokens, dtype="U32"),
        "gt_count": np.asarray(gt_counts, dtype=np.int16),
        "detection_error": np.asarray(detection_errors, dtype=np.float64),
    }
    for model in models:
        arrays[f"{model}_nll"] = np.asarray(score_rows[model], dtype=np.float64)
        arrays[f"{model}_best_nll"] = np.asarray(best_rows[model], dtype=np.float64)
        arrays[f"{model}_nll_k1"] = np.asarray(k1_rows[model], dtype=np.float64)
        arrays[f"{model}_nll_k5"] = np.asarray(k5_rows[model], dtype=np.float64)
        arrays[f"{model}_reconstruction_error"] = np.asarray(
            reconstruction_rows[model], dtype=np.float64
        )
        arrays[f"{model}_ppp_rate"] = np.asarray(
            ppp_rate_rows[model], dtype=np.float64
        )
        arrays[f"{model}_ppp_spatial_normalization_error"] = np.asarray(
            ppp_normalization_rows[model], dtype=np.float64
        )
        for name, values in decomposition_rows[model].items():
            arrays[f"{model}_{name}"] = np.asarray(values, dtype=np.float64)
    if not all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"):
        raise ValueError("non-finite real-data score")
    scientific = {
        "schema_version": "probset3d_f0_score_summary_v1",
        "run_id": args.run_id,
        "sample_count": len(tokens),
        "models": {},
        "k_best": args.k,
        "max_scenes": args.max_scenes,
        "score_array_digest": array_digest(arrays),
        "all_finite": True,
    }
    for model in models:
        nll = arrays[f"{model}_nll"]
        scientific["models"][model] = {
            "mean_nll": float(np.mean(nll)),
            "median_nll": float(np.median(nll)),
            "mean_nll_per_gt": float(np.mean(nll / np.maximum(arrays["gt_count"], 1))),
            "mean_k1_to_k5_gain": float(
                np.mean(arrays[f"{model}_nll_k1"] - arrays[f"{model}_nll_k5"])
            ),
            "mean_k5_to_k10_gain": float(
                np.mean(arrays[f"{model}_nll_k5"] - nll)
            ),
            "maximum_reconstruction_error": float(
                arrays[f"{model}_reconstruction_error"].max(initial=0.0)
            ),
            "mean_ppp_rate": float(np.mean(arrays[f"{model}_ppp_rate"])),
            "maximum_ppp_spatial_normalization_error": float(
                arrays[f"{model}_ppp_spatial_normalization_error"].max(initial=0.0)
            ),
            "mean_components": {
                name: float(np.mean(arrays[f"{model}_{name}"]))
                for name in decomposition_rows[model]
            },
        }
    write_npz(output / "per_scene_scores.npz", arrays)
    strict_write(output / "scientific_summary.json", scientific)
    strict_write(
        output / "runtime.json",
        {
            "wall_time_seconds": time.time() - started,
            "median_seconds_per_scene": float(np.median(runtimes)),
            "maximum_seconds_per_scene": float(np.max(runtimes)),
            "mean_seconds_per_scene": float(np.mean(runtimes)),
        },
    )
    print(json.dumps(scientific, indent=2, sort_keys=True), flush=True)


def fixture_parameters():
    covariance = np.diag([0.25, 0.25, 0.16, 0.04, 0.04, 0.04, 0.25, 0.25])
    bucket = {
        "sample_count": 100,
        "fallback_to_global": False,
        "gaussian_variance": np.diag(covariance).tolist(),
        "student_diagonal": np.diag(covariance).tolist(),
        "student_factor": np.zeros((STATE_DIM, 2)).tolist(),
        "student_scale": covariance.tolist(),
        "yaw_kappa": 20.0,
    }
    ppp_scale = np.diag([1.0, 0.25, 0.25, 0.25, 4.0, 4.0])
    prior = {
        "sample_count": 100,
        "fallback_to_global": False,
        "mean": [0.0, math.log(2.0), math.log(4.0), math.log(1.5), 0.0, 0.0],
        "diagonal": np.diag(ppp_scale).tolist(),
        "factor": np.zeros((PPP_STATE_DIM, 2)).tolist(),
        "scale": ppp_scale.tolist(),
    }
    return {
        "existence": {"slope": 1.0, "intercept": 0.0},
        "class_temperature": 1.0,
        "student_degrees_of_freedom": 5.0,
        "buckets": {str(index): dict(bucket) for index in range(30)},
        "ppp": {
            "mean_missed_count": 1.0,
            "state_priors": {str(index): dict(prior) for index in range(10)},
            "spatial_floor_mixture": 1e-6,
            "explained_kernel_radius_cells": 1,
            "explained_kernel_sigma_cells": 1.0,
        },
    }


def fixture_scene():
    boxes = np.asarray(
        [
            [0.1, 0.0, 0.0, 2.0, 4.0, 1.5, 0.05, 0.0, 0.0],
            [10.1, 0.0, 0.0, 2.5, 6.0, 2.5, -0.05, 0.0, 0.0],
            [25.0, 25.0, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    gt = np.asarray(
        [
            [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 2.5, 6.0, 2.5, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    logits = np.full((3, 10), -5.0, dtype=np.float64)
    logits[0, 0] = 5.0
    logits[1, 1] = 5.0
    logits[2, 0] = 2.0
    dense = np.full((10, 32, 32), -8.0, dtype=np.float64)
    dense[0, 4, 4] = 4.0
    dense[1, 4, 14] = 4.0
    dense[2, 20, 20] = 2.0
    return {
        "sample_token": "fixture",
        "query_indices": np.asarray([4 * 32 + 4, 4 * 32 + 14, 20 * 32 + 20]),
        "query_labels": np.asarray([0, 1, 2]),
        "query_logits": logits,
        "query_boxes": boxes,
        "query_scores": np.asarray([0.9, 0.8, 0.05]),
        "query_kept": np.ones(3, dtype=bool),
        "native_matched_gt": np.asarray([0, 1, -1]),
        "native_ignore": np.zeros(3, dtype=bool),
        "existence_override": np.asarray([0.9, 0.8, 0.05]),
        "gt_boxes": gt,
        "gt_labels": np.asarray([0, 1]),
        "dense_logits": dense,
        "final_boxes": boxes[:2],
        "final_scores": np.asarray([0.9, 0.8]),
        "final_labels": np.asarray([0, 1]),
    }


def copy_scene(scene):
    return {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in scene.items()
    }


def remove_query(scene, index):
    value = copy_scene(scene)
    query_count = len(value["query_boxes"])
    keep = np.ones(query_count, dtype=bool)
    keep[index] = False
    for name in (
        "query_indices",
        "query_labels",
        "query_logits",
        "query_boxes",
        "query_scores",
        "query_kept",
        "native_matched_gt",
        "native_ignore",
        "existence_override",
        "class_probability_override",
    ):
        if name in value:
            value[name] = value[name][keep]
    return value


def duplicate_query(scene, index):
    value = copy_scene(scene)
    for name in (
        "query_indices",
        "query_labels",
        "query_logits",
        "query_boxes",
        "query_scores",
        "query_kept",
        "native_matched_gt",
        "native_ignore",
        "existence_override",
        "class_probability_override",
    ):
        if name in value:
            value[name] = np.concatenate((value[name], value[name][index : index + 1]), axis=0)
    if "native_matched_gt" in value:
        value["native_matched_gt"][-1] = -1
    return value


def permute_scene(scene, query_permutation=None, gt_permutation=None):
    value = copy_scene(scene)
    if query_permutation is not None:
        for name in (
            "query_indices",
            "query_labels",
            "query_logits",
            "query_boxes",
            "query_scores",
            "query_kept",
            "native_matched_gt",
            "native_ignore",
            "existence_override",
            "class_probability_override",
        ):
            if name in value:
                value[name] = value[name][query_permutation]
    if gt_permutation is not None:
        value["gt_boxes"] = value["gt_boxes"][gt_permutation]
        value["gt_labels"] = value["gt_labels"][gt_permutation]
    return value


def run_fixtures(args):
    parameters = fixture_parameters()
    scene = fixture_scene()
    grid = {
        "voxel_size": [1.0, 1.0],
        "out_size_factor": 1,
        "point_cloud_range": [-4.0, -4.0],
    }
    base_mb = score_set(scene, parameters, "E2", grid, 10)
    base_pmb = score_set(scene, parameters, "E3", grid, 10)
    checks = {}

    def direction(name, changed, base=base_mb["nll"], relation="greater"):
        passed = changed > base if relation == "greater" else changed < base
        checks[name] = {
            "base": float(base),
            "changed": float(changed),
            "difference": float(changed - base),
            "passed": bool(passed),
        }

    direction("delete_useful_query", score_set(remove_query(scene, 0), parameters, "E2", grid)["nll"])
    false_positive = copy_scene(scene)
    false_box = np.asarray([[50.0, -50.0, 0.0, 2.0, 4.0, 1.5, 0.0, 0.0, 0.0]])
    for name, append in (
        ("query_indices", np.asarray([31 * 32 + 31])),
        ("query_labels", np.asarray([0])),
        ("query_logits", np.asarray([[5.0] + [-5.0] * 9])),
        ("query_boxes", false_box),
        ("query_scores", np.asarray([0.99])),
        ("query_kept", np.asarray([True])),
        ("native_matched_gt", np.asarray([-1])),
        ("native_ignore", np.asarray([False])),
        ("existence_override", np.asarray([0.99])),
    ):
        false_positive[name] = np.concatenate((false_positive[name], append), axis=0)
    direction("add_high_existence_false_positive", score_set(false_positive, parameters, "E2", grid)["nll"])
    direction("duplicate_correct_query", score_set(duplicate_query(scene, 0), parameters, "E2", grid)["nll"])
    perturbed_center = copy_scene(scene)
    perturbed_center["query_boxes"][0, 0] += 3.0
    direction("perturb_center", score_set(perturbed_center, parameters, "E2", grid)["nll"])
    perturbed_yaw = copy_scene(scene)
    perturbed_yaw["query_boxes"][0, 6] += math.pi / 2.0
    direction("perturb_yaw", score_set(perturbed_yaw, parameters, "E2", grid)["nll"])
    wrong_class = copy_scene(scene)
    wrong_class["query_logits"][0] = -5.0
    wrong_class["query_logits"][0, 2] = 5.0
    direction("reduce_correct_class_probability", score_set(wrong_class, parameters, "E2", grid)["nll"])
    collapsed = copy_scene(scene)
    collapsed["covariance_multiplier"] = 1e-4
    direction("collapse_covariance_around_wrong_mean", score_set(collapsed, parameters, "E1", grid)["nll"], score_set(scene, parameters, "E1", grid)["nll"])
    inflated = copy_scene(scene)
    inflated["covariance_multiplier"] = 1e4
    direction("inflate_covariance", score_set(inflated, parameters, "E1", grid)["nll"], score_set(scene, parameters, "E1", grid)["nll"])

    low_existence = copy_scene(scene)
    low_existence["existence_override"][2] = 1e-6
    direction(
        "unmatched_r_to_zero",
        score_set(low_existence, parameters, "E2", grid)["nll"],
        base_mb["nll"],
        relation="less",
    )
    high_existence = copy_scene(scene)
    high_existence["existence_override"][2] = 1.0 - 1e-6
    direction("unmatched_r_to_one", score_set(high_existence, parameters, "E2", grid)["nll"])

    missing = copy_scene(scene)
    missing["existence_override"][:] = 1e-6
    missing_positive_rate = score_set(missing, parameters, "E3", grid)["nll"]
    missing_zero_rate = copy_scene(missing)
    missing_zero_rate["ppp_rate_override"] = 1e-12
    direction(
        "zero_residual_rate_penalizes_unexplained_gt",
        score_set(missing_zero_rate, parameters, "E3", grid)["nll"],
        missing_positive_rate,
    )

    query_permutation = np.asarray([2, 0, 1])
    gt_permutation = np.asarray([1, 0])
    permuted = permute_scene(scene, query_permutation, gt_permutation)
    permuted_mb = score_set(permuted, parameters, "E2", grid)["nll"]
    permuted_pmb = score_set(permuted, parameters, "E3", grid)["nll"]
    checks["permutation_invariance_mb"] = {
        "absolute_error": abs(permuted_mb - base_mb["nll"]),
        "passed": abs(permuted_mb - base_mb["nll"]) <= 1e-10,
    }
    checks["permutation_invariance_pmb"] = {
        "absolute_error": abs(permuted_pmb - base_pmb["nll"]),
        "passed": abs(permuted_pmb - base_pmb["nll"]) <= 1e-10,
    }
    checks["decomposition_mb"] = {
        "absolute_error": base_mb["reconstruction_error"],
        "passed": base_mb["reconstruction_error"] <= 1e-10,
    }
    checks["decomposition_pmb"] = {
        "absolute_error": base_pmb["reconstruction_error"],
        "passed": base_pmb["reconstruction_error"] <= 1e-10,
    }
    checks["ppp_spatial_normalization"] = {
        "absolute_error": base_pmb["ppp_spatial_normalization_error"],
        "passed": base_pmb["ppp_spatial_normalization_error"] <= 1e-12,
    }
    k1 = score_set(scene, parameters, "E3", grid, 1)["nll"]
    k5 = score_set(scene, parameters, "E3", grid, 5)["nll"]
    k10 = base_pmb["nll"]
    checks["kbest_monotonicity"] = {
        "k1": k1,
        "k5": k5,
        "k10": k10,
        "passed": bool(k10 <= k5 <= k1),
    }
    cost = np.asarray([[1.0, 2.0, 4.0], [3.0, 1.0, 2.0]])
    solutions = murty_k_best(cost, 6)
    brute = sorted(
        cost[0, left] + cost[1, right]
        for left in range(3)
        for right in range(3)
        if left != right
    )
    murty = [value[0] for value in solutions]
    checks["murty_matches_bruteforce"] = {
        "murty": murty,
        "bruteforce": brute,
        "maximum_absolute_error": float(np.max(np.abs(np.asarray(murty) - np.asarray(brute)))),
        "passed": bool(np.allclose(murty, brute, atol=0.0, rtol=0.0)),
    }
    checks["all_scores_finite"] = {
        "passed": bool(
            all(
                math.isfinite(value)
                for value in (
                    base_mb["nll"],
                    base_pmb["nll"],
                    k1,
                    k5,
                    k10,
                )
            )
        )
    }
    report = {
        "schema_version": "probset3d_f0_mathematical_fixtures_v1",
        "checks": checks,
        "maximum_permutation_error": max(
            checks["permutation_invariance_mb"]["absolute_error"],
            checks["permutation_invariance_pmb"]["absolute_error"],
        ),
        "maximum_decomposition_error": max(
            checks["decomposition_mb"]["absolute_error"],
            checks["decomposition_pmb"]["absolute_error"],
        ),
        "all_passed": all(value["passed"] for value in checks.values()),
    }
    strict_write(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def real_injection_audit(args):
    run_root = Path(args.export_root).resolve() / "fusion-dev-clean-s0"
    parameters = strict_load(Path(args.parameters))
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite injection audit: {output}")
    tokens = []
    matched_delta = []
    bottom_delta = []
    paired_advantage = []
    duplicate_delta = []
    runtimes = []
    grid = None
    started = time.time()
    for manifest, scene in iter_scenes(run_root):
        if grid is None:
            grid = manifest["grid"]
        existence = prepared_queries(scene, parameters)[0]
        matched = np.flatnonzero(scene["native_matched_gt"] >= 0)
        unmatched = np.flatnonzero(
            (scene["native_matched_gt"] < 0) & ~scene["native_ignore"]
        )
        if not len(matched) or not len(unmatched):
            continue
        matched_index = int(matched[np.argmax(existence[matched])])
        bottom_index = int(unmatched[np.argmin(existence[unmatched])])
        scene_started = time.perf_counter()
        baseline = score_set(scene, parameters, "E3", grid, K_BEST)["nll"]
        removed_matched = score_set(
            remove_query(scene, matched_index), parameters, "E3", grid, K_BEST
        )["nll"]
        removed_bottom = score_set(
            remove_query(scene, bottom_index), parameters, "E3", grid, K_BEST
        )["nll"]
        duplicated = score_set(
            duplicate_query(scene, matched_index), parameters, "E3", grid, K_BEST
        )["nll"]
        runtimes.append(time.perf_counter() - scene_started)
        matched_change = removed_matched - baseline
        bottom_change = removed_bottom - baseline
        tokens.append(scene["sample_token"])
        matched_delta.append(matched_change)
        bottom_delta.append(bottom_change)
        paired_advantage.append(matched_change - bottom_change)
        duplicate_delta.append(duplicated - baseline)
        if len(tokens) % 100 == 0:
            print(
                json.dumps(
                    {
                        "real_injection_scenes": len(tokens),
                        "elapsed_seconds": time.time() - started,
                    }
                ),
                flush=True,
            )
    arrays = {
        "sample_tokens": np.asarray(tokens, dtype="U32"),
        "matched_query_deletion_delta": np.asarray(matched_delta, dtype=np.float64),
        "bottom_unmatched_deletion_delta": np.asarray(bottom_delta, dtype=np.float64),
        "matched_over_bottom_advantage": np.asarray(paired_advantage, dtype=np.float64),
        "duplicate_delta": np.asarray(duplicate_delta, dtype=np.float64),
    }
    if not len(tokens) or not all(
        np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"
    ):
        raise ValueError("empty or non-finite real injection audit")
    advantage = arrays["matched_over_bottom_advantage"]
    duplicate = arrays["duplicate_delta"]
    report = {
        "schema_version": "probset3d_f0_real_injection_audit_v1",
        "eligible_scene_count": len(tokens),
        "sample_array_digest": array_digest(arrays),
        "matched_deletion_over_bottom": {
            "positive_fraction": float(np.mean(advantage > 0.0)),
            "median_paired_advantage": float(np.median(advantage)),
            "passed": bool(np.mean(advantage > 0.0) > 0.5 and np.median(advantage) > 0.0),
        },
        "duplicate_contribution": {
            "positive_fraction": float(np.mean(duplicate > 0.0)),
            "median_delta": float(np.median(duplicate)),
            "passed": bool(np.mean(duplicate > 0.0) > 0.5 and np.median(duplicate) > 0.0),
        },
    }
    write_npz(output / "per_scene_injections.npz", arrays)
    strict_write(output / "scientific_summary.json", report)
    strict_write(
        output / "runtime.json",
        {
            "wall_time_seconds": time.time() - started,
            "median_seconds_per_scene": float(np.median(runtimes)),
            "maximum_seconds_per_scene": float(np.max(runtimes)),
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def analyze_real_information(args):
    score_path = Path(args.scores) / "per_scene_scores.npz"
    with np.load(score_path, allow_pickle=False) as value:
        count = np.asarray(value["gt_count"], dtype=np.float64)
        nll = np.asarray(value["E3_nll"], dtype=np.float64)
        error = np.asarray(value["detection_error"], dtype=np.float64)
    design = np.column_stack((np.ones(len(count)), count))
    predicted_nll = design @ np.linalg.lstsq(design, nll, rcond=None)[0]
    predicted_error = design @ np.linalg.lstsq(design, error, rcond=None)[0]
    nll_residual = nll - predicted_nll
    error_residual = error - predicted_error
    total_variance = float(np.sum((nll - nll.mean()) ** 2))
    count_r2 = (
        1.0 - float(np.sum(nll_residual**2)) / total_variance
        if total_variance > 0.0
        else 1.0
    )
    partial = spearmanr(nll_residual, error_residual)
    raw_spearman = spearmanr(nll, error)
    raw_pearson = pearsonr(nll, error)
    report = {
        "schema_version": "probset3d_f0_real_information_audit_v1",
        "sample_count": len(count),
        "model": "E3",
        "gt_count_only_linear_r2": float(count_r2),
        "partial_spearman_after_linear_count_residualization": {
            "rho": float(partial.statistic),
            "two_sided_p": float(partial.pvalue),
        },
        "raw_spearman": {
            "rho": float(raw_spearman.statistic),
            "two_sided_p": float(raw_spearman.pvalue),
        },
        "raw_pearson": {
            "r": float(raw_pearson.statistic),
            "two_sided_p": float(raw_pearson.pvalue),
        },
        "passed": bool(
            count_r2 < 0.80
            and partial.statistic > 0.10
            and partial.pvalue < 0.05
        ),
    }
    strict_write(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def finalize_audit(args):
    root = Path(args.root).resolve()
    p0_authority = strict_load(P0)
    train_tokens = strict_load(
        ROOT / p0_authority["splits"]["train"]["sample_tokens_path"]
    )["sample_tokens"]
    expected_train_token_set_sha256 = hashlib.sha256(
        "\n".join(sorted(train_tokens)).encode("ascii")
    ).hexdigest()
    fixtures = strict_load(root / "fixtures/mathematical_fixtures.json")
    information = strict_load(root / "real_information_audit.json")
    injections = strict_load(root / "real_injections/scientific_summary.json")
    parameters = strict_load(root / "reference_parameters.json")
    export_root = root / "exports"
    conditions = (
        "fusion-dev-clean-s0",
        "fusion-dev-camera-lowlight-s3",
        "fusion-dev-lidar-dropout-s3",
        "fusion-dev-lidar-beam-reduction-s3",
    )
    score_summaries = {}
    runtime_summaries = {}
    deterministic = {}
    for condition in conditions:
        first_root = root / "scores" / condition / "run1"
        second_root = root / "scores" / condition / "run2"
        first_summary_path = first_root / "scientific_summary.json"
        second_summary_path = second_root / "scientific_summary.json"
        first = strict_load(first_summary_path)
        second = strict_load(second_summary_path)
        score_summaries[condition] = first
        runtime_summaries[condition] = {
            "run1": strict_load(first_root / "runtime.json"),
            "run2": strict_load(second_root / "runtime.json"),
        }
        deterministic[condition] = {
            "strict_json_identical": (
                first_summary_path.read_bytes() == second_summary_path.read_bytes()
            ),
            "array_digest_identical": (
                first["score_array_digest"] == second["score_array_digest"]
            ),
        }

    dev_manifests = {}
    for condition in conditions:
        manifests = [
            value
            for _, value in iter_export_manifests(export_root / condition)
        ]
        dev_manifests[condition] = manifests
    train_manifests = [
        value
        for _, value in iter_export_manifests(export_root / "fusion-train-clean-s0")
    ]

    fixture_directions = (
        "delete_useful_query",
        "add_high_existence_false_positive",
        "duplicate_correct_query",
        "perturb_center",
        "perturb_yaw",
        "reduce_correct_class_probability",
        "collapse_covariance_around_wrong_mean",
        "inflate_covariance",
        "unmatched_r_to_zero",
        "unmatched_r_to_one",
        "zero_residual_rate_penalizes_unexplained_gt",
    )
    maximum_reconstruction = max(
        model["maximum_reconstruction_error"]
        for summary in score_summaries.values()
        for model in summary["models"].values()
    )
    maximum_median_runtime = max(
        run["median_seconds_per_scene"]
        for value in runtime_summaries.values()
        for run in value.values()
    )
    maximum_scene_runtime = max(
        run["maximum_seconds_per_scene"]
        for value in runtime_summaries.values()
        for run in value.values()
    )
    all_dev_manifests = [
        manifest for manifests in dev_manifests.values() for manifest in manifests
    ]
    checks = [
        (
            "G01_MATHEMATICAL_FIXTURES",
            fixtures["all_passed"],
            f"{sum(value['passed'] for value in fixtures['checks'].values())}/"
            f"{len(fixtures['checks'])} fixtures passed",
        ),
        (
            "G02_PERMUTATION_INVARIANCE",
            fixtures["maximum_permutation_error"] <= 1e-10,
            f"maximum absolute error={fixtures['maximum_permutation_error']:.3g}",
        ),
        (
            "G03_REAL_SCENES_FINITE",
            all(
                summary["sample_count"] == 2808 and summary["all_finite"]
                for summary in score_summaries.values()
            ),
            "clean and three S3 degradation exports: 2,808 finite scenes each",
        ),
        (
            "G04_RUNTIME",
            maximum_median_runtime < 0.5 and maximum_scene_runtime < 2.0,
            f"worst median={maximum_median_runtime:.6f}s; "
            f"worst maximum={maximum_scene_runtime:.6f}s",
        ),
        (
            "G05_ERROR_INJECTION_DIRECTIONS",
            all(fixtures["checks"][name]["passed"] for name in fixture_directions),
            f"{len(fixture_directions)}/{len(fixture_directions)} preregistered directions checked",
        ),
        (
            "G06_COUNT_CONTROLLED_INFORMATION",
            information["passed"],
            "R2={:.6f}; partial rho={:.6f}; p={:.3g}".format(
                information["gt_count_only_linear_r2"],
                information["partial_spearman_after_linear_count_residualization"]["rho"],
                information["partial_spearman_after_linear_count_residualization"]["two_sided_p"],
            ),
        ),
        (
            "G07_DECOMPOSITION_RECONSTRUCTION",
            maximum_reconstruction <= 1e-10
            and fixtures["maximum_decomposition_error"] <= 1e-10,
            f"maximum real-data absolute error={maximum_reconstruction:.3g}",
        ),
        (
            "G08_TWO_RUN_DETERMINISM",
            all(
                value["strict_json_identical"] and value["array_digest_identical"]
                for value in deterministic.values()
            ),
            "strict scientific JSON and per-scene array digests compared for all conditions",
        ),
        (
            "G09_MATCHED_QUERY_DELETION",
            injections["matched_deletion_over_bottom"]["passed"],
            "positive fraction={:.6f}; median paired advantage={:.6f}".format(
                injections["matched_deletion_over_bottom"]["positive_fraction"],
                injections["matched_deletion_over_bottom"]["median_paired_advantage"],
            ),
        ),
        (
            "G10_DUPLICATE_CONTRIBUTION",
            injections["duplicate_contribution"]["passed"],
            "positive fraction={:.6f}; median delta={:.6f}".format(
                injections["duplicate_contribution"]["positive_fraction"],
                injections["duplicate_contribution"]["median_delta"],
            ),
        ),
        (
            "G11_EXACT_TOPK_AUTHORITY",
            len(all_dev_manifests) == 4
            and all(
                manifest["accepted"]
                and manifest["sample_count"] == 2808
                and manifest["checks"]["topk_identity_failures"] == 0
                and manifest["checks"]["dense_finite_failures"] == 0
                for manifest in all_dev_manifests
            ),
            f"{len(all_dev_manifests)}/4 accepted complete dev manifests",
        ),
        (
            "G12_TRAIN_DEV_AUTHORITY",
            parameters["source_sample_count"] == 22508
            and parameters["source_token_set_sha256"]
            == expected_train_token_set_sha256
            and sum(value["sample_count"] for value in train_manifests) == 22508
            and all(
                not manifest["checks"]["official_val_used"]
                and not manifest["checks"]["internal_test_used"]
                and not manifest["checks"]["dc0_artifact_used"]
                and not manifest["checks"]["training_performed"]
                for manifest in train_manifests + all_dev_manifests
            ),
            "train=22,508; each dev condition=2,808; forbidden authorities unused",
        ),
    ]
    gate_records = [
        {"gate": name, "passed": bool(passed), "evidence": evidence}
        for name, passed, evidence in checks
    ]
    passed = all(value["passed"] for value in gate_records)
    failed_gates = [value["gate"] for value in gate_records if not value["passed"]]
    failure_classification = {
        "mathematical_or_numerical": [
            name
            for name in failed_gates
            if name
            in {
                "G01_MATHEMATICAL_FIXTURES",
                "G02_PERMUTATION_INVARIANCE",
                "G03_REAL_SCENES_FINITE",
                "G05_ERROR_INJECTION_DIRECTIONS",
                "G07_DECOMPOSITION_RECONSTRUCTION",
            }
        ],
        "empirical_information": [
            name
            for name in failed_gates
            if name
            in {
                "G06_COUNT_CONTROLLED_INFORMATION",
                "G09_MATCHED_QUERY_DELETION",
                "G10_DUPLICATE_CONTRIBUTION",
            }
        ],
        "infrastructure_or_authority": [
            name
            for name in failed_gates
            if name
            in {
                "G04_RUNTIME",
                "G08_TWO_RUN_DETERMINISM",
                "G11_EXACT_TOPK_AUTHORITY",
                "G12_TRAIN_DEV_AUTHORITY",
            }
        ],
    }
    endpoint = (
        "GO TO F1 PROBABILISTIC QUERY"
        if passed
        else "STOP PROBABILISTIC SET DIRECTION"
    )
    decision = {
        "schema_version": "probset3d_f0_decision_v1",
        "endpoint": endpoint,
        "passed_gate_count": sum(value["passed"] for value in gate_records),
        "required_gate_count": len(gate_records),
        "gates": gate_records,
        "failure_classification": failure_classification,
        "f0_scope": {
            "training_performed": False,
            "bevfusion_modified": False,
            "official_val_used": False,
            "internal_test_used": False,
            "dc0_partial_used": False,
        },
        "information_audit": information,
        "real_injection_audit": injections,
        "determinism": deterministic,
        "kbest_saturation": {
            condition: {
                model: {
                    "mean_k1_to_k5_gain": values["mean_k1_to_k5_gain"],
                    "mean_k5_to_k10_gain": values["mean_k5_to_k10_gain"],
                }
                for model, values in summary["models"].items()
            }
            for condition, summary in score_summaries.items()
        },
    }
    decision_root = root / "decision"
    if decision_root.exists() and any(decision_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite decision: {decision_root}")
    strict_write(decision_root / "decision.json", decision)
    worktree = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    excluded_archive_manifests = sorted(
        (root / "export_archive").glob("*/shard-*/accepted.json")
    )
    authority = {
        "schema_version": "probset3d_f0_authority_audit_v1",
        "reference_parameter_sha256": sha256(root / "reference_parameters.json"),
        "fixture_sha256": sha256(root / "fixtures/mathematical_fixtures.json"),
        "scorer_sha256": sha256(Path(__file__)),
        "dirty_worktree": bool(worktree),
        "git_status_sha256": hashlib.sha256(worktree.encode("utf-8")).hexdigest(),
        "git_status_entry_count": len(worktree.splitlines()),
        "exporter_manifest_sha256": {
            condition: [
                sha256(path)
                for path, _ in iter_export_manifests(export_root / condition)
            ]
            for condition in conditions + ("fusion-train-clean-s0",)
        },
        "forbidden_authorities_used": False,
        "excluded_non_authority_archives": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "reason": "pre-device-selection-fix export; never scored",
            }
            for path in excluded_archive_manifests
        ],
    }
    strict_write(decision_root / "authority_audit.json", authority)
    lines = [
        "# ProbSet3D F0 GO/STOP Decision",
        "",
        f"# `{endpoint}`",
        "",
        f"Result: {decision['passed_gate_count']}/{decision['required_gate_count']} frozen gates passed.",
        "",
        "| Gate | Result | Evidence |",
        "|---|---:|---|",
    ]
    for value in gate_records:
        lines.append(
            f"| `{value['gate']}` | {'PASS' if value['passed'] else 'FAIL'} | "
            f"{value['evidence']} |"
        )
    if failed_gates:
        lines.extend(
            [
                "",
                "## Failure classification",
                "",
                *[
                    f"- `{category}`: "
                    + (
                        ", ".join(f"`{name}`" for name in names)
                        if names
                        else "none"
                    )
                    for category, names in failure_classification.items()
                ],
            ]
        )
    lines.extend(
        [
            "",
            "## Scope integrity",
            "",
            "No training was performed; BEVFusion was not modified; official validation, "
            "internal test, and sealed DC0 partial artifacts were not used.",
            "",
            "This is the sole legal F0 endpoint under the preregistered two-terminal rule.",
        ]
    )
    report_path = decision_root / "decision_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".md.partial")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    generated = [
        *sorted((ROOT / "probset3d").glob("P0_*")),
        ROOT / "tools/probset3d_f0_export.py",
        ROOT / "tools/probset3d_f0_score.py",
        ROOT / "tests/test_probset3d_f0.py",
        root / "reference_parameters.json",
        root / "fixtures/mathematical_fixtures.json",
        root / "real_information_audit.json",
        root / "real_injections/per_scene_injections.npz",
        root / "real_injections/scientific_summary.json",
        root / "real_injections/runtime.json",
        *[
            path
            for condition in conditions
            for run in ("run1", "run2")
            for path in sorted((root / "scores" / condition / run).glob("*"))
        ],
        *[
            path
            for condition in conditions + ("fusion-train-clean-s0",)
            for path, _ in iter_export_manifests(export_root / condition)
        ],
        *excluded_archive_manifests,
        decision_root / "decision.json",
        decision_root / "authority_audit.json",
        decision_root / "decision_report.md",
    ]
    strict_write(
        decision_root / "manifest.json",
        {
            "schema_version": "probset3d_f0_decision_manifest_v1",
            "endpoint": endpoint,
            "artifacts": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in generated
            ],
        },
    )
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--export-root", default=str(EXPORT_ROOT))
    fit.add_argument("--output", default=str(OUT / "reference_parameters.json"))
    fit.set_defaults(function=fit_reference)
    score = subparsers.add_parser("score")
    score.add_argument("--export-root", default=str(EXPORT_ROOT))
    score.add_argument("--parameters", default=str(OUT / "reference_parameters.json"))
    score.add_argument("--run-id", required=True)
    score.add_argument("--models", nargs="+", default=["E0", "E1", "E2", "E3"])
    score.add_argument("--k", type=int, default=K_BEST)
    score.add_argument("--max-scenes", type=int)
    score.add_argument("--output", required=True)
    score.set_defaults(function=score_export)
    fixtures = subparsers.add_parser("fixtures")
    fixtures.add_argument(
        "--output", default=str(OUT / "fixtures/mathematical_fixtures.json")
    )
    fixtures.set_defaults(function=run_fixtures)
    injections = subparsers.add_parser("real-injections")
    injections.add_argument("--export-root", default=str(EXPORT_ROOT))
    injections.add_argument(
        "--parameters", default=str(OUT / "reference_parameters.json")
    )
    injections.add_argument(
        "--output", default=str(OUT / "real_injections")
    )
    injections.set_defaults(function=real_injection_audit)
    information = subparsers.add_parser("analyze-information")
    information.add_argument(
        "--scores", default=str(OUT / "scores/fusion-dev-clean-s0/run1")
    )
    information.add_argument(
        "--output", default=str(OUT / "real_information_audit.json")
    )
    information.set_defaults(function=analyze_real_information)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--root", default=str(OUT))
    finalize.set_defaults(function=finalize_audit)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.function(parsed)
