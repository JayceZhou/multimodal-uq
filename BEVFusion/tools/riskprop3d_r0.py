#!/usr/bin/env python3
"""Minimal perception-to-forecast uncertainty propagation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/riskprop3d-matplotlib")

import numpy as np
import torch
from pyquaternion import Quaternion
from scipy.special import logsumexp
from scipy.stats import chi2, spearmanr, vonmises
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from nuscenes.nuscenes import NuScenes
from probset3d.f1.implementation.probabilistic_query import ProbabilisticQueryHead
from tools import probset3d_f0_score as f0
from tools import probset3d_v3_formal as formal
from tools import probset3d_v3_pilot as pilot
from tools.geoprob3d_g0 import CONDITIONS
from tools.stage3d_b_native_assignment import detection_gt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/riskprop3d_r0"
AUTHORITY = ROOT / "outputs/probset3d_v3/f1_authority/run_a"
P0_AUTHORITY = ROOT / "probset3d/P0_DATA_SPLIT_AUTHORITY.json"
BASELINE = ROOT / "outputs/probset3d_v3/f1_formal/baseline.json"
CHECKPOINT_ROOT = ROOT / "outputs/probset3d_v3/f1_formal/checkpoints"
HEAD_SEEDS = (20260731, 20260801, 20260802)
PROPAGATION_SEEDS = (20260806, 20260807, 20260808)
PREDICTOR_SEED = 20260804
BOOTSTRAP_SEED = 20260805
HISTORY = 4
MIN_HISTORY = 2
HORIZON = 6
DT = 0.5
SAMPLES = 64
LEVELS = (0.50, 0.80, 0.90)
PREDICTORS = ("CV", "MLP")
GROUPS = ("P0", "P1", "P2", "P3")
STATE_FEATURES = 10
INPUT_DIM = (HISTORY + 1) * STATE_FEATURES + HISTORY + 10
TRAIN_EPOCHS = 15
TRAIN_BATCH = 1024
EVAL_CHUNK = 256
METRICS = (
    "energy", "nll", "mean_ade", "mean_fde", "min_ade", "min_fde",
    "miss", "coverage_50", "coverage_80", "coverage_90", "final_maha2",
    "final_det", "area90", "log_area90", "collision_probability",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_npz(path: Path, arrays) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def write_torch(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def variable(part, name: str, row: int):
    offsets = part[name + "_offsets"]
    return np.asarray(part[name][offsets[row] : offsets[row + 1]])


def load_infos(split: str):
    authority = strict_load(P0_AUTHORITY)["splits"][split]
    path = ROOT / authority["info_path"]
    if sha256(path) != authority["info_sha256"]:
        raise ValueError(f"info hash mismatch: {split}")
    with path.open("rb") as stream:
        value = pickle.load(stream)
    infos = value["infos"] if isinstance(value, dict) else value
    if len(infos) != authority["sample_count"]:
        raise ValueError(f"info count mismatch: {split}")
    return {item["token"]: item for item in infos}


def iter_clean_parts(split: str):
    manifest = strict_load(AUTHORITY / "manifest.json")
    if not manifest.get("accepted"):
        raise ValueError("F1-V3 authority is not accepted")
    for record in sorted(manifest["splits"][split]["files"], key=lambda item: item["file"]):
        path = AUTHORITY / split / record["file"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"authority part hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as value:
            yield {name: value[name] for name in value.files}


def frame_authority(infos):
    frames = {}
    for token, info in infos.items():
        lidar_to_ego = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        ego_to_global = Quaternion(info["ego2global_rotation"]).rotation_matrix
        rotation = ego_to_global @ lidar_to_ego
        origin = ego_to_global @ np.asarray(info["lidar2ego_translation"], dtype=np.float64) + np.asarray(info["ego2global_translation"], dtype=np.float64)
        frames[token] = (origin, rotation)
    return frames


def global_to_local(point, frame):
    origin, rotation = frame
    return rotation.T @ (np.asarray(point, dtype=np.float64) - origin)


def transform_box(box, source_frame, target_frame):
    box = np.asarray(box, dtype=np.float64)
    source_origin, source_rotation = source_frame
    target_origin, target_rotation = target_frame
    position = target_rotation.T @ (source_rotation @ box[:3] + source_origin - target_origin)
    velocity = target_rotation.T @ (source_rotation @ np.asarray([box[7], box[8], 0.0]))
    heading = target_rotation.T @ (source_rotation @ np.asarray([math.cos(box[6]), math.sin(box[6]), 0.0]))
    return np.asarray([
        position[0], position[1], position[2], box[3], box[4], box[5],
        math.atan2(heading[1], heading[0]), velocity[0], velocity[1],
    ], dtype=np.float64)


def state_features(box):
    box = np.asarray(box, dtype=np.float64)
    return np.asarray([
        box[0], box[1], box[2], *np.log(np.clip(box[3:6], 1e-6, None)),
        box[7], box[8], math.sin(box[6]), math.cos(box[6]),
    ], dtype=np.float32)


def predictor_input(history, current, mask, class_index):
    one_hot = np.zeros(10, dtype=np.float32)
    one_hot[int(class_index)] = 1.0
    return np.concatenate([*(state_features(box) for box in history), state_features(current), np.asarray(mask, dtype=np.float32), one_hot])


def source_rows(parts, clean: bool):
    for part in parts:
        count = len(part["sample_tokens"])
        for row in range(count):
            if clean:
                yield {
                    "token": str(part["sample_tokens"][row]),
                    "boxes": np.asarray(part["decoded_box_mean"][row]),
                    "features": np.asarray(part["query_features"][row]),
                    "logits": np.asarray(part["class_logits"][row]),
                    "seed": np.asarray(part["seed_score"][row]),
                    "matched": np.asarray(part["native_matched_gt"][row]),
                    "gt_boxes": variable(part, "gt_boxes", row),
                    "gt_labels": variable(part, "gt_labels", row),
                }
            else:
                yield {
                    "token": str(part["sample_tokens"][row]),
                    "boxes": np.asarray(part["query_boxes"][row]),
                    "features": np.asarray(part["query_features"][row]),
                    "logits": np.asarray(part["query_logits"][row]),
                    "seed": np.asarray(part["seed_probabilities"][row]),
                    "matched": np.asarray(part["native_matched_gt"][row]),
                    "gt_boxes": variable(part, "gt_boxes", row),
                    "gt_labels": variable(part, "gt_labels", row),
                }


def observation_map(rows, infos, frames, nusc, include_head: bool):
    baseline = strict_load(BASELINE)
    observations, order = {}, []
    for row in rows:
        token = row["token"]
        if token in observations or token not in infos:
            raise ValueError(f"invalid sample token in observation source: {token}")
        source_indices, gt_boxes, gt_labels = detection_gt(infos[token])
        if not np.array_equal(np.asarray(row["gt_labels"], dtype=np.int64), gt_labels) or np.max(np.abs(np.asarray(row["gt_boxes"]) - gt_boxes), initial=0.0) > 1e-4:
            raise ValueError(f"GT authority mismatch: {token}")
        sample = nusc.get("sample", token)
        if len(sample["anns"]) != len(infos[token]["gt_boxes"]):
            raise ValueError(f"annotation row mismatch: {token}")
        matched = np.asarray(row["matched"], dtype=np.int64)
        query_by_gt = np.full(len(gt_boxes), -1, dtype=np.int64)
        positive = np.flatnonzero(matched >= 0)
        if len(set(matched[positive].tolist())) != len(positive):
            raise ValueError(f"assignment is not one-to-one: {token}")
        query_by_gt[matched[positive]] = positive
        current = {}
        predicted = np.asarray(row["logits"]).argmax(-1)
        densities = formal.density_score(np.asarray(row["features"]), predicted, baseline["density"])
        for gt_index, query in enumerate(query_by_gt):
            if query < 0:
                continue
            annotation_token = sample["anns"][int(source_indices[gt_index])]
            annotation = nusc.get("sample_annotation", annotation_token)
            if np.linalg.norm(global_to_local(annotation["translation"], frames[token]) - gt_boxes[gt_index, :3]) > 1e-4:
                raise ValueError(f"annotation center mismatch: {token}/{gt_index}")
            instance = annotation["instance_token"]
            if instance in current:
                raise ValueError(f"duplicate instance assignment: {token}/{instance}")
            value = {
                "annotation": annotation_token,
                "box": np.asarray(row["boxes"][query], dtype=np.float32),
                "gt_box": np.asarray(gt_boxes[gt_index], dtype=np.float32),
                "class": int(gt_labels[gt_index]),
                "query_id": int(query),
                "score": float(np.asarray(row["logits"])[query].max()),
                "density": float(densities[query]),
            }
            if include_head:
                value.update(
                    feature=np.asarray(row["features"][query], dtype=np.float32),
                    logits=np.asarray(row["logits"][query], dtype=np.float32),
                    seed=float(np.asarray(row["seed"])[query]),
                )
            current[instance] = value
        observations[token] = current
        order.append(token)
    if len(order) != len(infos) or set(order) != set(infos):
        raise ValueError("observation source does not conserve the split")
    return order, observations


def future_tokens(nusc, token):
    values, current = [], token
    for _ in range(HORIZON):
        current = nusc.get("sample", current)["next"]
        if not current:
            return None
        values.append(current)
    return values


def build_tracks(order, observations, infos, frames, nusc, include_head: bool):
    fields = {name: [] for name in (
        "input", "gt_input", "future", "ego_future", "scene_token",
        "sample_token", "instance_token", "class", "range", "velocity",
        "score", "density", "history_length", "history_innovation",
        "collision_event",
    )}
    if include_head:
        fields.update({name: [] for name in ("feature", "logits", "box", "seed")})
    for token in order:
        sample = nusc.get("sample", token)
        targets = future_tokens(nusc, token)
        if targets is None:
            continue
        target_frame = frames[token]
        for instance, current in observations[token].items():
            annotation = nusc.get("sample_annotation", current["annotation"])
            future_annotations = []
            ann = annotation
            valid = True
            for target_token in targets:
                if not ann["next"]:
                    valid = False
                    break
                ann = nusc.get("sample_annotation", ann["next"])
                if ann["sample_token"] != target_token:
                    valid = False
                    break
                future_annotations.append(ann)
            if not valid:
                continue
            history_values, previous = [], sample["prev"]
            while previous and len(history_values) < HISTORY:
                value = observations.get(previous, {}).get(instance)
                if value is None:
                    break
                history_values.append((previous, value))
                previous = nusc.get("sample", previous)["prev"]
            if len(history_values) < MIN_HISTORY:
                continue
            history_values.reverse()
            transformed = [transform_box(value["box"], frames[past_token], target_frame) for past_token, value in history_values]
            mask = [0.0] * (HISTORY - len(transformed)) + [1.0] * len(transformed)
            history = [transformed[0]] * (HISTORY - len(transformed)) + transformed
            box = np.asarray(current["box"], dtype=np.float64)
            gt_box = np.asarray(current["gt_box"], dtype=np.float64)
            future = np.stack([global_to_local(value["translation"], target_frame)[:2] for value in future_annotations])
            ego_future = np.stack([global_to_local(frames[target_token][0], target_frame)[:2] for target_token in targets])
            innovation = np.linalg.norm(box[:2] - (history[-1][:2] + DT * history[-1][7:9]))
            fields["input"].append(predictor_input(history, box, mask, current["class"]))
            fields["gt_input"].append(predictor_input(history, gt_box, mask, current["class"]))
            fields["future"].append(future.astype(np.float32))
            fields["ego_future"].append(ego_future.astype(np.float32))
            fields["scene_token"].append(sample["scene_token"])
            fields["sample_token"].append(token)
            fields["instance_token"].append(instance)
            fields["class"].append(current["class"])
            fields["range"].append(float(np.linalg.norm(box[:2])))
            fields["velocity"].append(float(np.linalg.norm(box[7:9])))
            fields["score"].append(current["score"])
            fields["density"].append(current["density"])
            fields["history_length"].append(len(transformed))
            fields["history_innovation"].append(float(innovation))
            fields["collision_event"].append(float(np.min(np.linalg.norm(future - ego_future, axis=-1)) <= 2.5))
            if include_head:
                fields["feature"].append(current["feature"])
                fields["logits"].append(current["logits"])
                fields["box"].append(current["box"])
                fields["seed"].append(current["seed"])
    arrays = {}
    float_fields = {"input", "gt_input", "future", "ego_future", "range", "velocity", "score", "density", "history_innovation", "collision_event", "feature", "logits", "box", "seed"}
    int_fields = {"class", "history_length"}
    for name, values in fields.items():
        if name in float_fields:
            arrays[name] = np.asarray(values, dtype=np.float32)
        elif name in int_fields:
            arrays[name] = np.asarray(values, dtype=np.int16)
        else:
            arrays[name] = np.asarray(values)
    keys = np.char.add(np.char.add(arrays["sample_token"], "/"), arrays["instance_token"])
    if len(keys) == 0 or len(np.unique(keys)) != len(keys):
        raise ValueError("track keys are empty or non-unique")
    if arrays["input"].shape[1] != INPUT_DIM or not all(np.isfinite(value).all() for name, value in arrays.items() if name in float_fields):
        raise ValueError("non-finite or invalid track arrays")
    return arrays


def prepare() -> None:
    expected = [OUT / "tracks/train.npz", OUT / "tracks/label_cal.npz"] + [OUT / "tracks" / f"{condition}.npz" for condition in CONDITIONS]
    if any(path.exists() for path in expected):
        raise FileExistsError("R0 track output already exists")
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(ROOT / "data/nuscenes"), verbose=False)
    for split in ("train", "label_cal"):
        infos = load_infos(split)
        frames = frame_authority(infos)
        order, observations = observation_map(source_rows(iter_clean_parts(split), True), infos, frames, nusc, include_head=False)
        tracks = build_tracks(order, observations, infos, frames, nusc, include_head=False)
        tracks["meta__split"] = np.asarray(split)
        write_npz(OUT / "tracks" / f"{split}.npz", tracks)
        print(json.dumps({"prepared": split, "tracks": len(tracks["input"])}))
    infos = load_infos("dev")
    frames = frame_authority(infos)
    for condition, root in CONDITIONS.items():
        order, observations = observation_map(source_rows((part for _, part in f0.iter_parts(root)), False), infos, frames, nusc, include_head=True)
        tracks = build_tracks(order, observations, infos, frames, nusc, include_head=True)
        tracks["meta__condition"] = np.asarray(condition)
        write_npz(OUT / "tracks" / f"{condition}.npz", tracks)
        print(json.dumps({"prepared": condition, "tracks": len(tracks["input"])}))


class ForecastMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(INPUT_DIM, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, HORIZON * 2))

    def forward(self, value):
        return self.network(value).reshape(-1, HORIZON, 2)


def cv_prediction(inputs):
    inputs = np.asarray(inputs, dtype=np.float64)
    current = inputs[:, HISTORY * STATE_FEATURES : (HISTORY + 1) * STATE_FEATURES]
    times = np.arange(1, HORIZON + 1, dtype=np.float64) * DT
    return current[:, None, :2] + times[None, :, None] * current[:, None, 6:8]


def mlp_prediction(model, inputs, input_mean, input_scale, target_mean, target_scale, device):
    outputs = []
    for begin in range(0, len(inputs), 8192):
        value = torch.from_numpy(((np.asarray(inputs[begin : begin + 8192], dtype=np.float32) - input_mean) / input_scale)).to(device)
        with torch.inference_mode():
            residual = model(value).cpu().numpy() * target_scale + target_mean
        outputs.append(residual)
    return cv_prediction(inputs) + np.concatenate(outputs)


def covariance(errors):
    values = []
    for horizon in range(HORIZON):
        current = np.cov(np.asarray(errors[:, horizon], dtype=np.float64), rowvar=False)
        eigenvalue, eigenvector = np.linalg.eigh(current)
        values.append(eigenvector @ np.diag(np.maximum(eigenvalue, 1e-4)) @ eigenvector.T)
    return np.stack(values)


def train(device_index: int) -> None:
    if device_index not in range(4):
        raise ValueError("GPU must be in 0--3")
    random.seed(PREDICTOR_SEED)
    np.random.seed(PREDICTOR_SEED)
    torch.manual_seed(PREDICTOR_SEED)
    torch.cuda.manual_seed_all(PREDICTOR_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    train_tracks = np.load(OUT / "tracks/train.npz", allow_pickle=False)
    calibration = np.load(OUT / "tracks/label_cal.npz", allow_pickle=False)
    inputs = np.asarray(train_tracks["input"], dtype=np.float32)
    future = np.asarray(train_tracks["future"], dtype=np.float32)
    input_mean = inputs.mean(0)
    input_scale = np.maximum(inputs.std(0), 1e-4)
    residual = future - cv_prediction(inputs)
    target_mean = residual.mean((0, 1), keepdims=True).astype(np.float32)
    target_scale = np.maximum(residual.std((0, 1), keepdims=True), 1e-4).astype(np.float32)
    model = ForecastMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(PREDICTOR_SEED)
    target = ((residual - target_mean) / target_scale).astype(np.float32)
    for _ in range(TRAIN_EPOCHS):
        permutation = rng.permutation(len(inputs))
        for begin in range(0, len(inputs), TRAIN_BATCH):
            index = permutation[begin : begin + TRAIN_BATCH]
            x = torch.from_numpy((inputs[index] - input_mean) / input_scale).to(device)
            y = torch.from_numpy(target[index]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(x) - y) ** 2)
            if not torch.isfinite(loss):
                raise ValueError("non-finite predictor loss")
            loss.backward()
            optimizer.step()
    cal_input = np.asarray(calibration["gt_input"], dtype=np.float32)
    cal_future = np.asarray(calibration["future"], dtype=np.float64)
    cv_error = cal_future - cv_prediction(cal_input)
    mlp_error = cal_future - mlp_prediction(model, cal_input, input_mean, input_scale, target_mean, target_scale, device)
    checkpoint = {
        "architecture": [INPUT_DIM, 128, 128, HORIZON * 2],
        "epochs": TRAIN_EPOCHS,
        "input_mean": torch.from_numpy(input_mean.astype(np.float32)),
        "input_scale": torch.from_numpy(input_scale.astype(np.float32)),
        "model": model.state_dict(),
        "process_covariance": {"CV": torch.from_numpy(covariance(cv_error)), "MLP": torch.from_numpy(covariance(mlp_error))},
        "seed": PREDICTOR_SEED,
        "target_mean": torch.from_numpy(target_mean),
        "target_scale": torch.from_numpy(target_scale),
        "train_tracks_sha256": sha256(OUT / "tracks/train.npz"),
        "label_cal_tracks_sha256": sha256(OUT / "tracks/label_cal.npz"),
    }
    write_torch(OUT / "predictor.pt", checkpoint)
    train_tracks.close()
    calibration.close()
    print(json.dumps({"trained": True, "checkpoint": str((OUT / 'predictor.pt').relative_to(ROOT))}))


def load_predictor(device):
    checkpoint = torch.load(OUT / "predictor.pt", map_location="cpu", weights_only=True)
    model = ForecastMLP().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model, {
        "input_mean": checkpoint["input_mean"].numpy(),
        "input_scale": checkpoint["input_scale"].numpy(),
        "target_mean": checkpoint["target_mean"].numpy(),
        "target_scale": checkpoint["target_scale"].numpy(),
        "covariance": {name: checkpoint["process_covariance"][name].numpy() for name in PREDICTORS},
    }


def head_outputs(head, tracks, device):
    result = {name: [] for name in ("mean_state", "scale", "degrees_of_freedom", "yaw_mean", "yaw_scale", "yaw_kappa")}
    total = len(tracks["input"])
    for begin in range(0, total, 4096):
        stop = min(begin + 4096, total)
        feature = torch.from_numpy(np.asarray(tracks["feature"][begin:stop]))[:, None].to(device)
        logits = torch.from_numpy(np.asarray(tracks["logits"][begin:stop]))[:, None].to(device)
        box = torch.from_numpy(np.asarray(tracks["box"][begin:stop]))[:, None].to(device)
        seed = torch.from_numpy(np.asarray(tracks["seed"][begin:stop]))[:, None].to(device)
        with torch.inference_mode():
            output = head(feature, logits, box, seed, torch.zeros(stop - begin, 1, device=device))
        for name in result:
            result[name].append(output[name][:, 0].detach().cpu().numpy())
    return {name: np.concatenate(values) for name, values in result.items()}


def sample_states(output, rng, count, student: bool, z, radial_uniform, yaw_normal):
    mean = np.asarray(output["mean_state"], dtype=np.float64)
    scale = np.asarray(output["scale"], dtype=np.float64)
    if student:
        degrees = np.asarray(output["degrees_of_freedom"], dtype=np.float64)
        factor = np.sqrt(degrees[:, None] / chi2.ppf(radial_uniform, degrees[:, None]))
        linear = mean[:, None] + scale[:, None] * z * factor[..., None]
        yaw = rng.vonmises(np.asarray(output["yaw_mean"])[:, None], np.asarray(output["yaw_kappa"])[:, None], size=(len(mean), count))
    else:
        linear = mean[:, None] + scale[:, None] * z
        yaw = np.asarray(output["yaw_mean"])[:, None] + np.asarray(output["yaw_scale"])[:, None] * yaw_normal
    return linear, np.arctan2(np.sin(yaw), np.cos(yaw))


def replace_current(inputs, linear, yaw):
    value = np.repeat(np.asarray(inputs, dtype=np.float32)[:, None, :], linear.shape[1], axis=1)
    offset = HISTORY * STATE_FEATURES
    value[:, :, offset : offset + 8] = linear.astype(np.float32)
    value[:, :, offset + 8] = np.sin(yaw).astype(np.float32)
    value[:, :, offset + 9] = np.cos(yaw).astype(np.float32)
    return value


def predict_centers(predictor, inputs, model, parameters, device):
    shape = inputs.shape
    flat = np.asarray(inputs).reshape(-1, INPUT_DIM)
    if predictor == "CV":
        output = cv_prediction(flat)
    else:
        output = mlp_prediction(
            model, flat, parameters["input_mean"], parameters["input_scale"],
            parameters["target_mean"], parameters["target_scale"], device,
        )
    return output.reshape(*shape[:-1], HORIZON, 2)


def process_noise(rng, covariance, tracks, samples):
    standard = rng.standard_normal((tracks, samples, HORIZON, 2))
    cholesky = np.linalg.cholesky(np.asarray(covariance, dtype=np.float64))
    return np.einsum("nmhi,hji->nmhj", standard, cholesky)


def distribution_metrics(centers_a, centers_b, future, ego_future, covariance, noise_a, noise_b):
    centers_a = np.asarray(centers_a, dtype=np.float64)
    centers_b = np.asarray(centers_b, dtype=np.float64)
    future = np.asarray(future, dtype=np.float64)
    if centers_a.shape[1] == 1:
        sample_a = centers_a + noise_a
        sample_b = centers_b + noise_b
    else:
        sample_a = centers_a + noise_a
        sample_b = centers_b + noise_b
    first = np.linalg.norm((sample_a - future[:, None]).reshape(len(future), sample_a.shape[1], -1), axis=-1).mean(1)
    second = np.linalg.norm((sample_a - sample_b).reshape(len(future), sample_a.shape[1], -1), axis=-1).mean(1)
    energy = first - 0.5 * second

    inverse_process = np.linalg.inv(covariance)
    logdet_process = np.linalg.slogdet(covariance)[1]
    difference = future[:, None] - centers_a
    quadratic = np.einsum("nmhi,hij,nmhj->nmh", difference, inverse_process, difference)
    log_density = -0.5 * (2.0 * math.log(2.0 * math.pi) + logdet_process[None, None] + quadratic)
    nll = -(logsumexp(log_density.sum(-1), axis=1) - math.log(centers_a.shape[1]))

    mean = centers_a.mean(1)
    mean_error = np.linalg.norm(mean - future, axis=-1)
    sample_error = np.linalg.norm(centers_a - future[:, None], axis=-1)
    if centers_a.shape[1] > 1:
        centered = centers_a - mean[:, None]
        epistemic = np.einsum("nmhi,nmhj->nhij", centered, centered) / centers_a.shape[1]
    else:
        epistemic = np.zeros((len(future), HORIZON, 2, 2), dtype=np.float64)
    total_covariance = epistemic + covariance[None]
    determinant = np.maximum(np.linalg.det(total_covariance), 1e-12)
    inverse = np.linalg.inv(total_covariance)
    residual = future - mean
    mahalanobis = np.einsum("nhi,nhij,nhj->nh", residual, inverse, residual)
    result = {
        "energy": energy,
        "nll": nll,
        "mean_ade": mean_error.mean(-1),
        "mean_fde": mean_error[:, -1],
        "min_ade": sample_error.mean(-1).min(1),
        "min_fde": sample_error[:, :, -1].min(1),
        "miss": (sample_error[:, :, -1].min(1) > 2.0).astype(np.float64),
        "final_maha2": mahalanobis[:, -1],
        "final_det": determinant[:, -1],
        "area90": math.pi * chi2.ppf(0.90, 2) * np.sqrt(determinant[:, -1]),
        "log_area90": np.log(math.pi * chi2.ppf(0.90, 2) * np.sqrt(determinant[:, -1])),
        "collision_probability": (np.linalg.norm(sample_a - ego_future[:, None], axis=-1).min(-1) <= 2.5).mean(1),
    }
    for level in LEVELS:
        result[f"coverage_{int(level * 100)}"] = (mahalanobis <= chi2.ppf(level, 2)).mean(-1)
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("non-finite propagated metric")
    return result


def evaluate(replicate: int, device_index: int) -> None:
    if replicate not in range(3) or device_index not in range(4):
        raise ValueError("replicate must be 0--2 and GPU must be 0--3")
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    model, parameters = load_predictor(device)
    checkpoint = torch.load(CHECKPOINT_ROOT / f"seed_{HEAD_SEEDS[replicate]}.pt", map_location="cpu", weights_only=True)
    heads = {name: ProbabilisticQueryHead().to(device).eval() for name in ("M1", "M2")}
    for name, head in heads.items():
        head.load_state_dict(checkpoint["models"][name])
    arrays = {
        "meta__head_seed": np.asarray(HEAD_SEEDS[replicate], dtype=np.int64),
        "meta__propagation_seed": np.asarray(PROPAGATION_SEEDS[replicate], dtype=np.int64),
    }
    for condition_index, condition in enumerate(CONDITIONS):
        tracks = np.load(OUT / "tracks" / f"{condition}.npz", allow_pickle=False)
        outputs = {name: head_outputs(head, tracks, device) for name, head in heads.items()}
        storage = {predictor: {group: {metric: [] for metric in METRICS} for group in GROUPS} for predictor in PREDICTORS}
        rng = np.random.default_rng(PROPAGATION_SEEDS[replicate] + condition_index * 1009)
        for begin in range(0, len(tracks["input"]), EVAL_CHUNK):
            stop = min(begin + EVAL_CHUNK, len(tracks["input"]))
            source = np.asarray(tracks["input"][begin:stop])
            oracle = np.asarray(tracks["gt_input"][begin:stop])
            future = np.asarray(tracks["future"][begin:stop])
            ego_future = np.asarray(tracks["ego_future"][begin:stop])
            count = stop - begin
            sampled = {"P2": [], "P3": []}
            for _draw in range(2):
                z = rng.standard_normal((count, SAMPLES, 8))
                radial = rng.uniform(1e-7, 1.0 - 1e-7, (count, SAMPLES))
                yaw_normal = rng.standard_normal((count, SAMPLES))
                for name, student in (("P2", False), ("P3", True)):
                    output = {key: value[begin:stop] for key, value in outputs["M2" if student else "M1"].items()}
                    state = sample_states(output, rng, SAMPLES, student, z, radial, yaw_normal)
                    sampled[name].append(replace_current(source, *state))
            for predictor in PREDICTORS:
                covariance_value = parameters["covariance"][predictor]
                noise_a = process_noise(rng, covariance_value, stop - begin, SAMPLES)
                noise_b = process_noise(rng, covariance_value, stop - begin, SAMPLES)
                center_sets = {
                    "P0": (predict_centers(predictor, oracle[:, None], model, parameters, device),) * 2,
                    "P1": (predict_centers(predictor, source[:, None], model, parameters, device),) * 2,
                    "P2": tuple(predict_centers(predictor, value, model, parameters, device) for value in sampled["P2"]),
                    "P3": tuple(predict_centers(predictor, value, model, parameters, device) for value in sampled["P3"]),
                }
                for group, centers in center_sets.items():
                    metrics = distribution_metrics(centers[0], centers[1], future, ego_future, covariance_value, noise_a, noise_b)
                    for metric in METRICS:
                        storage[predictor][group][metric].append(np.asarray(metrics[metric], dtype=np.float32))
        for name in ("scene_token", "sample_token", "instance_token", "class", "range", "velocity", "score", "density", "history_length", "history_innovation", "collision_event"):
            arrays[f"{condition}__common__{name}"] = np.asarray(tracks[name])
        for predictor in PREDICTORS:
            for group in GROUPS:
                for metric in METRICS:
                    arrays[f"{condition}__{predictor}__{group}__{metric}"] = np.concatenate(storage[predictor][group][metric])
        tracks.close()
    path = OUT / "metrics" / f"replicate_{replicate}.npz"
    write_npz(path, arrays)
    print(json.dumps({"evaluated": True, "replicate": replicate, "output": str(path.relative_to(ROOT))}))


def scene_mean(value, scene_token):
    scenes, inverse = np.unique(scene_token, return_inverse=True)
    total = np.bincount(inverse, weights=np.asarray(value, dtype=np.float64), minlength=len(scenes))
    count = np.bincount(inverse, minlength=len(scenes))
    return scenes, total / np.maximum(count, 1)


def bootstrap(value, seed=BOOTSTRAP_SEED, repeats=10000):
    value = np.asarray(value, dtype=np.float64)
    value = value[np.isfinite(value)]
    if not len(value):
        return {"mean": None, "ci95": [None, None], "count": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        means[index] = value[rng.integers(0, len(value), len(value))].mean()
    return {
        "mean": float(value.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "count": int(len(value)),
    }


def risk_coverage(uncertainty, error):
    order = np.argsort(np.asarray(uncertainty), kind="stable")
    error = np.asarray(error, dtype=np.float64)
    fractions = np.linspace(0.10, 1.0, 91)
    curve = np.asarray([error[order[: max(1, int(len(error) * fraction))]].mean() for fraction in fractions])
    return float(np.trapz(curve, fractions))


def scene_risk_differences(scene, event, error, risk_p2, risk_p3, collision_event, collision_p2, collision_p3):
    values = {name: [] for name in ("high_fde_auprc", "high_fde_auroc", "risk_coverage", "false_safe", "collision_auprc")}
    for scene_token in np.unique(scene):
        selected = scene == scene_token
        current_event = np.asarray(event[selected], dtype=np.int64)
        if len(np.unique(current_event)) == 2:
            values["high_fde_auprc"].append(average_precision_score(current_event, risk_p3[selected]) - average_precision_score(current_event, risk_p2[selected]))
            values["high_fde_auroc"].append(roc_auc_score(current_event, risk_p3[selected]) - roc_auc_score(current_event, risk_p2[selected]))
        values["risk_coverage"].append(risk_coverage(risk_p2[selected], error[selected]) - risk_coverage(risk_p3[selected], error[selected]))
        if current_event.sum():
            safe_p2 = risk_p2[selected] <= np.quantile(risk_p2[selected], 0.80)
            safe_p3 = risk_p3[selected] <= np.quantile(risk_p3[selected], 0.80)
            values["false_safe"].append((current_event * safe_p2).sum() / current_event.sum() - (current_event * safe_p3).sum() / current_event.sum())
        current_collision = np.asarray(collision_event[selected], dtype=np.int64)
        if len(np.unique(current_collision)) == 2:
            values["collision_auprc"].append(average_precision_score(current_collision, collision_p3[selected]) - average_precision_score(current_collision, collision_p2[selected]))
    return {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}


def absolute_risk(event, error, risk, collision_event=None, collision_risk=None):
    event = np.asarray(event, dtype=np.int64)
    risk = np.asarray(risk, dtype=np.float64)
    safe = risk <= np.quantile(risk, 0.80)
    result = {
        "high_fde_auprc": float(average_precision_score(event, risk)) if len(np.unique(event)) == 2 else None,
        "high_fde_auroc": float(roc_auc_score(event, risk)) if len(np.unique(event)) == 2 else None,
        "risk_coverage": risk_coverage(risk, error),
        "false_safe": float((event * safe).sum() / event.sum()) if event.sum() else None,
    }
    if collision_event is not None and len(np.unique(collision_event)) == 2:
        result["collision_auprc"] = float(average_precision_score(collision_event, collision_risk))
    else:
        result["collision_auprc"] = None
    return result


def nested_information(runs, predictor):
    rows = []
    for condition in CONDITIONS:
        source = runs[0]
        common = lambda name: np.asarray(source[f"{condition}__common__{name}"])
        target = np.mean(np.stack([np.asarray(run[f"{condition}__{predictor}__P1__mean_fde"], dtype=np.float64) for run in runs]), axis=0)
        p2 = np.mean(np.stack([np.asarray(run[f"{condition}__{predictor}__P2__log_area90"], dtype=np.float64) for run in runs]), axis=0)
        p3 = np.mean(np.stack([np.asarray(run[f"{condition}__{predictor}__P3__log_area90"], dtype=np.float64) for run in runs]), axis=0)
        numeric = np.column_stack((
            common("score"), common("density"), common("range"), common("velocity"),
            common("history_length"), common("history_innovation"), p2,
        )).astype(np.float64)
        numeric = (numeric - numeric.mean(0)) / np.maximum(numeric.std(0), 1e-8)
        cls = common("class").astype(np.int64)
        rows.append((numeric, np.eye(10)[cls][:, 1:], np.log1p(target), p3))
    numeric = np.concatenate([row[0] for row in rows])
    one_hot = np.concatenate([row[1] for row in rows])
    target = np.concatenate([row[2] for row in rows])
    p3 = np.concatenate([row[3] for row in rows])
    base = np.column_stack((np.ones(len(target)), numeric, one_hot))
    beta = np.linalg.lstsq(base, target, rcond=None)[0]
    target_residual = target - base @ beta
    gamma = np.linalg.lstsq(base, p3, rcond=None)[0]
    p3_residual = p3 - base @ gamma
    extended = np.column_stack((base, p3))
    base_error = ((target - base @ beta) ** 2).sum()
    extended_error = ((target - extended @ np.linalg.lstsq(extended, target, rcond=None)[0]) ** 2).sum()
    total = ((target - target.mean()) ** 2).sum()
    return {
        "delta_r2": float((base_error - extended_error) / total),
        "partial_correlation": float(np.corrcoef(target_residual, p3_residual)[0, 1]),
        "spearman": float(spearmanr(p3_residual, target_residual).statistic),
        "track_count": int(len(target)),
    }


def finalize() -> None:
    runs = [np.load(OUT / "metrics" / f"replicate_{index}.npz", allow_pickle=False) for index in range(3)]
    for condition in CONDITIONS:
        for name in ("scene_token", "sample_token", "instance_token", "class", "range", "velocity", "score", "density", "history_length", "history_innovation", "collision_event"):
            reference = runs[0][f"{condition}__common__{name}"]
            if not all(np.array_equal(reference, run[f"{condition}__common__{name}"]) for run in runs[1:]):
                raise ValueError(f"replicate source mismatch: {condition}/{name}")

    def metric(run, condition, predictor, group, name):
        return np.asarray(run[f"{condition}__{predictor}__{group}__{name}"], dtype=np.float64)

    def paired(condition, predictor, baseline, name):
        per_replicate = []
        reference_scenes = None
        for run in runs:
            scene = np.asarray(run[f"{condition}__common__scene_token"])
            scenes, value = scene_mean(metric(run, condition, predictor, baseline, name) - metric(run, condition, predictor, "P3", name), scene)
            reference_scenes = scenes if reference_scenes is None else reference_scenes
            if not np.array_equal(scenes, reference_scenes):
                raise ValueError("scene authority mismatch")
            per_replicate.append(value)
        return bootstrap(np.mean(np.stack(per_replicate), axis=0))

    clean = {
        predictor: {
            name: {baseline: paired("clean_s0", predictor, baseline, name) for baseline in ("P1", "P2")}
            for name in ("energy", "nll")
        }
        for predictor in PREDICTORS
    }

    def family_paired(family, predictor, name, severities=(1, 2, 3)):
        rows = []
        for run in runs:
            current = []
            for severity in severities:
                condition = f"{family}_s{severity}"
                scene = np.asarray(run[f"{condition}__common__scene_token"])
                scenes, value = scene_mean(metric(run, condition, predictor, "P2", name) - metric(run, condition, predictor, "P3", name), scene)
                current.append(dict(zip(scenes.tolist(), value.tolist())))
            keys = sorted(set().union(*(set(value) for value in current)))
            rows.append(np.asarray([np.mean([value[key] for value in current if key in value]) for key in keys]))
        return bootstrap(np.mean(np.stack(rows), axis=0))

    family_energy = {predictor: {family: family_paired(family, predictor, "energy") for family in ("lowlight", "dropout", "beam")} for predictor in PREDICTORS}
    family_energy_no_s3 = {predictor: {family: family_paired(family, predictor, "energy", (1, 2)) for family in ("lowlight", "dropout", "beam")} for predictor in PREDICTORS}

    sharpness = {predictor: {} for predictor in PREDICTORS}
    point_harm = {predictor: {} for predictor in PREDICTORS}
    for predictor in PREDICTORS:
        condition_values = {}
        point_values = {}
        for condition in CONDITIONS:
            replicas, points = [], []
            for run in runs:
                p2_maha = metric(run, condition, predictor, "P2", "final_maha2")
                p3_maha = metric(run, condition, predictor, "P3", "final_maha2")
                p2_det = metric(run, condition, predictor, "P2", "final_det")
                p3_det = metric(run, condition, predictor, "P3", "final_det")
                q2, q3 = np.quantile(p2_maha, 0.90), np.quantile(p3_maha, 0.90)
                area2 = math.pi * q2 * np.sqrt(p2_det)
                area3 = math.pi * q3 * np.sqrt(p3_det)
                equal_threshold = chi2.ppf(0.90, 2) * np.sqrt(p2_det / p3_det)
                replicas.append((float(np.median(area3) / np.median(area2)), float(np.mean(p3_maha <= equal_threshold) - np.mean(p2_maha <= chi2.ppf(0.90, 2)))))
                points.append((
                    float((metric(run, condition, predictor, "P3", "mean_ade") - metric(run, condition, predictor, "P1", "mean_ade")).mean()),
                    float((metric(run, condition, predictor, "P3", "mean_fde") - metric(run, condition, predictor, "P1", "mean_fde")).mean()),
                ))
            condition_values[condition] = {"area_ratio": float(np.mean([value[0] for value in replicas])), "equal_area_coverage_gain": float(np.mean([value[1] for value in replicas]))}
            point_values[condition] = {"ade_delta": float(np.mean([value[0] for value in points])), "fde_delta": float(np.mean([value[1] for value in points]))}
        sharpness[predictor]["clean"] = condition_values["clean_s0"]
        point_harm[predictor]["clean"] = point_values["clean_s0"]
        for family in ("lowlight", "dropout", "beam"):
            sharpness[predictor][family] = {key: float(np.mean([condition_values[f"{family}_s{severity}"][key] for severity in (1, 2, 3)])) for key in condition_values["clean_s0"]}
            point_harm[predictor][family] = {key: float(np.mean([point_values[f"{family}_s{severity}"][key] for severity in (1, 2, 3)])) for key in point_values["clean_s0"]}

    risk = {predictor: {} for predictor in PREDICTORS}
    risk_absolute = {predictor: {} for predictor in PREDICTORS}
    risk_family_no_s3 = {predictor: {} for predictor in PREDICTORS}
    for predictor in PREDICTORS:
        condition_scene = {}
        for condition in CONDITIONS:
            replicate_values = []
            absolute_values = []
            for run in runs:
                scene = np.asarray(run[f"{condition}__common__scene_token"])
                error = metric(run, condition, predictor, "P1", "mean_fde")
                collision_event = np.asarray(run[f"{condition}__common__collision_event"])
                replicate_values.append(scene_risk_differences(
                    scene, error > 2.0, error,
                    metric(run, condition, predictor, "P2", "log_area90"),
                    metric(run, condition, predictor, "P3", "log_area90"),
                    collision_event,
                    metric(run, condition, predictor, "P2", "collision_probability"),
                    metric(run, condition, predictor, "P3", "collision_probability"),
                ))
                absolute_values.append({
                    "score": absolute_risk(error > 2.0, error, -np.asarray(run[f"{condition}__common__score"])),
                    "query_density": absolute_risk(error > 2.0, error, np.asarray(run[f"{condition}__common__density"])),
                    "P2": absolute_risk(error > 2.0, error, metric(run, condition, predictor, "P2", "log_area90"), collision_event, metric(run, condition, predictor, "P2", "collision_probability")),
                    "P3": absolute_risk(error > 2.0, error, metric(run, condition, predictor, "P3", "log_area90"), collision_event, metric(run, condition, predictor, "P3", "collision_probability")),
                })
            condition_scene[condition] = {
                name: np.mean(np.stack([value[name] for value in replicate_values]), axis=0)
                for name in replicate_values[0] if all(len(value[name]) == len(replicate_values[0][name]) for value in replicate_values)
            }
            risk[predictor][condition] = {name: bootstrap(value) for name, value in condition_scene[condition].items()}
            risk_absolute[predictor][condition] = {
                source: {
                    name: (None if all(value[source][name] is None for value in absolute_values) else float(np.mean([value[source][name] for value in absolute_values if value[source][name] is not None])))
                    for name in absolute_values[0][source]
                }
                for source in absolute_values[0]
            }
        for family in ("lowlight", "dropout", "beam"):
            risk_family_no_s3[predictor][family] = {}
            for name in ("high_fde_auprc", "high_fde_auroc", "risk_coverage", "false_safe", "collision_auprc"):
                values = [condition_scene[f"{family}_s{severity}"].get(name, np.asarray([])) for severity in (1, 2)]
                usable = [value for value in values if len(value)]
                risk_family_no_s3[predictor][family][name] = bootstrap(np.concatenate(usable) if usable else np.asarray([]))

    nontriviality = {predictor: nested_information(runs, predictor) for predictor in PREDICTORS}
    replicate_direction = {
        predictor: [float((metric(run, "clean_s0", predictor, "P2", "energy") - metric(run, "clean_s0", predictor, "P3", "energy")).mean()) for run in runs]
        for predictor in PREDICTORS
    }

    gate1_by_predictor = {predictor: all(clean[predictor]["energy"][baseline]["ci95"][0] > 0 for baseline in ("P1", "P2")) for predictor in PREDICTORS}
    gate2_by_predictor = {
        predictor: (
            sum(family_energy[predictor][family]["ci95"][0] > 0 for family in ("lowlight", "dropout", "beam")) >= 2
            and all(family_energy[predictor][family]["ci95"][1] >= 0 for family in ("lowlight", "dropout", "beam"))
            and sum(family_energy_no_s3[predictor][family]["ci95"][0] > 0 for family in ("lowlight", "dropout", "beam")) >= 2
            and all(family_energy_no_s3[predictor][family]["ci95"][1] >= 0 for family in ("lowlight", "dropout", "beam"))
        ) for predictor in PREDICTORS
    }
    gate3_by_predictor = {predictor: all(value["area_ratio"] <= 1.10 or value["equal_area_coverage_gain"] > 0 for value in sharpness[predictor].values()) for predictor in PREDICTORS}
    gate4_by_predictor = {predictor: all(value["ade_delta"] <= 0.05 and value["fde_delta"] <= 0.10 for value in point_harm[predictor].values()) for predictor in PREDICTORS}
    risk_names = ("high_fde_auprc", "risk_coverage", "false_safe")
    gate5_by_predictor = {}
    for predictor in PREDICTORS:
        clean_pass = any(risk[predictor]["clean_s0"].get(name, {}).get("ci95", [None])[0] is not None and risk[predictor]["clean_s0"][name]["ci95"][0] > 0 for name in risk_names)
        transfer_pass = any(risk_family_no_s3[predictor][family][name]["ci95"][0] is not None and risk_family_no_s3[predictor][family][name]["ci95"][0] > 0 for family in ("lowlight", "dropout", "beam") for name in risk_names)
        gate5_by_predictor[predictor] = clean_pass or transfer_pass
    gate6_by_predictor = {predictor: nontriviality[predictor]["delta_r2"] > 0 and nontriviality[predictor]["partial_correlation"] > 0 for predictor in PREDICTORS}
    predictor_flags = (gate1_by_predictor, gate2_by_predictor, gate3_by_predictor, gate4_by_predictor, gate5_by_predictor, gate6_by_predictor)
    gate7 = (
        all(all(value > 0 for value in replicate_direction[predictor]) for predictor in PREDICTORS)
        and all(flags["CV"] == flags["MLP"] for flags in predictor_flags)
    )
    gates = {
        "G1_future_probability_quality": all(gate1_by_predictor.values()),
        "G2_cross_degradation_stability": all(gate2_by_predictor.values()),
        "G3_downstream_sharpness": all(gate3_by_predictor.values()),
        "G4_point_prediction_not_harmed": all(gate4_by_predictor.values()),
        "G5_failure_identification": all(gate5_by_predictor.values()),
        "G6_M2_incremental_information": all(gate6_by_predictor.values()),
        "G7_seed_and_predictor_stability": gate7,
    }
    passed = all(gates.values())
    result = {
        "clean_proper_scores": clean,
        "endpoint": "GO TO RISKPROP3D METHOD DEVELOPMENT" if passed else "STOP PERCEPTION-TO-FORECAST PROPAGATION",
        "family_energy": family_energy,
        "family_energy_without_s3": family_energy_no_s3,
        "gates": gates,
        "gate_by_predictor": {
            "G1": gate1_by_predictor, "G2": gate2_by_predictor, "G3": gate3_by_predictor,
            "G4": gate4_by_predictor, "G5": gate5_by_predictor, "G6": gate6_by_predictor,
        },
        "input_sha256": {
            "code": sha256(Path(__file__)),
            "predictor": sha256(OUT / "predictor.pt"),
            "metrics": {str(index): sha256(OUT / "metrics" / f"replicate_{index}.npz") for index in range(3)},
            "heads": {str(seed): sha256(CHECKPOINT_ROOT / f"seed_{seed}.pt") for seed in HEAD_SEEDS},
        },
        "nontriviality": nontriviality,
        "official_val_read": False,
        "internal_test_read": False,
        "passed": passed,
        "point_harm": point_harm,
        "replicate_clean_energy_direction": replicate_direction,
        "risk": risk,
        "risk_absolute": risk_absolute,
        "risk_family_without_s3": risk_family_no_s3,
        "sharpness": sharpness,
        "track_counts": {condition: int(len(runs[0][f"{condition}__common__sample_token"])) for condition in CONDITIONS},
    }
    write_json(OUT / "result.json", result)
    for run in runs:
        run.close()
    print(json.dumps({"endpoint": result["endpoint"], "gates": gates, "passed": passed}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    trainer = subparsers.add_parser("train")
    trainer.add_argument("--device", type=int, choices=(0, 1, 2, 3), required=True)
    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("--replicate", type=int, choices=(0, 1, 2), required=True)
    evaluator.add_argument("--device", type=int, choices=(0, 1, 2, 3), required=True)
    subparsers.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "train":
        train(args.device)
    elif args.command == "evaluate":
        evaluate(args.replicate, args.device)
    else:
        finalize()


if __name__ == "__main__":
    main()
