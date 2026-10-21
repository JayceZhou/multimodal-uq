#!/usr/bin/env python3
"""RiskProp3D R2-Core capacity and distribution-mechanism audit."""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/riskprop3d-r2-core-matplotlib")

import numpy as np
import torch
from pyquaternion import Quaternion
from torch import nn
from torch.nn import functional as F

from nuscenes.nuscenes import NuScenes
from probset3d.f1.implementation.probabilistic_query import ProbabilisticQueryHead
from tools import riskprop3d_r0 as r0
from tools import riskprop3d_r1 as r1


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/riskprop3d_r2_core"
R15_RESULT = ROOT / "outputs/riskprop3d_r15/result.json"
CHECKPOINT_ROOT = r1.CHECKPOINT_ROOT
HEAD_SEEDS = r1.HEAD_SEEDS
MODEL_SEEDS = (20260815, 20260816, 20260817)
SAMPLE_SEEDS = (20260818, 20260819, 20260820)
BOOTSTRAP_SEED = r1.BOOTSTRAP_SEED
CONDITIONS = r1.CONDITIONS
FAMILIES = ("lowlight", "dropout", "beam")
GROUPS = (
    "C0_mean_matched",
    "C1_raw_concat",
    "C2_gaussian_dse",
    "C3_no_tail",
    "C4_no_circular",
    "C5_full_dse",
)
CAPACITY_GROUPS = (
    "C0_mean_matched",
    "C2_gaussian_dse",
    "C3_no_tail",
    "C4_no_circular",
    "C5_full_dse",
)
ABLATIONS = GROUPS[:-1]
DIST_DIM = 20
GAUSSIAN_DF = 1e6
MODES = r1.MODES
HISTORY = r1.HISTORY
HORIZON = r1.HORIZON
STATE_FEATURES = r1.STATE_FEATURES
INPUT_DIM = r1.INPUT_DIM
DT = r1.DT
TRAIN_EPOCHS = 15
TRAIN_BATCH = 1024
EVAL_CHUNK = 512
SAMPLES = r1.SAMPLES
SUBSET_NAMES = (
    "high_current_residual",
    "far_range",
    "sparse_lidar",
    "yaw_boundary",
    "turning",
    "future_heading_change",
    "crossing",
    "vehicle",
    "two_wheeler",
)
TAIL_SUBSETS = (
    "high_current_residual",
    "far_range",
    "sparse_lidar",
    "high_fde",
)
YAW_SUBSETS = (
    "yaw_boundary",
    "turning",
    "future_heading_change",
    "crossing",
    "vehicle",
    "two_wheeler",
)
CORE_YAW_SUBSETS = ("yaw_boundary", "turning", "future_heading_change")
RISK_NAMES = ("high_fde_auprc", "risk_coverage", "false_safe")
METRICS = r1.METRICS
CURRENT = HISTORY * STATE_FEATURES


def wrap_np(value):
    return (np.asarray(value) + math.pi) % (2.0 * math.pi) - math.pi


def tensor_state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def annotation_metadata(nusc, token, instance, cache):
    key = (str(token), str(instance))
    if key in cache:
        return cache[key]
    sample = nusc.get("sample", key[0])
    annotation = None
    for annotation_token in sample["anns"]:
        value = nusc.get("sample_annotation", annotation_token)
        if value["instance_token"] == key[1]:
            annotation = value
            break
    if annotation is None:
        raise ValueError(f"annotation not found: {key[0]}/{key[1]}")
    future = annotation
    for _ in range(HORIZON):
        if not future["next"]:
            raise ValueError(f"missing future annotation: {key[0]}/{key[1]}")
        future = nusc.get("sample_annotation", future["next"])
    current_yaw = Quaternion(annotation["rotation"]).yaw_pitch_roll[0]
    future_yaw = Quaternion(future["rotation"]).yaw_pitch_roll[0]
    cache[key] = (
        int(annotation["num_lidar_pts"]),
        float(abs(wrap_np(future_yaw - current_yaw))),
    )
    return cache[key]


def prepare() -> None:
    expected = [OUT / "subsets" / f"{condition}.npz" for condition in CONDITIONS]
    expected.append(OUT / "prepare_manifest.json")
    if any(path.exists() for path in expected):
        raise FileExistsError("R2-Core retained preparation output already exists")
    r15 = r1.strict_load(R15_RESULT)
    if not r15["valid"] or r15["endpoint"] != "GO TO RISKPROP3D-CORE":
        raise ValueError("R1.5 did not authorize R2-Core")
    with np.load(r0.OUT / "tracks/label_cal.npz", allow_pickle=False) as calibration:
        residual = np.linalg.norm(
            np.asarray(calibration["input"][:, CURRENT : CURRENT + 2])
            - np.asarray(calibration["gt_input"][:, CURRENT : CURRENT + 2]),
            axis=-1,
        )
    residual_threshold = float(np.quantile(residual, 0.80))
    nusc = NuScenes(
        version="v1.0-trainval", dataroot=str(ROOT / "data/nuscenes"), verbose=False
    )
    annotation_cache = {}
    source_metrics = np.load(r1.OUT / "metrics/replicate_0.npz", allow_pickle=False)
    records = {}
    for condition in CONDITIONS:
        with np.load(r0.OUT / "tracks" / f"{condition}.npz", allow_pickle=False) as tracks:
            scenes = r1.load_scene(r1.OUT / "scenes" / f"{condition}.npz")
            for name in ("sample_token", "instance_token"):
                if not np.array_equal(tracks[name], scenes[name]):
                    raise ValueError(f"R0/R1 row mismatch: {condition}/{name}")
            if not np.array_equal(tracks["future"], scenes["future"]):
                raise ValueError(f"R0/R1 target mismatch: {condition}")
            current_residual = np.linalg.norm(
                np.asarray(tracks["input"][:, CURRENT : CURRENT + 2])
                - np.asarray(tracks["gt_input"][:, CURRENT : CURRENT + 2]),
                axis=-1,
            )
            yaw = np.arctan2(
                np.asarray(scenes["input"][:, CURRENT + 8]),
                np.asarray(scenes["input"][:, CURRENT + 9]),
            )
            gt_current = np.asarray(tracks["gt_input"][:, CURRENT : CURRENT + 2])
            future = np.asarray(tracks["future"])
            first_segment = future[:, 0] - gt_current
            last_segment = future[:, -1] - future[:, -2]
            segment_valid = (
                np.linalg.norm(first_segment, axis=-1) > 0.5
            ) & (np.linalg.norm(last_segment, axis=-1) > 0.5)
            turn_angle = np.abs(
                wrap_np(
                    np.arctan2(last_segment[:, 1], last_segment[:, 0])
                    - np.arctan2(first_segment[:, 1], first_segment[:, 0])
                )
            )
            lidar_points = np.empty(len(gt_current), dtype=np.int16)
            future_yaw_change = np.empty(len(gt_current), dtype=np.float32)
            for row, (token, instance) in enumerate(
                zip(scenes["sample_token"], scenes["instance_token"])
            ):
                points, change = annotation_metadata(
                    nusc, str(token), str(instance), annotation_cache
                )
                lidar_points[row] = points
                future_yaw_change[row] = change
            crossing = np.asarray(
                source_metrics[f"{condition}__common__crossing"], dtype=np.uint8
            )
            classes = np.asarray(scenes["class"], dtype=np.int16)
            arrays = {
                "sample_token": np.asarray(scenes["sample_token"]),
                "instance_token": np.asarray(scenes["instance_token"]),
                "current_residual": current_residual.astype(np.float32),
                "num_lidar_pts": lidar_points,
                "future_yaw_change": future_yaw_change,
                "high_current_residual": (current_residual >= residual_threshold).astype(np.uint8),
                "far_range": (np.asarray(scenes["range"]) >= 40.0).astype(np.uint8),
                "sparse_lidar": (lidar_points <= 5).astype(np.uint8),
                "yaw_boundary": (np.abs(yaw) >= 5.0 * math.pi / 6.0).astype(np.uint8),
                "turning": (segment_valid & (turn_angle > math.pi / 6.0)).astype(np.uint8),
                "future_heading_change": (future_yaw_change > math.pi / 6.0).astype(np.uint8),
                "crossing": crossing,
                "vehicle": np.isin(classes, np.arange(5)).astype(np.uint8),
                "two_wheeler": np.isin(classes, (6, 7)).astype(np.uint8),
            }
        path = OUT / "subsets" / f"{condition}.npz"
        r1.write_npz(path, arrays)
        records[condition] = {
            "counts": {name: int(arrays[name].sum()) for name in SUBSET_NAMES},
            "rows": int(len(arrays["sample_token"])),
            "sha256": r1.sha256(path),
        }
        print(json.dumps({"condition": condition, **records[condition]}))
    source_metrics.close()
    manifest = {
        "checkpoint_sha256": {
            str(seed): r1.sha256(CHECKPOINT_ROOT / f"seed_{seed}.pt")
            for seed in HEAD_SEEDS
        },
        "conditions": list(CONDITIONS),
        "official_val_read": False,
        "internal_test_read": False,
        "records": records,
        "residual_q80_m": residual_threshold,
        "source_code_sha256": r1.sha256(Path(__file__)),
        "source_r15_result_sha256": r1.sha256(R15_RESULT),
        "source_r1_metrics_sha256": r15["input_sha256"]["metrics"],
    }
    r1.write_json(OUT / "prepare_manifest.json", manifest)


def load_heads(replicate, device):
    checkpoint = torch.load(
        CHECKPOINT_ROOT / f"seed_{HEAD_SEEDS[replicate]}.pt",
        map_location="cpu",
        weights_only=True,
    )
    heads = {
        name: ProbabilisticQueryHead().to(device).eval() for name in ("M1", "M2")
    }
    for name, head in heads.items():
        head.load_state_dict(checkpoint["models"][name])
    return heads


def agent_distributions(head, scenes, family, device):
    feature = np.asarray(scenes["agent_feature"])
    logits = np.asarray(scenes["agent_logits"])
    box = np.asarray(scenes["agent_box"])
    seed = np.asarray(scenes["agent_seed"])
    values = []
    for begin in range(0, len(box), 4096):
        stop = min(begin + 4096, len(box))
        with torch.inference_mode():
            output = head(
                torch.from_numpy(feature[begin:stop])[:, None].to(device),
                torch.from_numpy(logits[begin:stop])[:, None].to(device),
                torch.from_numpy(box[begin:stop])[:, None].to(device),
                torch.from_numpy(seed[begin:stop])[:, None].to(device),
                torch.zeros(stop - begin, 1, device=device),
            )
        if family == "M2":
            degrees = output["degrees_of_freedom"][:, 0, None]
            yaw_parameter = output["yaw_kappa"][:, 0, None]
            resultant = r1.mean_resultant_torch(output["yaw_kappa"][:, 0, None])
        elif family == "M1":
            degrees = torch.full(
                (stop - begin, 1), GAUSSIAN_DF, device=device
            )
            yaw_parameter = output["yaw_scale"][:, 0, None]
            resultant = torch.exp(-0.5 * yaw_parameter.square())
        else:
            raise ValueError(family)
        packed = torch.cat(
            (
                output["mean_state"][:, 0],
                output["scale"][:, 0],
                degrees,
                output["yaw_mean"][:, 0, None],
                yaw_parameter,
                resultant,
            ),
            dim=-1,
        )
        values.append(packed.cpu().numpy().astype(np.float32))
    result = np.concatenate(values)
    if result.shape != (len(box), DIST_DIM) or not np.isfinite(result).all():
        raise ValueError(f"invalid {family} distribution array")
    return result


class MechanismDSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(24, 48), nn.ReLU(), nn.Linear(48, 32)
        )
        self.summary = nn.Sequential(
            nn.Linear(7, 24), nn.ReLU(), nn.Linear(24, 16)
        )
        self.circular = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16)
        )

    def forward(self, dist, state_mean, state_scale, variant):
        mean = dist[..., :8]
        scale = torch.clamp(dist[..., 8:16], min=1e-6)
        degrees = torch.clamp(dist[..., 16], min=2.0001)
        variance = scale.square() * degrees[..., None] / (
            degrees[..., None] - 2.0
        )
        standard = torch.sqrt(torch.clamp(variance, min=1e-12))
        normalized_mean = (mean - state_mean) / state_scale
        zeros = torch.zeros_like(standard)
        if variant == "C0_mean_matched":
            linear = torch.stack((normalized_mean, zeros, zeros), dim=-1).flatten(-2)
            summary = torch.zeros(*mean.shape[:-1], 7, device=mean.device, dtype=mean.dtype)
        else:
            linear = torch.stack(
                (
                    normalized_mean,
                    torch.log1p(standard),
                    standard / (mean.abs() + 1.0),
                ),
                dim=-1,
            ).flatten(-2)
            xy = variance[..., :2]
            tail_log = torch.log(torch.clamp(degrees - 2.0, min=1e-4))
            tail_inverse = torch.reciprocal(degrees)
            if variant == "C3_no_tail":
                tail_log = torch.zeros_like(tail_log)
                tail_inverse = torch.zeros_like(tail_inverse)
            summary = torch.stack(
                (
                    torch.log1p(standard).mean(-1),
                    torch.log(torch.clamp(variance, min=1e-12)).sum(-1) / 8.0,
                    torch.log1p(torch.sqrt(xy.sum(-1))),
                    torch.log1p(torch.sqrt(torch.max(xy, dim=-1).values)),
                    (xy[..., 0] - xy[..., 1])
                    / torch.clamp(xy.sum(-1), min=1e-8),
                    tail_log,
                    tail_inverse,
                ),
                dim=-1,
            )
        yaw = dist[..., 17]
        resultant = dist[..., 19]
        if variant in ("C0_mean_matched", "C4_no_circular"):
            resultant = torch.ones_like(resultant)
        circular = torch.stack((torch.cos(yaw), torch.sin(yaw), resultant), dim=-1)
        return torch.cat(
            (self.linear(linear), self.summary(summary), self.circular(circular)),
            dim=-1,
        )


class CoreModel(nn.Module):
    def __init__(self, group, statistics):
        super().__init__()
        if group not in GROUPS:
            raise ValueError(group)
        self.group = group
        for name, value in statistics.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        if group == "C1_raw_concat":
            self.concat_encoder = nn.Sequential(
                nn.Linear(INPUT_DIM + 10, 128),
                nn.ReLU(),
                nn.Linear(128, r1.LATENT_DIM),
            )
        else:
            self.motion_encoder = nn.Sequential(
                nn.Linear(INPUT_DIM, 128),
                nn.ReLU(),
                nn.Linear(128, r1.LATENT_DIM),
            )
            self.dse = MechanismDSE()
            self.fusion = nn.Sequential(
                nn.Linear(2 * r1.LATENT_DIM, 128),
                nn.ReLU(),
                nn.Linear(128, r1.LATENT_DIM),
            )
        self.decoder = r1.TrajectoryDecoder()

    def forward(self, motion, m1_dist, m2_dist):
        normalized_motion = (motion - self.motion_mean) / self.motion_scale
        if self.group == "C1_raw_concat":
            raw = torch.cat(
                (m2_dist[..., 8:16], m2_dist[..., 16:17], m2_dist[..., 18:19]),
                dim=-1,
            )
            raw = (raw - self.raw_mean) / self.raw_scale
            latent = self.concat_encoder(torch.cat((normalized_motion, raw), dim=-1))
        else:
            source = m1_dist if self.group == "C2_gaussian_dse" else m2_dist
            motion_latent = self.motion_encoder(normalized_motion)
            distribution_latent = self.dse(
                source, self.state_mean, self.state_scale, self.group
            )
            latent = self.fusion(
                torch.cat((motion_latent, distribution_latent), dim=-1)
            )
        current = motion[:, CURRENT : CURRENT + STATE_FEATURES]
        times = (
            torch.arange(1, HORIZON + 1, device=motion.device, dtype=motion.dtype)
            * DT
        )
        cv = current[:, None, :2] + times[None, :, None] * current[:, None, 6:8]
        return self.decoder(latent, cv, self.target_mean, self.target_scale)


def raw_uncertainty(dist):
    return np.concatenate(
        (dist[..., 8:16], dist[..., 16:17], dist[..., 18:19]), axis=-1
    ).astype(np.float32)


def make_statistics(inputs, future, m2_target):
    inputs = np.asarray(inputs, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    residual = future - r0.cv_prediction(inputs).astype(np.float32)
    raw = raw_uncertainty(m2_target)
    return {
        "motion_mean": inputs.mean(0),
        "motion_scale": np.maximum(inputs.std(0), 1e-4),
        "raw_mean": raw.mean(0),
        "raw_scale": np.maximum(raw.std(0), 1e-4),
        "state_mean": m2_target[:, :8].mean(0),
        "state_scale": np.maximum(m2_target[:, :8].std(0), 1e-4),
        "target_mean": residual.mean(0),
        "target_scale": np.maximum(residual.std(0), 0.1),
    }


def gather_batch(scenes, m1_dist, m2_dist, indices, device):
    target = np.asarray(scenes["target_agent_index"])[indices]
    return (
        torch.from_numpy(np.asarray(scenes["input"])[indices]).to(device),
        torch.from_numpy(m1_dist[target]).to(device),
        torch.from_numpy(m2_dist[target]).to(device),
    )


def collect_outputs(model, scenes, m1_dist, m2_dist, device):
    storage = {name: [] for name in ("means", "scales", "logits")}
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(scenes["input"]), 2048):
            index = np.arange(begin, min(begin + 2048, len(scenes["input"])))
            output = model(*gather_batch(scenes, m1_dist, m2_dist, index, device))
            for name in storage:
                storage[name].append(output[name].cpu().numpy())
    return {name: np.concatenate(values) for name, values in storage.items()}


def set_determinism(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(device)


def train(replicate, device_index):
    if replicate not in range(3) or device_index not in range(4):
        raise ValueError("replicate must be 0--2 and GPU must be 0--3")
    path = OUT / "checkpoints" / f"replicate_{replicate}.pt"
    if path.exists():
        raise FileExistsError(path)
    device = torch.device(f"cuda:{device_index}")
    set_determinism(MODEL_SEEDS[replicate], device)
    manifest = r1.strict_load(OUT / "prepare_manifest.json")
    if manifest["source_r15_result_sha256"] != r1.sha256(R15_RESULT):
        raise ValueError("R1.5 result hash mismatch")
    train_scenes = r1.load_scene(r1.OUT / "scenes/train.npz")
    calibration = r1.load_scene(r1.OUT / "scenes/label_cal.npz")
    heads = load_heads(replicate, device)
    train_dist = {
        family: agent_distributions(heads[family], train_scenes, family, device)
        for family in ("M1", "M2")
    }
    calibration_dist = {
        family: agent_distributions(heads[family], calibration, family, device)
        for family in ("M1", "M2")
    }
    train_target = np.asarray(train_scenes["target_agent_index"])
    statistics = make_statistics(
        train_scenes["input"],
        train_scenes["future"],
        train_dist["M2"][train_target],
    )
    template = CoreModel("C5_full_dse", statistics).to(device)
    template_state = {
        name: value.detach().clone() for name, value in template.state_dict().items()
    }
    models = {}
    for group in CAPACITY_GROUPS:
        model = CoreModel(group, statistics).to(device)
        model.load_state_dict(template_state)
        models[group] = model
    models["C1_raw_concat"] = CoreModel("C1_raw_concat", statistics).to(device)
    models = {group: models[group] for group in GROUPS}
    initialization_hashes = {
        group: tensor_state_hash(models[group]) for group in GROUPS
    }
    parameter_counts = {
        group: int(sum(value.numel() for value in model.parameters()))
        for group, model in models.items()
    }
    if len({parameter_counts[group] for group in CAPACITY_GROUPS}) != 1:
        raise ValueError("capacity-matched parameter counts differ")
    if len({initialization_hashes[group] for group in CAPACITY_GROUPS}) != 1:
        raise ValueError("capacity-matched initialization differs")
    optimizers = {
        group: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for group, model in models.items()
    }
    rng = np.random.default_rng(MODEL_SEEDS[replicate])
    for epoch in range(TRAIN_EPOCHS):
        permutation = rng.permutation(len(train_scenes["input"]))
        losses = {group: 0.0 for group in GROUPS}
        seen = 0
        for begin in range(0, len(permutation), TRAIN_BATCH):
            index = permutation[begin : begin + TRAIN_BATCH]
            batch = gather_batch(
                train_scenes, train_dist["M1"], train_dist["M2"], index, device
            )
            target = torch.from_numpy(
                np.asarray(train_scenes["future"])[index]
            ).to(device)
            for group, model in models.items():
                model.train()
                optimizers[group].zero_grad(set_to_none=True)
                loss = r1.training_loss(model(*batch), target)
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite training loss: {group}")
                loss.backward()
                optimizers[group].step()
                losses[group] += float(loss.detach()) * len(index)
            seen += len(index)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "loss": {group: value / seen for group, value in losses.items()},
                    "replicate": replicate,
                }
            )
        )
    temperatures = {}
    for group, model in models.items():
        output = collect_outputs(
            model,
            calibration,
            calibration_dist["M1"],
            calibration_dist["M2"],
            device,
        )
        temperatures[group] = r1.fit_temperature(output, calibration["future"])
    checkpoint = {
        "architecture": {
            "capacity_groups": list(CAPACITY_GROUPS),
            "groups": list(GROUPS),
            "modes": MODES,
        },
        "epochs": TRAIN_EPOCHS,
        "head_seed": HEAD_SEEDS[replicate],
        "initialization_hashes": initialization_hashes,
        "model_seed": MODEL_SEEDS[replicate],
        "models": {group: model.state_dict() for group, model in models.items()},
        "parameter_counts": parameter_counts,
        "prepare_manifest_sha256": r1.sha256(OUT / "prepare_manifest.json"),
        "sampling_seed": SAMPLE_SEEDS[replicate],
        "statistics": {
            name: torch.as_tensor(value) for name, value in statistics.items()
        },
        "temperatures": temperatures,
        "training_code_sha256": r1.sha256(Path(__file__)),
    }
    r1.write_torch(path, checkpoint)
    print(
        json.dumps(
            {
                "trained": True,
                "replicate": replicate,
                "parameter_counts": parameter_counts,
                "temperatures": temperatures,
            }
        )
    )


def load_models(replicate, device):
    checkpoint = torch.load(
        OUT / "checkpoints" / f"replicate_{replicate}.pt",
        map_location="cpu",
        weights_only=True,
    )
    if (
        checkpoint["head_seed"] != HEAD_SEEDS[replicate]
        or checkpoint["model_seed"] != MODEL_SEEDS[replicate]
        or checkpoint["sampling_seed"] != SAMPLE_SEEDS[replicate]
    ):
        raise ValueError("checkpoint seed mismatch")
    statistics = {
        name: value.numpy() for name, value in checkpoint["statistics"].items()
    }
    models = {}
    for group in GROUPS:
        model = CoreModel(group, statistics).to(device).eval()
        model.load_state_dict(checkpoint["models"][group])
        models[group] = model
    return models, checkpoint


def evaluate(replicate, device_index):
    if replicate not in range(3) or device_index not in range(4):
        raise ValueError("replicate must be 0--2 and GPU must be 0--3")
    path = OUT / "metrics" / f"replicate_{replicate}.npz"
    if path.exists():
        raise FileExistsError(path)
    device = torch.device(f"cuda:{device_index}")
    set_determinism(SAMPLE_SEEDS[replicate], device)
    manifest = r1.strict_load(OUT / "prepare_manifest.json")
    models, checkpoint = load_models(replicate, device)
    heads = load_heads(replicate, device)
    arrays = {
        "meta__head_seed": np.asarray(HEAD_SEEDS[replicate], dtype=np.int64),
        "meta__model_seed": np.asarray(MODEL_SEEDS[replicate], dtype=np.int64),
        "meta__sampling_seed": np.asarray(SAMPLE_SEEDS[replicate], dtype=np.int64),
    }
    for condition_index, condition in enumerate(CONDITIONS):
        scenes = r1.load_scene(r1.OUT / "scenes" / f"{condition}.npz")
        subset_path = OUT / "subsets" / f"{condition}.npz"
        if r1.sha256(subset_path) != manifest["records"][condition]["sha256"]:
            raise ValueError(f"subset hash mismatch: {condition}")
        with np.load(subset_path, allow_pickle=False) as source:
            subsets = {name: source[name] for name in source.files}
        for name in ("sample_token", "instance_token"):
            if not np.array_equal(subsets[name], scenes[name]):
                raise ValueError(f"subset row mismatch: {condition}/{name}")
        distributions = {
            family: agent_distributions(heads[family], scenes, family, device)
            for family in ("M1", "M2")
        }
        storage = {
            group: {name: [] for name in METRICS} for group in GROUPS
        }
        runtime = {group: 0.0 for group in GROUPS}
        rng = np.random.default_rng(SAMPLE_SEEDS[replicate] + condition_index * 1009)
        warmed = False
        for begin in range(0, len(scenes["input"]), EVAL_CHUNK):
            stop = min(begin + EVAL_CHUNK, len(scenes["input"]))
            index = np.arange(begin, stop)
            batch = gather_batch(
                scenes,
                distributions["M1"],
                distributions["M2"],
                index,
                device,
            )
            if not warmed:
                with torch.inference_mode():
                    for model in models.values():
                        model(*batch)
                r1.synchronize(device)
                warmed = True
            outputs = {}
            for group, model in models.items():
                r1.synchronize(device)
                start = time.perf_counter()
                with torch.inference_mode():
                    outputs[group] = r1.numpy_output(model(*batch))
                r1.synchronize(device)
                runtime[group] += time.perf_counter() - start
            count = stop - begin
            uniforms = (
                rng.random((count, SAMPLES)),
                rng.random((count, SAMPLES)),
            )
            normals = (
                rng.standard_normal((count, SAMPLES, HORIZON, 2)),
                rng.standard_normal((count, SAMPLES, HORIZON, 2)),
            )
            future = np.asarray(scenes["future"])[index]
            ego_future = np.asarray(scenes["ego_future"])[index]
            for group in GROUPS:
                temperature = checkpoint["temperatures"][group]
                sample_a = r1.sample_mixture(
                    outputs[group], temperature, uniforms[0], normals[0]
                )
                sample_b = r1.sample_mixture(
                    outputs[group], temperature, uniforms[1], normals[1]
                )
                metrics = r1.trajectory_metrics(
                    outputs[group],
                    sample_a,
                    sample_b,
                    future,
                    ego_future,
                    temperature,
                )
                for name in METRICS:
                    storage[group][name].append(
                        np.asarray(metrics[name], dtype=np.float32)
                    )
        for name in r1.COMMON_FIELDS[:11]:
            arrays[f"{condition}__common__{name}"] = np.asarray(scenes[name])
        for name in SUBSET_NAMES:
            arrays[f"{condition}__common__{name}"] = np.asarray(
                subsets[name], dtype=np.uint8
            )
        for group in GROUPS:
            arrays[f"{condition}__runtime__{group}__seconds"] = np.asarray(
                runtime[group], dtype=np.float64
            )
            arrays[f"{condition}__runtime__{group}__tracks"] = np.asarray(
                len(scenes["input"]), dtype=np.int64
            )
            for name in METRICS:
                arrays[f"{condition}__{group}__{name}"] = np.concatenate(
                    storage[group][name]
                )
    r1.write_npz(path, arrays)
    print(
        json.dumps(
            {
                "evaluated": True,
                "output": str(path.relative_to(ROOT)),
                "replicate": replicate,
            }
        )
    )


def metric(run, condition, group, name):
    return np.asarray(run[f"{condition}__{group}__{name}"], dtype=np.float64)


def selected_rows(run, condition, subset):
    if subset is None:
        return None
    if subset == "high_fde":
        return metric(run, condition, "C0_mean_matched", "mean_fde") > 2.0
    return np.asarray(run[f"{condition}__common__{subset}"], dtype=bool)


def paired(runs, condition, baseline, candidate, name, subset=None):
    maps = []
    for run in runs:
        maps.append(
            r1.scene_mean_map(
                metric(run, condition, baseline, name)
                - metric(run, condition, candidate, name),
                run[f"{condition}__common__scene_token"],
                selected_rows(run, condition, subset),
            )
        )
    return r1.bootstrap(r1.average_maps(maps))


def family_paired(runs, family, baseline, candidate, name, severities):
    replicate_maps = []
    for run in runs:
        severity_maps = []
        for severity in severities:
            condition = f"{family}_s{severity}"
            severity_maps.append(
                r1.scene_mean_map(
                    metric(run, condition, baseline, name)
                    - metric(run, condition, candidate, name),
                    run[f"{condition}__common__scene_token"],
                )
            )
        keys = sorted(set().union(*(set(value) for value in severity_maps)))
        replicate_maps.append(
            {
                key: float(
                    np.mean([value[key] for value in severity_maps if key in value])
                )
                for key in keys
            }
        )
    return r1.bootstrap(r1.average_maps(replicate_maps))


def risk_difference(runs, condition, baseline, candidate):
    replicate = []
    for run in runs:
        replicate.append(
            r1.scene_risk_maps(
                run[f"{condition}__common__scene_token"],
                metric(run, condition, "C0_mean_matched", "mean_fde"),
                metric(run, condition, baseline, "log_area90"),
                metric(run, condition, candidate, "log_area90"),
            )
        )
    return {
        name: r1.bootstrap(r1.average_maps([value[name] for value in replicate]))
        for name in replicate[0]
    }


def family_risk(runs, family, baseline, candidate, severities=(1, 2)):
    replicate = []
    for run in runs:
        current = {
            name: {}
            for name in (
                "high_fde_auprc",
                "high_fde_auroc",
                "risk_coverage",
                "false_safe",
            )
        }
        for severity in severities:
            condition = f"{family}_s{severity}"
            values = r1.scene_risk_maps(
                run[f"{condition}__common__scene_token"],
                metric(run, condition, "C0_mean_matched", "mean_fde"),
                metric(run, condition, baseline, "log_area90"),
                metric(run, condition, candidate, "log_area90"),
            )
            for name, mapping in values.items():
                current[name].update(
                    {f"s{severity}/{key}": value for key, value in mapping.items()}
                )
        replicate.append(current)
    return {
        name: r1.bootstrap(r1.average_maps([value[name] for value in replicate]))
        for name in replicate[0]
    }


def median_equal_coverage_area(run, condition, group):
    maha = metric(run, condition, group, "final_maha2")
    determinant = metric(run, condition, group, "final_det")
    return float(
        np.median(math.pi * np.quantile(maha, 0.90) * np.sqrt(determinant))
    )


def significant(report):
    return report["ci95"][0] is not None and report["ci95"][0] > 0


def not_significantly_reversed(report):
    return report["ci95"][1] is not None and report["ci95"][1] >= 0


def family_rule(values):
    return (
        sum(significant(values[family]) for family in FAMILIES) >= 2
        and all(not_significantly_reversed(values[family]) for family in FAMILIES)
    )


def audit_inputs(runs, manifest, source_r15):
    finite_values = 0
    finite = True
    seeds = True
    common = True
    metric_hashes = {}
    for index, run in enumerate(runs):
        metric_path = OUT / "metrics" / f"replicate_{index}.npz"
        metric_hashes[str(index)] = r1.sha256(metric_path)
        seeds &= (
            int(run["meta__head_seed"]) == HEAD_SEEDS[index]
            and int(run["meta__model_seed"]) == MODEL_SEEDS[index]
            and int(run["meta__sampling_seed"]) == SAMPLE_SEEDS[index]
        )
        for name in run.files:
            value = np.asarray(run[name])
            if np.issubdtype(value.dtype, np.number):
                finite &= bool(np.isfinite(value).all())
                finite_values += int(value.size)
    common_names = r1.COMMON_FIELDS[:11] + SUBSET_NAMES
    for condition in CONDITIONS:
        for name in common_names:
            reference = runs[0][f"{condition}__common__{name}"]
            common &= all(
                np.array_equal(reference, run[f"{condition}__common__{name}"])
                for run in runs[1:]
            )
    subset_hashes = {
        condition: r1.sha256(OUT / "subsets" / f"{condition}.npz")
        for condition in CONDITIONS
    }
    subset_valid = all(
        subset_hashes[condition] == manifest["records"][condition]["sha256"]
        for condition in CONDITIONS
    )
    checkpoint_hashes = {}
    checkpoint_valid = True
    parameter_counts = {}
    initialization_hashes = {}
    for index in range(3):
        path = OUT / "checkpoints" / f"replicate_{index}.pt"
        checkpoint_hashes[str(index)] = r1.sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint_valid &= (
            checkpoint["head_seed"] == HEAD_SEEDS[index]
            and checkpoint["model_seed"] == MODEL_SEEDS[index]
            and checkpoint["sampling_seed"] == SAMPLE_SEEDS[index]
            and checkpoint["prepare_manifest_sha256"]
            == r1.sha256(OUT / "prepare_manifest.json")
        )
        parameter_counts[str(index)] = checkpoint["parameter_counts"]
        initialization_hashes[str(index)] = checkpoint["initialization_hashes"]
        checkpoint_valid &= (
            len(
                {
                    checkpoint["parameter_counts"][group]
                    for group in CAPACITY_GROUPS
                }
            )
            == 1
            and len(
                {
                    checkpoint["initialization_hashes"][group]
                    for group in CAPACITY_GROUPS
                }
            )
            == 1
        )
        tensors = list(checkpoint["statistics"].values())
        for state in checkpoint["models"].values():
            tensors.extend(state.values())
        for value in tensors:
            finite &= bool(torch.isfinite(value).all())
            finite_values += int(value.numel())
    r1_manifest = r1.strict_load(r1.OUT / "prepare_manifest.json")
    source_hashes = all(
        r1.sha256(r1.OUT / "scenes" / f"{name}.npz")
        == record["sha256"]
        for name, record in r1_manifest["records"].items()
    ) and all(
        r1.sha256(CHECKPOINT_ROOT / f"seed_{seed}.pt")
        == manifest["checkpoint_sha256"][str(seed)]
        for seed in HEAD_SEEDS
    )
    validity = {
        "V1_R15_authorizes_core_and_sources_frozen": bool(
            source_r15["valid"]
            and source_r15["endpoint"] == "GO TO RISKPROP3D-CORE"
            and manifest["source_r15_result_sha256"] == r1.sha256(R15_RESULT)
            and source_hashes
        ),
        "V2_subset_authority_and_scene_rows_frozen": bool(subset_valid and common),
        "V3_capacity_and_initialization_matched": bool(checkpoint_valid),
        "V4_three_preregistered_replicates": bool(seeds),
        "V5_all_retained_numeric_values_finite": bool(finite and finite_values > 0),
        "V6_forbidden_splits_unread": bool(
            not manifest["official_val_read"]
            and not manifest["internal_test_read"]
            and not source_r15["official_val_read"]
            and not source_r15["internal_test_read"]
        ),
    }
    evidence = {
        "checkpoints": checkpoint_hashes,
        "initialization_hashes": initialization_hashes,
        "metrics": metric_hashes,
        "parameter_counts": parameter_counts,
        "subsets": subset_hashes,
    }
    return validity, evidence, finite_values


def finalize():
    result_path = OUT / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    manifest = r1.strict_load(OUT / "prepare_manifest.json")
    source_r15 = r1.strict_load(R15_RESULT)
    runs = [
        np.load(OUT / "metrics" / f"replicate_{index}.npz", allow_pickle=False)
        for index in range(3)
    ]
    validity, hashes, finite_values = audit_inputs(runs, manifest, source_r15)

    per_condition = {
        condition: {
            baseline: {
                name: paired(runs, condition, baseline, "C5_full_dse", name)
                for name in ("energy", "nll")
            }
            for baseline in ABLATIONS
        }
        for condition in CONDITIONS
    }
    transfer = {
        baseline: {
            scope: {
                family: family_paired(
                    runs,
                    family,
                    baseline,
                    "C5_full_dse",
                    "energy",
                    severities,
                )
                for family in FAMILIES
            }
            for scope, severities in (
                ("full", (1, 2, 3)),
                ("without_s3", (1, 2)),
            )
        }
        for baseline in ABLATIONS
    }
    mechanism_subsets = {
        "tail_C5_over_C3": {
            subset: paired(
                runs,
                "clean_s0",
                "C3_no_tail",
                "C5_full_dse",
                "energy",
                subset,
            )
            for subset in TAIL_SUBSETS
        },
        "circular_C5_over_C4": {
            subset: paired(
                runs,
                "clean_s0",
                "C4_no_circular",
                "C5_full_dse",
                "energy",
                subset,
            )
            for subset in YAW_SUBSETS
        },
    }

    risk = {}
    for baseline in ABLATIONS:
        risk[baseline] = {
            "clean": risk_difference(runs, "clean_s0", baseline, "C5_full_dse"),
            "without_s3": {
                family: family_risk(
                    runs, family, baseline, "C5_full_dse", severities=(1, 2)
                )
                for family in FAMILIES
            },
        }

    condition_sharpness = {}
    for condition in CONDITIONS:
        ratios, areas = [], []
        for run in runs:
            current = {
                group: median_equal_coverage_area(run, condition, group)
                for group in GROUPS
            }
            ratios.append(
                current["C5_full_dse"]
                / min(current[group] for group in ABLATIONS)
            )
            areas.append(current)
        condition_sharpness[condition] = {
            "C5_over_best_ablation": float(np.mean(ratios)),
            "median_area_by_group": {
                group: float(np.mean([value[group] for value in areas]))
                for group in GROUPS
            },
        }
    sharpness = {
        "clean": condition_sharpness["clean_s0"],
        "full": {
            family: {
                "C5_over_best_ablation": float(
                    np.mean(
                        [
                            condition_sharpness[f"{family}_s{severity}"][
                                "C5_over_best_ablation"
                            ]
                            for severity in (1, 2, 3)
                        ]
                    )
                )
            }
            for family in FAMILIES
        },
        "without_s3": {
            family: {
                "C5_over_best_ablation": float(
                    np.mean(
                        [
                            condition_sharpness[f"{family}_s{severity}"][
                                "C5_over_best_ablation"
                            ]
                            for severity in (1, 2)
                        ]
                    )
                )
            }
            for family in FAMILIES
        },
        "per_condition": condition_sharpness,
    }

    condition_accuracy = {}
    for condition in CONDITIONS:
        condition_accuracy[condition] = {
            name: float(
                np.mean(
                    [
                        (
                            metric(run, condition, "C5_full_dse", name)
                            - metric(run, condition, "C0_mean_matched", name)
                        ).mean()
                        for run in runs
                    ]
                )
            )
            for name in ("mean_ade", "mean_fde")
        }
    accuracy = {
        "clean": condition_accuracy["clean_s0"],
        "full": {
            family: {
                name: float(
                    np.mean(
                        [
                            condition_accuracy[f"{family}_s{severity}"][name]
                            for severity in (1, 2, 3)
                        ]
                    )
                )
                for name in ("mean_ade", "mean_fde")
            }
            for family in FAMILIES
        },
        "per_condition": condition_accuracy,
    }

    seed_directions = {
        baseline: [
            float(
                (
                    metric(run, "clean_s0", baseline, "energy")
                    - metric(run, "clean_s0", "C5_full_dse", "energy")
                ).mean()
            )
            for run in runs
        ]
        for baseline in ABLATIONS
    }

    clean_capacity = per_condition["clean_s0"]["C0_mean_matched"]["energy"]
    clean_gaussian = per_condition["clean_s0"]["C2_gaussian_dse"]["energy"]
    clean_tail = per_condition["clean_s0"]["C3_no_tail"]["energy"]
    clean_circular = per_condition["clean_s0"]["C4_no_circular"]["energy"]
    g1 = (
        significant(clean_capacity)
        and family_rule(transfer["C0_mean_matched"]["full"])
        and family_rule(transfer["C0_mean_matched"]["without_s3"])
    )
    g2 = (
        significant(clean_gaussian)
        and family_rule(transfer["C2_gaussian_dse"]["full"])
        and family_rule(transfer["C2_gaussian_dse"]["without_s3"])
    )
    tail_sensitive = mechanism_subsets["tail_C5_over_C3"]
    g3 = (
        not_significantly_reversed(clean_tail)
        and any(
            significant(tail_sensitive[subset])
            and tail_sensitive[subset]["mean"] > clean_tail["mean"]
            for subset in TAIL_SUBSETS
        )
        and any(
            significant(transfer["C3_no_tail"]["full"][family])
            for family in FAMILIES
        )
        and all(
            not_significantly_reversed(
                transfer["C3_no_tail"]["full"][family]
            )
            for family in FAMILIES
        )
    )
    yaw_sensitive = mechanism_subsets["circular_C5_over_C4"]
    g4 = (
        not_significantly_reversed(clean_circular)
        and (
            significant(clean_circular)
            or any(significant(yaw_sensitive[subset]) for subset in YAW_SUBSETS)
        )
        and any(
            significant(yaw_sensitive[subset])
            and yaw_sensitive[subset]["mean"] > clean_circular["mean"]
            for subset in CORE_YAW_SUBSETS
        )
    )

    risk_pass_by_ablation = {}
    for baseline in ABLATIONS:
        risk_pass_by_ablation[baseline] = any(
            significant(risk[baseline]["clean"][name]) for name in RISK_NAMES
        ) or any(
            significant(risk[baseline]["without_s3"][family][name])
            for family in FAMILIES
            for name in RISK_NAMES
        )
    g5 = all(risk_pass_by_ablation.values())
    sharpness_ratios = [sharpness["clean"]["C5_over_best_ablation"]]
    sharpness_ratios.extend(
        sharpness[scope][family]["C5_over_best_ablation"]
        for scope in ("full", "without_s3")
        for family in FAMILIES
    )
    g6 = all(value <= 1.05 for value in sharpness_ratios)
    g7 = all(
        value["mean_ade"] <= 0.05 and value["mean_fde"] <= 0.10
        for value in (accuracy["clean"], *accuracy["full"].values())
    )
    g8 = all(
        all(value > 0 for value in directions)
        for directions in seed_directions.values()
    )
    gates = {
        "G1_capacity": g1,
        "G2_distribution_family": g2,
        "G3_tail_contribution": g3,
        "G4_circular_contribution": g4,
        "G5_risk_utility": g5,
        "G6_sharpness": g6,
        "G7_accuracy": g7,
        "G8_seed_stability": g8,
    }
    valid = all(validity.values())
    ideal = valid and all(gates.values())
    foundation = all(gates[name] for name in (
        "G1_capacity",
        "G2_distribution_family",
        "G5_risk_utility",
        "G6_sharpness",
        "G7_accuracy",
        "G8_seed_stability",
    ))
    if not valid:
        endpoint = "INVALID R2-CORE RUN"
    elif ideal:
        endpoint = "GO TO R3 STRONG-HOST GENERALIZATION"
    elif foundation and (g3 or g4):
        supported = "+".join(
            name
            for name, passed in (("TAIL", g3), ("CIRCULAR", g4))
            if passed
        )
        endpoint = f"GO TO R3 WITH SCOPED {supported} MECHANISM CLAIM"
    elif g1 and not g2:
        endpoint = "REASSESS RISKPROP3D PAPER POSITIONING"
    else:
        endpoint = "STOP RISKPROP3D-CORE MECHANISM CLAIM"
    result = {
        "accuracy": accuracy,
        "endpoint": endpoint,
        "gates": gates,
        "input_sha256": {
            **hashes,
            "code": r1.sha256(Path(__file__)),
            "prepare_manifest": r1.sha256(OUT / "prepare_manifest.json"),
            "source_r15_result": r1.sha256(R15_RESULT),
        },
        "internal_test_read": False,
        "mechanism_subsets": mechanism_subsets,
        "numeric_value_count": finite_values,
        "official_val_read": False,
        "parameter_match": {
            "capacity_groups": list(CAPACITY_GROUPS),
            "counts": hashes["parameter_counts"],
            "initialization_hashes": hashes["initialization_hashes"],
        },
        "passed": ideal,
        "per_condition_proper_scores": per_condition,
        "risk": risk,
        "risk_pass_by_ablation": risk_pass_by_ablation,
        "seed_directions": seed_directions,
        "sharpness": sharpness,
        "source_r15_endpoint": source_r15["endpoint"],
        "subset_counts": {
            condition: manifest["records"][condition]["counts"]
            for condition in CONDITIONS
        },
        "track_counts": {
            condition: int(
                len(runs[0][f"{condition}__common__sample_token"])
            )
            for condition in CONDITIONS
        },
        "transfer_energy": transfer,
        "valid": valid,
        "validity": validity,
    }
    r1.write_json(result_path, result)
    for run in runs:
        run.close()
    print(
        json.dumps(
            {"endpoint": endpoint, "gates": gates, "validity": validity},
            indent=2,
            sort_keys=True,
        )
    )


def self_check():
    rng = np.random.default_rng(17)
    count = 6
    motion = np.zeros((count, INPUT_DIM), dtype=np.float32)
    motion[:, CURRENT : CURRENT + 2] = rng.normal(size=(count, 2))
    motion[:, CURRENT + 6 : CURRENT + 8] = rng.normal(size=(count, 2))
    motion[:, -10] = 1.0
    future = r0.cv_prediction(motion).astype(np.float32)
    m2 = np.zeros((count, DIST_DIM), dtype=np.float32)
    m2[:, :8] = motion[:, CURRENT : CURRENT + 8]
    m2[:, 8:16] = 0.2
    m2[:, 16] = 5.0
    m2[:, 17] = 0.3
    m2[:, 18] = 10.0
    with torch.no_grad():
        m2[:, 19] = r1.mean_resultant_torch(torch.full((count,), 10.0)).numpy()
    m1 = m2.copy()
    m1[:, 16] = GAUSSIAN_DF
    m1[:, 18] = 0.2
    m1[:, 19] = math.exp(-0.5 * 0.2**2)
    statistics = make_statistics(motion, future, m2)
    torch.manual_seed(9)
    template = CoreModel("C5_full_dse", statistics)
    state = {name: value.clone() for name, value in template.state_dict().items()}
    models = {}
    for group in CAPACITY_GROUPS:
        model = CoreModel(group, statistics)
        model.load_state_dict(state)
        models[group] = model
    if len({sum(p.numel() for p in model.parameters()) for model in models.values()}) != 1:
        raise AssertionError("capacity parameter check failed")
    if len({tensor_state_hash(model) for model in models.values()}) != 1:
        raise AssertionError("initialization match check failed")
    models["C1_raw_concat"] = CoreModel("C1_raw_concat", statistics)
    tensors = (
        torch.from_numpy(motion), torch.from_numpy(m1), torch.from_numpy(m2)
    )
    for group in GROUPS:
        output = models[group](*tensors)
        if output["means"].shape != (count, MODES, HORIZON, 2):
            raise AssertionError(f"invalid shape: {group}")
        if not torch.isfinite(r1.training_loss(output, torch.from_numpy(future))):
            raise AssertionError(f"non-finite loss: {group}")
    print(
        json.dumps(
            {
                "capacity_parameters": sum(
                    p.numel() for p in models["C5_full_dse"].parameters()
                ),
                "groups": list(GROUPS),
                "self_check": "passed",
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("self-check")
    for command in ("train", "evaluate"):
        current = subparsers.add_parser(command)
        current.add_argument("--replicate", type=int, choices=(0, 1, 2), required=True)
        current.add_argument("--device", type=int, choices=(0, 1, 2, 3), required=True)
    subparsers.add_parser("finalize")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "self-check":
        self_check()
    elif args.command == "train":
        train(args.replicate, args.device)
    elif args.command == "evaluate":
        evaluate(args.replicate, args.device)
    else:
        finalize()


if __name__ == "__main__":
    main()
