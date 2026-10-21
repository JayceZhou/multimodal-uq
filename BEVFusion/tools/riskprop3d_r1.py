#!/usr/bin/env python3
"""RiskProp3D R1: minimal distribution-aware multi-agent forecasting audit."""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/riskprop3d-r1-matplotlib")

import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

from nuscenes.nuscenes import NuScenes
from probset3d.f1.implementation.probabilistic_query import ProbabilisticQueryHead
from tools import probset3d_f0_score as f0
from tools import riskprop3d_r0 as r0


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/riskprop3d_r1"
CHECKPOINT_ROOT = ROOT / "outputs/probset3d_v3/f1_formal/checkpoints"
HEAD_SEEDS = (20260731, 20260801, 20260802)
MODEL_SEEDS = (20260809, 20260810, 20260811)
SAMPLE_SEEDS = (20260812, 20260813, 20260814)
BOOTSTRAP_SEED = 20260805
CONDITIONS = tuple(r0.CONDITIONS)
GROUPS = ("B0", "B1_MC", "B2_concat", "R1_DSE", "R1_interaction")
LEARNED_GROUPS = ("B0", "B2_concat", "R1_DSE", "R1_interaction")
HISTORY = r0.HISTORY
HORIZON = r0.HORIZON
STATE_FEATURES = r0.STATE_FEATURES
INPUT_DIM = r0.INPUT_DIM
DT = r0.DT
MAX_NEIGHBORS = 8
NEIGHBOR_RADIUS = 30.0
DIST_DIM = 19
RAW_UNCERTAINTY_DIM = 10
LATENT_DIM = 64
MODES = 6
SAMPLES = 64
TRAIN_EPOCHS = 15
TRAIN_BATCH = 1024
EVAL_CHUNK = 128
LEVELS = (0.50, 0.80, 0.90)
METRICS = (
    "energy", "nll", "mean_ade", "mean_fde", "min_ade", "min_fde",
    "miss", "coverage_50", "coverage_80", "coverage_90", "final_maha2",
    "final_det", "area90", "log_area90", "collision_probability",
)
COMMON_FIELDS = (
    "scene_token", "sample_token", "instance_token", "class", "range",
    "velocity", "score", "density", "history_length", "history_innovation",
    "collision_event", "crossing", "close_following", "overtaking",
    "high_density", "interaction_heavy", "high_uncertainty",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_load(path: Path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
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


def scene_sets(nusc) -> dict:
    result = {}
    for split in ("train", "label_cal", "dev"):
        result[split] = sorted(
            {
                str(nusc.get("sample", token)["scene_token"])
                for token in r0.load_infos(split)
            }
        )
    if any(
        set(result[left]) & set(result[right])
        for left, right in (("train", "label_cal"), ("train", "dev"), ("label_cal", "dev"))
    ):
        raise ValueError("train/label-cal/dev scenes are not disjoint")
    return result


def attach_scene_graph(tracks, order, observations):
    """Replace duplicated target query tensors with one indexed per-frame agent table."""
    agent_fields = {name: [] for name in ("feature", "logits", "box", "seed", "class")}
    agent_index = {}
    sample_agents = {}
    for token in order:
        current = []
        for instance, value in observations[token].items():
            index = len(agent_fields["seed"])
            agent_index[(token, instance)] = index
            current.append((instance, index, value))
            for name in ("feature", "logits", "box", "seed", "class"):
                agent_fields[name].append(value[name])
        sample_agents[token] = current

    count = len(tracks["input"])
    target_index = np.empty(count, dtype=np.int32)
    neighbor_index = np.full((count, MAX_NEIGHBORS), -1, dtype=np.int32)
    for row, (token, instance) in enumerate(zip(tracks["sample_token"], tracks["instance_token"])):
        token, instance = str(token), str(instance)
        target = agent_index[(token, instance)]
        target_index[row] = target
        center = np.asarray(agent_fields["box"][target], dtype=np.float64)[:2]
        candidates = []
        for other_instance, index, value in sample_agents[token]:
            if other_instance == instance:
                continue
            distance = float(np.linalg.norm(np.asarray(value["box"], dtype=np.float64)[:2] - center))
            if distance <= NEIGHBOR_RADIUS:
                candidates.append((distance, other_instance, index))
        candidates.sort(key=lambda value: (value[0], value[1]))
        selected = [value[2] for value in candidates[:MAX_NEIGHBORS]]
        neighbor_index[row, : len(selected)] = selected

    arrays = {
        name: np.asarray(value)
        for name, value in tracks.items()
        if name not in {"gt_input", "feature", "logits", "box", "seed"}
    }
    arrays.update(
        agent_feature=np.asarray(agent_fields["feature"], dtype=np.float32),
        agent_logits=np.asarray(agent_fields["logits"], dtype=np.float32),
        agent_box=np.asarray(agent_fields["box"], dtype=np.float32),
        agent_seed=np.asarray(agent_fields["seed"], dtype=np.float32),
        agent_class=np.asarray(agent_fields["class"], dtype=np.int16),
        target_agent_index=target_index,
        neighbor_agent_index=neighbor_index,
    )
    validate_scene_arrays(arrays)
    return arrays


def validate_scene_arrays(arrays) -> None:
    count = len(arrays["input"])
    agents = len(arrays["agent_box"])
    target = np.asarray(arrays["target_agent_index"])
    neighbor = np.asarray(arrays["neighbor_agent_index"])
    if "gt_input" in arrays or arrays["input"].shape != (count, INPUT_DIM):
        raise ValueError("forbidden GT current input or invalid motion input")
    if neighbor.shape != (count, MAX_NEIGHBORS):
        raise ValueError("invalid neighbor layout")
    if not np.all((target >= 0) & (target < agents)):
        raise ValueError("invalid target agent index")
    if not np.all((neighbor >= -1) & (neighbor < agents)):
        raise ValueError("invalid neighbor agent index")
    if np.any(neighbor == target[:, None]):
        raise ValueError("target appears in its own neighborhood")
    numeric = (
        "input", "future", "ego_future", "agent_feature", "agent_logits",
        "agent_box", "agent_seed",
    )
    if not all(np.isfinite(np.asarray(arrays[name])).all() for name in numeric):
        raise ValueError("non-finite scene array")


def load_scene(path: Path):
    """Materialize a compressed scene archive once; repeated NPZ slicing re-decompresses it."""
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    validate_scene_arrays(arrays)
    return arrays


def prepare() -> None:
    expected = [OUT / "scenes" / f"{name}.npz" for name in ("train", "label_cal", *CONDITIONS)]
    expected.append(OUT / "prepare_manifest.json")
    if any(path.exists() for path in expected):
        raise FileExistsError("R1 retained preparation output already exists")
    nusc = NuScenes(
        version="v1.0-trainval", dataroot=str(ROOT / "data/nuscenes"), verbose=False
    )
    splits = scene_sets(nusc)
    records = {}
    for split in ("train", "label_cal"):
        infos = r0.load_infos(split)
        frames = r0.frame_authority(infos)
        rows = r0.source_rows(r0.iter_clean_parts(split), clean=True)
        order, observations = r0.observation_map(
            rows, infos, frames, nusc, include_head=True
        )
        tracks = r0.build_tracks(
            order, observations, infos, frames, nusc, include_head=True
        )
        arrays = attach_scene_graph(tracks, order, observations)
        arrays["meta__split"] = np.asarray(split)
        path = OUT / "scenes" / f"{split}.npz"
        write_npz(path, arrays)
        records[split] = {
            "agents": int(len(arrays["agent_box"])),
            "sha256": sha256(path),
            "tracks": int(len(arrays["input"])),
        }
        print(json.dumps({"prepared": split, **records[split]}))

    infos = r0.load_infos("dev")
    frames = r0.frame_authority(infos)
    for condition, root in r0.CONDITIONS.items():
        rows = r0.source_rows((part for _, part in f0.iter_parts(root)), clean=False)
        order, observations = r0.observation_map(
            rows, infos, frames, nusc, include_head=True
        )
        tracks = r0.build_tracks(
            order, observations, infos, frames, nusc, include_head=True
        )
        arrays = attach_scene_graph(tracks, order, observations)
        arrays["meta__condition"] = np.asarray(condition)
        path = OUT / "scenes" / f"{condition}.npz"
        write_npz(path, arrays)
        records[condition] = {
            "agents": int(len(arrays["agent_box"])),
            "sha256": sha256(path),
            "tracks": int(len(arrays["input"])),
        }
        print(json.dumps({"prepared": condition, **records[condition]}))

    authority_manifest = r0.AUTHORITY / "manifest.json"
    manifest = {
        "conditions": list(CONDITIONS),
        "f1_v3_authority_manifest_sha256": sha256(authority_manifest),
        "head_sha256": {
            str(seed): sha256(CHECKPOINT_ROOT / f"seed_{seed}.pt") for seed in HEAD_SEEDS
        },
        "neighbor_radius_m": NEIGHBOR_RADIUS,
        "max_neighbors": MAX_NEIGHBORS,
        "official_val_read": False,
        "internal_test_read": False,
        "records": records,
        "scene_tokens": splits,
        "source_code_sha256": sha256(Path(__file__)),
    }
    write_json(OUT / "prepare_manifest.json", manifest)


def load_m2(replicate: int, device: torch.device):
    checkpoint = torch.load(
        CHECKPOINT_ROOT / f"seed_{HEAD_SEEDS[replicate]}.pt",
        map_location="cpu",
        weights_only=True,
    )
    head = ProbabilisticQueryHead().to(device).eval()
    head.load_state_dict(checkpoint["models"]["M2"])
    return head


def agent_distributions(head, scenes, device: torch.device) -> np.ndarray:
    values = []
    total = len(scenes["agent_box"])
    for begin in range(0, total, 4096):
        stop = min(begin + 4096, total)
        feature = torch.from_numpy(np.asarray(scenes["agent_feature"][begin:stop]))[:, None].to(device)
        logits = torch.from_numpy(np.asarray(scenes["agent_logits"][begin:stop]))[:, None].to(device)
        box = torch.from_numpy(np.asarray(scenes["agent_box"][begin:stop]))[:, None].to(device)
        seed = torch.from_numpy(np.asarray(scenes["agent_seed"][begin:stop]))[:, None].to(device)
        with torch.inference_mode():
            output = head(feature, logits, box, seed, torch.zeros(stop - begin, 1, device=device))
        packed = torch.cat(
            (
                output["mean_state"][:, 0],
                output["scale"][:, 0],
                output["degrees_of_freedom"][:, 0, None],
                output["yaw_mean"][:, 0, None],
                output["yaw_kappa"][:, 0, None],
            ),
            dim=-1,
        )
        values.append(packed.cpu().numpy().astype(np.float32))
    result = np.concatenate(values) if values else np.empty((0, DIST_DIM), dtype=np.float32)
    if result.shape != (total, DIST_DIM) or not np.isfinite(result).all():
        raise ValueError("invalid M2 agent distribution")
    return result


def distribution_variance_np(dist):
    dist = np.asarray(dist, dtype=np.float64)
    degrees = np.maximum(dist[..., 16], 2.0001)
    return np.square(dist[..., 8:16]) * degrees[..., None] / (degrees[..., None] - 2.0)


def raw_uncertainty_np(dist):
    dist = np.asarray(dist, dtype=np.float32)
    return np.concatenate((dist[..., 8:16], dist[..., 16:17], dist[..., 18:19]), axis=-1)


def mean_resultant_torch(kappa):
    kappa = torch.clamp(kappa, min=1e-6, max=1e4)
    return torch.special.i1e(kappa) / torch.special.i0e(kappa)


class DistributionalStateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(24, 48), nn.ReLU(), nn.Linear(48, 32))
        self.summary = nn.Sequential(nn.Linear(7, 24), nn.ReLU(), nn.Linear(24, 16))
        self.circular = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16))

    def forward(self, dist, state_mean, state_scale):
        mean = dist[..., :8]
        scale = torch.clamp(dist[..., 8:16], min=1e-6)
        degrees = torch.clamp(dist[..., 16], min=2.0001)
        variance = scale.square() * degrees[..., None] / (degrees[..., None] - 2.0)
        standard = torch.sqrt(torch.clamp(variance, min=1e-12))
        normalized_mean = (mean - state_mean) / state_scale
        linear = torch.stack(
            (normalized_mean, torch.log1p(standard), standard / (mean.abs() + 1.0)),
            dim=-1,
        ).flatten(-2)
        xy = variance[..., :2]
        summary = torch.stack(
            (
                torch.log1p(standard).mean(-1),
                torch.log(torch.clamp(variance, min=1e-12)).sum(-1) / 8.0,
                torch.log1p(torch.sqrt(xy.sum(-1))),
                torch.log1p(torch.sqrt(torch.max(xy, dim=-1).values)),
                (xy[..., 0] - xy[..., 1]) / torch.clamp(xy.sum(-1), min=1e-8),
                torch.log(torch.clamp(degrees - 2.0, min=1e-4)),
                torch.reciprocal(degrees),
            ),
            dim=-1,
        )
        yaw = dist[..., 17]
        resultant = mean_resultant_torch(dist[..., 18])
        circular = torch.stack((torch.cos(yaw), torch.sin(yaw), resultant), dim=-1)
        return torch.cat((self.linear(linear), self.summary(summary), self.circular(circular)), dim=-1)


class TrajectoryDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        output_dim = MODES * HORIZON * 4 + MODES
        self.network = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.ReLU(), nn.Linear(128, output_dim)
        )

    def forward(self, latent, cv, target_mean, target_scale):
        value = self.network(latent)
        split = MODES * HORIZON * 2
        residual = value[:, :split].reshape(-1, MODES, HORIZON, 2)
        raw_scale = value[:, split : 2 * split].reshape(-1, MODES, HORIZON, 2)
        logits = value[:, 2 * split :]
        means = cv[:, None] + residual * target_scale + target_mean
        scales = (F.softplus(raw_scale) + 1e-3) * target_scale
        return {"means": means, "scales": scales, "logits": logits}


class RiskPropModel(nn.Module):
    def __init__(self, group: str, statistics: dict):
        super().__init__()
        if group not in LEARNED_GROUPS:
            raise ValueError(group)
        self.group = group
        for name, value in statistics.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        self.motion_encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 128), nn.ReLU(), nn.Linear(128, LATENT_DIM)
        )
        if group == "B2_concat":
            self.concat_encoder = nn.Sequential(
                nn.Linear(INPUT_DIM + RAW_UNCERTAINTY_DIM, 128),
                nn.ReLU(),
                nn.Linear(128, LATENT_DIM),
            )
        if group in ("R1_DSE", "R1_interaction"):
            self.dse = DistributionalStateEncoder()
            self.fusion = nn.Sequential(
                nn.Linear(2 * LATENT_DIM, 128), nn.ReLU(), nn.Linear(128, LATENT_DIM)
            )
        if group == "R1_interaction":
            self.neighbor_actor = nn.Sequential(
                nn.Linear(LATENT_DIM + 10, 96), nn.ReLU(), nn.Linear(96, LATENT_DIM)
            )
            self.query = nn.Linear(LATENT_DIM, LATENT_DIM, bias=False)
            self.key = nn.Linear(LATENT_DIM, LATENT_DIM, bias=False)
            self.value = nn.Linear(LATENT_DIM, LATENT_DIM, bias=False)
            self.edge_bias = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
            self.edge_value = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, LATENT_DIM))
            self.interaction = nn.Sequential(
                nn.Linear(2 * LATENT_DIM, 96), nn.ReLU(), nn.Linear(96, LATENT_DIM)
            )
        self.decoder = TrajectoryDecoder()

    def probabilistic_edges(self, target, neighbor):
        delta_position = neighbor[..., :2] - target[:, None, :2]
        delta_velocity = neighbor[..., 6:8] - target[:, None, 6:8]
        target_variance = self.distribution_variance(target)
        neighbor_variance = self.distribution_variance(neighbor)
        relative_variance = target_variance[:, None] + neighbor_variance
        xy_variance = torch.clamp(relative_variance[..., :2], min=1e-8)
        maha2 = (delta_position.square() / xy_variance).sum(-1)
        radial = torch.sqrt(xy_variance.sum(-1))
        target_yaw, neighbor_yaw = target[:, None, 17], neighbor[..., 17]
        concentration = mean_resultant_torch(target[:, None, 18]) * mean_resultant_torch(neighbor[..., 18])
        heading_conflict = 1.0 - concentration * torch.cos(target_yaw - neighbor_yaw)
        velocity_norm2 = torch.clamp(delta_velocity.square().sum(-1), min=1e-6)
        closest_time = torch.clamp(
            -(delta_position * delta_velocity).sum(-1) / velocity_norm2,
            min=0.0,
            max=HORIZON * DT,
        )
        closest = delta_position + closest_time[..., None] * delta_velocity
        closest_maha = torch.sqrt(torch.clamp((closest.square() / xy_variance).sum(-1), min=0.0))
        return torch.cat(
            (
                delta_position / NEIGHBOR_RADIUS,
                delta_velocity / 10.0,
                torch.log1p(radial)[..., None],
                torch.log1p(torch.sqrt(torch.clamp(maha2, min=0.0)))[..., None],
                torch.exp(-0.5 * torch.clamp(maha2, max=80.0))[..., None],
                heading_conflict[..., None],
                (closest_time / (HORIZON * DT))[..., None],
                torch.log1p(closest_maha)[..., None],
            ),
            dim=-1,
        )

    @staticmethod
    def distribution_variance(dist):
        degrees = torch.clamp(dist[..., 16], min=2.0001)
        return dist[..., 8:16].square() * degrees[..., None] / (degrees[..., None] - 2.0)

    def forward(self, motion, target_dist, neighbor_dist, neighbor_class, neighbor_mask):
        normalized_motion = (motion - self.motion_mean) / self.motion_scale
        if self.group == "B2_concat":
            raw = torch.cat(
                (target_dist[..., 8:16], target_dist[..., 16:17], target_dist[..., 18:19]),
                dim=-1,
            )
            raw = (raw - self.raw_mean) / self.raw_scale
            latent = self.concat_encoder(torch.cat((normalized_motion, raw), dim=-1))
        else:
            motion_latent = self.motion_encoder(normalized_motion)
            if self.group == "B0":
                latent = motion_latent
            else:
                distribution_latent = self.dse(target_dist, self.state_mean, self.state_scale)
                latent = self.fusion(torch.cat((motion_latent, distribution_latent), dim=-1))
        if self.group == "R1_interaction":
            neighbor_latent = self.dse(neighbor_dist, self.state_mean, self.state_scale)
            neighbor_one_hot = F.one_hot(neighbor_class.long(), 10).to(motion.dtype)
            neighbor_actor = self.neighbor_actor(torch.cat((neighbor_latent, neighbor_one_hot), dim=-1))
            edge = self.probabilistic_edges(target_dist, neighbor_dist)
            score = (
                self.query(latent)[:, None] * self.key(neighbor_actor)
            ).sum(-1) / math.sqrt(LATENT_DIM)
            score = score + self.edge_bias(edge).squeeze(-1)
            score = score.masked_fill(~neighbor_mask, -1e4)
            weight = torch.softmax(score, dim=-1) * neighbor_mask.to(score.dtype)
            weight = weight / torch.clamp(weight.sum(-1, keepdim=True), min=1e-8)
            value = self.value(neighbor_actor) + self.edge_value(edge)
            context = (weight[..., None] * value).sum(-2)
            latent = self.interaction(torch.cat((latent, context), dim=-1))
        current = motion[:, HISTORY * STATE_FEATURES : (HISTORY + 1) * STATE_FEATURES]
        times = torch.arange(1, HORIZON + 1, device=motion.device, dtype=motion.dtype) * DT
        cv = current[:, None, :2] + times[None, :, None] * current[:, None, 6:8]
        return self.decoder(latent, cv, self.target_mean, self.target_scale)


def make_statistics(inputs, future, target_dist):
    inputs = np.asarray(inputs, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    target_dist = np.asarray(target_dist, dtype=np.float32)
    residual = future - r0.cv_prediction(inputs).astype(np.float32)
    raw = raw_uncertainty_np(target_dist)
    return {
        "motion_mean": inputs.mean(0),
        "motion_scale": np.maximum(inputs.std(0), 1e-4),
        "raw_mean": raw.mean(0),
        "raw_scale": np.maximum(raw.std(0), 1e-4),
        "state_mean": target_dist[:, :8].mean(0),
        "state_scale": np.maximum(target_dist[:, :8].std(0), 1e-4),
        "target_mean": residual.mean(0),
        "target_scale": np.maximum(residual.std(0), 0.1),
    }


def gather_batch(scenes, distributions, indices, device):
    target_index = np.asarray(scenes["target_agent_index"])[indices]
    neighbor_index = np.asarray(scenes["neighbor_agent_index"])[indices]
    neighbor_mask = neighbor_index >= 0
    safe_neighbor = np.maximum(neighbor_index, 0)
    return (
        torch.from_numpy(np.asarray(scenes["input"])[indices]).to(device),
        torch.from_numpy(distributions[target_index]).to(device),
        torch.from_numpy(distributions[safe_neighbor]).to(device),
        torch.from_numpy(np.asarray(scenes["agent_class"])[safe_neighbor].astype(np.int64)).to(device),
        torch.from_numpy(neighbor_mask).to(device),
    )


def mixture_nll_torch(output, target, temperature=1.0):
    scale = torch.clamp(output["scales"] * temperature, min=1e-4)
    residual = (target[:, None] - output["means"]) / scale
    log_density = -0.5 * (
        residual.square() + 2.0 * torch.log(scale) + math.log(2.0 * math.pi)
    ).sum((-1, -2))
    return -torch.logsumexp(F.log_softmax(output["logits"], dim=-1) + log_density, dim=-1)


def training_loss(output, target):
    nll = mixture_nll_torch(output, target).mean()
    best_ade = torch.linalg.vector_norm(output["means"] - target[:, None], dim=-1).mean(-1).min(-1).values.mean()
    return nll + 0.1 * best_ade


def collect_outputs(model, scenes, distributions, device):
    storage = {name: [] for name in ("means", "scales", "logits")}
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(scenes["input"]), 2048):
            index = np.arange(begin, min(begin + 2048, len(scenes["input"])))
            batch = gather_batch(scenes, distributions, index, device)
            output = model(*batch)
            for name in storage:
                storage[name].append(output[name].cpu().numpy())
    return {name: np.concatenate(value) for name, value in storage.items()}


def mixture_nll_np(output, target, temperature=1.0):
    means = np.asarray(output["means"], dtype=np.float64)
    scale = np.maximum(np.asarray(output["scales"], dtype=np.float64) * temperature, 1e-6)
    target = np.asarray(target, dtype=np.float64)
    residual = (target[:, None] - means) / scale
    log_density = -0.5 * (
        np.square(residual) + 2.0 * np.log(scale) + math.log(2.0 * math.pi)
    ).sum((-1, -2))
    logits = np.asarray(output["logits"], dtype=np.float64)
    log_weight = logits - logsumexp(logits, axis=-1, keepdims=True)
    return -(logsumexp(log_weight + log_density, axis=-1))


def fit_temperature(output, future):
    candidates = np.exp(np.linspace(-1.5, 1.5, 31))
    losses = np.asarray([mixture_nll_np(output, future, value).mean() for value in candidates])
    return float(candidates[int(np.argmin(losses))])


def set_determinism(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(device)


def train(replicate: int, device_index: int) -> None:
    if replicate not in range(3) or device_index not in range(4):
        raise ValueError("replicate must be 0--2 and GPU must be 0--3")
    path = OUT / "checkpoints" / f"replicate_{replicate}.pt"
    if path.exists():
        raise FileExistsError(path)
    device = torch.device(f"cuda:{device_index}")
    set_determinism(MODEL_SEEDS[replicate], device)
    manifest = strict_load(OUT / "prepare_manifest.json")
    train_scenes = load_scene(OUT / "scenes/train.npz")
    calibration = load_scene(OUT / "scenes/label_cal.npz")
    for name, value in (("train", train_scenes), ("label_cal", calibration)):
        if sha256(OUT / "scenes" / f"{name}.npz") != manifest["records"][name]["sha256"]:
            raise ValueError(f"prepared scene hash mismatch: {name}")
    head = load_m2(replicate, device)
    train_dist = agent_distributions(head, train_scenes, device)
    calibration_dist = agent_distributions(head, calibration, device)
    target_dist = train_dist[np.asarray(train_scenes["target_agent_index"])]
    statistics = make_statistics(train_scenes["input"], train_scenes["future"], target_dist)
    models = {name: RiskPropModel(name, statistics).to(device) for name in LEARNED_GROUPS}
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for name, model in models.items()
    }
    rng = np.random.default_rng(MODEL_SEEDS[replicate])
    for epoch in range(TRAIN_EPOCHS):
        permutation = rng.permutation(len(train_scenes["input"]))
        epoch_loss = {name: 0.0 for name in LEARNED_GROUPS}
        seen = 0
        for begin in range(0, len(permutation), TRAIN_BATCH):
            index = permutation[begin : begin + TRAIN_BATCH]
            batch = gather_batch(train_scenes, train_dist, index, device)
            target = torch.from_numpy(np.asarray(train_scenes["future"])[index]).to(device)
            for name, model in models.items():
                model.train()
                optimizers[name].zero_grad(set_to_none=True)
                loss = training_loss(model(*batch), target)
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite training loss: {name}")
                loss.backward()
                optimizers[name].step()
                epoch_loss[name] += float(loss.detach()) * len(index)
            seen += len(index)
        print(json.dumps({"epoch": epoch + 1, "loss": {name: value / seen for name, value in epoch_loss.items()}, "replicate": replicate}))

    temperatures = {}
    for name, model in models.items():
        output = collect_outputs(model, calibration, calibration_dist, device)
        temperatures[name] = fit_temperature(output, calibration["future"])
    label_target = calibration_dist[np.asarray(calibration["target_agent_index"])]
    logdet = np.log(np.maximum(distribution_variance_np(label_target), 1e-12)).sum(-1)
    checkpoint = {
        "architecture": {
            "decoder_modes": MODES,
            "hidden": [LATENT_DIM, 128],
            "groups": list(LEARNED_GROUPS),
        },
        "epochs": TRAIN_EPOCHS,
        "head_seed": HEAD_SEEDS[replicate],
        "high_uncertainty_logdet_q75": float(np.quantile(logdet, 0.75)),
        "model_seed": MODEL_SEEDS[replicate],
        "models": {name: model.state_dict() for name, model in models.items()},
        "prepare_manifest_sha256": sha256(OUT / "prepare_manifest.json"),
        "sampling_seed": SAMPLE_SEEDS[replicate],
        "statistics": {name: torch.as_tensor(value) for name, value in statistics.items()},
        "temperatures": temperatures,
        "training_code_sha256": sha256(Path(__file__)),
    }
    write_torch(path, checkpoint)
    print(json.dumps({"trained": True, "replicate": replicate, "temperatures": temperatures}))


def load_models(replicate: int, device: torch.device):
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
    statistics = {name: value.numpy() for name, value in checkpoint["statistics"].items()}
    models = {}
    for name in LEARNED_GROUPS:
        model = RiskPropModel(name, statistics).to(device).eval()
        model.load_state_dict(checkpoint["models"][name])
        models[name] = model
    return models, checkpoint


def numpy_output(output):
    return {name: value.detach().cpu().numpy() for name, value in output.items()}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sample_m2_motion(motion, target_dist, rng):
    count = len(motion)
    z = rng.standard_normal((count, SAMPLES, 8))
    uniform = rng.uniform(1e-7, 1.0 - 1e-7, (count, SAMPLES))
    degrees = np.asarray(target_dist[:, 16], dtype=np.float64)
    factor = np.sqrt(degrees[:, None] / chi2.ppf(uniform, degrees[:, None]))
    linear = (
        np.asarray(target_dist[:, None, :8], dtype=np.float64)
        + np.asarray(target_dist[:, None, 8:16], dtype=np.float64) * z * factor[..., None]
    )
    yaw = rng.vonmises(
        np.asarray(target_dist[:, 17], dtype=np.float64)[:, None],
        np.asarray(target_dist[:, 18], dtype=np.float64)[:, None],
        size=(count, SAMPLES),
    )
    return r0.replace_current(motion, linear, yaw)


def mc_components(model, motion, target_dist, rng, device, measure=False):
    synchronize(device)
    start = time.perf_counter()
    sampled_motion = sample_m2_motion(motion, target_dist, rng)
    flat = torch.from_numpy(sampled_motion.reshape(-1, INPUT_DIM)).to(device)
    count = len(flat)
    dummy_target = torch.zeros(count, DIST_DIM, device=device)
    dummy_neighbor = torch.zeros(count, MAX_NEIGHBORS, DIST_DIM, device=device)
    dummy_class = torch.zeros(count, MAX_NEIGHBORS, dtype=torch.long, device=device)
    dummy_mask = torch.zeros(count, MAX_NEIGHBORS, dtype=torch.bool, device=device)
    with torch.inference_mode():
        output = numpy_output(model(flat, dummy_target, dummy_neighbor, dummy_class, dummy_mask))
    synchronize(device)
    elapsed = time.perf_counter() - start if measure else 0.0
    batch = len(motion)
    means = output["means"].reshape(batch, SAMPLES * MODES, HORIZON, 2)
    scales = output["scales"].reshape(batch, SAMPLES * MODES, HORIZON, 2)
    logits = output["logits"].reshape(batch, SAMPLES, MODES).astype(np.float64)
    logits = logits - logsumexp(logits, axis=-1, keepdims=True) - math.log(SAMPLES)
    return {
        "means": means,
        "scales": scales,
        "logits": logits.reshape(batch, SAMPLES * MODES),
    }, elapsed


def sample_mixture(output, temperature, uniform, normal):
    means = np.asarray(output["means"], dtype=np.float64)
    scales = np.asarray(output["scales"], dtype=np.float64) * float(temperature)
    logits = np.asarray(output["logits"], dtype=np.float64)
    weights = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
    cumulative = np.cumsum(weights, axis=-1)
    component = (uniform[..., None] > cumulative[:, None]).sum(-1)
    component = np.minimum(component, means.shape[1] - 1)
    row = np.arange(len(means))[:, None]
    return means[row, component] + scales[row, component] * normal


def trajectory_metrics(output, sample_a, sample_b, future, ego_future, temperature):
    future = np.asarray(future, dtype=np.float64)
    sample_a = np.asarray(sample_a, dtype=np.float64)
    sample_b = np.asarray(sample_b, dtype=np.float64)
    first = np.linalg.norm(
        (sample_a - future[:, None]).reshape(len(future), len(sample_a[0]), -1), axis=-1
    ).mean(1)
    second = np.linalg.norm(
        (sample_a - sample_b).reshape(len(future), len(sample_a[0]), -1), axis=-1
    ).mean(1)
    means = np.asarray(output["means"], dtype=np.float64)
    scales = np.asarray(output["scales"], dtype=np.float64) * float(temperature)
    logits = np.asarray(output["logits"], dtype=np.float64)
    weight = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
    predictive_mean = np.einsum("nk,nkhi->nhi", weight, means)
    centered = means - predictive_mean[:, None]
    covariance = np.einsum("nk,nkhi,nkhj->nhij", weight, centered, centered)
    covariance[..., 0, 0] += np.einsum("nk,nkh->nh", weight, np.square(scales[..., 0]))
    covariance[..., 1, 1] += np.einsum("nk,nkh->nh", weight, np.square(scales[..., 1]))
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    covariance = np.einsum(
        "nhik,nhk,nhjk->nhij", eigenvector, np.maximum(eigenvalue, 1e-6), eigenvector
    )
    determinant = np.maximum(np.linalg.det(covariance), 1e-12)
    inverse = np.linalg.inv(covariance)
    residual = future - predictive_mean
    mahalanobis = np.einsum("nhi,nhij,nhj->nh", residual, inverse, residual)
    mean_error = np.linalg.norm(predictive_mean - future, axis=-1)
    sample_error = np.linalg.norm(sample_a - future[:, None], axis=-1)
    result = {
        "energy": first - 0.5 * second,
        "nll": mixture_nll_np(output, future, temperature),
        "mean_ade": mean_error.mean(-1),
        "mean_fde": mean_error[:, -1],
        "min_ade": sample_error.mean(-1).min(1),
        "min_fde": sample_error[:, :, -1].min(1),
        "miss": (sample_error[:, :, -1].min(1) > 2.0).astype(np.float64),
        "final_maha2": mahalanobis[:, -1],
        "final_det": determinant[:, -1],
        "area90": math.pi * chi2.ppf(0.90, 2) * np.sqrt(determinant[:, -1]),
        "log_area90": np.log(math.pi * chi2.ppf(0.90, 2) * np.sqrt(determinant[:, -1])),
        "collision_probability": (
            np.linalg.norm(sample_a - np.asarray(ego_future)[:, None], axis=-1).min(-1) <= 2.5
        ).mean(1),
    }
    for level in LEVELS:
        result[f"coverage_{int(level * 100)}"] = (mahalanobis <= chi2.ppf(level, 2)).mean(-1)
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("non-finite trajectory metric")
    return result


def wrap_np(value):
    return (np.asarray(value) + math.pi) % (2.0 * math.pi) - math.pi


def interaction_subsets(target, neighbor, mask, uncertainty_threshold):
    target = np.asarray(target, dtype=np.float64)
    neighbor = np.asarray(neighbor, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    delta = neighbor[..., :2] - target[:, None, :2]
    distance = np.linalg.norm(delta, axis=-1)
    heading = np.abs(wrap_np(neighbor[..., 17] - target[:, None, 17]))
    target_heading = target[:, 17]
    lateral = np.abs(
        -np.sin(target_heading)[:, None] * delta[..., 0]
        + np.cos(target_heading)[:, None] * delta[..., 1]
    )
    relative_velocity = neighbor[..., 6:8] - target[:, None, 6:8]
    longitudinal_speed = np.abs(
        np.cos(target_heading)[:, None] * relative_velocity[..., 0]
        + np.sin(target_heading)[:, None] * relative_velocity[..., 1]
    )
    crossing = np.any(mask & (distance <= 20.0) & (heading >= math.pi / 4) & (heading <= 3 * math.pi / 4), axis=1)
    following = np.any(mask & (distance <= 15.0) & (heading <= math.pi / 6) & (lateral <= 3.5), axis=1)
    overtaking = np.any(mask & (distance <= 20.0) & (heading <= math.pi / 6) & (lateral <= 4.0) & (longitudinal_speed >= 2.0), axis=1)
    density = mask.sum(-1) >= 4
    logdet = np.log(np.maximum(distribution_variance_np(target), 1e-12)).sum(-1)
    return {
        "crossing": crossing.astype(np.uint8),
        "close_following": following.astype(np.uint8),
        "overtaking": overtaking.astype(np.uint8),
        "high_density": density.astype(np.uint8),
        "interaction_heavy": (crossing | following | overtaking | density).astype(np.uint8),
        "high_uncertainty": (logdet >= uncertainty_threshold).astype(np.uint8),
    }


def evaluate(replicate: int, device_index: int) -> None:
    if replicate not in range(3) or device_index not in range(4):
        raise ValueError("replicate must be 0--2 and GPU must be 0--3")
    path = OUT / "metrics" / f"replicate_{replicate}.npz"
    if path.exists():
        raise FileExistsError(path)
    device = torch.device(f"cuda:{device_index}")
    set_determinism(SAMPLE_SEEDS[replicate], device)
    manifest = strict_load(OUT / "prepare_manifest.json")
    models, checkpoint = load_models(replicate, device)
    head = load_m2(replicate, device)
    arrays = {
        "meta__head_seed": np.asarray(HEAD_SEEDS[replicate], dtype=np.int64),
        "meta__model_seed": np.asarray(MODEL_SEEDS[replicate], dtype=np.int64),
        "meta__sampling_seed": np.asarray(SAMPLE_SEEDS[replicate], dtype=np.int64),
    }
    for condition_index, condition in enumerate(CONDITIONS):
        scene_path = OUT / "scenes" / f"{condition}.npz"
        if sha256(scene_path) != manifest["records"][condition]["sha256"]:
            raise ValueError(f"prepared scene hash mismatch: {condition}")
        scenes = load_scene(scene_path)
        distributions = agent_distributions(head, scenes, device)
        target_all = distributions[np.asarray(scenes["target_agent_index"])]
        neighbor_indices = np.asarray(scenes["neighbor_agent_index"])
        mask_all = neighbor_indices >= 0
        neighbor_all = distributions[np.maximum(neighbor_indices, 0)]
        subsets = interaction_subsets(
            target_all, neighbor_all, mask_all,
            checkpoint["high_uncertainty_logdet_q75"],
        )
        storage = {group: {metric: [] for metric in METRICS} for group in GROUPS}
        runtime = {group: 0.0 for group in GROUPS}
        rng = np.random.default_rng(SAMPLE_SEEDS[replicate] + condition_index * 1009)
        warmed = False
        for begin in range(0, len(scenes["input"]), EVAL_CHUNK):
            stop = min(begin + EVAL_CHUNK, len(scenes["input"]))
            index = np.arange(begin, stop)
            batch = gather_batch(scenes, distributions, index, device)
            if not warmed:
                with torch.inference_mode():
                    for model in models.values():
                        model(*batch)
                synchronize(device)
                warmed = True
            outputs = {}
            for group in LEARNED_GROUPS:
                synchronize(device)
                start = time.perf_counter()
                with torch.inference_mode():
                    outputs[group] = numpy_output(models[group](*batch))
                synchronize(device)
                runtime[group] += time.perf_counter() - start
            motion = np.asarray(scenes["input"])[index]
            target_dist = target_all[index]
            mc_a, elapsed = mc_components(models["B0"], motion, target_dist, rng, device, measure=True)
            mc_b, _ = mc_components(models["B0"], motion, target_dist, rng, device, measure=False)
            outputs["B1_MC"] = mc_a
            runtime["B1_MC"] += elapsed
            count = stop - begin
            uniforms = (rng.random((count, SAMPLES)), rng.random((count, SAMPLES)))
            normals = (
                rng.standard_normal((count, SAMPLES, HORIZON, 2)),
                rng.standard_normal((count, SAMPLES, HORIZON, 2)),
            )
            future = np.asarray(scenes["future"])[index]
            ego_future = np.asarray(scenes["ego_future"])[index]
            for group in GROUPS:
                temperature = checkpoint["temperatures"]["B0" if group == "B1_MC" else group]
                first_output = outputs[group]
                second_output = mc_b if group == "B1_MC" else first_output
                sample_a = sample_mixture(first_output, temperature, uniforms[0], normals[0])
                sample_b = sample_mixture(second_output, temperature, uniforms[1], normals[1])
                metrics = trajectory_metrics(
                    first_output, sample_a, sample_b, future, ego_future, temperature
                )
                for name in METRICS:
                    storage[group][name].append(np.asarray(metrics[name], dtype=np.float32))
        for name in COMMON_FIELDS[:11]:
            arrays[f"{condition}__common__{name}"] = np.asarray(scenes[name])
        for name, value in subsets.items():
            arrays[f"{condition}__common__{name}"] = value
        for group in GROUPS:
            arrays[f"{condition}__runtime__{group}__seconds"] = np.asarray(runtime[group], dtype=np.float64)
            arrays[f"{condition}__runtime__{group}__tracks"] = np.asarray(len(scenes["input"]), dtype=np.int64)
            for name in METRICS:
                arrays[f"{condition}__{group}__{name}"] = np.concatenate(storage[group][name])
    write_npz(path, arrays)
    print(json.dumps({"evaluated": True, "replicate": replicate, "output": str(path.relative_to(ROOT))}))


def scene_mean_map(value, scene, selected=None):
    value = np.asarray(value, dtype=np.float64)
    scene = np.asarray(scene)
    selected = np.ones(len(value), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    result = {}
    for token in np.unique(scene[selected]):
        current = selected & (scene == token)
        if current.any():
            result[str(token)] = float(value[current].mean())
    return result


def average_maps(maps):
    keys = sorted(set().union(*(set(value) for value in maps)))
    return np.asarray(
        [np.mean([value[key] for value in maps if key in value]) for key in keys],
        dtype=np.float64,
    )


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
    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    order = np.argsort(uncertainty, kind="stable")
    fractions = np.linspace(0.10, 1.0, 91)
    curve = np.asarray(
        [error[order[: max(1, int(len(error) * fraction))]].mean() for fraction in fractions]
    )
    return float(np.trapz(curve, fractions))


def scene_risk_maps(scene, error, baseline_risk, candidate_risk, selected=None):
    scene = np.asarray(scene)
    error = np.asarray(error, dtype=np.float64)
    baseline_risk = np.asarray(baseline_risk, dtype=np.float64)
    candidate_risk = np.asarray(candidate_risk, dtype=np.float64)
    selected = np.ones(len(scene), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    values = {name: {} for name in ("high_fde_auprc", "high_fde_auroc", "risk_coverage", "false_safe")}
    for token in np.unique(scene[selected]):
        current = selected & (scene == token)
        if current.sum() < 2:
            continue
        event = error[current] > 2.0
        base, candidate = baseline_risk[current], candidate_risk[current]
        if len(np.unique(event)) == 2:
            values["high_fde_auprc"][str(token)] = float(
                average_precision_score(event, candidate) - average_precision_score(event, base)
            )
            values["high_fde_auroc"][str(token)] = float(
                roc_auc_score(event, candidate) - roc_auc_score(event, base)
            )
        values["risk_coverage"][str(token)] = risk_coverage(base, error[current]) - risk_coverage(candidate, error[current])
        if event.any():
            base_safe = base <= np.quantile(base, 0.80)
            candidate_safe = candidate <= np.quantile(candidate, 0.80)
            values["false_safe"][str(token)] = float(
                (event * base_safe).sum() / event.sum()
                - (event * candidate_safe).sum() / event.sum()
            )
    return values


def finalize() -> None:
    result_path = OUT / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    manifest = strict_load(OUT / "prepare_manifest.json")
    runs = [np.load(OUT / "metrics" / f"replicate_{index}.npz", allow_pickle=False) for index in range(3)]

    finite_values = 0
    seed_valid = True
    common_valid = True
    for index, run in enumerate(runs):
        seed_valid &= (
            int(run["meta__head_seed"]) == HEAD_SEEDS[index]
            and int(run["meta__model_seed"]) == MODEL_SEEDS[index]
            and int(run["meta__sampling_seed"]) == SAMPLE_SEEDS[index]
        )
        for name in run.files:
            value = np.asarray(run[name])
            if np.issubdtype(value.dtype, np.number):
                if not np.isfinite(value).all():
                    raise ValueError(f"non-finite retained metric: replicate {index}/{name}")
                finite_values += int(value.size)
    for name in ("train", "label_cal", *CONDITIONS):
        with np.load(OUT / "scenes" / f"{name}.npz", allow_pickle=False) as scenes:
            validate_scene_arrays(scenes)
            for field in scenes.files:
                value = np.asarray(scenes[field])
                if np.issubdtype(value.dtype, np.number):
                    if not np.isfinite(value).all():
                        raise ValueError(f"non-finite prepared scene value: {name}/{field}")
                    finite_values += int(value.size)
    checkpoints_valid = True
    for index in range(3):
        checkpoint = torch.load(
            OUT / "checkpoints" / f"replicate_{index}.pt",
            map_location="cpu",
            weights_only=True,
        )
        checkpoints_valid &= (
            checkpoint["head_seed"] == HEAD_SEEDS[index]
            and checkpoint["model_seed"] == MODEL_SEEDS[index]
            and checkpoint["sampling_seed"] == SAMPLE_SEEDS[index]
            and checkpoint["prepare_manifest_sha256"] == sha256(OUT / "prepare_manifest.json")
        )
        tensors = list(checkpoint["statistics"].values())
        for state in checkpoint["models"].values():
            tensors.extend(state.values())
        for value in tensors:
            if not torch.isfinite(value).all():
                raise ValueError(f"non-finite checkpoint tensor: replicate {index}")
            finite_values += int(value.numel())
    for condition in CONDITIONS:
        for name in COMMON_FIELDS:
            if name == "high_uncertainty":
                continue
            reference = runs[0][f"{condition}__common__{name}"]
            common_valid &= all(
                np.array_equal(reference, run[f"{condition}__common__{name}"])
                for run in runs[1:]
            )

    def metric(run, condition, group, name):
        return np.asarray(run[f"{condition}__{group}__{name}"], dtype=np.float64)

    def paired(condition, baseline, candidate, name, subset=None):
        maps = []
        for run in runs:
            scene = run[f"{condition}__common__scene_token"]
            selected = None if subset is None else run[f"{condition}__common__{subset}"].astype(bool)
            maps.append(
                scene_mean_map(
                    metric(run, condition, baseline, name) - metric(run, condition, candidate, name),
                    scene,
                    selected,
                )
            )
        return bootstrap(average_maps(maps))

    def family_paired(family, baseline, candidate, name, severities=(1, 2, 3)):
        replicate_maps = []
        for run in runs:
            severity_maps = []
            for severity in severities:
                condition = f"{family}_s{severity}"
                severity_maps.append(
                    scene_mean_map(
                        metric(run, condition, baseline, name) - metric(run, condition, candidate, name),
                        run[f"{condition}__common__scene_token"],
                    )
                )
            keys = sorted(set().union(*(set(value) for value in severity_maps)))
            replicate_maps.append(
                {
                    key: float(np.mean([value[key] for value in severity_maps if key in value]))
                    for key in keys
                }
            )
        return bootstrap(average_maps(replicate_maps))

    def risk_difference(condition, baseline, candidate, subset=None):
        replicate = []
        for run in runs:
            selected = None if subset is None else run[f"{condition}__common__{subset}"].astype(bool)
            replicate.append(
                scene_risk_maps(
                    run[f"{condition}__common__scene_token"],
                    metric(run, condition, "B0", "mean_fde"),
                    metric(run, condition, baseline, "log_area90"),
                    metric(run, condition, candidate, "log_area90"),
                    selected,
                )
            )
        return {
            name: bootstrap(average_maps([value[name] for value in replicate]))
            for name in replicate[0]
        }

    def family_risk(family, baseline, candidate, severities=(1, 2)):
        replicate = []
        for run in runs:
            current = {name: {} for name in ("high_fde_auprc", "high_fde_auroc", "risk_coverage", "false_safe")}
            for severity in severities:
                condition = f"{family}_s{severity}"
                values = scene_risk_maps(
                    run[f"{condition}__common__scene_token"],
                    metric(run, condition, "B0", "mean_fde"),
                    metric(run, condition, baseline, "log_area90"),
                    metric(run, condition, candidate, "log_area90"),
                )
                for name, mapping in values.items():
                    current[name].update({f"s{severity}/{key}": value for key, value in mapping.items()})
            replicate.append(current)
        return {
            name: bootstrap(average_maps([value[name] for value in replicate]))
            for name in replicate[0]
        }

    distribution_encoding = {
        name: paired("clean_s0", "B2_concat", "R1_DSE", name)
        for name in ("energy", "nll")
    }
    interaction = {
        "energy": paired("clean_s0", "R1_DSE", "R1_interaction", "energy"),
        "interaction_heavy_energy": paired(
            "clean_s0", "R1_DSE", "R1_interaction", "energy", "interaction_heavy"
        ),
        "risk": risk_difference("clean_s0", "R1_DSE", "R1_interaction"),
    }
    against_mc = paired("clean_s0", "B1_MC", "R1_interaction", "energy")
    runtime = {}
    for group in GROUPS:
        seconds = np.asarray(
            [float(run[f"clean_s0__runtime__{group}__seconds"]) for run in runs]
        )
        tracks = np.asarray(
            [int(run[f"clean_s0__runtime__{group}__tracks"]) for run in runs]
        )
        runtime[group] = {
            "milliseconds_per_track": float(np.mean(1000.0 * seconds / tracks)),
            "replicate_seconds": seconds.tolist(),
        }
    runtime_ratio = runtime["B1_MC"]["milliseconds_per_track"] / max(
        runtime["R1_interaction"]["milliseconds_per_track"], 1e-12
    )

    family_energy = {
        family: family_paired(family, "B2_concat", "R1_interaction", "energy")
        for family in ("lowlight", "dropout", "beam")
    }
    family_energy_no_s3 = {
        family: family_paired(
            family, "B2_concat", "R1_interaction", "energy", severities=(1, 2)
        )
        for family in ("lowlight", "dropout", "beam")
    }

    point_harm = {}
    sharpness = {}
    condition_point = {}
    condition_sharpness = {}
    for condition in CONDITIONS:
        point_values, sharp_values = [], []
        for run in runs:
            point_values.append(
                (
                    float((metric(run, condition, "R1_interaction", "mean_ade") - metric(run, condition, "B0", "mean_ade")).mean()),
                    float((metric(run, condition, "R1_interaction", "mean_fde") - metric(run, condition, "B0", "mean_fde")).mean()),
                )
            )
            mc_maha = metric(run, condition, "B1_MC", "final_maha2")
            method_maha = metric(run, condition, "R1_interaction", "final_maha2")
            mc_area = math.pi * np.quantile(mc_maha, 0.90) * np.sqrt(metric(run, condition, "B1_MC", "final_det"))
            method_area = math.pi * np.quantile(method_maha, 0.90) * np.sqrt(metric(run, condition, "R1_interaction", "final_det"))
            sharp_values.append(float(np.median(method_area) / np.median(mc_area)))
        condition_point[condition] = {
            "ade_delta": float(np.mean([value[0] for value in point_values])),
            "fde_delta": float(np.mean([value[1] for value in point_values])),
        }
        condition_sharpness[condition] = float(np.mean(sharp_values))
    point_harm["clean"] = condition_point["clean_s0"]
    sharpness["clean"] = {"equal_coverage_area_ratio": condition_sharpness["clean_s0"]}
    for family in ("lowlight", "dropout", "beam"):
        point_harm[family] = {
            name: float(np.mean([condition_point[f"{family}_s{severity}"][name] for severity in (1, 2, 3)]))
            for name in ("ade_delta", "fde_delta")
        }
        sharpness[family] = {
            "equal_coverage_area_ratio": float(
                np.mean([condition_sharpness[f"{family}_s{severity}"] for severity in (1, 2, 3)])
            )
        }

    risk = risk_difference("clean_s0", "B1_MC", "R1_interaction")
    risk_family_no_s3 = {
        family: family_risk(family, "B1_MC", "R1_interaction")
        for family in ("lowlight", "dropout", "beam")
    }
    replicate_directions = {
        "R1_DSE_over_B2": [
            float((metric(run, "clean_s0", "B2_concat", "energy") - metric(run, "clean_s0", "R1_DSE", "energy")).mean())
            for run in runs
        ],
        "R1_interaction_over_R1_DSE": [
            float((metric(run, "clean_s0", "R1_DSE", "energy") - metric(run, "clean_s0", "R1_interaction", "energy")).mean())
            for run in runs
        ],
        "R1_interaction_over_B1_MC": [
            float((metric(run, "clean_s0", "B1_MC", "energy") - metric(run, "clean_s0", "R1_interaction", "energy")).mean())
            for run in runs
        ],
    }

    def significant(report):
        return report["ci95"][0] is not None and report["ci95"][0] > 0

    gate1 = all(significant(distribution_encoding[name]) for name in ("energy", "nll"))
    gate2 = (
        significant(interaction["energy"])
        and significant(interaction["risk"]["high_fde_auprc"])
        and significant(interaction["interaction_heavy_energy"])
        and interaction["interaction_heavy_energy"]["mean"] > interaction["energy"]["mean"]
    )
    gate3 = significant(against_mc) or (
        against_mc["ci95"][1] is not None
        and against_mc["ci95"][1] >= 0
        and runtime_ratio >= 5.0
    )
    gate4 = (
        sum(significant(family_energy[family]) for family in family_energy) >= 2
        and all(family_energy[family]["ci95"][1] is not None and family_energy[family]["ci95"][1] >= 0 for family in family_energy)
        and sum(significant(family_energy_no_s3[family]) for family in family_energy_no_s3) >= 2
        and all(family_energy_no_s3[family]["ci95"][1] is not None and family_energy_no_s3[family]["ci95"][1] >= 0 for family in family_energy_no_s3)
    )
    gate5 = all(
        value["ade_delta"] <= 0.05 and value["fde_delta"] <= 0.10
        for value in point_harm.values()
    )
    gate6 = all(value["equal_coverage_area_ratio"] <= 1.10 for value in sharpness.values())
    risk_names = ("high_fde_auprc", "risk_coverage", "false_safe")
    gate7 = any(significant(risk[name]) for name in risk_names) or any(
        significant(risk_family_no_s3[family][name])
        for family in risk_family_no_s3
        for name in risk_names
    )
    gates = {
        "G1_distribution_encoding": gate1,
        "G2_distributional_interaction": gate2,
        "G3_against_monte_carlo": gate3,
        "G4_cross_degradation": gate4,
        "G5_accuracy_preservation": gate5,
        "G6_equal_coverage_sharpness": gate6,
        "G7_future_risk_utility": gate7,
    }

    scene_tokens = manifest["scene_tokens"]
    disjoint = not any(
        set(scene_tokens[left]) & set(scene_tokens[right])
        for left, right in (("train", "label_cal"), ("train", "dev"), ("label_cal", "dev"))
    )
    input_hashes_valid = (
        sha256(r0.AUTHORITY / "manifest.json") == manifest["f1_v3_authority_manifest_sha256"]
        and checkpoints_valid
        and all(
        sha256(OUT / "scenes" / f"{name}.npz") == record["sha256"]
        for name, record in manifest["records"].items()
        )
        and all(
        sha256(CHECKPOINT_ROOT / f"seed_{seed}.pt") == manifest["head_sha256"][str(seed)]
        for seed in HEAD_SEEDS
    )
    )
    no_forbidden_fields = True
    for name in ("train", "label_cal", *CONDITIONS):
        with np.load(OUT / "scenes" / f"{name}.npz", allow_pickle=False) as scenes:
            no_forbidden_fields &= "gt_input" not in scenes.files
    validity = {
        "V1_frozen_input_hashes": input_hashes_valid,
        "V2_scene_disjoint": disjoint,
        "V3_no_GT_residual_or_corruption_input": no_forbidden_fields,
        "V4_three_preregistered_replicates": bool(seed_valid and common_valid),
        "V5_all_numeric_values_finite": finite_values > 0,
        "V6_forbidden_splits_unread": not manifest["official_val_read"] and not manifest["internal_test_read"],
    }
    valid = all(validity.values())
    passed = valid and all(gates.values())
    endpoint = (
        "INVALID R1 RUN"
        if not valid
        else "GO TO R2 FULL RISKPROP3D"
        if passed
        else "STOP RISKPROP3D METHOD ESCALATION"
    )
    result = {
        "against_monte_carlo": {"energy": against_mc, "runtime_ratio": runtime_ratio},
        "distribution_encoding": distribution_encoding,
        "endpoint": endpoint,
        "family_energy": family_energy,
        "family_energy_without_s3": family_energy_no_s3,
        "gates": gates,
        "input_sha256": {
            "code": sha256(Path(__file__)),
            "metrics": {str(index): sha256(OUT / "metrics" / f"replicate_{index}.npz") for index in range(3)},
            "prepare_manifest": sha256(OUT / "prepare_manifest.json"),
        },
        "interaction": interaction,
        "internal_test_read": False,
        "numeric_value_count": finite_values,
        "official_val_read": False,
        "passed": passed,
        "point_harm": point_harm,
        "replicate_clean_energy_directions": replicate_directions,
        "risk": risk,
        "risk_family_without_s3": risk_family_no_s3,
        "runtime": runtime,
        "sharpness": sharpness,
        "track_counts": {
            condition: int(len(runs[0][f"{condition}__common__sample_token"]))
            for condition in CONDITIONS
        },
        "valid": valid,
        "validity": validity,
    }
    write_json(result_path, result)
    for run in runs:
        run.close()
    print(json.dumps({"endpoint": endpoint, "gates": gates, "validity": validity}, indent=2, sort_keys=True))


def self_check() -> None:
    rng = np.random.default_rng(7)
    count = 5
    motion = np.zeros((count, INPUT_DIM), dtype=np.float32)
    current = HISTORY * STATE_FEATURES
    motion[:, current : current + 2] = rng.normal(size=(count, 2))
    motion[:, current + 6 : current + 8] = rng.normal(size=(count, 2))
    motion[:, -10] = 1.0
    future = r0.cv_prediction(motion).astype(np.float32)
    dist = np.zeros((count, DIST_DIM), dtype=np.float32)
    dist[:, :8] = motion[:, current : current + 8]
    dist[:, 8:16] = 0.2
    dist[:, 16] = 5.0
    dist[:, 17] = 0.3
    dist[:, 18] = 10.0
    statistics = make_statistics(motion, future, dist)
    neighbor = np.repeat(dist[:, None], MAX_NEIGHBORS, axis=1)
    neighbor[:, :, 0] += np.arange(1, MAX_NEIGHBORS + 1)
    neighbor_class = np.zeros((count, MAX_NEIGHBORS), dtype=np.int64)
    mask = np.zeros((count, MAX_NEIGHBORS), dtype=bool)
    mask[:, :2] = True
    tensors = (
        torch.from_numpy(motion), torch.from_numpy(dist), torch.from_numpy(neighbor),
        torch.from_numpy(neighbor_class), torch.from_numpy(mask),
    )
    for group in LEARNED_GROUPS:
        torch.manual_seed(3)
        model = RiskPropModel(group, statistics).eval()
        with torch.inference_mode():
            output = model(*tensors)
        if output["means"].shape != (count, MODES, HORIZON, 2):
            raise AssertionError(f"invalid output shape: {group}")
        if not all(torch.isfinite(value).all() for value in output.values()):
            raise AssertionError(f"non-finite model output: {group}")
        if not torch.isfinite(training_loss(output, torch.from_numpy(future))):
            raise AssertionError(f"non-finite loss: {group}")
    model = RiskPropModel("R1_interaction", statistics).eval()
    no_mask = torch.zeros_like(tensors[-1])
    with torch.inference_mode():
        first = model(*tensors[:-1], no_mask)["means"]
        changed = tensors[2].clone()
        changed[..., :2] += 1000.0
        second = model(tensors[0], tensors[1], changed, tensors[3], no_mask)["means"]
    torch.testing.assert_close(first, second)
    subsets = interaction_subsets(dist, neighbor, mask, uncertainty_threshold=1e9)
    if any(value.shape != (count,) for value in subsets.values()):
        raise AssertionError("invalid subset shape")
    exact = {
        "means": np.zeros((count, MODES, HORIZON, 2), dtype=np.float64),
        "scales": np.full((count, MODES, HORIZON, 2), 0.2, dtype=np.float64),
        "logits": np.zeros((count, MODES), dtype=np.float64),
    }
    shifted = {name: value.copy() for name, value in exact.items()}
    shifted["means"] += 2.0
    uniform = rng.random((count, SAMPLES))
    normal_a = rng.standard_normal((count, SAMPLES, HORIZON, 2))
    normal_b = rng.standard_normal((count, SAMPLES, HORIZON, 2))
    ego = np.full_like(future, 100.0)
    exact_metric = trajectory_metrics(
        exact,
        sample_mixture(exact, 1.0, uniform, normal_a),
        sample_mixture(exact, 1.0, uniform, normal_b),
        np.zeros_like(future),
        ego,
        1.0,
    )
    shifted_metric = trajectory_metrics(
        shifted,
        sample_mixture(shifted, 1.0, uniform, normal_a),
        sample_mixture(shifted, 1.0, uniform, normal_b),
        np.zeros_like(future),
        ego,
        1.0,
    )
    if not (
        exact_metric["energy"].mean() < shifted_metric["energy"].mean()
        and exact_metric["nll"].mean() < shifted_metric["nll"].mean()
    ):
        raise AssertionError("proper-score direction check failed")
    print(json.dumps({"self_check": "passed", "groups": list(GROUPS)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    for command in ("train", "evaluate"):
        current = subparsers.add_parser(command)
        current.add_argument("--replicate", type=int, choices=(0, 1, 2), required=True)
        current.add_argument("--device", type=int, choices=(0, 1, 2, 3), required=True)
    subparsers.add_parser("finalize")
    subparsers.add_parser("self-check")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "train":
        train(args.replicate, args.device)
    elif args.command == "evaluate":
        evaluate(args.replicate, args.device)
    elif args.command == "finalize":
        finalize()
    else:
        self_check()


if __name__ == "__main__":
    main()
