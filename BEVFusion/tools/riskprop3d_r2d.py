#!/usr/bin/env python3
"""RiskProp3D R2-D risk-preserving amortized propagation audit."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/riskprop3d-r2d-matplotlib")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import riskprop3d_r0 as r0
import riskprop3d_r1 as r1
from probset3d.f1.implementation.probabilistic_query import ProbabilisticQueryHead


OUT = ROOT / "outputs/riskprop3d_r2d"
R0_RESULT = r0.OUT / "result.json"
R2_CORE_RESULT = ROOT / "outputs/riskprop3d_r2_core/result.json"
R1_OUT = ROOT / "outputs/riskprop3d_r1"

CONDITIONS = tuple(r0.CONDITIONS)
FAMILIES = ("lowlight", "dropout", "beam")
GROUPS = ("D0_deterministic", "D1_mc_teacher", "D2_gt_only", "D3_risk_distilled")
LEARNED = ("D2_gt_only", "D3_risk_distilled")
HEAD_SEEDS = r0.HEAD_SEEDS
R0_PROPAGATION_SEEDS = r0.PROPAGATION_SEEDS
TRAIN_TEACHER_SEEDS = (20260821, 20260822, 20260823)
MODEL_SEEDS = (20260824, 20260825, 20260826)
STUDENT_SAMPLE_SEEDS = (20260827, 20260828, 20260829)
BOOTSTRAP_SEED = 20260805

HISTORY = r0.HISTORY
HORIZON = r0.HORIZON
STATE_FEATURES = r0.STATE_FEATURES
INPUT_DIM = r0.INPUT_DIM
RAW_DIM = 20
MODES = 6
SAMPLES = 64
TRAIN_EPOCHS = 15
TRAIN_BATCH = 512
EVAL_CHUNK = r0.EVAL_CHUNK
METRICS = r0.METRICS
FIDELITY = (
    "teacher_energy_distance",
    "first_moment_error",
    "second_moment_error",
    "coverage90_gap",
)
COMMON_FIELDS = (
    "scene_token",
    "sample_token",
    "instance_token",
    "class",
    "range",
    "velocity",
    "score",
    "density",
    "history_length",
    "history_innovation",
    "collision_event",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_tracks(name):
    with np.load(r0.OUT / "tracks" / f"{name}.npz", allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def load_scenes(name):
    return r1.load_scene(R1_OUT / "scenes" / f"{name}.npz")


def assert_rows(name, tracks, scenes):
    for field in ("scene_token", "sample_token", "instance_token", "input", "future"):
        if not np.array_equal(np.asarray(tracks[field]), np.asarray(scenes[field])):
            raise ValueError(f"R0/R1 row mismatch: {name}/{field}")


def prepare():
    path = OUT / "prepare_manifest.json"
    if path.exists():
        raise FileExistsError(path)
    r0_result = r0.strict_load(R0_RESULT)
    r2_result = r0.strict_load(R2_CORE_RESULT)
    if r0_result["endpoint"] != "GO TO RISKPROP3D METHOD DEVELOPMENT":
        raise ValueError("R0 does not authorize propagation")
    if not r2_result["valid"] or r2_result["endpoint"] != "STOP RISKPROP3D-CORE MECHANISM CLAIM":
        raise ValueError("R2-Core boundary is not frozen")

    names = ("train", "label_cal", *CONDITIONS)
    records = {}
    split_scenes = {}
    for name in names:
        tracks = load_tracks(name)
        scenes = load_scenes(name)
        assert_rows(name, tracks, scenes)
        split_scenes[name] = set(np.asarray(tracks["scene_token"]).astype(str).tolist())
        records[name] = {
            "rows": int(len(tracks["input"])),
            "r0_tracks_sha256": sha256(r0.OUT / "tracks" / f"{name}.npz"),
            "r1_scenes_sha256": sha256(R1_OUT / "scenes" / f"{name}.npz"),
        }
    dev_scenes = set().union(*(split_scenes[name] for name in CONDITIONS))
    if (
        split_scenes["train"] & split_scenes["label_cal"]
        or split_scenes["train"] & dev_scenes
        or split_scenes["label_cal"] & dev_scenes
    ):
        raise ValueError("scene splits overlap")

    manifest = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "conditions": list(CONDITIONS),
        "corruption_input": False,
        "current_gt_residual_input": False,
        "head_checkpoint_sha256": {
            str(seed): sha256(r0.CHECKPOINT_ROOT / f"seed_{seed}.pt")
            for seed in HEAD_SEEDS
        },
        "head_seeds": list(HEAD_SEEDS),
        "internal_test_read": False,
        "model_seeds": list(MODEL_SEEDS),
        "official_val_read": False,
        "predictor_sha256": sha256(r0.OUT / "predictor.pt"),
        "r0_result_sha256": sha256(R0_RESULT),
        "r2_core_result_sha256": sha256(R2_CORE_RESULT),
        "records": records,
        "r0_propagation_seeds": list(R0_PROPAGATION_SEEDS),
        "student_sample_seeds": list(STUDENT_SAMPLE_SEEDS),
        "train_teacher_seeds": list(TRAIN_TEACHER_SEEDS),
    }
    r0.write_json(path, manifest)
    print(json.dumps({"prepared": True, "rows": {k: v["rows"] for k, v in records.items()}}))


def set_determinism(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.cuda.set_device(device)


def load_head(replicate, device):
    checkpoint = torch.load(
        r0.CHECKPOINT_ROOT / f"seed_{HEAD_SEEDS[replicate]}.pt",
        map_location="cpu",
        weights_only=True,
    )
    head = ProbabilisticQueryHead().to(device).eval()
    head.load_state_dict(checkpoint["models"]["M2"])
    return head


def target_head_outputs(head, scenes, device):
    target = np.asarray(scenes["target_agent_index"], dtype=np.int64)
    storage = {
        name: []
        for name in (
            "mean_state",
            "scale",
            "degrees_of_freedom",
            "yaw_mean",
            "yaw_kappa",
        )
    }
    for begin in range(0, len(target), 4096):
        index = target[begin : begin + 4096]
        with torch.inference_mode():
            output = head(
                torch.from_numpy(np.asarray(scenes["agent_feature"])[index])[:, None].to(device),
                torch.from_numpy(np.asarray(scenes["agent_logits"])[index])[:, None].to(device),
                torch.from_numpy(np.asarray(scenes["agent_box"])[index])[:, None].to(device),
                torch.from_numpy(np.asarray(scenes["agent_seed"])[index])[:, None].to(device),
                torch.zeros(len(index), 1, device=device),
            )
        for name in storage:
            storage[name].append(output[name][:, 0].detach().cpu().numpy())
    result = {name: np.concatenate(value) for name, value in storage.items()}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("non-finite M2 head output")
    return result


def dev_head_outputs(head, tracks, device):
    output = r0.head_outputs(head, tracks, device)
    return {
        name: output[name]
        for name in ("mean_state", "scale", "degrees_of_freedom", "yaw_mean", "yaw_kappa")
    }


def raw_token(output):
    scale = np.maximum(np.asarray(output["scale"], dtype=np.float32), 1e-6)
    degrees = np.maximum(np.asarray(output["degrees_of_freedom"], dtype=np.float32), 2.0001)
    yaw = np.asarray(output["yaw_mean"], dtype=np.float32)
    kappa = np.maximum(np.asarray(output["yaw_kappa"], dtype=np.float32), 0.0)
    token = np.concatenate(
        (
            np.asarray(output["mean_state"], dtype=np.float32),
            np.log(scale),
            np.log(np.maximum(degrees - 2.0, 1e-4))[:, None],
            np.sin(yaw)[:, None],
            np.cos(yaw)[:, None],
            np.log1p(kappa)[:, None],
        ),
        axis=-1,
    )
    if token.shape[1] != RAW_DIM or not np.isfinite(token).all():
        raise ValueError("invalid raw probability token")
    return token.astype(np.float32)


class Student(nn.Module):
    def __init__(self, statistics):
        super().__init__()
        for name, value in statistics.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        output_dim = MODES + 2 * MODES * HORIZON * 2
        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM + RAW_DIM, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, motion, token):
        feature = torch.cat((motion, token), dim=-1)
        raw = self.network((feature - self.feature_mean) / self.feature_scale)
        logits = raw[:, :MODES]
        offset = MODES
        means = raw[:, offset : offset + MODES * HORIZON * 2].reshape(
            -1, MODES, HORIZON, 2
        )
        offset += MODES * HORIZON * 2
        scales = raw[:, offset:].reshape(-1, MODES, HORIZON, 2)
        current = motion[:, HISTORY * STATE_FEATURES : (HISTORY + 1) * STATE_FEATURES]
        times = (
            torch.arange(1, HORIZON + 1, device=motion.device, dtype=motion.dtype)
            * r0.DT
        )
        cv = current[:, None, :2] + times[None, :, None] * current[:, None, 6:8]
        return {
            "logits": logits,
            "means": cv[:, None] + means * self.target_scale + self.target_mean,
            "scales": F.softplus(scales) * self.target_scale + 1e-4,
        }


def make_statistics(motion, token, future):
    feature = np.concatenate((motion, token), axis=-1).astype(np.float32)
    residual = np.asarray(future, dtype=np.float32) - r0.cv_prediction(motion).astype(np.float32)
    return {
        "feature_mean": feature.mean(0),
        "feature_scale": np.maximum(feature.std(0), 1e-4),
        "target_mean": residual.mean(0),
        "target_scale": np.maximum(residual.std(0), 0.1),
    }


def mixture_moments_torch(output, temperature=1.0):
    weight = torch.softmax(output["logits"], dim=-1)
    means = output["means"]
    scales = output["scales"] * temperature
    mean = torch.einsum("nk,nkhi->nhi", weight, means)
    centered = means - mean[:, None]
    covariance = torch.einsum("nk,nkhi,nkhj->nhij", weight, centered, centered)
    covariance[..., 0, 0] += torch.einsum("nk,nkh->nh", weight, scales[..., 0].square())
    covariance[..., 1, 1] += torch.einsum("nk,nkh->nh", weight, scales[..., 1].square())
    eye = torch.eye(2, device=mean.device, dtype=mean.dtype)
    return mean, covariance + 1e-5 * eye


def empirical_moments_torch(samples):
    mean = samples.mean(1)
    centered = samples - mean[:, None]
    covariance = torch.einsum("nmhi,nmhj->nhij", centered, centered) / samples.shape[1]
    eye = torch.eye(2, device=samples.device, dtype=samples.dtype)
    return mean, covariance + 1e-5 * eye


def mahalanobis_torch(target, mean, covariance):
    residual = target - mean
    inverse = torch.linalg.inv(covariance)
    return torch.einsum("nhi,nhij,nhj->nh", residual, inverse, residual)


def gt_loss(output, future, unit):
    nll = r1.mixture_nll_torch(output, future).mean() / (HORIZON * 2)
    best = (
        torch.linalg.vector_norm(output["means"] - future[:, None], dim=-1)
        .mean(-1)
        .min(-1)
        .values.mean()
        / unit
    )
    return nll + 0.1 * best


def distillation_losses(output, teacher, future, risk_mean, risk_scale, target_scale):
    weight = torch.softmax(output["logits"], dim=-1)
    normal_a = torch.randn_like(output["means"])
    normal_b = torch.randn_like(output["means"])
    sample_a = output["means"] + output["scales"] * normal_a
    sample_b = output["means"] + output["scales"] * normal_b
    scale = target_scale[None, None]
    student_a = (sample_a / scale).flatten(-2)
    student_b = (sample_b / scale).flatten(-2)
    teacher_flat = (teacher / scale).flatten(-2)
    cross = torch.linalg.vector_norm(
        student_a[:, :, None] - teacher_flat[:, None], dim=-1
    ).mean(-1)
    self_distance = torch.linalg.vector_norm(
        student_a[:, :, None] - student_b[:, None], dim=-1
    )
    transport = (
        2.0 * (weight * cross).sum(-1)
        - torch.einsum("nk,nl,nkl->n", weight, weight, self_distance)
    ).mean()

    student_mean, student_cov = mixture_moments_torch(output)
    teacher_mean, teacher_cov = empirical_moments_torch(teacher)
    normalized_mean = F.smooth_l1_loss(
        student_mean / target_scale, teacher_mean / target_scale
    )
    student_var = torch.diagonal(student_cov, dim1=-2, dim2=-1)
    teacher_var = torch.diagonal(teacher_cov, dim1=-2, dim2=-1)
    log_variance = F.smooth_l1_loss(torch.log(student_var), torch.log(teacher_var))
    student_corr = student_cov[..., 0, 1] / torch.sqrt(student_var.prod(-1))
    teacher_corr = teacher_cov[..., 0, 1] / torch.sqrt(teacher_var.prod(-1))
    correlation = F.smooth_l1_loss(student_corr, teacher_corr)
    moment = normalized_mean + log_variance + correlation

    student_maha = mahalanobis_torch(future, student_mean, student_cov)
    teacher_maha = mahalanobis_torch(future, teacher_mean, teacher_cov)
    calibration = F.smooth_l1_loss(torch.log1p(student_maha), torch.log1p(teacher_maha))
    student_logdet = torch.linalg.slogdet(student_cov[:, -1])[1]
    teacher_logdet = torch.linalg.slogdet(teacher_cov[:, -1])[1]
    sharpness = F.relu(student_logdet - teacher_logdet).square().mean()
    calsharp = calibration + sharpness

    student_risk = (student_logdet - risk_mean) / risk_scale
    teacher_risk = (teacher_logdet - risk_mean) / risk_scale
    risk_mse = F.smooth_l1_loss(student_risk, teacher_risk)
    centered_student = student_risk - student_risk.mean()
    centered_teacher = teacher_risk - teacher_risk.mean()
    corr = (centered_student * centered_teacher).mean() / torch.clamp(
        centered_student.square().mean().sqrt()
        * centered_teacher.square().mean().sqrt(),
        min=1e-6,
    )
    risk = risk_mse + 1.0 - corr
    return {
        "transport": transport,
        "moment": moment,
        "calsharp": calsharp,
        "risk": risk,
    }


def sliced_output(output, begin, stop):
    return {name: np.asarray(value[begin:stop]) for name, value in output.items()}


def generate_teacher_samples(motion, output, predictor, parameters, rng, device):
    samples = []
    covariance = parameters["covariance"]["MLP"]
    for begin in range(0, len(motion), EVAL_CHUNK):
        stop = min(begin + EVAL_CHUNK, len(motion))
        count = stop - begin
        z = rng.standard_normal((count, SAMPLES, 8))
        radial = rng.uniform(1e-7, 1.0 - 1e-7, (count, SAMPLES))
        yaw_normal = rng.standard_normal((count, SAMPLES))
        state = r0.sample_states(
            sliced_output(output, begin, stop),
            rng,
            SAMPLES,
            True,
            z,
            radial,
            yaw_normal,
        )
        sampled_motion = r0.replace_current(motion[begin:stop], *state)
        centers = r0.predict_centers(
            "MLP", sampled_motion, predictor, parameters, device
        )
        noise = r0.process_noise(rng, covariance, count, SAMPLES)
        samples.append((centers + noise).astype(np.float32))
    result = np.concatenate(samples)
    if result.shape != (len(motion), SAMPLES, HORIZON, 2):
        raise ValueError("teacher sample shape mismatch")
    if not np.isfinite(result).all():
        raise ValueError("non-finite teacher samples")
    return result


def teacher_risk_statistics(teacher):
    values = []
    for begin in range(0, len(teacher), 4096):
        final = np.asarray(teacher[begin : begin + 4096, :, -1], dtype=np.float64)
        centered = final - final.mean(1, keepdims=True)
        covariance = np.einsum("nmi,nmj->nij", centered, centered) / final.shape[1]
        covariance[:, 0, 0] += 1e-5
        covariance[:, 1, 1] += 1e-5
        values.append(np.linalg.slogdet(covariance)[1])
    logdet = np.concatenate(values)
    return float(logdet.mean()), float(max(logdet.std(), 1e-4))


def collect_student(model, motion, token, device):
    storage = {name: [] for name in ("means", "scales", "logits")}
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(motion), 4096):
            stop = min(begin + 4096, len(motion))
            output = model(
                torch.from_numpy(np.asarray(motion[begin:stop], dtype=np.float32)).to(device),
                torch.from_numpy(np.asarray(token[begin:stop], dtype=np.float32)).to(device),
            )
            for name in storage:
                storage[name].append(output[name].detach().cpu().numpy())
    return {name: np.concatenate(value) for name, value in storage.items()}


def train(replicate, device_index):
    if replicate not in range(3) or device_index not in range(4):
        raise ValueError("replicate must be 0--2 and GPU must be 0--3")
    path = OUT / "checkpoints" / f"replicate_{replicate}.pt"
    if path.exists():
        raise FileExistsError(path)
    device = torch.device(f"cuda:{device_index}")
    set_determinism(MODEL_SEEDS[replicate], device)
    manifest = r0.strict_load(OUT / "prepare_manifest.json")
    if manifest["r0_result_sha256"] != sha256(R0_RESULT):
        raise ValueError("R0 result hash changed")

    tracks = load_tracks("train")
    scenes = load_scenes("train")
    assert_rows("train", tracks, scenes)
    head = load_head(replicate, device)
    head_output = target_head_outputs(head, scenes, device)
    token = raw_token(head_output)
    predictor, parameters = r0.load_predictor(device)
    teacher = generate_teacher_samples(
        np.asarray(tracks["input"], dtype=np.float32),
        head_output,
        predictor,
        parameters,
        np.random.default_rng(TRAIN_TEACHER_SEEDS[replicate]),
        device,
    )
    risk_mean, risk_scale = teacher_risk_statistics(teacher)
    statistics = make_statistics(
        np.asarray(tracks["input"], dtype=np.float32),
        token,
        np.asarray(tracks["future"], dtype=np.float32),
    )
    template = Student(statistics).to(device)
    initial_state = {
        name: value.detach().clone() for name, value in template.state_dict().items()
    }
    models = {name: Student(statistics).to(device) for name in LEARNED}
    for model in models.values():
        model.load_state_dict(initial_state)
    initial_hashes = {name: tensor_state_hash(model) for name, model in models.items()}
    parameter_counts = {
        name: int(sum(value.numel() for value in model.parameters()))
        for name, model in models.items()
    }
    if len(set(initial_hashes.values())) != 1 or len(set(parameter_counts.values())) != 1:
        raise ValueError("D2/D3 architecture or initialization mismatch")
    optimizers = {
        name: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for name, model in models.items()
    }
    motion = np.asarray(tracks["input"], dtype=np.float32)
    future = np.asarray(tracks["future"], dtype=np.float32)
    unit = float(np.mean(statistics["target_scale"]))
    rng = np.random.default_rng(MODEL_SEEDS[replicate])
    epoch_records = []
    for epoch in range(TRAIN_EPOCHS):
        permutation = rng.permutation(len(motion))
        totals = {
            name: {key: 0.0 for key in ("total", "gt", "transport", "moment", "calsharp", "risk")}
            for name in LEARNED
        }
        seen = 0
        for begin in range(0, len(permutation), TRAIN_BATCH):
            index = permutation[begin : begin + TRAIN_BATCH]
            batch_motion = torch.from_numpy(motion[index]).to(device)
            batch_token = torch.from_numpy(token[index]).to(device)
            batch_future = torch.from_numpy(future[index]).to(device)
            batch_teacher = torch.from_numpy(teacher[index]).to(device)
            count = len(index)

            for name, model in models.items():
                model.train()
                optimizers[name].zero_grad(set_to_none=True)
                output = model(batch_motion, batch_token)
                current_gt = gt_loss(output, batch_future, unit)
                components = {key: torch.zeros((), device=device) for key in ("transport", "moment", "calsharp", "risk")}
                if name == "D3_risk_distilled":
                    components = distillation_losses(
                        output,
                        batch_teacher,
                        batch_future,
                        torch.as_tensor(risk_mean, device=device),
                        torch.as_tensor(risk_scale, device=device),
                        model.target_scale,
                    )
                    loss = (
                        current_gt
                        + components["transport"]
                        + 0.25 * components["moment"]
                        + 0.25 * components["calsharp"]
                        + 0.25 * components["risk"]
                    )
                else:
                    loss = current_gt
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite loss: {name}")
                loss.backward()
                optimizers[name].step()
                totals[name]["total"] += float(loss.detach()) * count
                totals[name]["gt"] += float(current_gt.detach()) * count
                for key, value in components.items():
                    totals[name][key] += float(value.detach()) * count
            seen += count
        record = {
            name: {key: value / seen for key, value in current.items()}
            for name, current in totals.items()
        }
        epoch_records.append(record)
        print(json.dumps({"epoch": epoch + 1, "loss": record, "replicate": replicate}))

    calibration_tracks = load_tracks("label_cal")
    calibration_scenes = load_scenes("label_cal")
    assert_rows("label_cal", calibration_tracks, calibration_scenes)
    calibration_output = target_head_outputs(head, calibration_scenes, device)
    calibration_token = raw_token(calibration_output)
    temperatures = {}
    for name, model in models.items():
        output = collect_student(
            model,
            np.asarray(calibration_tracks["input"], dtype=np.float32),
            calibration_token,
            device,
        )
        temperatures[name] = r1.fit_temperature(output, calibration_tracks["future"])

    checkpoint = {
        "architecture": [INPUT_DIM + RAW_DIM, 96, 64, 150],
        "epochs": TRAIN_EPOCHS,
        "head_seed": HEAD_SEEDS[replicate],
        "initialization_hashes": initial_hashes,
        "loss_weights": {
            "gt": 1.0,
            "transport": 1.0,
            "moment": 0.25,
            "calsharp": 0.25,
            "risk": 0.25,
        },
        "model_seed": MODEL_SEEDS[replicate],
        "models": {name: model.state_dict() for name, model in models.items()},
        "parameter_counts": parameter_counts,
        "prepare_manifest_sha256": sha256(OUT / "prepare_manifest.json"),
        "risk_mean": risk_mean,
        "risk_scale": risk_scale,
        "statistics": {name: torch.as_tensor(value) for name, value in statistics.items()},
        "student_sample_seed": STUDENT_SAMPLE_SEEDS[replicate],
        "temperatures": temperatures,
        "train_teacher_seed": TRAIN_TEACHER_SEEDS[replicate],
        "training_code_sha256": sha256(Path(__file__)),
    }
    r0.write_torch(path, checkpoint)
    print(
        json.dumps(
            {
                "trained": True,
                "replicate": replicate,
                "parameters": parameter_counts,
                "temperatures": temperatures,
            }
        )
    )


def load_students(replicate, device):
    checkpoint = torch.load(
        OUT / "checkpoints" / f"replicate_{replicate}.pt",
        map_location="cpu",
        weights_only=True,
    )
    if (
        checkpoint["head_seed"] != HEAD_SEEDS[replicate]
        or checkpoint["train_teacher_seed"] != TRAIN_TEACHER_SEEDS[replicate]
        or checkpoint["model_seed"] != MODEL_SEEDS[replicate]
        or checkpoint["student_sample_seed"] != STUDENT_SAMPLE_SEEDS[replicate]
    ):
        raise ValueError("checkpoint seed mismatch")
    statistics = {
        name: value.numpy() for name, value in checkpoint["statistics"].items()
    }
    models = {}
    for name in LEARNED:
        model = Student(statistics).to(device).eval()
        model.load_state_dict(checkpoint["models"][name])
        models[name] = model
    return models, checkpoint


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def numpy_output(output):
    return {name: value.detach().cpu().numpy() for name, value in output.items()}


def mixture_moments_np(output, temperature):
    means = np.asarray(output["means"], dtype=np.float64)
    scales = np.asarray(output["scales"], dtype=np.float64) * float(temperature)
    logits = np.asarray(output["logits"], dtype=np.float64)
    weight = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
    mean = np.einsum("nk,nkhi->nhi", weight, means)
    centered = means - mean[:, None]
    covariance = np.einsum("nk,nkhi,nkhj->nhij", weight, centered, centered)
    covariance[..., 0, 0] += np.einsum("nk,nkh->nh", weight, np.square(scales[..., 0]))
    covariance[..., 1, 1] += np.einsum("nk,nkh->nh", weight, np.square(scales[..., 1]))
    covariance[..., 0, 0] += 1e-6
    covariance[..., 1, 1] += 1e-6
    return mean, covariance


def replay_teacher_chunk(motion, output, predictor, parameters, rng, device):
    count = len(motion)
    centers = []
    for _draw in range(2):
        z = rng.standard_normal((count, SAMPLES, 8))
        radial = rng.uniform(1e-7, 1.0 - 1e-7, (count, SAMPLES))
        yaw_normal = rng.standard_normal((count, SAMPLES))
        state = r0.sample_states(output, rng, SAMPLES, True, z, radial, yaw_normal)
        sampled_motion = r0.replace_current(motion, *state)
        centers.append(
            r0.predict_centers(
                "MLP", sampled_motion, predictor, parameters, device
            )
        )
    # R0 evaluated CV before MLP. Preserve its two discarded process-noise draws.
    r0.process_noise(rng, parameters["covariance"]["CV"], count, SAMPLES)
    r0.process_noise(rng, parameters["covariance"]["CV"], count, SAMPLES)
    noise_a = r0.process_noise(
        rng, parameters["covariance"]["MLP"], count, SAMPLES
    )
    noise_b = r0.process_noise(
        rng, parameters["covariance"]["MLP"], count, SAMPLES
    )
    deterministic = r0.predict_centers(
        "MLP", motion[:, None], predictor, parameters, device
    )
    teacher_covariance = np.asarray(parameters["covariance"]["MLP"], dtype=np.float64)[None] + np.einsum(
        "nmhi,nmhj->nhij",
        centers[0] - centers[0].mean(1, keepdims=True),
        centers[0] - centers[0].mean(1, keepdims=True),
    ) / SAMPLES
    return {
        "D0_centers": deterministic,
        "D1_centers_a": centers[0],
        "D1_centers_b": centers[1],
        "noise_a": noise_a,
        "noise_b": noise_b,
        "teacher_a": centers[0] + noise_a,
        "teacher_b": centers[1] + noise_b,
        "teacher_mean": centers[0].mean(1),
        "teacher_covariance": teacher_covariance,
    }


def timed_teacher_chunk(motion, output, predictor, parameters, rng, device):
    synchronize(device)
    start = time.perf_counter()
    count = len(motion)
    z = rng.standard_normal((count, SAMPLES, 8))
    radial = rng.uniform(1e-7, 1.0 - 1e-7, (count, SAMPLES))
    yaw_normal = rng.standard_normal((count, SAMPLES))
    state = r0.sample_states(output, rng, SAMPLES, True, z, radial, yaw_normal)
    sampled_motion = r0.replace_current(motion, *state)
    r0.predict_centers("MLP", sampled_motion, predictor, parameters, device)
    r0.process_noise(rng, parameters["covariance"]["MLP"], count, SAMPLES)
    synchronize(device)
    return time.perf_counter() - start


def student_fidelity(
    output,
    temperature,
    sample_a,
    sample_b,
    teacher,
    student_metrics,
    teacher_metrics,
):
    student_mean, student_covariance = mixture_moments_np(output, temperature)
    teacher_a = np.asarray(teacher["teacher_a"], dtype=np.float64)
    teacher_b = np.asarray(teacher["teacher_b"], dtype=np.float64)
    sample_a = np.asarray(sample_a, dtype=np.float64)
    sample_b = np.asarray(sample_b, dtype=np.float64)
    cross = np.linalg.norm(
        (sample_a - teacher_a).reshape(len(sample_a), SAMPLES, -1), axis=-1
    ).mean(1)
    student_self = np.linalg.norm(
        (sample_a - sample_b).reshape(len(sample_a), SAMPLES, -1), axis=-1
    ).mean(1)
    teacher_self = np.linalg.norm(
        (teacher_a - teacher_b).reshape(len(sample_a), SAMPLES, -1), axis=-1
    ).mean(1)
    first = np.linalg.norm(
        student_mean - np.asarray(teacher["teacher_mean"]), axis=-1
    ).mean(-1)
    difference = student_covariance - np.asarray(teacher["teacher_covariance"])
    numerator = np.linalg.norm(difference, axis=(-2, -1))
    denominator = np.maximum(
        np.linalg.norm(np.asarray(teacher["teacher_covariance"]), axis=(-2, -1)),
        1e-6,
    )
    result = {
        "teacher_energy_distance": 2.0 * cross - student_self - teacher_self,
        "first_moment_error": first,
        "second_moment_error": (numerator / denominator).mean(-1),
        "coverage90_gap": np.abs(
            np.asarray(student_metrics["coverage_90"])
            - np.asarray(teacher_metrics["coverage_90"])
        ),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("non-finite fidelity metric")
    return result


def evaluate(replicate, device_index):
    if replicate not in range(3) or device_index not in range(4):
        raise ValueError("replicate must be 0--2 and GPU must be 0--3")
    path = OUT / "metrics" / f"replicate_{replicate}.npz"
    if path.exists():
        raise FileExistsError(path)
    device = torch.device(f"cuda:{device_index}")
    set_determinism(STUDENT_SAMPLE_SEEDS[replicate], device)
    manifest = r0.strict_load(OUT / "prepare_manifest.json")
    models, checkpoint = load_students(replicate, device)
    head = load_head(replicate, device)
    predictor, parameters = r0.load_predictor(device)
    r0_archive = np.load(
        r0.OUT / "metrics" / f"replicate_{replicate}.npz", allow_pickle=False
    )
    arrays = {
        "meta__head_seed": np.asarray(HEAD_SEEDS[replicate], dtype=np.int64),
        "meta__model_seed": np.asarray(MODEL_SEEDS[replicate], dtype=np.int64),
        "meta__r0_propagation_seed": np.asarray(
            R0_PROPAGATION_SEEDS[replicate], dtype=np.int64
        ),
        "meta__student_sample_seed": np.asarray(
            STUDENT_SAMPLE_SEEDS[replicate], dtype=np.int64
        ),
        "meta__train_teacher_seed": np.asarray(
            TRAIN_TEACHER_SEEDS[replicate], dtype=np.int64
        ),
    }
    for condition_index, condition in enumerate(CONDITIONS):
        tracks = load_tracks(condition)
        scenes = load_scenes(condition)
        assert_rows(condition, tracks, scenes)
        head_output = dev_head_outputs(head, tracks, device)
        token = raw_token(head_output)
        authority_rng = np.random.default_rng(
            R0_PROPAGATION_SEEDS[replicate] + condition_index * 1009
        )
        student_rng = np.random.default_rng(
            STUDENT_SAMPLE_SEEDS[replicate] + condition_index * 1009
        )
        runtime_rng = np.random.default_rng(
            STUDENT_SAMPLE_SEEDS[replicate] + 100000 + condition_index * 1009
        )
        storage = {
            group: {name: [] for name in METRICS} for group in GROUPS
        }
        fidelity = {
            group: {name: [] for name in FIDELITY} for group in LEARNED
        }
        runtime = {"D1_mc_teacher": 0.0, "D3_risk_distilled": 0.0}
        warmed = False
        motion = np.asarray(tracks["input"], dtype=np.float32)
        future_all = np.asarray(tracks["future"], dtype=np.float32)
        ego_all = np.asarray(tracks["ego_future"], dtype=np.float32)
        for begin in range(0, len(motion), EVAL_CHUNK):
            stop = min(begin + EVAL_CHUNK, len(motion))
            current_motion = motion[begin:stop]
            current_output = sliced_output(head_output, begin, stop)
            future = future_all[begin:stop]
            ego = ego_all[begin:stop]
            teacher = replay_teacher_chunk(
                current_motion,
                current_output,
                predictor,
                parameters,
                authority_rng,
                device,
            )
            teacher_metrics = {
                "D0_deterministic": r0.distribution_metrics(
                    teacher["D0_centers"],
                    teacher["D0_centers"],
                    future,
                    ego,
                    parameters["covariance"]["MLP"],
                    teacher["noise_a"],
                    teacher["noise_b"],
                ),
                "D1_mc_teacher": r0.distribution_metrics(
                    teacher["D1_centers_a"],
                    teacher["D1_centers_b"],
                    future,
                    ego,
                    parameters["covariance"]["MLP"],
                    teacher["noise_a"],
                    teacher["noise_b"],
                ),
            }
            for group, values in teacher_metrics.items():
                for name in METRICS:
                    storage[group][name].append(
                        np.asarray(values[name], dtype=np.float32)
                    )

            batch_motion = torch.from_numpy(current_motion).to(device)
            batch_token = torch.from_numpy(token[begin:stop]).to(device)
            if not warmed:
                with torch.inference_mode():
                    for model in models.values():
                        model(batch_motion, batch_token)
                timed_teacher_chunk(
                    current_motion,
                    current_output,
                    predictor,
                    parameters,
                    runtime_rng,
                    device,
                )
                warmed = True
            outputs = {}
            for group, model in models.items():
                if group == "D3_risk_distilled":
                    synchronize(device)
                    start = time.perf_counter()
                    measured_token = raw_token(current_output)
                    measured_motion = torch.from_numpy(current_motion).to(device)
                    measured_token = torch.from_numpy(measured_token).to(device)
                    with torch.inference_mode():
                        outputs[group] = numpy_output(
                            model(measured_motion, measured_token)
                        )
                    synchronize(device)
                    runtime[group] += time.perf_counter() - start
                else:
                    with torch.inference_mode():
                        outputs[group] = numpy_output(model(batch_motion, batch_token))
            runtime["D1_mc_teacher"] += timed_teacher_chunk(
                current_motion,
                current_output,
                predictor,
                parameters,
                runtime_rng,
                device,
            )

            count = stop - begin
            uniforms = (
                student_rng.random((count, SAMPLES)),
                student_rng.random((count, SAMPLES)),
            )
            normals = (
                student_rng.standard_normal((count, SAMPLES, HORIZON, 2)),
                student_rng.standard_normal((count, SAMPLES, HORIZON, 2)),
            )
            for group in LEARNED:
                temperature = checkpoint["temperatures"][group]
                sample_a = r1.sample_mixture(
                    outputs[group], temperature, uniforms[0], normals[0]
                )
                sample_b = r1.sample_mixture(
                    outputs[group], temperature, uniforms[1], normals[1]
                )
                values = r1.trajectory_metrics(
                    outputs[group], sample_a, sample_b, future, ego, temperature
                )
                for name in METRICS:
                    storage[group][name].append(
                        np.asarray(values[name], dtype=np.float32)
                    )
                current_fidelity = student_fidelity(
                    outputs[group],
                    temperature,
                    sample_a,
                    sample_b,
                    teacher,
                    values,
                    teacher_metrics["D1_mc_teacher"],
                )
                for name in FIDELITY:
                    fidelity[group][name].append(
                        np.asarray(current_fidelity[name], dtype=np.float32)
                    )

        for name in COMMON_FIELDS:
            arrays[f"{condition}__common__{name}"] = np.asarray(tracks[name])
        replay_max = 0.0
        for group, r0_group in (
            ("D0_deterministic", "P1"),
            ("D1_mc_teacher", "P3"),
        ):
            for name in METRICS:
                value = np.concatenate(storage[group][name])
                arrays[f"{condition}__{group}__{name}"] = value
                authority = np.asarray(
                    r0_archive[f"{condition}__MLP__{r0_group}__{name}"]
                )
                replay_max = max(
                    replay_max,
                    float(np.max(np.abs(value.astype(np.float64) - authority.astype(np.float64)))),
                )
        arrays[f"{condition}__replay__max_abs"] = np.asarray(
            replay_max, dtype=np.float64
        )
        for group in LEARNED:
            for name in METRICS:
                arrays[f"{condition}__{group}__{name}"] = np.concatenate(
                    storage[group][name]
                )
            for name in FIDELITY:
                arrays[f"{condition}__{group}__{name}"] = np.concatenate(
                    fidelity[group][name]
                )
        for group in runtime:
            arrays[f"{condition}__runtime__{group}__seconds"] = np.asarray(
                runtime[group], dtype=np.float64
            )
            arrays[f"{condition}__runtime__{group}__tracks"] = np.asarray(
                len(motion), dtype=np.int64
            )
    r0_archive.close()
    r0.write_npz(path, arrays)
    print(json.dumps({"evaluated": True, "replicate": replicate, "output": str(path.relative_to(ROOT))}))


def metric(run, condition, group, name):
    return np.asarray(run[f"{condition}__{group}__{name}"], dtype=np.float64)


def paired(runs, condition, baseline, candidate, name):
    maps = [
        r1.scene_mean_map(
            metric(run, condition, baseline, name)
            - metric(run, condition, candidate, name),
            run[f"{condition}__common__scene_token"],
        )
        for run in runs
    ]
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


def bootstrap_intervals(value):
    value = np.asarray(value, dtype=np.float64)
    value = value[np.isfinite(value)]
    if not len(value):
        return {
            "mean": None,
            "ci90": [None, None],
            "ci95": [None, None],
            "count": 0,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(10000, dtype=np.float64)
    for index in range(len(means)):
        means[index] = value[rng.integers(0, len(value), len(value))].mean()
    return {
        "mean": float(value.mean()),
        "ci90": [float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))],
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "count": int(len(value)),
    }


def averaged_scope_map(run, conditions, group, name):
    maps = [
        r1.scene_mean_map(
            metric(run, condition, group, name),
            run[f"{condition}__common__scene_token"],
        )
        for condition in conditions
    ]
    keys = sorted(set().union(*(set(value) for value in maps)))
    return {
        key: float(np.mean([value[key] for value in maps if key in value]))
        for key in keys
    }


def equivalence_report(runs, conditions):
    differences, teachers = [], []
    for run in runs:
        teacher = averaged_scope_map(run, conditions, "D1_mc_teacher", "energy")
        student = averaged_scope_map(run, conditions, "D3_risk_distilled", "energy")
        keys = sorted(set(teacher) | set(student))
        differences.append(
            {key: student[key] - teacher[key] for key in keys if key in teacher and key in student}
        )
        teachers.append(teacher)
    difference_values = r1.average_maps(differences)
    teacher_values = r1.average_maps(teachers)
    report = bootstrap_intervals(difference_values)
    teacher_mean = float(teacher_values.mean())
    margin = 0.02 * teacher_mean
    superiority = report["ci95"][1] < 0
    equivalent = report["ci90"][0] > -margin and report["ci90"][1] < margin
    return {
        **report,
        "equivalence_margin": margin,
        "teacher_mean_energy": teacher_mean,
        "superiority": bool(superiority),
        "tost_equivalent": bool(equivalent),
        "pass": bool(superiority or equivalent),
    }


def scene_risk_retention(run, condition):
    scene = np.asarray(run[f"{condition}__common__scene_token"])
    error = metric(run, condition, "D0_deterministic", "mean_fde")
    event = error > 2.0
    risks = {
        group: metric(run, condition, group, "log_area90")
        for group in ("D0_deterministic", "D1_mc_teacher", "D3_risk_distilled")
    }
    result = {
        name: {key: {} for key in ("teacher_gain", "student_gain", "retention_contrast")}
        for name in ("high_fde_auprc", "risk_coverage")
    }
    for scene_token in np.unique(scene):
        selected = scene == scene_token
        current_event = event[selected]
        if len(np.unique(current_event)) == 2:
            scores = {
                group: average_precision_score(current_event, risk[selected])
                for group, risk in risks.items()
            }
            teacher_gain = scores["D1_mc_teacher"] - scores["D0_deterministic"]
            student_gain = scores["D3_risk_distilled"] - scores["D0_deterministic"]
            key = str(scene_token)
            result["high_fde_auprc"]["teacher_gain"][key] = teacher_gain
            result["high_fde_auprc"]["student_gain"][key] = student_gain
            result["high_fde_auprc"]["retention_contrast"][key] = (
                student_gain - 0.9 * teacher_gain
            )
        coverages = {
            group: r0.risk_coverage(risk[selected], error[selected])
            for group, risk in risks.items()
        }
        teacher_gain = coverages["D0_deterministic"] - coverages["D1_mc_teacher"]
        student_gain = coverages["D0_deterministic"] - coverages["D3_risk_distilled"]
        key = str(scene_token)
        result["risk_coverage"]["teacher_gain"][key] = teacher_gain
        result["risk_coverage"]["student_gain"][key] = student_gain
        result["risk_coverage"]["retention_contrast"][key] = (
            student_gain - 0.9 * teacher_gain
        )
    return result


def risk_retention(runs):
    replicate = [scene_risk_retention(run, "clean_s0") for run in runs]
    result = {}
    for name in ("high_fde_auprc", "risk_coverage"):
        result[name] = {
            key: r1.bootstrap(
                r1.average_maps([value[name][key] for value in replicate])
            )
            for key in ("teacher_gain", "student_gain", "retention_contrast")
        }
        result[name]["pass"] = bool(
            result[name]["teacher_gain"]["ci95"][0] > 0
            and result[name]["retention_contrast"]["ci95"][0] > 0
        )
    return result


def median_equal_coverage_area(run, condition, group):
    maha = metric(run, condition, group, "final_maha2")
    determinant = metric(run, condition, group, "final_det")
    return float(np.median(math.pi * np.quantile(maha, 0.90) * np.sqrt(determinant)))


def significant(report):
    return report["ci95"][0] is not None and report["ci95"][0] > 0


def not_reversed(report):
    return report["ci95"][1] is not None and report["ci95"][1] >= 0


def family_rule(values):
    return (
        sum(significant(values[family]) for family in FAMILIES) >= 2
        and all(not_reversed(values[family]) for family in FAMILIES)
    )


def audit_inputs(runs, manifest):
    r0_result = r0.strict_load(R0_RESULT)
    r2_result = r0.strict_load(R2_CORE_RESULT)
    source_valid = (
        r0_result["endpoint"] == "GO TO RISKPROP3D METHOD DEVELOPMENT"
        and r2_result["valid"]
        and r2_result["endpoint"] == "STOP RISKPROP3D-CORE MECHANISM CLAIM"
        and manifest["r0_result_sha256"] == sha256(R0_RESULT)
        and manifest["r2_core_result_sha256"] == sha256(R2_CORE_RESULT)
        and manifest["predictor_sha256"] == sha256(r0.OUT / "predictor.pt")
    )
    source_valid &= all(
        manifest["head_checkpoint_sha256"][str(seed)]
        == sha256(r0.CHECKPOINT_ROOT / f"seed_{seed}.pt")
        for seed in HEAD_SEEDS
    )
    row_hashes = True
    for name, record in manifest["records"].items():
        row_hashes &= (
            record["r0_tracks_sha256"]
            == sha256(r0.OUT / "tracks" / f"{name}.npz")
            and record["r1_scenes_sha256"]
            == sha256(R1_OUT / "scenes" / f"{name}.npz")
        )

    finite = True
    finite_count = 0
    seeds = True
    common = True
    replay = True
    metric_hashes = {}
    for index, run in enumerate(runs):
        metric_path = OUT / "metrics" / f"replicate_{index}.npz"
        metric_hashes[str(index)] = sha256(metric_path)
        seeds &= (
            int(run["meta__head_seed"]) == HEAD_SEEDS[index]
            and int(run["meta__model_seed"]) == MODEL_SEEDS[index]
            and int(run["meta__r0_propagation_seed"])
            == R0_PROPAGATION_SEEDS[index]
            and int(run["meta__student_sample_seed"])
            == STUDENT_SAMPLE_SEEDS[index]
            and int(run["meta__train_teacher_seed"])
            == TRAIN_TEACHER_SEEDS[index]
        )
        replay &= all(
            float(run[f"{condition}__replay__max_abs"]) <= 1e-5
            for condition in CONDITIONS
        )
        for name in run.files:
            value = np.asarray(run[name])
            if np.issubdtype(value.dtype, np.number):
                finite &= bool(np.isfinite(value).all())
                finite_count += int(value.size)
    for condition in CONDITIONS:
        for name in COMMON_FIELDS:
            reference = runs[0][f"{condition}__common__{name}"]
            common &= all(
                np.array_equal(reference, run[f"{condition}__common__{name}"])
                for run in runs[1:]
            )

    checkpoint_valid = True
    checkpoint_hashes = {}
    parameter_counts = {}
    initialization_hashes = {}
    code_hash = sha256(Path(__file__))
    for index in range(3):
        path = OUT / "checkpoints" / f"replicate_{index}.pt"
        checkpoint_hashes[str(index)] = sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint_valid &= (
            checkpoint["head_seed"] == HEAD_SEEDS[index]
            and checkpoint["train_teacher_seed"] == TRAIN_TEACHER_SEEDS[index]
            and checkpoint["model_seed"] == MODEL_SEEDS[index]
            and checkpoint["student_sample_seed"] == STUDENT_SAMPLE_SEEDS[index]
            and checkpoint["prepare_manifest_sha256"]
            == sha256(OUT / "prepare_manifest.json")
            and checkpoint["training_code_sha256"] == code_hash
            and len(set(checkpoint["parameter_counts"].values())) == 1
            and len(set(checkpoint["initialization_hashes"].values())) == 1
        )
        parameter_counts[str(index)] = checkpoint["parameter_counts"]
        initialization_hashes[str(index)] = checkpoint["initialization_hashes"]
        tensors = list(checkpoint["statistics"].values())
        for state in checkpoint["models"].values():
            tensors.extend(state.values())
        for value in tensors:
            finite &= bool(torch.isfinite(value).all())
            finite_count += int(value.numel())

    validity = {
        "V1_R0_and_R2_boundaries_and_sources_frozen": bool(source_valid),
        "V2_R0_R1_rows_and_split_authority_frozen": bool(row_hashes),
        "V3_R0_teacher_replay_exact": bool(replay),
        "V4_D2_D3_capacity_initialization_and_seeds_frozen": bool(
            checkpoint_valid and seeds
        ),
        "V5_common_rows_and_all_numeric_values_finite": bool(
            common and finite and finite_count > 0
        ),
        "V6_forbidden_inputs_and_splits_unread": bool(
            not manifest["corruption_input"]
            and not manifest["current_gt_residual_input"]
            and not manifest["official_val_read"]
            and not manifest["internal_test_read"]
            and not r0_result["official_val_read"]
            and not r0_result["internal_test_read"]
            and not r2_result["official_val_read"]
            and not r2_result["internal_test_read"]
        ),
    }
    evidence = {
        "checkpoints": checkpoint_hashes,
        "code": code_hash,
        "initialization_hashes": initialization_hashes,
        "metrics": metric_hashes,
        "parameter_counts": parameter_counts,
        "prepare_manifest": sha256(OUT / "prepare_manifest.json"),
    }
    return validity, evidence, finite_count


def finalize():
    path = OUT / "result.json"
    if path.exists():
        raise FileExistsError(path)
    manifest = r0.strict_load(OUT / "prepare_manifest.json")
    runs = [
        np.load(OUT / "metrics" / f"replicate_{index}.npz", allow_pickle=False)
        for index in range(3)
    ]
    validity, hashes, finite_count = audit_inputs(runs, manifest)

    fidelity = {
        name: paired(
            runs,
            "clean_s0",
            "D2_gt_only",
            "D3_risk_distilled",
            name,
        )
        for name in FIDELITY
    }
    downstream = {
        "clean": paired(
            runs,
            "clean_s0",
            "D2_gt_only",
            "D3_risk_distilled",
            "energy",
        ),
        "full": {
            family: family_paired(
                runs,
                family,
                "D2_gt_only",
                "D3_risk_distilled",
                "energy",
                (1, 2, 3),
            )
            for family in FAMILIES
        },
        "without_s3": {
            family: family_paired(
                runs,
                family,
                "D2_gt_only",
                "D3_risk_distilled",
                "energy",
                (1, 2),
            )
            for family in FAMILIES
        },
    }
    equivalence = {
        "clean": equivalence_report(runs, ("clean_s0",)),
        "full": {
            family: equivalence_report(
                runs, tuple(f"{family}_s{severity}" for severity in (1, 2, 3))
            )
            for family in FAMILIES
        },
        "without_s3": {
            family: equivalence_report(
                runs, tuple(f"{family}_s{severity}" for severity in (1, 2))
            )
            for family in FAMILIES
        },
    }
    risk = risk_retention(runs)

    condition_sharpness = {}
    for condition in CONDITIONS:
        ratios = []
        areas = {group: [] for group in ("D1_mc_teacher", "D3_risk_distilled")}
        for run in runs:
            current = {
                group: median_equal_coverage_area(run, condition, group)
                for group in areas
            }
            ratios.append(current["D3_risk_distilled"] / current["D1_mc_teacher"])
            for group, value in current.items():
                areas[group].append(value)
        condition_sharpness[condition] = {
            "D3_over_D1": float(np.mean(ratios)),
            "area": {group: float(np.mean(value)) for group, value in areas.items()},
        }
    sharpness = {
        "clean": condition_sharpness["clean_s0"],
        "full": {
            family: {
                "D3_over_D1": float(
                    np.mean(
                        [
                            condition_sharpness[f"{family}_s{severity}"]["D3_over_D1"]
                            for severity in (1, 2, 3)
                        ]
                    )
                )
            }
            for family in FAMILIES
        },
        "without_s3": {
            family: {
                "D3_over_D1": float(
                    np.mean(
                        [
                            condition_sharpness[f"{family}_s{severity}"]["D3_over_D1"]
                            for severity in (1, 2)
                        ]
                    )
                )
            }
            for family in FAMILIES
        },
        "per_condition": condition_sharpness,
    }

    runtime_seconds = {
        group: [
            float(run[f"clean_s0__runtime__{group}__seconds"])
            for run in runs
        ]
        for group in ("D1_mc_teacher", "D3_risk_distilled")
    }
    speedups = [
        teacher / student
        for teacher, student in zip(
            runtime_seconds["D1_mc_teacher"],
            runtime_seconds["D3_risk_distilled"],
        )
    ]
    checkpoint = torch.load(
        OUT / "checkpoints/replicate_0.pt", map_location="cpu", weights_only=True
    )
    student_parameters = int(checkpoint["parameter_counts"]["D3_risk_distilled"])
    r0_parameters = int(sum(value.numel() for value in r0.ForecastMLP().parameters()))
    efficiency = {
        "D1_seconds": runtime_seconds["D1_mc_teacher"],
        "D3_seconds": runtime_seconds["D3_risk_distilled"],
        "parameter_ratio_to_R0_MLP": student_parameters / r0_parameters,
        "r0_mlp_parameters": r0_parameters,
        "speedup_by_replicate": speedups,
        "speedup_mean": float(np.mean(speedups)),
        "student_parameters": student_parameters,
    }

    diagnostics = {
        "clean_proper_scores": {
            baseline: {
                name: paired(
                    runs,
                    "clean_s0",
                    baseline,
                    "D3_risk_distilled",
                    name,
                )
                for name in ("energy", "nll")
            }
            for baseline in ("D0_deterministic", "D1_mc_teacher", "D2_gt_only")
        },
        "accuracy_D3_minus_D0": {
            condition: {
                name: float(
                    np.mean(
                        [
                            (
                                metric(run, condition, "D3_risk_distilled", name)
                                - metric(run, condition, "D0_deterministic", name)
                            ).mean()
                            for run in runs
                        ]
                    )
                )
                for name in ("mean_ade", "mean_fde")
            }
            for condition in CONDITIONS
        },
        "seed_clean_energy_D2_minus_D3": [
            float(
                (
                    metric(run, "clean_s0", "D2_gt_only", "energy")
                    - metric(run, "clean_s0", "D3_risk_distilled", "energy")
                ).mean()
            )
            for run in runs
        ],
    }

    g1 = all(significant(value) for value in fidelity.values())
    g2 = (
        significant(downstream["clean"])
        and family_rule(downstream["full"])
        and family_rule(downstream["without_s3"])
    )
    g3 = equivalence["clean"]["pass"] and all(
        equivalence["full"][family]["pass"] for family in FAMILIES
    )
    g4 = any(value["pass"] for value in risk.values())
    g5 = sharpness["clean"]["D3_over_D1"] <= 1.05 and all(
        sharpness["full"][family]["D3_over_D1"] <= 1.05
        for family in FAMILIES
    )
    g6 = (
        efficiency["speedup_mean"] >= 10.0
        and efficiency["parameter_ratio_to_R0_MLP"] <= 1.10
    )
    gates = {
        "G1_teacher_fidelity": bool(g1),
        "G2_downstream_proper_score": bool(g2),
        "G3_mc_equivalence": bool(g3),
        "G4_risk_retention": bool(g4),
        "G5_sharpness": bool(g5),
        "G6_efficiency": bool(g6),
    }
    valid = all(validity.values())
    passed = valid and all(gates.values())
    endpoint = (
        "INVALID R2-D RUN"
        if not valid
        else (
            "GO TO R3 STRONG-HOST AMORTIZED PROPAGATION"
            if passed
            else "STOP LEARNED AMORTIZED PROPAGATION"
        )
    )
    result = {
        "diagnostics": diagnostics,
        "downstream_energy_D3_over_D2": downstream,
        "efficiency": efficiency,
        "endpoint": endpoint,
        "gates": gates,
        "input_sha256": {
            **hashes,
            "r0_result": sha256(R0_RESULT),
            "r2_core_result": sha256(R2_CORE_RESULT),
        },
        "internal_test_read": False,
        "mc_equivalence_D3_vs_D1": equivalence,
        "numeric_value_count": int(finite_count),
        "official_val_read": False,
        "passed": bool(passed),
        "risk_retention": risk,
        "sharpness": sharpness,
        "teacher_fidelity_D3_over_D2": fidelity,
        "track_counts": {
            condition: int(len(runs[0][f"{condition}__common__scene_token"]))
            for condition in CONDITIONS
        },
        "valid": bool(valid),
        "validity": validity,
    }
    r0.write_json(path, result)
    print(json.dumps({"endpoint": endpoint, "gates": gates, "validity": validity}, indent=2))


def self_check():
    rng = np.random.default_rng(17)
    statistics = {
        "feature_mean": np.zeros(INPUT_DIM + RAW_DIM, dtype=np.float32),
        "feature_scale": np.ones(INPUT_DIM + RAW_DIM, dtype=np.float32),
        "target_mean": np.zeros((HORIZON, 2), dtype=np.float32),
        "target_scale": np.ones((HORIZON, 2), dtype=np.float32),
    }
    first = Student(statistics)
    second = Student(statistics)
    second.load_state_dict(first.state_dict())
    count = int(sum(value.numel() for value in first.parameters()))
    assert count == 24118
    assert tensor_state_hash(first) == tensor_state_hash(second)
    batch = 8
    motion = torch.from_numpy(rng.normal(size=(batch, INPUT_DIM)).astype(np.float32))
    token = torch.from_numpy(rng.normal(size=(batch, RAW_DIM)).astype(np.float32))
    future = torch.from_numpy(rng.normal(size=(batch, HORIZON, 2)).astype(np.float32))
    teacher = torch.from_numpy(
        rng.normal(size=(batch, SAMPLES, HORIZON, 2)).astype(np.float32)
    )
    output = first(motion, token)
    assert output["means"].shape == (batch, MODES, HORIZON, 2)
    assert output["scales"].shape == (batch, MODES, HORIZON, 2)
    assert output["logits"].shape == (batch, MODES)
    current_gt = gt_loss(output, future, 1.0)
    values = distillation_losses(
        output,
        teacher,
        future,
        torch.tensor(0.0),
        torch.tensor(1.0),
        first.target_scale,
    )
    loss = current_gt + values["transport"] + 0.25 * sum(
        values[name] for name in ("moment", "calsharp", "risk")
    )
    assert torch.isfinite(loss)
    loss.backward()
    synthetic_output = {
        "mean_state": rng.normal(size=(batch, 8)).astype(np.float32),
        "scale": np.exp(rng.normal(size=(batch, 8))).astype(np.float32),
        "degrees_of_freedom": (2.1 + rng.random(batch) * 10).astype(np.float32),
        "yaw_mean": rng.uniform(-math.pi, math.pi, batch).astype(np.float32),
        "yaw_kappa": np.exp(rng.normal(size=batch)).astype(np.float32),
    }
    assert raw_token(synthetic_output).shape == (batch, RAW_DIM)
    r0_count = int(sum(value.numel() for value in r0.ForecastMLP().parameters()))
    assert count <= 1.10 * r0_count
    print(
        json.dumps(
            {
                "loss": float(loss.detach()),
                "r0_mlp_parameters": r0_count,
                "self_check": "passed",
                "student_parameters": count,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("self-check")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--replicate", type=int, required=True)
    train_parser.add_argument("--device", type=int, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--replicate", type=int, required=True)
    evaluate_parser.add_argument("--device", type=int, required=True)
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
