#!/usr/bin/env python3
"""RiskProp3D R3-P probability-token necessity audit."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/riskprop3d-r3p-matplotlib")

import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import riskprop3d_r0 as r0
import riskprop3d_r1 as r1
import riskprop3d_r2d as r2d
from probset3d.f1.implementation.probabilistic_query import ProbabilisticQueryHead


OUT = ROOT / "outputs/riskprop3d_r3p"
R0_RESULT = r0.OUT / "result.json"
R2_CORE_RESULT = ROOT / "outputs/riskprop3d_r2_core/result.json"
R2D_RESULT = r2d.OUT / "result.json"
R1_OUT = ROOT / "outputs/riskprop3d_r1"

CONDITIONS = tuple(r0.CONDITIONS)
FAMILIES = ("lowlight", "dropout", "beam")
GROUPS = ("P0_mean_only", "P1_confidence", "P2_gaussian", "P3_m2_probability")
BASELINES = GROUPS[:3]
HEAD_SEEDS = r0.HEAD_SEEDS
MODEL_SEEDS = (20260830, 20260831, 20260832)
SAMPLE_SEEDS = (20260833, 20260834, 20260835)
BOOTSTRAP_SEED = 20260805
TRAIN_EPOCHS = 15
TRAIN_BATCH = 512
EVAL_CHUNK = 2048
SAMPLES = 64
METRICS = r0.METRICS
COMMON_FIELDS = r2d.COMMON_FIELDS
RISK_NAMES = ("high_fde_auprc", "risk_coverage", "false_safe")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_determinism(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.cuda.set_device(device)


def prepare():
    path = OUT / "prepare_manifest.json"
    if path.exists():
        raise FileExistsError(path)
    r0_result = r0.strict_load(R0_RESULT)
    r2_result = r0.strict_load(R2_CORE_RESULT)
    r2d_result = r0.strict_load(R2D_RESULT)
    if r0_result["endpoint"] != "GO TO RISKPROP3D METHOD DEVELOPMENT":
        raise ValueError("R0 propagation authority missing")
    if not r2_result["valid"] or r2_result["endpoint"] != "STOP RISKPROP3D-CORE MECHANISM CLAIM":
        raise ValueError("R2-Core boundary changed")
    if not r2d_result["valid"] or r2d_result["endpoint"] != "STOP LEARNED AMORTIZED PROPAGATION":
        raise ValueError("R2-D boundary changed")

    names = ("train", "label_cal", *CONDITIONS)
    records = {}
    split_scenes = {}
    for name in names:
        tracks = r2d.load_tracks(name)
        scenes = r2d.load_scenes(name)
        r2d.assert_rows(name, tracks, scenes)
        split_scenes[name] = set(np.asarray(tracks["scene_token"]).astype(str).tolist())
        records[name] = {
            "rows": int(len(tracks["input"])),
            "r0_tracks_sha256": sha256(r0.OUT / "tracks" / f"{name}.npz"),
            "r1_scenes_sha256": sha256(R1_OUT / "scenes" / f"{name}.npz"),
        }
    dev = set().union(*(split_scenes[name] for name in CONDITIONS))
    if (
        split_scenes["train"] & split_scenes["label_cal"]
        or split_scenes["train"] & dev
        or split_scenes["label_cal"] & dev
    ):
        raise ValueError("scene split overlap")
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
        "r0_result_sha256": sha256(R0_RESULT),
        "r2_core_result_sha256": sha256(R2_CORE_RESULT),
        "r2d_result_sha256": sha256(R2D_RESULT),
        "records": records,
        "sample_seeds": list(SAMPLE_SEEDS),
    }
    r0.write_json(path, manifest)
    print(json.dumps({"prepared": True, "rows": {k: v["rows"] for k, v in records.items()}}))


def load_heads(replicate, device):
    checkpoint = torch.load(
        r0.CHECKPOINT_ROOT / f"seed_{HEAD_SEEDS[replicate]}.pt",
        map_location="cpu",
        weights_only=True,
    )
    heads = {name: ProbabilisticQueryHead().to(device).eval() for name in ("M1", "M2")}
    for name, head in heads.items():
        head.load_state_dict(checkpoint["models"][name])
    return heads


def target_head_outputs(head, scenes, family, device):
    target = np.asarray(scenes["target_agent_index"], dtype=np.int64)
    names = ["mean_state", "scale", "yaw_mean"]
    names += ["yaw_scale"] if family == "M1" else ["degrees_of_freedom", "yaw_kappa"]
    storage = {name: [] for name in names}
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
        for name in names:
            storage[name].append(output[name][:, 0].detach().cpu().numpy())
    result = {name: np.concatenate(value) for name, value in storage.items()}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"non-finite {family} head output")
    return result


def dev_head_outputs(head, tracks, family, device):
    output = r0.head_outputs(head, tracks, device)
    names = ["mean_state", "scale", "yaw_mean"]
    names += ["yaw_scale"] if family == "M1" else ["degrees_of_freedom", "yaw_kappa"]
    return {name: output[name] for name in names}


def make_tokens(m1, m2, tracks):
    count = len(m2["mean_state"])
    mean = np.asarray(m2["mean_state"], dtype=np.float32)
    yaw = np.asarray(m2["yaw_mean"], dtype=np.float32)
    base = np.zeros((count, r2d.RAW_DIM), dtype=np.float32)
    base[:, :8] = mean
    base[:, 17] = np.sin(yaw)
    base[:, 18] = np.cos(yaw)
    p0 = base.copy()
    p1 = base.copy()
    score = np.clip(np.asarray(tracks["score"], dtype=np.float32), 1e-6, 1 - 1e-6)
    p1[:, 8] = np.log(score / (1.0 - score))
    p1[:, 9] = np.log1p(np.maximum(np.asarray(tracks["density"], dtype=np.float32), 0.0))
    p2 = base.copy()
    p2[:, 8:16] = np.log(np.maximum(np.asarray(m1["scale"], dtype=np.float32), 1e-6))
    yaw_scale = np.maximum(np.asarray(m1["yaw_scale"], dtype=np.float32), 1e-4)
    p2[:, 19] = np.log1p(1.0 / np.square(yaw_scale))
    p3 = r2d.raw_token(m2)
    result = {
        "P0_mean_only": p0,
        "P1_confidence": p1,
        "P2_gaussian": p2,
        "P3_m2_probability": p3,
    }
    if not all(value.shape == (count, r2d.RAW_DIM) and np.isfinite(value).all() for value in result.values()):
        raise ValueError("invalid probability token")
    return result


def copy_parameters(source, target):
    source_values = dict(source.named_parameters())
    with torch.no_grad():
        for name, value in target.named_parameters():
            value.copy_(source_values[name])


def collect(model, motion, token, device):
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
    if manifest["r2d_result_sha256"] != sha256(R2D_RESULT):
        raise ValueError("R2-D result hash changed")
    tracks = r2d.load_tracks("train")
    scenes = r2d.load_scenes("train")
    r2d.assert_rows("train", tracks, scenes)
    heads = load_heads(replicate, device)
    outputs = {
        family: target_head_outputs(heads[family], scenes, family, device)
        for family in ("M1", "M2")
    }
    tokens = make_tokens(outputs["M1"], outputs["M2"], tracks)
    motion = np.asarray(tracks["input"], dtype=np.float32)
    future = np.asarray(tracks["future"], dtype=np.float32)
    statistics = {
        group: r2d.make_statistics(motion, tokens[group], future) for group in GROUPS
    }
    template = r2d.Student(statistics["P3_m2_probability"]).to(device)
    models = {group: r2d.Student(statistics[group]).to(device) for group in GROUPS}
    for model in models.values():
        copy_parameters(template, model)
    parameter_hashes = {group: parameter_hash(model) for group, model in models.items()}
    parameter_counts = {
        group: int(sum(value.numel() for value in model.parameters()))
        for group, model in models.items()
    }
    if len(set(parameter_hashes.values())) != 1 or len(set(parameter_counts.values())) != 1:
        raise ValueError("architecture or initialization mismatch")
    optimizers = {
        group: torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        for group, model in models.items()
    }
    unit = float(np.mean(statistics["P3_m2_probability"]["target_scale"]))
    rng = np.random.default_rng(MODEL_SEEDS[replicate])
    for epoch in range(TRAIN_EPOCHS):
        permutation = rng.permutation(len(motion))
        totals = {group: 0.0 for group in GROUPS}
        seen = 0
        for begin in range(0, len(permutation), TRAIN_BATCH):
            index = permutation[begin : begin + TRAIN_BATCH]
            batch_motion = torch.from_numpy(motion[index]).to(device)
            batch_future = torch.from_numpy(future[index]).to(device)
            for group, model in models.items():
                model.train()
                optimizers[group].zero_grad(set_to_none=True)
                output = model(
                    batch_motion, torch.from_numpy(tokens[group][index]).to(device)
                )
                loss = r2d.gt_loss(output, batch_future, unit)
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite loss: {group}")
                loss.backward()
                optimizers[group].step()
                totals[group] += float(loss.detach()) * len(index)
            seen += len(index)
        print(json.dumps({"epoch": epoch + 1, "loss": {k: v / seen for k, v in totals.items()}, "replicate": replicate}))

    calibration_tracks = r2d.load_tracks("label_cal")
    calibration_scenes = r2d.load_scenes("label_cal")
    r2d.assert_rows("label_cal", calibration_tracks, calibration_scenes)
    calibration_outputs = {
        family: target_head_outputs(
            heads[family], calibration_scenes, family, device
        )
        for family in ("M1", "M2")
    }
    calibration_tokens = make_tokens(
        calibration_outputs["M1"], calibration_outputs["M2"], calibration_tracks
    )
    temperatures = {
        group: r1.fit_temperature(
            collect(
                model,
                np.asarray(calibration_tracks["input"], dtype=np.float32),
                calibration_tokens[group],
                device,
            ),
            calibration_tracks["future"],
        )
        for group, model in models.items()
    }
    checkpoint = {
        "architecture": [r0.INPUT_DIM + r2d.RAW_DIM, 96, 64, 150],
        "epochs": TRAIN_EPOCHS,
        "head_seed": HEAD_SEEDS[replicate],
        "loss": "GT_ONLY",
        "model_seed": MODEL_SEEDS[replicate],
        "models": {group: model.state_dict() for group, model in models.items()},
        "parameter_counts": parameter_counts,
        "parameter_initialization_hashes": parameter_hashes,
        "prepare_manifest_sha256": sha256(OUT / "prepare_manifest.json"),
        "sample_seed": SAMPLE_SEEDS[replicate],
        "statistics": {
            group: {name: torch.as_tensor(value) for name, value in current.items()}
            for group, current in statistics.items()
        },
        "temperatures": temperatures,
        "training_code_sha256": sha256(Path(__file__)),
    }
    r0.write_torch(path, checkpoint)
    print(json.dumps({"trained": True, "replicate": replicate, "parameters": parameter_counts, "temperatures": temperatures}))


def load_models(replicate, device):
    checkpoint = torch.load(
        OUT / "checkpoints" / f"replicate_{replicate}.pt",
        map_location="cpu",
        weights_only=True,
    )
    if (
        checkpoint["head_seed"] != HEAD_SEEDS[replicate]
        or checkpoint["model_seed"] != MODEL_SEEDS[replicate]
        or checkpoint["sample_seed"] != SAMPLE_SEEDS[replicate]
    ):
        raise ValueError("checkpoint seed mismatch")
    models = {}
    for group in GROUPS:
        statistics = {
            name: value.numpy()
            for name, value in checkpoint["statistics"][group].items()
        }
        model = r2d.Student(statistics).to(device).eval()
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
    manifest = r0.strict_load(OUT / "prepare_manifest.json")
    if manifest["r2d_result_sha256"] != sha256(R2D_RESULT):
        raise ValueError("R2-D result hash changed")
    models, checkpoint = load_models(replicate, device)
    heads = load_heads(replicate, device)
    arrays = {
        "meta__head_seed": np.asarray(HEAD_SEEDS[replicate], dtype=np.int64),
        "meta__model_seed": np.asarray(MODEL_SEEDS[replicate], dtype=np.int64),
        "meta__sample_seed": np.asarray(SAMPLE_SEEDS[replicate], dtype=np.int64),
    }
    for condition_index, condition in enumerate(CONDITIONS):
        tracks = r2d.load_tracks(condition)
        scenes = r2d.load_scenes(condition)
        r2d.assert_rows(condition, tracks, scenes)
        outputs = {
            family: dev_head_outputs(heads[family], tracks, family, device)
            for family in ("M1", "M2")
        }
        tokens = make_tokens(outputs["M1"], outputs["M2"], tracks)
        rng = np.random.default_rng(
            SAMPLE_SEEDS[replicate] + condition_index * 1009
        )
        storage = {group: {name: [] for name in METRICS} for group in GROUPS}
        motion = np.asarray(tracks["input"], dtype=np.float32)
        future_all = np.asarray(tracks["future"], dtype=np.float32)
        ego_all = np.asarray(tracks["ego_future"], dtype=np.float32)
        for begin in range(0, len(motion), EVAL_CHUNK):
            stop = min(begin + EVAL_CHUNK, len(motion))
            current_motion = torch.from_numpy(motion[begin:stop]).to(device)
            model_outputs = {}
            with torch.inference_mode():
                for group, model in models.items():
                    model_outputs[group] = r2d.numpy_output(
                        model(
                            current_motion,
                            torch.from_numpy(tokens[group][begin:stop]).to(device),
                        )
                    )
            count = stop - begin
            uniforms = (
                rng.random((count, SAMPLES)),
                rng.random((count, SAMPLES)),
            )
            normals = (
                rng.standard_normal((count, SAMPLES, r0.HORIZON, 2)),
                rng.standard_normal((count, SAMPLES, r0.HORIZON, 2)),
            )
            future = future_all[begin:stop]
            ego = ego_all[begin:stop]
            for group in GROUPS:
                temperature = checkpoint["temperatures"][group]
                sample_a = r1.sample_mixture(
                    model_outputs[group], temperature, uniforms[0], normals[0]
                )
                sample_b = r1.sample_mixture(
                    model_outputs[group], temperature, uniforms[1], normals[1]
                )
                values = r1.trajectory_metrics(
                    model_outputs[group], sample_a, sample_b, future, ego, temperature
                )
                for name in METRICS:
                    storage[group][name].append(
                        np.asarray(values[name], dtype=np.float32)
                    )
        for name in COMMON_FIELDS:
            arrays[f"{condition}__common__{name}"] = np.asarray(tracks[name])
        for group in GROUPS:
            for name in METRICS:
                arrays[f"{condition}__{group}__{name}"] = np.concatenate(
                    storage[group][name]
                )
    r0.write_npz(path, arrays)
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


def paired(runs, condition, baseline, name):
    maps = [
        r1.scene_mean_map(
            metric(run, condition, baseline, name)
            - metric(run, condition, "P3_m2_probability", name),
            run[f"{condition}__common__scene_token"],
        )
        for run in runs
    ]
    return r1.bootstrap(r1.average_maps(maps))


def family_paired(runs, family, baseline, name, severities):
    replicate_maps = []
    for run in runs:
        severity_maps = []
        for severity in severities:
            condition = f"{family}_s{severity}"
            severity_maps.append(
                r1.scene_mean_map(
                    metric(run, condition, baseline, name)
                    - metric(run, condition, "P3_m2_probability", name),
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


def significant(report):
    return report["ci95"][0] is not None and report["ci95"][0] > 0


def not_reversed(report):
    return report["ci95"][1] is not None and report["ci95"][1] >= 0


def family_rule(values):
    return (
        sum(significant(values[family]) for family in FAMILIES) >= 2
        and all(not_reversed(values[family]) for family in FAMILIES)
    )


def scene_risk_maps(run, condition, baseline):
    scene = np.asarray(run[f"{condition}__common__scene_token"])
    error = metric(run, condition, "P0_mean_only", "mean_fde")
    event = error > 2.0
    baseline_risk = metric(run, condition, baseline, "log_area90")
    candidate_risk = metric(run, condition, "P3_m2_probability", "log_area90")
    result = {name: {} for name in RISK_NAMES}
    for scene_token in np.unique(scene):
        selected = scene == scene_token
        if selected.sum() < 2:
            continue
        current_event = event[selected]
        base = baseline_risk[selected]
        candidate = candidate_risk[selected]
        key = str(scene_token)
        if len(np.unique(current_event)) == 2:
            result["high_fde_auprc"][key] = float(
                average_precision_score(current_event, candidate)
                - average_precision_score(current_event, base)
            )
        result["risk_coverage"][key] = float(
            r0.risk_coverage(base, error[selected])
            - r0.risk_coverage(candidate, error[selected])
        )
        if current_event.any():
            base_safe = base <= np.quantile(base, 0.80)
            candidate_safe = candidate <= np.quantile(candidate, 0.80)
            result["false_safe"][key] = float(
                (current_event * base_safe).sum() / current_event.sum()
                - (current_event * candidate_safe).sum() / current_event.sum()
            )
    return result


def risk_difference(runs, baseline):
    replicate = [scene_risk_maps(run, "clean_s0", baseline) for run in runs]
    return {
        name: r1.bootstrap(
            r1.average_maps([current[name] for current in replicate])
        )
        for name in RISK_NAMES
    }


def median_equal_coverage_area(run, condition, group):
    mahalanobis = metric(run, condition, group, "final_maha2")
    determinant = metric(run, condition, group, "final_det")
    return float(
        np.median(
            math.pi * np.quantile(mahalanobis, 0.90) * np.sqrt(determinant)
        )
    )


def sharpness_report(runs):
    per_condition = {}
    for condition in CONDITIONS:
        ratios = []
        areas = {group: [] for group in GROUPS}
        for run in runs:
            current = {
                group: median_equal_coverage_area(run, condition, group)
                for group in GROUPS
            }
            sharpest = min(current[group] for group in BASELINES)
            ratios.append(current["P3_m2_probability"] / sharpest)
            for group, value in current.items():
                areas[group].append(value)
        per_condition[condition] = {
            "P3_over_sharpest_baseline": float(np.mean(ratios)),
            "area": {
                group: float(np.mean(values)) for group, values in areas.items()
            },
        }
    return {
        "clean": per_condition["clean_s0"],
        "full": {
            family: {
                "P3_over_sharpest_baseline": float(
                    np.mean(
                        [
                            per_condition[f"{family}_s{severity}"][
                                "P3_over_sharpest_baseline"
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
                "P3_over_sharpest_baseline": float(
                    np.mean(
                        [
                            per_condition[f"{family}_s{severity}"][
                                "P3_over_sharpest_baseline"
                            ]
                            for severity in (1, 2)
                        ]
                    )
                )
            }
            for family in FAMILIES
        },
        "per_condition": per_condition,
    }


def accuracy_report(runs):
    per_condition = {
        condition: {
            name: float(
                np.mean(
                    [
                        (
                            metric(run, condition, "P3_m2_probability", name)
                            - metric(run, condition, "P0_mean_only", name)
                        ).mean()
                        for run in runs
                    ]
                )
            )
            for name in ("mean_ade", "mean_fde")
        }
        for condition in CONDITIONS
    }
    return {
        "clean": per_condition["clean_s0"],
        "full": {
            family: {
                name: float(
                    np.mean(
                        [
                            per_condition[f"{family}_s{severity}"][name]
                            for severity in (1, 2, 3)
                        ]
                    )
                )
                for name in ("mean_ade", "mean_fde")
            }
            for family in FAMILIES
        },
        "per_condition": per_condition,
    }


def audit_inputs(runs, manifest):
    r0_result = r0.strict_load(R0_RESULT)
    r2_result = r0.strict_load(R2_CORE_RESULT)
    r2d_result = r0.strict_load(R2D_RESULT)
    source_valid = (
        r0_result["endpoint"] == "GO TO RISKPROP3D METHOD DEVELOPMENT"
        and r2_result["valid"]
        and r2_result["endpoint"] == "STOP RISKPROP3D-CORE MECHANISM CLAIM"
        and r2d_result["valid"]
        and r2d_result["endpoint"] == "STOP LEARNED AMORTIZED PROPAGATION"
        and manifest["r0_result_sha256"] == sha256(R0_RESULT)
        and manifest["r2_core_result_sha256"] == sha256(R2_CORE_RESULT)
        and manifest["r2d_result_sha256"] == sha256(R2D_RESULT)
    )
    source_valid &= all(
        manifest["head_checkpoint_sha256"][str(seed)]
        == sha256(r0.CHECKPOINT_ROOT / f"seed_{seed}.pt")
        for seed in HEAD_SEEDS
    )
    row_hashes = all(
        record["r0_tracks_sha256"]
        == sha256(r0.OUT / "tracks" / f"{name}.npz")
        and record["r1_scenes_sha256"]
        == sha256(R1_OUT / "scenes" / f"{name}.npz")
        for name, record in manifest["records"].items()
    )
    finite = True
    finite_count = 0
    seeds = True
    common = True
    metric_hashes = {}
    for index, run in enumerate(runs):
        metric_path = OUT / "metrics" / f"replicate_{index}.npz"
        metric_hashes[str(index)] = sha256(metric_path)
        seeds &= (
            int(run["meta__head_seed"]) == HEAD_SEEDS[index]
            and int(run["meta__model_seed"]) == MODEL_SEEDS[index]
            and int(run["meta__sample_seed"]) == SAMPLE_SEEDS[index]
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
        checkpoint_path = OUT / "checkpoints" / f"replicate_{index}.pt"
        checkpoint_hashes[str(index)] = sha256(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        counts = checkpoint["parameter_counts"]
        hashes = checkpoint["parameter_initialization_hashes"]
        checkpoint_valid &= (
            checkpoint["head_seed"] == HEAD_SEEDS[index]
            and checkpoint["model_seed"] == MODEL_SEEDS[index]
            and checkpoint["sample_seed"] == SAMPLE_SEEDS[index]
            and checkpoint["loss"] == "GT_ONLY"
            and checkpoint["epochs"] == TRAIN_EPOCHS
            and checkpoint["prepare_manifest_sha256"]
            == sha256(OUT / "prepare_manifest.json")
            and checkpoint["training_code_sha256"] == code_hash
            and set(counts) == set(GROUPS)
            and set(hashes) == set(GROUPS)
            and len(set(counts.values())) == 1
            and len(set(hashes.values())) == 1
        )
        parameter_counts[str(index)] = counts
        initialization_hashes[str(index)] = hashes
        tensors = []
        for current in checkpoint["statistics"].values():
            tensors.extend(current.values())
        for state in checkpoint["models"].values():
            tensors.extend(state.values())
        for value in tensors:
            finite &= bool(torch.isfinite(value).all())
            finite_count += int(value.numel())

    validity = {
        "V1_source_boundaries_and_hashes_frozen": bool(source_valid),
        "V2_rows_and_split_authority_frozen": bool(row_hashes),
        "V3_architecture_parameter_count_and_initialization_matched": bool(
            checkpoint_valid
        ),
        "V4_GT_only_training_contract_frozen": bool(checkpoint_valid),
        "V5_preregistered_seeds_and_common_rows_present": bool(seeds and common),
        "V6_all_retained_numeric_values_finite": bool(
            finite and finite_count > 0
        ),
        "V7_forbidden_model_inputs_absent": bool(
            not manifest["corruption_input"]
            and not manifest["current_gt_residual_input"]
        ),
        "V8_official_validation_and_internal_test_unread": bool(
            not manifest["official_val_read"]
            and not manifest["internal_test_read"]
            and not r0_result["official_val_read"]
            and not r0_result["internal_test_read"]
            and not r2_result["official_val_read"]
            and not r2_result["internal_test_read"]
            and not r2d_result["official_val_read"]
            and not r2d_result["internal_test_read"]
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

    energy = {
        baseline: {
            "clean": paired(runs, "clean_s0", baseline, "energy"),
            "full": {
                family: family_paired(
                    runs, family, baseline, "energy", (1, 2, 3)
                )
                for family in FAMILIES
            },
            "without_s3": {
                family: family_paired(
                    runs, family, baseline, "energy", (1, 2)
                )
                for family in FAMILIES
            },
        }
        for baseline in BASELINES
    }
    risk = {
        baseline: risk_difference(runs, baseline)
        for baseline in ("P0_mean_only", "P1_confidence")
    }
    sharpness = sharpness_report(runs)
    accuracy = accuracy_report(runs)
    seed_directions = {
        baseline: [
            float(
                (
                    metric(run, "clean_s0", baseline, "energy")
                    - metric(run, "clean_s0", "P3_m2_probability", "energy")
                ).mean()
            )
            for run in runs
        ]
        for baseline in BASELINES
    }
    diagnostics = {
        "clean_scores": {
            baseline: {
                name: paired(runs, "clean_s0", baseline, name)
                for name in ("energy", "nll")
            }
            for baseline in BASELINES
        },
        "transfer_energy": energy,
        "risk": risk,
    }

    g1 = (
        significant(energy["P0_mean_only"]["clean"])
        and family_rule(energy["P0_mean_only"]["full"])
        and family_rule(energy["P0_mean_only"]["without_s3"])
    )
    g2 = significant(energy["P1_confidence"]["clean"])
    g3 = significant(energy["P2_gaussian"]["clean"])
    passing_risk_metrics = [
        name
        for name in RISK_NAMES
        if significant(risk["P0_mean_only"][name])
        and significant(risk["P1_confidence"][name])
    ]
    g4 = bool(passing_risk_metrics)
    g5 = (
        sharpness["clean"]["P3_over_sharpest_baseline"] <= 1.05
        and all(
            sharpness["full"][family]["P3_over_sharpest_baseline"] <= 1.05
            for family in FAMILIES
        )
        and accuracy["clean"]["mean_ade"] <= 0.05
        and accuracy["clean"]["mean_fde"] <= 0.10
        and all(
            accuracy["full"][family]["mean_ade"] <= 0.05
            and accuracy["full"][family]["mean_fde"] <= 0.10
            for family in FAMILIES
        )
    )
    g6 = all(
        all(value > 0 for value in seed_directions[baseline])
        for baseline in BASELINES
    )
    gates = {
        "G1_mean_only_necessity": bool(g1),
        "G2_beyond_confidence": bool(g2),
        "G3_M2_necessity": bool(g3),
        "G4_risk_increment": bool(g4),
        "G5_sharpness_and_accuracy": bool(g5),
        "G6_seed_stability": bool(g6),
    }
    valid = all(validity.values())
    passed = valid and all(gates.values())
    endpoint = (
        "INVALID R3-P RUN"
        if not valid
        else (
            "PERCEPTION-DISTRIBUTION CONDITIONING SUPPORTED"
            if passed
            else "STOP RISKPROP3D DIRECTION"
        )
    )
    result = {
        "accuracy_P3_minus_P0": accuracy,
        "diagnostics": diagnostics,
        "endpoint": endpoint,
        "energy_P3_over_baselines": energy,
        "gates": gates,
        "input_sha256": {
            **hashes,
            "r0_result": sha256(R0_RESULT),
            "r2_core_result": sha256(R2_CORE_RESULT),
            "r2d_result": sha256(R2D_RESULT),
        },
        "internal_test_read": False,
        "numeric_value_count": int(finite_count),
        "official_val_read": False,
        "passed": bool(passed),
        "risk_P3_over_baselines": risk,
        "risk_metrics_passing_both_baselines": passing_risk_metrics,
        "seed_clean_energy_directions": seed_directions,
        "sharpness": sharpness,
        "track_counts": {
            condition: int(len(runs[0][f"{condition}__common__scene_token"]))
            for condition in CONDITIONS
        },
        "valid": bool(valid),
        "validity": validity,
    }
    r0.write_json(path, result)
    for run in runs:
        run.close()
    print(
        json.dumps(
            {"endpoint": endpoint, "gates": gates, "validity": validity},
            indent=2,
        )
    )


def self_check():
    rng = np.random.default_rng(23)
    batch = 8
    motion = rng.normal(size=(batch, r0.INPUT_DIM)).astype(np.float32)
    future = rng.normal(size=(batch, r0.HORIZON, 2)).astype(np.float32)
    tracks = {
        "score": rng.uniform(0.05, 0.95, batch).astype(np.float32),
        "density": rng.integers(1, 30, batch).astype(np.float32),
    }
    mean = rng.normal(size=(batch, 8)).astype(np.float32)
    yaw = rng.uniform(-math.pi, math.pi, batch).astype(np.float32)
    m1 = {
        "mean_state": mean.copy(),
        "scale": np.exp(rng.normal(size=(batch, 8))).astype(np.float32),
        "yaw_mean": yaw.copy(),
        "yaw_scale": np.exp(rng.normal(size=batch)).astype(np.float32),
    }
    m2 = {
        "mean_state": mean.copy(),
        "scale": np.exp(rng.normal(size=(batch, 8))).astype(np.float32),
        "degrees_of_freedom": (2.1 + 10.0 * rng.random(batch)).astype(np.float32),
        "yaw_mean": yaw.copy(),
        "yaw_kappa": np.exp(rng.normal(size=batch)).astype(np.float32),
    }
    tokens = make_tokens(m1, m2, tracks)
    assert all(value.shape == (batch, r2d.RAW_DIM) for value in tokens.values())
    assert all(
        not np.array_equal(tokens[left], tokens[right])
        for index, left in enumerate(GROUPS)
        for right in GROUPS[index + 1 :]
    )
    statistics = {
        group: r2d.make_statistics(motion, tokens[group], future)
        for group in GROUPS
    }
    torch.manual_seed(29)
    template = r2d.Student(statistics["P3_m2_probability"])
    models = {group: r2d.Student(statistics[group]) for group in GROUPS}
    for model in models.values():
        copy_parameters(template, model)
    counts = {
        group: int(sum(value.numel() for value in model.parameters()))
        for group, model in models.items()
    }
    hashes = {group: parameter_hash(model) for group, model in models.items()}
    assert set(counts.values()) == {24118}
    assert len(set(hashes.values())) == 1
    losses = {}
    for group, model in models.items():
        output = model(torch.from_numpy(motion), torch.from_numpy(tokens[group]))
        loss = r2d.gt_loss(output, torch.from_numpy(future), 1.0)
        assert torch.isfinite(loss)
        loss.backward()
        losses[group] = float(loss.detach())
    print(
        json.dumps(
            {
                "groups": list(GROUPS),
                "losses": losses,
                "parameters": counts,
                "self_check": "passed",
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
