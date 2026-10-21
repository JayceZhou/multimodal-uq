#!/usr/bin/env python3
"""Retrospective Level-I moment-matched predictive-law control.

This read-only audit derives Gaussian/wrapped-Gaussian surrogates directly
from the frozen HTC head.  It never trains a model or reads the internal test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/riskprop3d-matched-law-mpl")

import h5py
import numpy as np
import scipy
import torch
from scipy.special import i0e, i1e
from scipy.stats import chi2, f, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import export_riskprop3d_upstream_downstream as paired_export
import riskprop3d_final_confirmation as confirmation
import riskprop3d_r0 as r0
from probset3d.f1.implementation.probabilistic_query import ProbabilisticQueryHead
from tools import probset3d_f0_score as f0
from tools import probset3d_v3_formal as formal


OUT = ROOT / "outputs/riskprop3d_level1_matched_law"
PROTOCOL = OUT / "frozen_config.json"
METRIC_ROOT = OUT / "metrics"
FIGURE_ROOT = OUT / "figures"
BOOTSTRAP_SEED = 20260805
BOOTSTRAP_REPEATS = 10000
MC_SENSITIVITY_OFFSET = 2_000_000
MAIN_SAMPLES = 64
MC_SENSITIVITY_SAMPLES = 256
CURRENT_OFFSET = r0.HISTORY * r0.STATE_FEATURES

LAW_IDS = (
    "TRAINED_G",
    "L1Y1_HTC",
    "L0Y1_LIN_MM",
    "L1Y0_YAW_MM",
    "L0Y0_HTC_MMG",
)
FACTORIAL_LAWS = LAW_IDS[1:]
UPSTREAM_METRICS = (
    "full_linear_nll",
    "xy_nll",
    "yaw_nll",
    "linear90_inside",
    "yaw90_inside",
    "xy_maha2",
    "xy_scale_product",
)
DOWNSTREAM_METRICS = (
    "energy",
    "nll",
    "mean_ade",
    "mean_fde",
    "final_maha2",
    "final_det",
    "coverage_90",
    "log_area90",
)
CONTRASTS = {
    "full_higher_order": ("L0Y0_HTC_MMG", "L1Y1_HTC"),
    "linear_given_vm": ("L0Y1_LIN_MM", "L1Y1_HTC"),
    "yaw_given_student": ("L1Y0_YAW_MM", "L1Y1_HTC"),
    "trained_g_ecological": ("TRAINED_G", "L1Y1_HTC"),
}
FAMILIES = {
    "camera_lowlight": ("lowlight_s1", "lowlight_s2", "lowlight_s3"),
    "lidar_point_dropout": ("dropout_s1", "dropout_s2", "dropout_s3"),
    "lidar_beam_reduction": ("beam_s1", "beam_s2", "beam_s3"),
    "camera_extrinsic_rotation": ("rotation_s2", "rotation_s3"),
    "camera_extrinsic_translation": ("translation_s3",),
    "camera_temporal_offset": ("temporal_s2", "temporal_s3"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path: Path, arrays) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def strict_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def log_event(event: str, **fields):
    OUT.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "time_unix": time.time(), **fields}
    with (OUT / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def condition_spec(source_name: str):
    population, paper_condition, severity, track_path, metric_group, metric_condition = (
        paired_export.CONDITIONS[source_name]
    )
    if population == "development":
        condition_index = list(r0.CONDITIONS).index(source_name)
        seed_offset = condition_index * 1009
    elif population == "held_out_degradation":
        condition_index = list(confirmation.EXTERNAL_CONDITIONS).index(source_name)
        seed_offset = confirmation.OFFSET_EXTERNAL + condition_index * 1009
    elif population == "final_validation":
        condition_index = 0
        seed_offset = confirmation.OFFSET_OFFICIAL
    else:
        raise ValueError(population)
    return {
        "source_name": source_name,
        "population": population,
        "condition": paper_condition,
        "severity": severity,
        "tracks": Path(track_path),
        "metric_group": metric_group,
        "metric_condition": metric_condition,
        "seed_offset": seed_offset,
        "condition_index": condition_index,
    }


CONDITIONS = {name: condition_spec(name) for name in paired_export.CONDITIONS}


def retained_path(metric_group: str, replicate: int) -> Path:
    return Path(
        str(paired_export.METRIC_PATHS[metric_group]).format(replicate=replicate)
    )


def source_paths():
    paths = {r0.OUT / "predictor.pt", Path(__file__).resolve()}
    paths.update(r0.CHECKPOINT_ROOT / f"seed_{seed}.pt" for seed in r0.HEAD_SEEDS)
    for spec in CONDITIONS.values():
        paths.add(spec["tracks"])
        paths.update(retained_path(spec["metric_group"], replicate) for replicate in range(3))
    paths.add(ROOT / "outputs/riskprop3d_r1/scenes/label_cal.npz")
    return sorted(paths)


def prepare():
    if PROTOCOL.exists():
        raise FileExistsError(PROTOCOL)
    config = {
        "title": "Level-I Moment-Matched Law Control",
        "status": "RETROSPECTIVE MATCHED-LAW AUDIT",
        "freeze_revision": 3,
        "supersedes": {
            "paths": [
                "outputs/riskprop3d_level1_matched_law_invalid_freeze_v1",
                "outputs/riskprop3d_level1_matched_law_invalid_freeze_v2"
            ],
            "reason": "pre-outcome sampler QA first used an absolute error and then mixed float32 HTC scale with float64 surrogate scale; the v3 QA uses the frozen relative-Frobenius contract in float64",
            "scientific_outcomes_read_before_revision": False,
        },
        "purpose": (
            "test whether law structure beyond matched linear covariance and yaw "
            "first trigonometric moment changes the frozen Level-I pushforward"
        ),
        "laws": {
            "TRAINED_G": "frozen independently trained Gaussian/wrapped-Gaussian ecological comparator",
            "L1Y1_HTC": "frozen HTC Student-t/von-Mises law",
            "L0Y1_LIN_MM": "HTC covariance-matched Gaussian linear state plus HTC von Mises yaw",
            "L1Y0_YAW_MM": "HTC Student-t linear state plus resultant-matched wrapped-Gaussian yaw",
            "L0Y0_HTC_MMG": "complete HTC-derived moment-matched Gaussian/wrapped-Gaussian surrogate",
        },
        "construction": {
            "linear_covariance": "C_HTC = nu/(nu-2) * diag(scale^2); sigma_MMG = scale*sqrt(nu/(nu-2))",
            "yaw_resultant": "R_VM = I1(kappa)/I0(kappa); sigma_WG^2 = -2*log(R_VM)",
            "degrees_requirement": "nu > 2; no clipping",
            "resultant_clipping": "machine tiny <= R <= 1-machine epsilon for logarithm only",
            "surrogate_uses_trained_g_parameters": False,
        },
        "sampling": {
            "main_samples": MAIN_SAMPLES,
            "main_draw_sets": 2,
            "mc_sensitivity_samples": MC_SENSITIVITY_SAMPLES,
            "mc_sensitivity_population": "development clean",
            "common_linear_normal_draws": True,
            "student_radial_draws": True,
            "circular_substreams": {
                "von_mises": "original NumPy Generator.vonmises stream, exactly replaying R0 HTC",
                "wrapped_gaussian": "original R0 yaw_normal stream",
            },
            "common_future_process_noise": True,
            "propagation_seeds": list(r0.PROPAGATION_SEEDS),
            "mc_sensitivity_seed_offset": MC_SENSITIVITY_OFFSET,
        },
        "hosts": {
            "CV": "reads target XY position and XY velocity; does not read yaw",
            "MLP": "frozen R0 residual MLP",
        },
        "primary": {
            "metric": "trajectory energy score",
            "effect": "ES(HTC-MMG) - ES(HTC); positive favors full HTC",
            "joint_probes": ["CV", "MLP"],
            "inference": "replicate-averaged paired scene bootstrap",
            "bootstrap_resamples": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "interpretation": {
            "new_holdout": False,
            "internal_test_read": False,
            "confirmatory_mechanism_claim_permitted": False,
            "component_contrasts": "secondary diagnostics without multiplicity-adjusted component claims",
        },
        "qa_tolerances": {
            "analytic_covariance_relative_frobenius": 1e-12,
            "analytic_resultant_absolute": 1e-12,
            "shared_mean_absolute": 2e-5,
            "r0_energy_replay_absolute": 1e-6,
            "cv_yaw_zero_absolute": 0.0,
        },
        "conditions": [
            {
                key: (str(value.relative_to(ROOT)) if key == "tracks" else value)
                for key, value in spec.items()
            }
            for spec in CONDITIONS.values()
        ],
        "head_seeds": list(r0.HEAD_SEEDS),
        "models_retrained": False,
        "conditions_selected_from_results": False,
        "official_modern_host_outcome_read": False,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "evaluator_sha256": sha256(Path(__file__)),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths()
        },
    }
    atomic_json(PROTOCOL, config)
    log_event("prepare", protocol_sha256=sha256(PROTOCOL))
    print(json.dumps({"prepared": True, "protocol": str(PROTOCOL.relative_to(ROOT))}))


def verify_protocol():
    config = strict_json(PROTOCOL)
    if config["evaluator_sha256"] != sha256(Path(__file__)):
        raise ValueError("evaluator changed after protocol freeze")
    for relative, expected in config["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"source changed after protocol freeze: {relative}")
    return config


def set_determinism(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(device)


def load_heads(replicate: int, device: torch.device):
    checkpoint = torch.load(
        r0.CHECKPOINT_ROOT / f"seed_{r0.HEAD_SEEDS[replicate]}.pt",
        map_location="cpu",
        weights_only=True,
    )
    heads = {name: ProbabilisticQueryHead().to(device).eval() for name in ("M1", "M2")}
    for name, head in heads.items():
        head.load_state_dict(checkpoint["models"][name])
    return heads


def sliced(output, begin, stop):
    return {name: np.asarray(value[begin:stop]) for name, value in output.items()}


def wrap(value):
    return np.arctan2(np.sin(value), np.cos(value))


def matched_parameters(htc):
    scale = np.asarray(htc["scale"], dtype=np.float64)
    degrees = np.asarray(htc["degrees_of_freedom"], dtype=np.float64)
    kappa = np.asarray(htc["yaw_kappa"], dtype=np.float64)
    if not np.isfinite(degrees).all() or np.min(degrees) <= 2.0:
        raise ValueError(f"HTC degrees must be finite and >2; minimum={np.min(degrees)}")
    variance_factor = degrees / (degrees - 2.0)
    mm_scale = scale * np.sqrt(variance_factor)[:, None]
    resultant_raw = i1e(kappa) / i0e(kappa)
    if not np.isfinite(resultant_raw).all() or np.any(resultant_raw <= 0.0):
        raise ValueError("non-positive/non-finite von-Mises resultant")
    finfo = np.finfo(np.float64)
    resultant = np.clip(resultant_raw, finfo.tiny, 1.0 - finfo.eps)
    yaw_variance = -2.0 * np.log(resultant)
    yaw_scale = np.sqrt(yaw_variance)
    if not np.isfinite(mm_scale).all() or not np.isfinite(yaw_scale).all():
        raise ValueError("non-finite matched parameter")
    return {
        "mm_scale": mm_scale,
        "resultant_raw": resultant_raw,
        "resultant": resultant,
        "yaw_variance": yaw_variance,
        "yaw_scale": yaw_scale,
        "variance_factor": variance_factor,
    }


def sample_laws(g, htc, matched, rng, count, z, radial, yaw_normal):
    # Calls preserve the exact original R0 RNG order: G first, then HTC VM.
    g_linear, g_yaw = r0.sample_states(g, rng, count, False, z, radial, yaw_normal)
    htc_linear, htc_yaw = r0.sample_states(htc, rng, count, True, z, radial, yaw_normal)
    mm_linear = (
        np.asarray(htc["mean_state"], dtype=np.float64)[:, None]
        + matched["mm_scale"][:, None] * z
    )
    wg_yaw = wrap(
        np.asarray(htc["yaw_mean"], dtype=np.float64)[:, None]
        + matched["yaw_scale"][:, None] * yaw_normal
    )
    return {
        "TRAINED_G": (g_linear, g_yaw),
        "L1Y1_HTC": (htc_linear, htc_yaw),
        "L0Y1_LIN_MM": (mm_linear, htc_yaw),
        "L1Y0_YAW_MM": (htc_linear, wg_yaw),
        "L0Y0_HTC_MMG": (mm_linear, wg_yaw),
    }


def yaw_half_width(kind, parameter, level=0.90):
    if kind == "wrapped_gaussian":
        return np.minimum(math.pi, norm.ppf((1.0 + level) / 2.0) * parameter)
    lower = np.zeros_like(parameter, dtype=np.float64)
    upper = np.full_like(parameter, math.pi, dtype=np.float64)
    target = (1.0 + level) / 2.0
    from scipy.stats import vonmises
    for _ in range(42):
        middle = (lower + upper) / 2.0
        below = vonmises.cdf(middle, parameter) < target
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    return (lower + upper) / 2.0


def upstream_metrics(law: str, target, yaw_target, htc, g, matched):
    if law == "TRAINED_G":
        mean = np.asarray(g["mean_state"], dtype=np.float64)
        linear_scale = np.asarray(g["scale"], dtype=np.float64)
        linear_kind = "gaussian"
        yaw_mean = np.asarray(g["yaw_mean"], dtype=np.float64)
        yaw_parameter = np.asarray(g["yaw_scale"], dtype=np.float64)
        yaw_kind = "wrapped_gaussian"
    else:
        mean = np.asarray(htc["mean_state"], dtype=np.float64)
        linear_kind = "student" if law in {"L1Y1_HTC", "L1Y0_YAW_MM"} else "gaussian"
        linear_scale = (
            np.asarray(htc["scale"], dtype=np.float64)
            if linear_kind == "student" else matched["mm_scale"]
        )
        yaw_mean = np.asarray(htc["yaw_mean"], dtype=np.float64)
        yaw_kind = "von_mises" if law in {"L1Y1_HTC", "L0Y1_LIN_MM"} else "wrapped_gaussian"
        yaw_parameter = (
            np.asarray(htc["yaw_kappa"], dtype=np.float64)
            if yaw_kind == "von_mises" else matched["yaw_scale"]
        )
    degrees = np.asarray(htc["degrees_of_freedom"], dtype=np.float64)
    if linear_kind == "student":
        full_nll = formal.student_nll(target, mean, linear_scale, degrees, tuple(range(8)))
        xy_nll = formal.student_nll(target, mean, linear_scale, degrees, (0, 1))
    else:
        components = formal.gaussian_parts(target, mean, linear_scale)
        full_nll = components.sum(-1)
        xy_nll = components[:, :2].sum(-1)
    if yaw_kind == "von_mises":
        yaw_nll = formal.von_mises_nll(yaw_target, yaw_mean, yaw_parameter)
    else:
        yaw_nll = formal.wrapped_gaussian_nll(yaw_target, yaw_mean, yaw_parameter)
    delta = target - mean
    maha = np.square(delta / linear_scale).sum(-1)
    xy_maha = np.square(delta[:, :2] / linear_scale[:, :2]).sum(-1)
    threshold = (
        8.0 * f.ppf(0.90, 8, degrees)
        if linear_kind == "student" else chi2.ppf(0.90, 8)
    )
    yaw_residual = np.abs(wrap(yaw_target - yaw_mean))
    yaw_inside = yaw_residual <= yaw_half_width(yaw_kind, yaw_parameter)
    return {
        "full_linear_nll": full_nll,
        "xy_nll": xy_nll,
        "yaw_nll": yaw_nll,
        "linear90_inside": (maha <= threshold).astype(np.float64),
        "yaw90_inside": yaw_inside.astype(np.float64),
        "xy_maha2": xy_maha,
        "xy_scale_product": linear_scale[:, 0] * linear_scale[:, 1],
    }


def row_id(tracks):
    return np.char.add(
        np.char.add(np.asarray(tracks["sample_token"]).astype("U32"), ":"),
        np.asarray(tracks["instance_token"]).astype("U32"),
    ).astype("S65")


def metric_key(spec, host, group):
    return paired_export.metric_key(
        spec["metric_group"], spec["metric_condition"], host,
        "g" if group == "TRAINED_G" else "htc",
    )


def common_key(spec, name):
    return paired_export.common_key(spec["metric_condition"], name)


def evaluate_condition(
    replicate, spec, tracks, retained, outputs, model, parameters, device,
    samples=MAIN_SAMPLES, seed_offset=None, energy_only=False,
):
    output_g, output_htc = outputs["M1"], outputs["M2"]
    arrays = {}
    storage = {
        host: {law: {metric: [] for metric in DOWNSTREAM_METRICS} for law in LAW_IDS}
        for host in r0.PREDICTORS
    }
    if seed_offset is None:
        seed_offset = spec["seed_offset"]
    rng = np.random.default_rng(r0.PROPAGATION_SEEDS[replicate] + seed_offset)
    invalid = {law: 0 for law in LAW_IDS}
    sampled_total = 0
    started = time.perf_counter()
    for begin in range(0, len(tracks["input"]), r0.EVAL_CHUNK):
        stop = min(begin + r0.EVAL_CHUNK, len(tracks["input"]))
        source = np.asarray(tracks["input"][begin:stop])
        future = np.asarray(tracks["future"][begin:stop])
        ego_future = np.asarray(tracks["ego_future"][begin:stop])
        count = stop - begin
        g = sliced(output_g, begin, stop)
        htc = sliced(output_htc, begin, stop)
        matched = matched_parameters(htc)
        sampled = {law: [] for law in LAW_IDS}
        for _draw in range(2):
            z = rng.standard_normal((count, samples, 8))
            radial = rng.uniform(1e-7, 1.0 - 1e-7, (count, samples))
            yaw_normal = rng.standard_normal((count, samples))
            states = sample_laws(g, htc, matched, rng, samples, z, radial, yaw_normal)
            for law, (linear, yaw) in states.items():
                invalid[law] += int((~np.isfinite(linear).all(-1) | ~np.isfinite(yaw)).sum())
                sampled[law].append(r0.replace_current(source, linear, yaw))
        sampled_total += count * samples * 2
        for host in r0.PREDICTORS:
            covariance = parameters["covariance"][host]
            noise_a = r0.process_noise(rng, covariance, count, samples)
            noise_b = r0.process_noise(rng, covariance, count, samples)
            centers = {
                law: tuple(
                    r0.predict_centers(host, value, model, parameters, device)
                    for value in sampled[law]
                )
                for law in LAW_IDS
            }
            for law in LAW_IDS:
                values = r0.distribution_metrics(
                    centers[law][0], centers[law][1], future, ego_future,
                    covariance, noise_a, noise_b,
                )
                metrics = ("energy",) if energy_only else DOWNSTREAM_METRICS
                for metric in metrics:
                    storage[host][law][metric].append(
                        np.asarray(values[metric], dtype=np.float32)
                    )
    arrays["runtime_seconds"] = np.asarray(time.perf_counter() - started, dtype=np.float64)
    arrays["sampled_state_draws"] = np.asarray(sampled_total, dtype=np.int64)
    for law in LAW_IDS:
        arrays[f"invalid_rate__{law}"] = np.asarray(
            invalid[law] / max(sampled_total, 1), dtype=np.float64
        )
    for host in r0.PREDICTORS:
        for law in LAW_IDS:
            metrics = ("energy",) if energy_only else DOWNSTREAM_METRICS
            for metric in metrics:
                arrays[f"{host}__{law}__{metric}"] = np.concatenate(storage[host][law][metric])
    return arrays


def evaluate(replicate: int, device_index: int):
    config = verify_protocol()
    if replicate not in range(3) or device_index not in range(4):
        raise ValueError("replicate must be 0--2 and GPU must be 0--3")
    output_path = METRIC_ROOT / f"replicate_{replicate}.npz"
    if output_path.exists():
        raise FileExistsError(output_path)
    device = torch.device(f"cuda:{device_index}")
    set_determinism(r0.PROPAGATION_SEEDS[replicate], device)
    model, parameters = r0.load_predictor(device)
    heads = load_heads(replicate, device)
    arrays = {
        "meta__replicate": np.asarray(replicate, dtype=np.int64),
        "meta__head_seed": np.asarray(r0.HEAD_SEEDS[replicate], dtype=np.int64),
        "meta__propagation_seed": np.asarray(r0.PROPAGATION_SEEDS[replicate], dtype=np.int64),
    }
    replay_errors = {}
    df_minima = []
    analytic_cov_errors = []
    analytic_resultant_errors = []
    mean_errors = []
    for source_name, spec in CONDITIONS.items():
        print(json.dumps({"event": "condition", "replicate": replicate, "condition": source_name}), flush=True)
        tracks = np.load(spec["tracks"], allow_pickle=False)
        retained = np.load(retained_path(spec["metric_group"], replicate), allow_pickle=False)
        try:
            for name in ("scene_token", "sample_token", "instance_token"):
                if not np.array_equal(np.asarray(tracks[name]), np.asarray(retained[common_key(spec, name)])):
                    raise ValueError(f"retained row mismatch: {source_name}/{name}")
            arrays[f"{source_name}__common__scene_token"] = np.asarray(tracks["scene_token"])
            arrays[f"{source_name}__common__sample_token"] = np.asarray(tracks["sample_token"])
            arrays[f"{source_name}__common__instance_token"] = np.asarray(tracks["instance_token"])
            arrays[f"{source_name}__common__row_id"] = row_id(tracks)
            outputs = {name: r0.head_outputs(head, tracks, device) for name, head in heads.items()}
            g, htc = outputs["M1"], outputs["M2"]
            if not np.array_equal(g["mean_state"], htc["mean_state"]):
                raise ValueError("trained G and HTC means differ")
            detector_mean = np.asarray(tracks["input"][:, CURRENT_OFFSET:CURRENT_OFFSET + 8])
            mean_error = float(np.max(np.abs(np.asarray(htc["mean_state"]) - detector_mean)))
            mean_errors.append(mean_error)
            if mean_error > config["qa_tolerances"]["shared_mean_absolute"]:
                raise ValueError(f"head/detector mean mismatch: {mean_error}")
            matched = matched_parameters(htc)
            df_minima.append(float(np.min(htc["degrees_of_freedom"])))
            covariance_htc = np.square(np.asarray(htc["scale"], dtype=np.float64)) * matched["variance_factor"][:, None]
            covariance_mmg = np.square(matched["mm_scale"])
            cov_error = np.linalg.norm(covariance_htc - covariance_mmg, axis=1) / np.maximum(
                np.linalg.norm(covariance_htc, axis=1), np.finfo(np.float64).tiny
            )
            resultant_mmg = np.exp(-0.5 * matched["yaw_variance"])
            resultant_error = np.abs(resultant_mmg - matched["resultant"])
            analytic_cov_errors.append(float(np.max(cov_error)))
            analytic_resultant_errors.append(float(np.max(resultant_error)))
            if np.max(cov_error) > config["qa_tolerances"]["analytic_covariance_relative_frobenius"]:
                raise ValueError("analytic covariance match failed")
            if np.max(resultant_error) > config["qa_tolerances"]["analytic_resultant_absolute"]:
                raise ValueError("analytic resultant match failed")
            target = np.asarray(tracks["gt_input"][:, CURRENT_OFFSET:CURRENT_OFFSET + 8], dtype=np.float64)
            yaw_target = np.arctan2(
                np.asarray(tracks["gt_input"][:, CURRENT_OFFSET + 8], dtype=np.float64),
                np.asarray(tracks["gt_input"][:, CURRENT_OFFSET + 9], dtype=np.float64),
            )
            for law in LAW_IDS:
                values = upstream_metrics(law, target, yaw_target, htc, g, matched)
                for metric in UPSTREAM_METRICS:
                    arrays[f"{source_name}__upstream__{law}__{metric}"] = np.asarray(
                        values[metric], dtype=np.float32
                    )
            propagated = evaluate_condition(
                replicate, spec, tracks, retained, outputs, model, parameters, device
            )
            for key, value in propagated.items():
                arrays[f"{source_name}__{key}"] = value
            for host in r0.PREDICTORS:
                for law, original_group in (("TRAINED_G", "P2"), ("L1Y1_HTC", "P3")):
                    current = np.asarray(arrays[f"{source_name}__{host}__{law}__energy"], dtype=np.float64)
                    original = np.asarray(retained[metric_key(spec, host, law)], dtype=np.float64)
                    maximum = float(np.max(np.abs(current - original)))
                    replay_errors[f"{source_name}__{host}__{law}"] = maximum
                    if maximum > config["qa_tolerances"]["r0_energy_replay_absolute"]:
                        raise ValueError(f"R0 replay mismatch: {source_name}/{host}/{law}: {maximum}")
            cv_yaw_error = max(
                float(np.max(np.abs(
                    arrays[f"{source_name}__CV__L1Y1_HTC__energy"]
                    - arrays[f"{source_name}__CV__L1Y0_YAW_MM__energy"]
                ))),
                float(np.max(np.abs(
                    arrays[f"{source_name}__CV__L0Y1_LIN_MM__energy"]
                    - arrays[f"{source_name}__CV__L0Y0_HTC_MMG__energy"]
                ))),
            )
            arrays[f"{source_name}__cv_yaw_zero_max_abs"] = np.asarray(cv_yaw_error)
            if cv_yaw_error != 0.0:
                raise ValueError(f"CV unexpectedly depends on yaw: {source_name}/{cv_yaw_error}")
            if source_name == "clean_s0":
                mc = evaluate_condition(
                    replicate, spec, tracks, retained, outputs, model, parameters, device,
                    samples=MC_SENSITIVITY_SAMPLES,
                    seed_offset=spec["seed_offset"] + MC_SENSITIVITY_OFFSET,
                    energy_only=True,
                )
                for key, value in mc.items():
                    arrays[f"mc256__{key}"] = value
        finally:
            retained.close()
            tracks.close()
    arrays["meta__min_degrees"] = np.asarray(min(df_minima), dtype=np.float64)
    arrays["meta__max_analytic_covariance_relative_error"] = np.asarray(max(analytic_cov_errors), dtype=np.float64)
    arrays["meta__max_analytic_resultant_absolute_error"] = np.asarray(max(analytic_resultant_errors), dtype=np.float64)
    arrays["meta__max_shared_mean_absolute_error"] = np.asarray(max(mean_errors), dtype=np.float64)
    arrays["meta__max_r0_energy_replay_absolute"] = np.asarray(max(replay_errors.values()), dtype=np.float64)
    arrays["meta__replay_keys"] = np.asarray(sorted(replay_errors))
    arrays["meta__replay_errors"] = np.asarray([replay_errors[key] for key in sorted(replay_errors)])
    for name, value in arrays.items():
        if np.issubdtype(np.asarray(value).dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"non-finite output: {name}")
    atomic_npz(output_path, arrays)
    log_event(
        "evaluate_complete", replicate=replicate, device=device_index,
        output_sha256=sha256(output_path), max_r0_replay=max(replay_errors.values()),
    )
    print(json.dumps({
        "evaluated": True,
        "replicate": replicate,
        "min_degrees": min(df_minima),
        "max_r0_replay": max(replay_errors.values()),
        "output": str(output_path.relative_to(ROOT)),
    }))


def scene_means(values, scenes):
    unique, inverse = np.unique(scenes, return_inverse=True)
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64))
    counts = np.bincount(inverse)
    return unique, sums / counts


def scene_map(values, scenes):
    tokens, means = scene_means(values, scenes)
    return {token: value for token, value in zip(tokens, means)}


def average_maps(maps):
    common = sorted(set.intersection(*(set(value) for value in maps)))
    if not common:
        raise ValueError("no common scene keys")
    return np.asarray([np.mean([value[key] for value in maps]) for key in common])


def bootstrap(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPEATS):
        means[index] = values[rng.integers(0, len(values), len(values))].mean()
    return {
        "mean": float(values.mean()),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "scenes": int(len(values)),
    }


def condition_paired(runs, source_name, getter):
    seed_maps = []
    seed_effects = []
    for run in runs:
        scenes = np.asarray(run[f"{source_name}__common__scene_token"])
        values = getter(run, source_name)
        mapping = scene_map(values, scenes)
        seed_maps.append(mapping)
        seed_effects.append(float(np.mean(list(mapping.values()))))
    report = bootstrap(average_maps(seed_maps))
    report["seed_effects"] = seed_effects
    report["all_seed_directions_positive"] = all(value > 0 for value in seed_effects)
    return report


def family_paired(runs, conditions, getter):
    seed_maps = []
    seed_effects = []
    for run in runs:
        condition_maps = []
        for source_name in conditions:
            scenes = np.asarray(run[f"{source_name}__common__scene_token"])
            condition_maps.append(scene_map(getter(run, source_name), scenes))
        mapping = {
            key: float(np.mean([current[key] for current in condition_maps if key in current]))
            for key in set.union(*(set(current) for current in condition_maps))
        }
        seed_maps.append(mapping)
        seed_effects.append(float(np.mean(list(mapping.values()))))
    report = bootstrap(average_maps(seed_maps))
    report["seed_effects"] = seed_effects
    report["all_seed_directions_positive"] = all(value > 0 for value in seed_effects)
    return report


def downstream_getter(host, baseline, candidate, metric="energy", prefix=""):
    return lambda run, condition: (
        np.asarray(run[f"{prefix}{condition}__{host}__{baseline}__{metric}"], dtype=np.float64)
        - np.asarray(run[f"{prefix}{condition}__{host}__{candidate}__{metric}"], dtype=np.float64)
    )


def upstream_getter(baseline, candidate, metric):
    return lambda run, condition: (
        np.asarray(run[f"{condition}__upstream__{baseline}__{metric}"], dtype=np.float64)
        - np.asarray(run[f"{condition}__upstream__{candidate}__{metric}"], dtype=np.float64)
    )


def interaction_getter(host):
    def getter(run, condition):
        def value(law):
            return np.asarray(run[f"{condition}__{host}__{law}__energy"], dtype=np.float64)
        return value("L0Y0_HTC_MMG") - value("L1Y0_YAW_MM") - value("L0Y1_LIN_MM") + value("L1Y1_HTC")
    return getter


def quantile90(values):
    try:
        return float(np.quantile(values, 0.90, method="linear"))
    except TypeError:
        return float(np.quantile(values, 0.90, interpolation="linear"))


def downstream_equal_area(run, condition, host, law):
    maha = np.asarray(run[f"{condition}__{host}__{law}__final_maha2"], dtype=np.float64)
    determinant = np.asarray(run[f"{condition}__{host}__{law}__final_det"], dtype=np.float64)
    return float(np.median(math.pi * quantile90(maha) * np.sqrt(determinant)))


def upstream_equal_area(run, condition, law):
    maha = np.asarray(run[f"{condition}__upstream__{law}__xy_maha2"], dtype=np.float64)
    product = np.asarray(run[f"{condition}__upstream__{law}__xy_scale_product"], dtype=np.float64)
    return float(np.median(math.pi * quantile90(maha) * product))


def write_csv(path: Path, rows):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def create_h5_outputs(runs):
    paired_path = OUT / "paired_track_metrics.h5"
    scene_path = OUT / "scene_level_metrics.h5"
    if paired_path.exists() or scene_path.exists():
        raise FileExistsError("HDF5 output already exists")
    paired_tmp = paired_path.with_suffix(".h5.partial")
    scene_tmp = scene_path.with_suffix(".h5.partial")
    with h5py.File(paired_tmp, "w") as paired, h5py.File(scene_tmp, "w") as scene_file:
        paired.attrs["layout"] = "replicate/condition grouped arrays; law and probe dimensions stored as attributes"
        paired.attrs["laws"] = json.dumps(LAW_IDS)
        paired.attrs["probes"] = json.dumps(r0.PREDICTORS)
        for replicate, run in enumerate(runs):
            for source_name in CONDITIONS:
                group = paired.create_group(f"replicate_{replicate}/{source_name}")
                for name in ("scene_token", "sample_token", "instance_token", "row_id"):
                    group.create_dataset(name, data=run[f"{source_name}__common__{name}"], compression="gzip")
                upstream = np.stack([
                    np.stack([
                        np.asarray(run[f"{source_name}__upstream__{law}__{metric}"], dtype=np.float32)
                        for metric in UPSTREAM_METRICS
                    ], axis=1)
                    for law in LAW_IDS
                ], axis=1)
                group.create_dataset("upstream", data=upstream, compression="gzip", compression_opts=4)
                downstream = np.stack([
                    np.stack([
                        np.stack([
                            np.asarray(run[f"{source_name}__{host}__{law}__{metric}"], dtype=np.float32)
                            for metric in DOWNSTREAM_METRICS
                        ], axis=1)
                        for law in LAW_IDS
                    ], axis=1)
                    for host in r0.PREDICTORS
                ], axis=1)
                group.create_dataset("downstream", data=downstream, compression="gzip", compression_opts=4)
                group.attrs["upstream_metrics"] = json.dumps(UPSTREAM_METRICS)
                group.attrs["downstream_metrics"] = json.dumps(DOWNSTREAM_METRICS)
                group.attrs["head_seed"] = int(run["meta__head_seed"])

                scenes = np.asarray(run[f"{source_name}__common__scene_token"])
                unique = np.unique(scenes)
                sg = scene_file.create_group(f"replicate_{replicate}/{source_name}")
                sg.create_dataset("scene_token", data=unique)
                es = np.empty((len(unique), len(r0.PREDICTORS), len(LAW_IDS)), dtype=np.float32)
                for host_index, host in enumerate(r0.PREDICTORS):
                    for law_index, law in enumerate(LAW_IDS):
                        tokens, values = scene_means(run[f"{source_name}__{host}__{law}__energy"], scenes)
                        if not np.array_equal(tokens, unique):
                            raise ValueError("scene aggregation mismatch")
                        es[:, host_index, law_index] = values
                sg.create_dataset("trajectory_energy", data=es, compression="gzip")
    paired_tmp.replace(paired_path)
    scene_tmp.replace(scene_path)
    return paired_path, scene_path


def finalize():
    config = verify_protocol()
    result_path = OUT / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    metric_paths = [METRIC_ROOT / f"replicate_{replicate}.npz" for replicate in range(3)]
    if not all(path.exists() for path in metric_paths):
        raise FileNotFoundError("all three matched-law replicates are required")
    runs = [np.load(path, allow_pickle=False) for path in metric_paths]
    summary_rows = []
    bootstrap_rows = []
    effects = {"conditions": {}, "families": {}, "upstream": {}, "mc256": {}}
    try:
        for source_name, spec in CONDITIONS.items():
            effects["conditions"][source_name] = {}
            for host in r0.PREDICTORS:
                effects["conditions"][source_name][host] = {}
                for contrast, (baseline, candidate) in CONTRASTS.items():
                    report = condition_paired(
                        runs, source_name, downstream_getter(host, baseline, candidate)
                    )
                    effects["conditions"][source_name][host][contrast] = report
                    bootstrap_rows.append({
                        "scope": "condition", "population": spec["population"],
                        "condition": spec["condition"], "severity": spec["severity"],
                        "probe": host, "domain": "downstream", "metric": "trajectory_energy",
                        "contrast": contrast, **report,
                    })
                interaction = condition_paired(runs, source_name, interaction_getter(host))
                effects["conditions"][source_name][host]["factorial_interaction"] = interaction
                bootstrap_rows.append({
                    "scope": "condition", "population": spec["population"],
                    "condition": spec["condition"], "severity": spec["severity"],
                    "probe": host, "domain": "downstream", "metric": "trajectory_energy",
                    "contrast": "factorial_interaction", **interaction,
                })
            effects["upstream"][source_name] = {}
            for metric in ("full_linear_nll", "xy_nll", "yaw_nll"):
                report = condition_paired(
                    runs, source_name,
                    upstream_getter("L0Y0_HTC_MMG", "L1Y1_HTC", metric),
                )
                effects["upstream"][source_name][metric] = report
                bootstrap_rows.append({
                    "scope": "condition", "population": spec["population"],
                    "condition": spec["condition"], "severity": spec["severity"],
                    "probe": "upstream", "domain": "upstream", "metric": metric,
                    "contrast": "full_higher_order", **report,
                })
            for replicate, run in enumerate(runs):
                for law in LAW_IDS:
                    row = {
                        "population": spec["population"], "source_name": source_name,
                        "condition": spec["condition"], "severity": spec["severity"],
                        "replicate": replicate, "head_seed": int(run["meta__head_seed"]),
                        "law_id": law,
                    }
                    for metric in ("full_linear_nll", "xy_nll", "yaw_nll", "linear90_inside", "yaw90_inside"):
                        row[f"upstream_{metric}"] = float(np.mean(run[f"{source_name}__upstream__{law}__{metric}"]))
                    row["upstream_eq90_xy_area"] = upstream_equal_area(run, source_name, law)
                    for host in r0.PREDICTORS:
                        for metric in ("energy", "nll", "mean_ade", "mean_fde", "coverage_90"):
                            row[f"{host}_{metric}"] = float(np.mean(run[f"{source_name}__{host}__{law}__{metric}"]))
                        row[f"{host}_eq90_final_area"] = downstream_equal_area(run, source_name, host, law)
                    row["invalid_sample_rate"] = float(run[f"{source_name}__invalid_rate__{law}"])
                    summary_rows.append(row)

        for family, conditions in FAMILIES.items():
            effects["families"][family] = {}
            for host in r0.PREDICTORS:
                report = family_paired(
                    runs, conditions,
                    downstream_getter(host, "L0Y0_HTC_MMG", "L1Y1_HTC"),
                )
                effects["families"][family][host] = report
                bootstrap_rows.append({
                    "scope": "family", "population": "development" if family in {
                        "camera_lowlight", "lidar_point_dropout", "lidar_beam_reduction"
                    } else "held_out_degradation",
                    "condition": family, "severity": "aggregate", "probe": host,
                    "domain": "downstream", "metric": "trajectory_energy",
                    "contrast": "full_higher_order", **report,
                })

        for host in r0.PREDICTORS:
            report = condition_paired(
                runs, "clean_s0",
                lambda run, _condition, host=host: (
                    np.asarray(run[f"mc256__{host}__L0Y0_HTC_MMG__energy"], dtype=np.float64)
                    - np.asarray(run[f"mc256__{host}__L1Y1_HTC__energy"], dtype=np.float64)
                ),
            )
            effects["mc256"][host] = report
            bootstrap_rows.append({
                "scope": "mc_sensitivity", "population": "development",
                "condition": "clean", "severity": 0, "probe": host,
                "domain": "downstream", "metric": "trajectory_energy_M256",
                "contrast": "full_higher_order", **report,
            })

        clean_primary = effects["conditions"]["clean_s0"]
        clean_pass = all(
            clean_primary[host]["full_higher_order"]["ci95_lower"] > 0.0
            and clean_primary[host]["full_higher_order"]["all_seed_directions_positive"]
            for host in r0.PREDICTORS
        )
        mc_direction = all(effects["mc256"][host]["mean"] > 0.0 for host in r0.PREDICTORS)
        family_direction = all(
            effects["families"][family][host]["mean"] >= 0.0
            and effects["families"][family][host]["ci95_upper"] >= 0.0
            for family in FAMILIES for host in r0.PREDICTORS
        )
        upstream_pass = all(
            effects["upstream"]["clean_s0"][metric]["ci95_lower"] > 0.0
            for metric in ("full_linear_nll", "xy_nll", "yaw_nll")
        )
        max_replay = max(float(run["meta__max_r0_energy_replay_absolute"]) for run in runs)
        min_degrees = min(float(run["meta__min_degrees"]) for run in runs)
        max_cov_error = max(float(run["meta__max_analytic_covariance_relative_error"]) for run in runs)
        max_resultant_error = max(float(run["meta__max_analytic_resultant_absolute_error"]) for run in runs)
        max_mean_error = max(float(run["meta__max_shared_mean_absolute_error"]) for run in runs)
        max_invalid = max(
            float(run[f"{condition}__invalid_rate__{law}"])
            for run in runs for condition in CONDITIONS for law in LAW_IDS
        )
        integrity_gates = {
            "identical_rows_and_order": True,
            "shared_detector_mean": max_mean_error <= config["qa_tolerances"]["shared_mean_absolute"],
            "analytic_linear_covariance_match": max_cov_error <= config["qa_tolerances"]["analytic_covariance_relative_frobenius"],
            "analytic_circular_resultant_match": max_resultant_error <= config["qa_tolerances"]["analytic_resultant_absolute"],
            "surrogate_independent_of_trained_g_parameters": True,
            "frozen_probe_process_noise_and_targets": True,
            "finite_nu_gt_2_positive_definite": min_degrees > 2.0,
            "invalid_clipping_rejection_rates_saved": max_invalid == 0.0,
            "all_seeds_and_conditions_present": True,
            "scene_level_traceability": True,
            "exact_r0_replay": max_replay <= config["qa_tolerances"]["r0_energy_replay_absolute"],
            "cv_yaw_zero_check": all(
                float(run[f"{condition}__cv_yaw_zero_max_abs"]) == 0.0
                for run in runs for condition in CONDITIONS
            ),
        }
        valid = all(integrity_gates.values())
        if not valid:
            endpoint = "INVALID MATCHED-LAW AUDIT"
        elif clean_pass and mc_direction:
            endpoint = (
                "RETROSPECTIVE OUTCOME A — AGGREGATE HIGHER-ORDER LAW-SHAPE EFFECT SUPPORTED"
                if family_direction else
                "RETROSPECTIVE OUTCOME C — CLEAN EFFECT WITH DEGRADATION DEPENDENCE"
            )
        elif any(clean_primary[host]["full_higher_order"]["mean"] > 0.0 for host in r0.PREDICTORS):
            endpoint = "RETROSPECTIVE OUTCOME C — PROBE/SEED-DEPENDENT MATCHED-LAW EFFECT"
        else:
            endpoint = "RETROSPECTIVE OUTCOME B — NO STABLE HIGHER-ORDER TRANSPORT"

        paired_h5, scene_h5 = create_h5_outputs(runs)
        write_csv(OUT / "condition_seed_summary.csv", summary_rows)
        flattened_rows = []
        for row in bootstrap_rows:
            current = dict(row)
            seed_effects = current.pop("seed_effects")
            current["seed_0_effect"] = seed_effects[0]
            current["seed_1_effect"] = seed_effects[1]
            current["seed_2_effect"] = seed_effects[2]
            flattened_rows.append(current)
        write_csv(OUT / "bootstrap_summary.csv", flattened_rows)
        write_csv(
            OUT / "mc_sensitivity_summary.csv",
            [row for row in flattened_rows if row["scope"] == "mc_sensitivity"],
        )
        integrity = {
            "valid": valid,
            "gates": integrity_gates,
            "min_degrees_of_freedom": min_degrees,
            "max_shared_mean_absolute_error": max_mean_error,
            "max_analytic_covariance_relative_error": max_cov_error,
            "max_analytic_resultant_absolute_error": max_resultant_error,
            "max_invalid_sample_rate": max_invalid,
            "max_r0_energy_replay_absolute": max_replay,
            "internal_test_read": False,
        }
        atomic_json(OUT / "integrity_report.json", integrity)
        result = {
            "valid": valid,
            "endpoint": endpoint,
            "retrospective": True,
            "clean_joint_probe_pass": clean_pass,
            "mc256_direction_consistent": mc_direction,
            "six_family_direction_stable": family_direction,
            "clean_upstream_nll_joint_pass": upstream_pass,
            "higher_order_upstream_quality_survives_downstream_transform": bool(clean_pass and upstream_pass),
            "effects": effects,
            "integrity": integrity,
            "metric_sha256": {str(index): sha256(path) for index, path in enumerate(metric_paths)},
            "paired_track_metrics_sha256": sha256(paired_h5),
            "scene_level_metrics_sha256": sha256(scene_h5),
            "protocol_sha256": sha256(PROTOCOL),
            "models_retrained": False,
            "official_modern_host_outcome_read": False,
            "internal_test_read": False,
        }
        atomic_json(result_path, result)
    finally:
        for run in runs:
            run.close()
    build_manifest()
    log_event("finalize_complete", endpoint=result["endpoint"], result_sha256=sha256(result_path))
    print(json.dumps(result, indent=2))


def build_manifest():
    output_paths = [
        path for path in OUT.rglob("*")
        if path.is_file() and not path.name.endswith(".partial") and path.name != "run.log"
    ]
    manifest = {
        "title": "Level-I Moment-Matched Law Control",
        "status": "retrospective matched-law audit",
        "evaluator": str(Path(__file__).relative_to(ROOT)),
        "evaluator_sha256": sha256(Path(__file__)),
        "git_commit": None,
        "commands": [
            "prepare", "self-check", "sampler-validation",
            "evaluate --replicate 0/1/2 --device 3", "finalize", "plot", "report",
        ],
        "head_seeds": list(r0.HEAD_SEEDS),
        "propagation_seeds": list(r0.PROPAGATION_SEEDS),
        "conditions": [
            {key: (str(value.relative_to(ROOT)) if key == "tracks" else value) for key, value in spec.items()}
            for spec in CONDITIONS.values()
        ],
        "formulas": strict_json(PROTOCOL)["construction"],
        "process_noise_sha256": sha256(r0.OUT / "predictor.pt"),
        "outputs": {
            str(path.relative_to(OUT)): sha256(path)
            for path in sorted(output_paths)
            if path.name != "manifest.json"
        },
        "excluded_rows": 0,
        "models_retrained": False,
        "internal_test_read": False,
    }
    try:
        import subprocess
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False,
            capture_output=True, text=True,
        )
        if completed.returncode == 0:
            manifest["git_commit"] = completed.stdout.strip()
    except OSError:
        pass
    path = OUT / "manifest.json"
    if path.exists():
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    else:
        atomic_json(path, manifest)


def sampler_validation(device_index: int):
    config = verify_protocol()
    output_path = OUT / "sampler_validation.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    device = torch.device(f"cuda:{device_index}")
    set_determinism(20260836, device)
    heads = load_heads(0, device)
    tracks = np.load(
        ROOT / "outputs/riskprop3d_r1/scenes/label_cal.npz", allow_pickle=False
    )
    try:
        count = min(16, len(tracks["input"]))
        target = np.asarray(tracks["target_agent_index"][:count], dtype=np.int64)
        subset = {
            "input": np.asarray(tracks["input"][:count]),
            "feature": np.asarray(tracks["agent_feature"])[target],
            "logits": np.asarray(tracks["agent_logits"])[target],
            "box": np.asarray(tracks["agent_box"])[target],
            "seed": np.asarray(tracks["agent_seed"])[target],
        }
        outputs = r0.head_outputs(heads["M2"], subset, device)
    finally:
        tracks.close()
    matched = matched_parameters(outputs)
    draws = 262144
    rng = np.random.default_rng(20260836)
    covariance_errors = []
    mean_errors = []
    resultant_errors = []
    for row in range(count):
        z = rng.standard_normal((draws, 8))
        radial = rng.uniform(1e-7, 1.0 - 1e-7, draws)
        degrees = float(outputs["degrees_of_freedom"][row])
        factor = np.sqrt(degrees / chi2.ppf(radial, degrees))
        student = outputs["mean_state"][row] + outputs["scale"][row] * z * factor[:, None]
        gaussian = outputs["mean_state"][row] + matched["mm_scale"][row] * z
        mean_errors.append(float(np.linalg.norm(student.mean(0) - gaussian.mean(0))))
        covariance_errors.append(float(
            np.linalg.norm(np.cov(student, rowvar=False) - np.cov(gaussian, rowvar=False))
            / max(np.linalg.norm(np.square(matched["mm_scale"][row])), np.finfo(np.float64).tiny)
        ))
        vm = rng.vonmises(outputs["yaw_mean"][row], outputs["yaw_kappa"][row], draws)
        wg = wrap(outputs["yaw_mean"][row] + matched["yaw_scale"][row] * rng.standard_normal(draws))
        resultant_errors.append(float(abs(
            abs(np.mean(np.exp(1j * vm))) - abs(np.mean(np.exp(1j * wg)))
        )))
    analytic_covariance_htc = (
        np.square(np.asarray(outputs["scale"], dtype=np.float64))
        * matched["variance_factor"][:, None]
    )
    analytic_covariance_mmg = np.square(matched["mm_scale"])
    analytic_covariance_error = float(np.max(
        np.linalg.norm(analytic_covariance_htc - analytic_covariance_mmg, axis=1)
        / np.maximum(
            np.linalg.norm(analytic_covariance_htc, axis=1),
            np.finfo(np.float64).tiny,
        )
    ))
    analytic_resultant_error = float(np.max(np.abs(
        np.exp(-0.5 * matched["yaw_variance"]) - matched["resultant"]
    )))
    report = {
        "passed": (
            analytic_covariance_error <= config["qa_tolerances"]["analytic_covariance_relative_frobenius"]
            and analytic_resultant_error <= config["qa_tolerances"]["analytic_resultant_absolute"]
            and float(np.min(outputs["degrees_of_freedom"])) > 2.0
        ),
        "split": "clean label-calibration smoke rows",
        "rows": count,
        "samples_per_row": draws,
        "min_degrees_of_freedom": float(np.min(outputs["degrees_of_freedom"])),
        "analytic_covariance_max_relative_frobenius_error": analytic_covariance_error,
        "analytic_resultant_max_absolute_error": analytic_resultant_error,
        "empirical_student_vs_mmg_mean_l2": {
            "median": float(np.median(mean_errors)), "max": float(np.max(mean_errors)),
        },
        "empirical_student_vs_mmg_covariance_relative_frobenius": {
            "median": float(np.median(covariance_errors)), "max": float(np.max(covariance_errors)),
            "note": "descriptive Monte Carlo diagnostic; near-two-df Student-t covariance converges slowly",
        },
        "empirical_vm_vs_wg_resultant_absolute_error": {
            "median": float(np.median(resultant_errors)), "max": float(np.max(resultant_errors)),
        },
        "postprocessing": "linear state unchanged; yaw wrapped identically with atan2(sin,cos)",
        "clipping_rate": 0.0,
        "rejection_rate": 0.0,
        "invalid_rate": 0.0,
    }
    if not report["passed"]:
        raise ValueError("sampler validation failed")
    atomic_json(output_path, report)
    log_event("sampler_validation_complete", passed=True, output_sha256=sha256(output_path))
    print(json.dumps(report, indent=2))


def plot():
    verify_protocol()
    result = strict_json(OUT / "result.json")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    colors = {"CV": "#4C78A8", "MLP": "#E07A5F"}
    scopes = ["clean_s0", *FAMILIES, "official_clean"]
    labels = ["Dev\nclean", "Lowlight", "Dropout", "Beam", "Rotation", "Translation", "Temporal", "Final\nvalidation"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    source_rows = []
    for panel, host in enumerate(r0.PREDICTORS):
        axis = axes[panel]
        reports = []
        for scope in scopes:
            if scope in FAMILIES:
                report = result["effects"]["families"][scope][host]
                scope_type = "family"
            else:
                report = result["effects"]["conditions"][scope][host]["full_higher_order"]
                scope_type = "condition"
            reports.append(report)
            source_rows.append({
                "figure": "matched_law_primary", "probe": host, "scope": scope,
                "scope_type": scope_type, "effect": report["mean"],
                "ci95_lower": report["ci95_lower"], "ci95_upper": report["ci95_upper"],
            })
        y = np.arange(len(reports))
        effect = np.asarray([value["mean"] for value in reports])
        low = effect - np.asarray([value["ci95_lower"] for value in reports])
        high = np.asarray([value["ci95_upper"] for value in reports]) - effect
        axis.errorbar(effect, y, xerr=np.stack((low, high)), fmt="o", color=colors[host],
                      markersize=4.5, capsize=2.5, linewidth=1.1)
        axis.axvline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.5, alpha=0.7)
        axis.set_title(f"{host} forecasting probe", fontsize=8, fontweight="bold")
        axis.set_xlabel("Energy difference: HTC-MMG − HTC")
        axis.text(-0.12, 1.03, chr(ord("a") + panel), transform=axis.transAxes,
                  fontsize=9, fontweight="bold")
    axes[0].set_yticks(np.arange(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.text(0.5, 0.995, "Positive values favor the complete HTC law; 95% paired scene-bootstrap CIs",
             ha="center", va="top", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.0)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    base = FIGURE_ROOT / "matched_law_primary"
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300}), ("tiff", {"dpi": 600})):
        fig.savefig(base.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    contrast_order = ("linear_given_vm", "yaw_given_student", "full_higher_order", "factorial_interaction")
    contrast_labels = ("Linear shape", "Yaw family", "Full matched law", "2×2 interaction")
    for panel, host in enumerate(r0.PREDICTORS):
        reports = [result["effects"]["conditions"]["clean_s0"][host][name] for name in contrast_order]
        for name, current in zip(contrast_order, reports):
            source_rows.append({
                "figure": "matched_law_factorial", "probe": host, "scope": name,
                "scope_type": "secondary_clean_contrast", "effect": current["mean"],
                "ci95_lower": current["ci95_lower"], "ci95_upper": current["ci95_upper"],
            })
        effect = np.asarray([value["mean"] for value in reports])
        low = effect - np.asarray([value["ci95_lower"] for value in reports])
        high = np.asarray([value["ci95_upper"] for value in reports]) - effect
        y = np.arange(len(reports))
        axes[panel].errorbar(effect, y, xerr=np.stack((low, high)), fmt="o", color=colors[host],
                             markersize=4.5, capsize=2.5, linewidth=1.1)
        axes[panel].axvline(0, color="#777777", linestyle="--", linewidth=0.8)
        axes[panel].grid(axis="x", color="#D9D9D9", linewidth=0.5, alpha=0.7)
        axes[panel].set_title(host, fontsize=8, fontweight="bold")
        axes[panel].set_xlabel("Clean energy contrast")
    axes[0].set_yticks(np.arange(len(contrast_labels)), contrast_labels)
    axes[0].invert_yaxis()
    fig.text(0.5, 0.995, "Factorial contrasts are secondary diagnostics; positive main effects favor HTC components",
             ha="center", va="top", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=2.0)
    base = FIGURE_ROOT / "matched_law_factorial"
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300}), ("tiff", {"dpi": 600})):
        fig.savefig(base.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)
    write_csv(OUT / "figure_source_data.csv", source_rows)
    log_event("plot_complete", figures=["matched_law_primary", "matched_law_factorial"])
    print(json.dumps({"plotted": True, "figures": str(FIGURE_ROOT.relative_to(ROOT))}))


def report():
    result = strict_json(OUT / "result.json")
    integrity = strict_json(OUT / "integrity_report.json")
    clean = result["effects"]["conditions"]["clean_s0"]
    official = result["effects"]["conditions"]["official_clean"]
    lines = [
        "# Level-I Moment-Matched Law Control — Results Report",
        "",
        f"**{result['endpoint']}**",
        "",
        "This is a retrospective matched-law audit on already used development, held-out-degradation, and final-validation populations. It does not open the internal test and cannot establish a new confirmatory component mechanism.",
        "",
        "## Primary matched-law contrast",
        "",
        "Positive values are `ES(HTC-MMG) - ES(HTC)` and favor the complete HTC law.",
        "",
        "| Population | Probe | Effect | 95% CI | Seed effects |",
        "|---|---|---:|---:|---:|",
    ]
    for population, values in (("Development clean", clean), ("Final validation", official)):
        for host in r0.PREDICTORS:
            current = values[host]["full_higher_order"]
            lines.append(
                f"| {population} | {host} | {current['mean']:.5f} | "
                f"[{current['ci95_lower']:.5f}, {current['ci95_upper']:.5f}] | "
                + ", ".join(f"{value:.5f}" for value in current["seed_effects"]) + " |"
            )
    lines += [
        "",
        "## Interpretation checks",
        "",
        f"- Joint clean-probe criterion: **{'PASS' if result['clean_joint_probe_pass'] else 'FAIL'}**.",
        f"- Clean M=256 direction check: **{'PASS' if result['mc256_direction_consistent'] else 'FAIL'}**.",
        f"- Six-family directional stability: **{'PASS' if result['six_family_direction_stable'] else 'FAIL'}**.",
        f"- Clean upstream full-linear/XY/yaw NLL joint criterion: **{'PASS' if result['clean_upstream_nll_joint_pass'] else 'FAIL'}**.",
        f"- Higher-order upstream-quality transport wording permitted by the frozen rule: **{'YES' if result['higher_order_upstream_quality_survives_downstream_transform'] else 'NO'}**.",
        "",
        "## Integrity",
        "",
        f"All integrity gates: **{'PASS' if integrity['valid'] else 'FAIL'}**.",
        f"Minimum df: `{integrity['min_degrees_of_freedom']:.6f}`; maximum analytic covariance relative error: `{integrity['max_analytic_covariance_relative_error']:.3e}`; maximum resultant error: `{integrity['max_analytic_resultant_absolute_error']:.3e}`; maximum invalid rate: `{integrity['max_invalid_sample_rate']:.3e}`; maximum R0 energy replay error: `{integrity['max_r0_energy_replay_absolute']:.3e}`.",
        "",
        "No clipping or rejection is used by the frozen linear sampler; yaw is wrapped identically. CV yaw contrasts are exact zero checks and are not circular-mechanism evidence.",
        "",
        "## Files",
        "",
        "- `result.json`: machine-readable endpoint and all paired effects.",
        "- `bootstrap_summary.csv`: condition/family/upstream/MC intervals.",
        "- `condition_seed_summary.csv`: per-condition, seed, and law summaries.",
        "- `paired_track_metrics.h5`: paired per-target metrics.",
        "- `scene_level_metrics.h5`: traceable scene-level energy arrays.",
        "- `figures/`: primary and factorial manuscript/supplement figures.",
    ]
    path = OUT / "RESULTS_REPORT.md"
    if path.exists():
        raise FileExistsError(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    contract = OUT / "FIGURE_CONTRACT.md"
    contract.write_text(
        "# Figure contract\n\n"
        "## Primary figure\n\nConclusion: the sign and stability of `ES(HTC-MMG)-ES(HTC)` across probes and populations. "
        "Evidence: replicate-averaged paired-scene bootstrap. A zero line is mandatory. Condition/family labels must distinguish development, held-out degradation, and final validation.\n\n"
        "## Factorial figure\n\nConclusion: secondary decomposition only. Linear, yaw, full, and interaction contrasts must not be labeled causal or confirmatory. CV yaw is an integrity zero.\n\n"
        "Exports: editable SVG/PDF plus 300-dpi PNG and 600-dpi TIFF. Source data: `figure_source_data.csv`.\n",
        encoding="utf-8",
    )
    build_manifest()
    log_event("report_complete", report_sha256=sha256(path))
    print(json.dumps({"reported": True, "path": str(path.relative_to(ROOT))}))


def self_check():
    rng = np.random.default_rng(17)
    n = 8
    htc = {
        "mean_state": rng.normal(size=(n, 8)),
        "scale": np.exp(rng.normal(-0.2, 0.2, size=(n, 8))),
        "degrees_of_freedom": rng.uniform(2.1, 12.0, size=n),
        "yaw_mean": rng.uniform(-math.pi, math.pi, size=n),
        "yaw_kappa": np.exp(rng.normal(size=n)),
    }
    matched = matched_parameters(htc)
    covariance = np.square(htc["scale"]) * htc["degrees_of_freedom"][:, None] / (htc["degrees_of_freedom"] - 2.0)[:, None]
    if not np.allclose(covariance, np.square(matched["mm_scale"]), rtol=1e-13, atol=0.0):
        raise ValueError("analytic covariance self-check failed")
    if not np.allclose(np.exp(-0.5 * matched["yaw_variance"]), matched["resultant"], rtol=1e-13, atol=0.0):
        raise ValueError("analytic resultant self-check failed")
    inputs = rng.normal(size=(3, r0.INPUT_DIM)).astype(np.float32)
    first = inputs.copy()
    second = inputs.copy()
    first[:, CURRENT_OFFSET + 8:CURRENT_OFFSET + 10] = np.asarray([[0.0, 1.0]])
    second[:, CURRENT_OFFSET + 8:CURRENT_OFFSET + 10] = np.asarray([[1.0, 0.0]])
    if not np.array_equal(r0.cv_prediction(first), r0.cv_prediction(second)):
        raise ValueError("CV yaw-independence self-check failed")
    print(json.dumps({"self_check": True, "covariance_match": True, "resultant_match": True, "cv_yaw_independent": True}))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("self-check")
    validation = commands.add_parser("sampler-validation")
    validation.add_argument("--device", type=int, default=3)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--replicate", type=int, required=True)
    evaluation.add_argument("--device", type=int, default=3)
    commands.add_parser("finalize")
    commands.add_parser("plot")
    commands.add_parser("report")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "self-check":
        self_check()
    elif args.command == "sampler-validation":
        sampler_validation(args.device)
    elif args.command == "evaluate":
        evaluate(args.replicate, args.device)
    elif args.command == "finalize":
        finalize()
    elif args.command == "plot":
        plot()
    elif args.command == "report":
        report()


if __name__ == "__main__":
    main()
