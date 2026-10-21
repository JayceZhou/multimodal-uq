#!/usr/bin/env python3
"""Read-only sensitivity audit for the R0 additive process-noise covariance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/riskprop3d-process-noise-mpl")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import riskprop3d_r0 as r0
from probset3d.f1.implementation.probabilistic_query import ProbabilisticQueryHead


OUT = ROOT / "outputs/riskprop3d_process_noise_sensitivity"
SCALES = (0.0, 0.5, 1.0, 2.0)  # covariance multipliers, not std multipliers
OFFICIAL_SEED_OFFSET = 900000
BOOTSTRAP_SEED = 20260805
BOOTSTRAP_REPEATS = 10000
POPULATIONS = {
    "development_clean": {
        "tracks": r0.OUT / "tracks/clean_s0.npz",
        "retained": r0.OUT / "metrics/replicate_{replicate}.npz",
        "retained_prefix": "clean_s0",
        "seed_offset": 0,
    },
    "final_validation": {
        "tracks": ROOT / "outputs/riskprop3d_final_confirmation/tracks/official/clean.npz",
        "retained": ROOT / "outputs/riskprop3d_final_confirmation/metrics/official/replicate_{replicate}.npz",
        "retained_prefix": "official_clean__r0",
        "seed_offset": OFFICIAL_SEED_OFFSET,
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path, arrays):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def strict_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_paths():
    paths = {r0.OUT / "predictor.pt"}
    paths.update(r0.CHECKPOINT_ROOT / f"seed_{seed}.pt" for seed in r0.HEAD_SEEDS)
    for spec in POPULATIONS.values():
        paths.add(spec["tracks"])
        paths.update(
            Path(str(spec["retained"]).format(replicate=replicate))
            for replicate in range(3)
        )
    return sorted(paths)


def prepare():
    protocol = OUT / "protocol.json"
    if protocol.exists():
        raise FileExistsError(protocol)
    value = {
        "purpose": "reviewer-requested R0 process-noise magnitude sensitivity",
        "covariance_multipliers": list(SCALES),
        "standard_deviation_multipliers": [math.sqrt(value) for value in SCALES],
        "populations": list(POPULATIONS),
        "hosts": list(r0.PREDICTORS),
        "comparators": {"G": "P2/M1", "HTC": "P3/M2"},
        "primary_metric": "trajectory energy score",
        "effect": "G minus HTC; positive favors HTC",
        "sampling": {
            "perception_samples": r0.SAMPLES,
            "perception_draw_sets": 2,
            "common_random_numbers": True,
            "same_base_process_noise_across_multipliers": True,
            "propagation_seeds": list(r0.PROPAGATION_SEEDS),
        },
        "inference": {
            "unit": "scene",
            "paired_scene_bootstrap_resamples": BOOTSTRAP_REPEATS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "replicates_averaged_before_scene_bootstrap": True,
            "multiplicity": "unadjusted directional sensitivity intervals; all prespecified contrasts reported",
        },
        "robustness_interpretation": (
            "directionally robust only if every population-host-multiplier point estimate "
            "is positive; interval-supported throughout only if every 95% CI lower bound is positive"
        ),
        "models_retrained": False,
        "conditions_selected_from_results": False,
        "official_modern_host_outcome_read": False,
        "internal_test_read": False,
        "evaluator_sha256": sha256(Path(__file__)),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths()
        },
    }
    atomic_json(protocol, value)
    print(json.dumps({"prepared": True, "protocol": str(protocol.relative_to(ROOT))}))


def verify_protocol():
    value = strict_json(OUT / "protocol.json")
    if value["evaluator_sha256"] != sha256(Path(__file__)):
        raise ValueError("evaluator changed after sensitivity protocol freeze")
    for relative, expected in value["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"source changed after sensitivity freeze: {relative}")
    return value


def set_determinism(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(device)


def load_heads(replicate, device):
    checkpoint = torch.load(
        r0.CHECKPOINT_ROOT / f"seed_{r0.HEAD_SEEDS[replicate]}.pt",
        map_location="cpu",
        weights_only=True,
    )
    heads = {name: ProbabilisticQueryHead().to(device).eval() for name in ("M1", "M2")}
    for name, head in heads.items():
        head.load_state_dict(checkpoint["models"][name])
    return heads


def slice_output(values, begin, stop):
    return {name: value[begin:stop] for name, value in values.items()}


def energy_score(centers_a, centers_b, future, noise_a, noise_b, covariance_multiplier):
    centers_a = np.asarray(centers_a, dtype=np.float64)
    centers_b = np.asarray(centers_b, dtype=np.float64)
    future = np.asarray(future, dtype=np.float64)
    factor = math.sqrt(covariance_multiplier)
    samples_a = centers_a + factor * noise_a
    samples_b = centers_b + factor * noise_b
    first = np.linalg.norm(
        (samples_a - future[:, None]).reshape(len(future), samples_a.shape[1], -1),
        axis=-1,
    ).mean(1)
    second = np.linalg.norm(
        (samples_a - samples_b).reshape(len(future), samples_a.shape[1], -1),
        axis=-1,
    ).mean(1)
    return first - 0.5 * second


def scale_tag(value):
    return str(value).replace(".", "p")


def retained_key(prefix, host, group):
    return f"{prefix}__{host}__{group}__energy"


def evaluate(replicate, device_index):
    verify_protocol()
    if replicate not in range(3) or device_index not in range(4):
        raise ValueError("replicate/device outside frozen range")
    output_path = OUT / "metrics" / f"replicate_{replicate}.npz"
    if output_path.exists():
        raise FileExistsError(output_path)
    device = torch.device(f"cuda:{device_index}")
    set_determinism(r0.PROPAGATION_SEEDS[replicate], device)
    model, parameters = r0.load_predictor(device)
    heads = load_heads(replicate, device)
    arrays = {
        "meta__replicate": np.asarray(replicate, dtype=np.int64),
        "meta__head_seed": np.asarray(r0.HEAD_SEEDS[replicate], dtype=np.int64),
    }
    replay_errors = {}

    for population, spec in POPULATIONS.items():
        tracks = np.load(spec["tracks"], allow_pickle=False)
        retained_path = Path(str(spec["retained"]).format(replicate=replicate))
        retained = np.load(retained_path, allow_pickle=False)
        try:
            prefix = spec["retained_prefix"]
            for name in ("scene_token", "sample_token", "instance_token"):
                retained_rows = np.asarray(retained[f"{prefix.split('__r0')[0]}__common__{name}"])
                if not np.array_equal(np.asarray(tracks[name]), retained_rows):
                    raise ValueError(f"row mismatch: {population}/{name}")
                arrays[f"{population}__common__{name}"] = np.asarray(tracks[name])

            outputs = {name: r0.head_outputs(head, tracks, device) for name, head in heads.items()}
            storage = {
                host: {
                    scale: {group: [] for group in ("P2", "P3")}
                    for scale in SCALES
                }
                for host in r0.PREDICTORS
            }
            rng = np.random.default_rng(
                r0.PROPAGATION_SEEDS[replicate] + spec["seed_offset"]
            )
            for begin in range(0, len(tracks["input"]), r0.EVAL_CHUNK):
                stop = min(begin + r0.EVAL_CHUNK, len(tracks["input"]))
                motion = np.asarray(tracks["input"][begin:stop])
                future = np.asarray(tracks["future"][begin:stop])
                count = stop - begin
                sampled = {"P2": [], "P3": []}
                for _draw in range(2):
                    z = rng.standard_normal((count, r0.SAMPLES, 8))
                    radial = rng.uniform(1e-7, 1.0 - 1e-7, (count, r0.SAMPLES))
                    yaw_normal = rng.standard_normal((count, r0.SAMPLES))
                    for group, family, student in (
                        ("P2", "M1", False),
                        ("P3", "M2", True),
                    ):
                        state = r0.sample_states(
                            slice_output(outputs[family], begin, stop),
                            rng,
                            r0.SAMPLES,
                            student,
                            z,
                            radial,
                            yaw_normal,
                        )
                        sampled[group].append(r0.replace_current(motion, *state))

                for host in r0.PREDICTORS:
                    covariance = parameters["covariance"][host]
                    noise_a = r0.process_noise(rng, covariance, count, r0.SAMPLES)
                    noise_b = r0.process_noise(rng, covariance, count, r0.SAMPLES)
                    centers = {
                        group: tuple(
                            r0.predict_centers(host, value, model, parameters, device)
                            for value in sampled[group]
                        )
                        for group in ("P2", "P3")
                    }
                    for scale in SCALES:
                        for group in ("P2", "P3"):
                            values = energy_score(
                                centers[group][0], centers[group][1], future,
                                noise_a, noise_b, scale,
                            )
                            storage[host][scale][group].append(values.astype(np.float32))

            for host in r0.PREDICTORS:
                for scale in SCALES:
                    for group in ("P2", "P3"):
                        key = f"{population}__{host}__cov_{scale_tag(scale)}__{group}__energy"
                        arrays[key] = np.concatenate(storage[host][scale][group])
                for group in ("P2", "P3"):
                    current = arrays[
                        f"{population}__{host}__cov_{scale_tag(1.0)}__{group}__energy"
                    ]
                    original = np.asarray(retained[retained_key(prefix, host, group)])
                    maximum = float(np.max(np.abs(current.astype(np.float64) - original.astype(np.float64))))
                    replay_errors[f"{population}__{host}__{group}"] = maximum
                    if maximum > 1e-6:
                        raise ValueError(f"1x replay mismatch: {population}/{host}/{group}: {maximum}")
        finally:
            retained.close()
            tracks.close()

    arrays["meta__max_abs_1x_replay"] = np.asarray(max(replay_errors.values()), dtype=np.float64)
    arrays["meta__replay_keys"] = np.asarray(sorted(replay_errors))
    arrays["meta__replay_max_abs"] = np.asarray([replay_errors[key] for key in sorted(replay_errors)])
    for name, value in arrays.items():
        if np.issubdtype(np.asarray(value).dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"non-finite output: {name}")
    atomic_npz(output_path, arrays)
    print(json.dumps({
        "evaluated": True,
        "replicate": replicate,
        "max_abs_1x_replay": max(replay_errors.values()),
        "output": str(output_path.relative_to(ROOT)),
    }))


def scene_means(values, scenes):
    unique, inverse = np.unique(scenes, return_inverse=True)
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64))
    counts = np.bincount(inverse)
    return unique, sums / counts


def bootstrap(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))
    means = values[draws].mean(1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "scenes": int(len(values)),
    }


def finalize():
    verify_protocol()
    result_path = OUT / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    paths = [OUT / "metrics" / f"replicate_{replicate}.npz" for replicate in range(3)]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError("all three sensitivity replicates are required")
    runs = [np.load(path, allow_pickle=False) for path in paths]
    effects = {}
    source_rows = []
    try:
        for population in POPULATIONS:
            reference_scene = np.asarray(runs[0][f"{population}__common__scene_token"])
            if not all(
                np.array_equal(reference_scene, run[f"{population}__common__scene_token"])
                for run in runs[1:]
            ):
                raise ValueError(f"replicate row mismatch: {population}")
            effects[population] = {}
            for host in r0.PREDICTORS:
                effects[population][host] = {}
                for scale in SCALES:
                    key = f"{population}__{host}__cov_{scale_tag(scale)}"
                    per_seed_scene = []
                    seed_effects = []
                    law_scene_means = {"G": [], "HTC": []}
                    common_scenes = None
                    for run in runs:
                        g = np.asarray(run[f"{key}__P2__energy"], dtype=np.float64)
                        htc = np.asarray(run[f"{key}__P3__energy"], dtype=np.float64)
                        scenes, effect = scene_means(g - htc, reference_scene)
                        if common_scenes is None:
                            common_scenes = scenes
                        elif not np.array_equal(common_scenes, scenes):
                            raise ValueError("scene mismatch after aggregation")
                        per_seed_scene.append(effect)
                        seed_effects.append(float(effect.mean()))
                        for name, values in (("G", g), ("HTC", htc)):
                            _scenes, current = scene_means(values, reference_scene)
                            law_scene_means[name].append(current)
                    paired = np.mean(np.stack(per_seed_scene), axis=0)
                    report = bootstrap(paired)
                    report["seed_effects"] = seed_effects
                    report["all_seed_directions_positive"] = all(value > 0 for value in seed_effects)
                    report["mean_es_g"] = float(np.mean(law_scene_means["G"]))
                    report["mean_es_htc"] = float(np.mean(law_scene_means["HTC"]))
                    effects[population][host][str(scale)] = report
                    source_rows.append({
                        "population": population,
                        "host": host,
                        "covariance_multiplier": scale,
                        "standard_deviation_multiplier": math.sqrt(scale),
                        "g_minus_htc_energy": report["mean"],
                        "ci95_lower": report["ci95"][0],
                        "ci95_upper": report["ci95"][1],
                        "mean_es_g": report["mean_es_g"],
                        "mean_es_htc": report["mean_es_htc"],
                        "scenes": report["scenes"],
                        "seed_0_effect": seed_effects[0],
                        "seed_1_effect": seed_effects[1],
                        "seed_2_effect": seed_effects[2],
                    })
        point_direction_stable = all(
            row["g_minus_htc_energy"] > 0 for row in source_rows
        )
        interval_supported_throughout = all(
            row["ci95_lower"] > 0 for row in source_rows
        )
        all_seed_directions_positive = all(
            row[f"seed_{index}_effect"] > 0
            for row in source_rows for index in range(3)
        )
        result = {
            "valid": True,
            "endpoint": (
                "PROCESS-NOISE SENSITIVITY SUPPORTED"
                if interval_supported_throughout
                else "PROCESS-NOISE DIRECTION STABLE BUT NOT INTERVAL-SUPPORTED THROUGHOUT"
                if point_direction_stable
                else "PROCESS-NOISE SENSITIVITY NOT SUPPORTED"
            ),
            "point_direction_stable": point_direction_stable,
            "interval_supported_throughout": interval_supported_throughout,
            "all_seed_directions_positive": all_seed_directions_positive,
            "effects": effects,
            "max_abs_1x_replay": float(max(run["meta__max_abs_1x_replay"] for run in runs)),
            "metric_sha256": {str(index): sha256(path) for index, path in enumerate(paths)},
            "protocol_sha256": sha256(OUT / "protocol.json"),
            "official_modern_host_outcome_read": False,
            "internal_test_read": False,
        }
        atomic_json(result_path, result)
        csv_path = OUT / "figure_source_data.csv"
        temporary = csv_path.with_suffix(csv_path.suffix + ".partial")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
            writer.writeheader()
            writer.writerows(source_rows)
        temporary.replace(csv_path)
    finally:
        for run in runs:
            run.close()
    print(json.dumps(result, indent=2))


def plot():
    verify_protocol()
    result = strict_json(OUT / "result.json")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
    colors = {"CV": "#4C78A8", "MLP": "#E07A5F"}
    titles = {
        "development_clean": "Development clean",
        "final_validation": "Final validation",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharex=True, sharey=True)
    for panel, (axis, population) in enumerate(zip(axes, POPULATIONS)):
        for host in r0.PREDICTORS:
            reports = [result["effects"][population][host][str(scale)] for scale in SCALES]
            y = np.asarray([report["mean"] for report in reports])
            lower = y - np.asarray([report["ci95"][0] for report in reports])
            upper = np.asarray([report["ci95"][1] for report in reports]) - y
            axis.errorbar(
                SCALES, y, yerr=np.stack((lower, upper)),
                color=colors[host], marker="o", markersize=4.2,
                linewidth=1.5, elinewidth=1.0, capsize=2.5, label=host,
            )
        axis.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
        axis.set_title(titles[population], fontsize=8, weight="bold", pad=6)
        axis.set_xticks(SCALES)
        axis.set_xlabel("Process-noise covariance multiplier")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
        axis.text(-0.13, 1.05, chr(ord("a") + panel), transform=axis.transAxes,
                  fontsize=9, fontweight="bold", va="top")
    axes[0].set_ylabel("Trajectory energy gain, G − HTC")
    axes[1].legend(loc="upper right", title="Forecasting probe", title_fontsize=7)
    fig.text(0.5, 0.995, "Positive values favor HTC; error bars are paired scene-bootstrap 95% CIs",
             ha="center", va="top", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=1.6)
    figure_root = OUT / "figure_process_noise_sensitivity"
    fig.savefig(figure_root.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(figure_root.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(figure_root.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(figure_root.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"plotted": True, "root": str(figure_root.relative_to(ROOT))}))


def self_check():
    rng = np.random.default_rng(11)
    a = rng.normal(size=(4, 64, 6, 2))
    b = rng.normal(size=(4, 64, 6, 2))
    future = rng.normal(size=(4, 6, 2))
    noise_a = rng.normal(size=a.shape)
    noise_b = rng.normal(size=b.shape)
    no_noise = energy_score(a, b, future, noise_a, noise_b, 0.0)
    direct = np.linalg.norm((a - future[:, None]).reshape(4, 64, -1), axis=-1).mean(1)
    direct -= 0.5 * np.linalg.norm((a - b).reshape(4, 64, -1), axis=-1).mean(1)
    if not np.array_equal(no_noise, direct):
        raise ValueError("zero-noise energy check failed")
    print(json.dumps({"self_check": True, "zero_noise_exact": True}))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("self-check")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--replicate", type=int, choices=range(3), required=True)
    evaluate_parser.add_argument("--device", type=int, choices=range(4), required=True)
    subparsers.add_parser("finalize")
    subparsers.add_parser("plot")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "self-check":
        self_check()
    elif args.command == "evaluate":
        evaluate(args.replicate, args.device)
    elif args.command == "finalize":
        finalize()
    elif args.command == "plot":
        plot()


if __name__ == "__main__":
    main()
