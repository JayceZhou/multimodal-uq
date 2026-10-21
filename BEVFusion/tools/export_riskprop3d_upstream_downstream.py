#!/usr/bin/env python3
"""Export paired upstream/downstream evidence for the RiskProp3D paper.

This is a read-only replay of the frozen Gaussian (G/M1) and
heavy-tailed/circular (HTC/M2) probability heads.  It never samples a new
forecast distribution: downstream energy scores are copied from the retained
64-sample R0/final-confirmation archives.

The requested ``eq90_volume_*`` names are retained for compatibility, but the
quantity is deliberately one common geometry: per-row XY ellipse area in m^2
after method-wise empirical rescaling to 90% coverage on the same condition,
seed, and matched rows.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import h5py
import numpy as np
import torch
from scipy.special import gammaln
from scipy.stats import chi2, f

from probset3d.f1.implementation.probabilistic_query import ProbabilisticQueryHead


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/riskprop3d_upstream_downstream_export"
CHECKPOINT_ROOT = ROOT / "outputs/probset3d_v3/f1_formal/checkpoints"
HEAD_SEEDS = (20260731, 20260801, 20260802)
CURRENT_OFFSET = 4 * 10
STATE_DIM = 8


# source_name: (population, paper condition, integer severity, track path,
#               metric archive group, metric key condition)
CONDITIONS = {
    "clean_s0": (
        "development", "clean", 0,
        ROOT / "outputs/riskprop3d_r0/tracks/clean_s0.npz", "development", "clean_s0",
    ),
    "lowlight_s1": (
        "development", "camera_lowlight", 1,
        ROOT / "outputs/riskprop3d_r0/tracks/lowlight_s1.npz", "development", "lowlight_s1",
    ),
    "lowlight_s2": (
        "development", "camera_lowlight", 2,
        ROOT / "outputs/riskprop3d_r0/tracks/lowlight_s2.npz", "development", "lowlight_s2",
    ),
    "lowlight_s3": (
        "development", "camera_lowlight", 3,
        ROOT / "outputs/riskprop3d_r0/tracks/lowlight_s3.npz", "development", "lowlight_s3",
    ),
    "dropout_s1": (
        "development", "lidar_point_dropout", 1,
        ROOT / "outputs/riskprop3d_r0/tracks/dropout_s1.npz", "development", "dropout_s1",
    ),
    "dropout_s2": (
        "development", "lidar_point_dropout", 2,
        ROOT / "outputs/riskprop3d_r0/tracks/dropout_s2.npz", "development", "dropout_s2",
    ),
    "dropout_s3": (
        "development", "lidar_point_dropout", 3,
        ROOT / "outputs/riskprop3d_r0/tracks/dropout_s3.npz", "development", "dropout_s3",
    ),
    "beam_s1": (
        "development", "lidar_beam_reduction", 1,
        ROOT / "outputs/riskprop3d_r0/tracks/beam_s1.npz", "development", "beam_s1",
    ),
    "beam_s2": (
        "development", "lidar_beam_reduction", 2,
        ROOT / "outputs/riskprop3d_r0/tracks/beam_s2.npz", "development", "beam_s2",
    ),
    "beam_s3": (
        "development", "lidar_beam_reduction", 3,
        ROOT / "outputs/riskprop3d_r0/tracks/beam_s3.npz", "development", "beam_s3",
    ),
    "rotation_s2": (
        "held_out_degradation", "camera_extrinsic_rotation", 2,
        ROOT / "outputs/riskprop3d_final_confirmation/tracks/external/rotation_s2.npz",
        "held_out_degradation", "rotation_s2",
    ),
    "rotation_s3": (
        "held_out_degradation", "camera_extrinsic_rotation", 3,
        ROOT / "outputs/riskprop3d_final_confirmation/tracks/external/rotation_s3.npz",
        "held_out_degradation", "rotation_s3",
    ),
    "translation_s3": (
        "held_out_degradation", "camera_extrinsic_translation", 3,
        ROOT / "outputs/riskprop3d_final_confirmation/tracks/external/translation_s3.npz",
        "held_out_degradation", "translation_s3",
    ),
    "temporal_s2": (
        "held_out_degradation", "camera_temporal_offset", 2,
        ROOT / "outputs/riskprop3d_final_confirmation/tracks/external/temporal_s2.npz",
        "held_out_degradation", "temporal_s2",
    ),
    "temporal_s3": (
        "held_out_degradation", "camera_temporal_offset", 3,
        ROOT / "outputs/riskprop3d_final_confirmation/tracks/external/temporal_s3.npz",
        "held_out_degradation", "temporal_s3",
    ),
    "official_clean": (
        "final_validation", "clean", 0,
        ROOT / "outputs/riskprop3d_final_confirmation/tracks/official/clean.npz",
        "final_validation", "official_clean",
    ),
}

METRIC_PATHS = {
    "development": ROOT / "outputs/riskprop3d_r0/metrics/replicate_{replicate}.npz",
    "held_out_degradation": ROOT / "outputs/riskprop3d_final_confirmation/metrics/external/replicate_{replicate}.npz",
    "final_validation": ROOT / "outputs/riskprop3d_final_confirmation/metrics/official/replicate_{replicate}.npz",
}

REQUIRED_COLUMNS = (
    "population", "condition", "severity", "head_seed", "row_id",
    "xy_nll_g", "xy_nll_htc",
    "linear90_inside_g", "linear90_inside_htc",
    "eq90_volume_g", "eq90_volume_htc",
    "downstream_es_g_cv", "downstream_es_htc_cv",
    "downstream_es_g_mlp", "downstream_es_htc_mlp",
)

EXTRA_COLUMNS = (
    "scene_token", "sample_token", "instance_token", "source_row",
    "xy_mahalanobis2_g", "xy_mahalanobis2_htc",
    "eq90_threshold_g", "eq90_threshold_htc",
)

FLOAT_COLUMNS = tuple(
    column for column in REQUIRED_COLUMNS + EXTRA_COLUMNS
    if column not in {
        "population", "condition", "severity", "head_seed", "row_id",
        "scene_token", "sample_token", "instance_token", "source_row",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Mapping) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_heads(seed: int) -> Dict[str, ProbabilisticQueryHead]:
    checkpoint = torch.load(
        CHECKPOINT_ROOT / f"seed_{seed}.pt", map_location="cpu", weights_only=True
    )
    heads = {name: ProbabilisticQueryHead().eval() for name in ("M1", "M2")}
    for name, head in heads.items():
        head.load_state_dict(checkpoint["models"][name])
    return heads


def head_outputs(
    head: ProbabilisticQueryHead, tracks: Mapping[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    storage = {
        name: []
        for name in (
            "mean_state", "scale", "degrees_of_freedom", "yaw_mean",
            "yaw_scale", "yaw_kappa",
        )
    }
    count = len(tracks["input"])
    for begin in range(0, count, 4096):
        stop = min(begin + 4096, count)
        batch = stop - begin
        with torch.inference_mode():
            output = head(
                torch.from_numpy(np.asarray(tracks["feature"][begin:stop]))[:, None],
                torch.from_numpy(np.asarray(tracks["logits"][begin:stop]))[:, None],
                torch.from_numpy(np.asarray(tracks["box"][begin:stop]))[:, None],
                torch.from_numpy(np.asarray(tracks["seed"][begin:stop]))[:, None],
                torch.zeros(batch, 1),
            )
        for name in storage:
            storage[name].append(output[name][:, 0].cpu().numpy())
    return {name: np.concatenate(values) for name, values in storage.items()}


def gaussian_xy_nll(target: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    delta = target[:, :2] - mean[:, :2]
    xy_scale = scale[:, :2]
    maha = np.square(delta / xy_scale).sum(axis=1)
    return (
        math.log(2.0 * math.pi)
        + np.log(xy_scale).sum(axis=1)
        + 0.5 * maha
    )


def student_xy_nll(
    target: np.ndarray, mean: np.ndarray, scale: np.ndarray, degrees: np.ndarray
) -> np.ndarray:
    delta = target[:, :2] - mean[:, :2]
    xy_scale = scale[:, :2]
    maha = np.square(delta / xy_scale).sum(axis=1)
    dimension = 2.0
    log_prob = (
        gammaln((degrees + dimension) / 2.0)
        - gammaln(degrees / 2.0)
        - 0.5 * dimension * np.log(degrees * math.pi)
        - np.log(xy_scale).sum(axis=1)
        - 0.5 * (degrees + dimension) * np.log1p(maha / degrees)
    )
    return -log_prob


def quantile90(values: np.ndarray) -> float:
    try:
        return float(np.quantile(values, 0.90, method="linear"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, 0.90, interpolation="linear"))


def metric_key(group: str, condition: str, host: str, law: str) -> str:
    p = "P2" if law == "g" else "P3"
    if group == "development":
        return f"{condition}__{host}__{p}__energy"
    return f"{condition}__r0__{host}__{p}__energy"


def common_key(condition: str, name: str) -> str:
    return f"{condition}__common__{name}"


def fixed_bytes(values: Iterable[str], width: int) -> np.ndarray:
    return np.asarray(list(values), dtype=f"S{width}")


def make_row_id(sample: np.ndarray, instance: np.ndarray) -> np.ndarray:
    result = np.char.add(np.char.add(sample.astype("U32"), ":"), instance.astype("U32"))
    if len(np.unique(result)) != len(result):
        raise ValueError("sample_token:instance_token is not unique within condition")
    return result.astype("S65")


def compute_arrays(
    population: str,
    condition: str,
    severity: int,
    seed: int,
    tracks: Mapping[str, np.ndarray],
    metrics: Mapping[str, np.ndarray],
    metric_group: str,
    metric_condition: str,
    outputs: Mapping[str, Mapping[str, np.ndarray]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    sample = np.asarray(tracks["sample_token"])
    instance = np.asarray(tracks["instance_token"])
    scene = np.asarray(tracks["scene_token"])
    count = len(sample)
    for name, source in (("sample_token", sample), ("instance_token", instance), ("scene_token", scene)):
        retained = np.asarray(metrics[common_key(metric_condition, name)])
        if not np.array_equal(source, retained):
            raise ValueError(f"row mismatch for {population}/{metric_condition}/{name}")

    g = outputs["M1"]
    htc = outputs["M2"]
    if not np.array_equal(g["mean_state"], htc["mean_state"]):
        raise ValueError("G and HTC do not share the exact decoded mean")
    detector_state = np.asarray(tracks["input"][:, CURRENT_OFFSET : CURRENT_OFFSET + STATE_DIM])
    if not np.allclose(g["mean_state"], detector_state, rtol=0.0, atol=2e-5):
        raise ValueError("probability-head mean does not match the frozen detector-state input")

    target = np.asarray(
        tracks["gt_input"][:, CURRENT_OFFSET : CURRENT_OFFSET + STATE_DIM],
        dtype=np.float64,
    )
    mean = np.asarray(g["mean_state"], dtype=np.float64)
    scale_g = np.asarray(g["scale"], dtype=np.float64)
    scale_htc = np.asarray(htc["scale"], dtype=np.float64)
    degrees = np.asarray(htc["degrees_of_freedom"], dtype=np.float64)
    delta = target - mean
    maha8_g = np.square(delta / scale_g).sum(axis=1)
    maha8_htc = np.square(delta / scale_htc).sum(axis=1)
    maha2_g = np.square(delta[:, :2] / scale_g[:, :2]).sum(axis=1)
    maha2_htc = np.square(delta[:, :2] / scale_htc[:, :2]).sum(axis=1)

    q_g = quantile90(maha2_g)
    q_htc = quantile90(maha2_htc)
    # Ellipse {(x-mu)' diag(scale^-2) (x-mu) <= q}; area = pi*q*sx*sy.
    area_g = math.pi * q_g * scale_g[:, 0] * scale_g[:, 1]
    area_htc = math.pi * q_htc * scale_htc[:, 0] * scale_htc[:, 1]
    inside_g = maha8_g <= chi2.ppf(0.90, STATE_DIM)
    inside_htc = maha8_htc <= STATE_DIM * f.ppf(0.90, STATE_DIM, degrees)

    arrays: Dict[str, np.ndarray] = {
        "population": np.full(count, population.encode("utf-8"), dtype="S24"),
        "condition": np.full(count, condition.encode("utf-8"), dtype="S32"),
        "severity": np.full(count, severity, dtype=np.int8),
        "head_seed": np.full(count, seed, dtype=np.int64),
        "row_id": make_row_id(sample, instance),
        "xy_nll_g": gaussian_xy_nll(target, mean, scale_g).astype(np.float32),
        "xy_nll_htc": student_xy_nll(target, mean, scale_htc, degrees).astype(np.float32),
        "linear90_inside_g": inside_g.astype(np.uint8),
        "linear90_inside_htc": inside_htc.astype(np.uint8),
        "eq90_volume_g": area_g.astype(np.float32),
        "eq90_volume_htc": area_htc.astype(np.float32),
        "downstream_es_g_cv": np.asarray(metrics[metric_key(metric_group, metric_condition, "CV", "g")], dtype=np.float32),
        "downstream_es_htc_cv": np.asarray(metrics[metric_key(metric_group, metric_condition, "CV", "htc")], dtype=np.float32),
        "downstream_es_g_mlp": np.asarray(metrics[metric_key(metric_group, metric_condition, "MLP", "g")], dtype=np.float32),
        "downstream_es_htc_mlp": np.asarray(metrics[metric_key(metric_group, metric_condition, "MLP", "htc")], dtype=np.float32),
        "scene_token": scene.astype("S32"),
        "sample_token": sample.astype("S32"),
        "instance_token": instance.astype("S32"),
        "source_row": np.arange(count, dtype=np.int64),
        "xy_mahalanobis2_g": maha2_g.astype(np.float32),
        "xy_mahalanobis2_htc": maha2_htc.astype(np.float32),
        "eq90_threshold_g": np.full(count, q_g, dtype=np.float32),
        "eq90_threshold_htc": np.full(count, q_htc, dtype=np.float32),
    }
    for name in REQUIRED_COLUMNS + EXTRA_COLUMNS:
        if len(arrays[name]) != count:
            raise ValueError(f"wrong row count for {name}")
    for name in FLOAT_COLUMNS:
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"non-finite values in {name}")

    summary = {
        "population": population,
        "condition": condition,
        "severity": severity,
        "head_seed": seed,
        "rows": count,
        "xy_nll_g_mean": float(np.mean(arrays["xy_nll_g"])),
        "xy_nll_htc_mean": float(np.mean(arrays["xy_nll_htc"])),
        "xy_nll_gain_htc": float(np.mean(arrays["xy_nll_g"] - arrays["xy_nll_htc"])),
        "linear90_coverage_g": float(np.mean(inside_g)),
        "linear90_coverage_htc": float(np.mean(inside_htc)),
        "eq90_xy_area_g_median": float(np.median(area_g)),
        "eq90_xy_area_htc_median": float(np.median(area_htc)),
        "eq90_xy_area_ratio_htc_g": float(np.median(area_htc) / np.median(area_g)),
        "downstream_es_g_cv_mean": float(np.mean(arrays["downstream_es_g_cv"])),
        "downstream_es_htc_cv_mean": float(np.mean(arrays["downstream_es_htc_cv"])),
        "downstream_es_gain_htc_cv": float(np.mean(arrays["downstream_es_g_cv"] - arrays["downstream_es_htc_cv"])),
        "downstream_es_g_mlp_mean": float(np.mean(arrays["downstream_es_g_mlp"])),
        "downstream_es_htc_mlp_mean": float(np.mean(arrays["downstream_es_htc_mlp"])),
        "downstream_es_gain_htc_mlp": float(np.mean(arrays["downstream_es_g_mlp"] - arrays["downstream_es_htc_mlp"])),
        "eq90_threshold_g": q_g,
        "eq90_threshold_htc": q_htc,
    }
    return arrays, summary


def create_h5(path: Path, total_rows: int) -> h5py.File:
    temporary = path.with_suffix(path.suffix + ".partial")
    handle = h5py.File(temporary, "w")
    handle.attrs["schema_version"] = "riskprop3d-upstream-downstream-v1"
    handle.attrs["rows"] = total_rows
    for name in REQUIRED_COLUMNS + EXTRA_COLUMNS:
        if name == "population":
            dtype = "S24"
        elif name == "condition":
            dtype = "S32"
        elif name == "row_id":
            dtype = "S65"
        elif name in {"scene_token", "sample_token", "instance_token"}:
            dtype = "S32"
        elif name == "severity":
            dtype = np.int8
        elif name in {"head_seed", "source_row"}:
            dtype = np.int64
        elif name.startswith("linear90_inside"):
            dtype = np.uint8
        else:
            dtype = np.float32
        handle.create_dataset(
            name, shape=(total_rows,), dtype=dtype,
            chunks=(min(65536, total_rows),), compression="gzip", compression_opts=4,
        )
    return handle


def csv_value(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="replace a prior derived export")
    parser.add_argument("--skip-csv", action="store_true", help="omit the portable CSV.gz copy")
    args = parser.parse_args()

    if OUT.exists() and any(OUT.iterdir()) and not args.overwrite:
        raise FileExistsError(f"derived export already exists: {OUT}; pass --overwrite to replace it")
    OUT.mkdir(parents=True, exist_ok=True)
    shard_root = OUT / "shards"
    shard_root.mkdir(exist_ok=True)

    total_rows = 3 * sum(len(np.load(spec[3], allow_pickle=False)["input"]) for spec in CONDITIONS.values())
    h5_path = OUT / "paired_rows.h5"
    h5_partial = h5_path.with_suffix(h5_path.suffix + ".partial")
    if h5_partial.exists():
        h5_partial.unlink()
    h5 = create_h5(h5_path, total_rows)
    csv_path = OUT / "paired_rows.csv.gz"
    csv_partial = csv_path.with_suffix(csv_path.suffix + ".partial")
    if csv_partial.exists():
        csv_partial.unlink()
    csv_stream = None
    csv_writer = None
    columns = REQUIRED_COLUMNS + EXTRA_COLUMNS
    if not args.skip_csv:
        csv_stream = gzip.open(csv_partial, "wt", encoding="utf-8", newline="", compresslevel=6)
        csv_writer = csv.writer(csv_stream, lineterminator="\n")
        csv_writer.writerow(columns)

    source_hashes = {}
    records = []
    summaries = []
    cursor = 0
    try:
        for replicate, seed in enumerate(HEAD_SEEDS):
            print(json.dumps({"event": "load_heads", "replicate": replicate, "head_seed": seed}), flush=True)
            heads = load_heads(seed)
            metric_archives = {
                group: np.load(Path(str(template).format(replicate=replicate)), allow_pickle=False)
                for group, template in METRIC_PATHS.items()
            }
            try:
                for group, archive in metric_archives.items():
                    retained_seed = int(np.asarray(archive["meta__head_seed"]).item())
                    if retained_seed != seed:
                        raise ValueError(
                            f"head-seed mismatch in {group}: {retained_seed} != {seed}"
                        )
                for source_name, spec in CONDITIONS.items():
                    population, condition, severity, track_path, metric_group, metric_condition = spec
                    print(json.dumps({
                        "event": "condition", "population": population,
                        "condition": condition, "severity": severity, "head_seed": seed,
                    }), flush=True)
                    tracks = np.load(track_path, allow_pickle=False)
                    try:
                        outputs = {name: head_outputs(head, tracks) for name, head in heads.items()}
                        arrays, summary = compute_arrays(
                            population, condition, severity, seed, tracks,
                            metric_archives[metric_group], metric_group, metric_condition, outputs,
                        )
                    finally:
                        tracks.close()

                    count = len(arrays["row_id"])
                    stop = cursor + count
                    for name in columns:
                        h5[name][cursor:stop] = arrays[name]
                    if csv_writer is not None:
                        for row in zip(*(arrays[name] for name in columns)):
                            csv_writer.writerow(tuple(csv_value(value) for value in row))

                    shard_path = shard_root / f"{population}__{source_name}__seed_{seed}.npz"
                    if shard_path.exists() and args.overwrite:
                        shard_path.unlink()
                    write_npz_atomic(shard_path, arrays)
                    record = {
                        "population": population,
                        "condition": condition,
                        "severity": severity,
                        "head_seed": seed,
                        "rows": count,
                        "h5_begin": cursor,
                        "h5_stop": stop,
                        "source_name": source_name,
                        "track_source": str(track_path.relative_to(ROOT)),
                        "metric_source": str(Path(str(METRIC_PATHS[metric_group]).format(replicate=replicate)).relative_to(ROOT)),
                        "shard": str(shard_path.relative_to(ROOT)),
                        "shard_sha256": sha256(shard_path),
                        "eq90_threshold_g": summary["eq90_threshold_g"],
                        "eq90_threshold_htc": summary["eq90_threshold_htc"],
                    }
                    records.append(record)
                    summaries.append(summary)
                    cursor = stop
            finally:
                for archive in metric_archives.values():
                    archive.close()
        if cursor != total_rows:
            raise ValueError(f"expected {total_rows} rows, wrote {cursor}")
    except BaseException:
        h5.close()
        if csv_stream is not None:
            csv_stream.close()
        raise
    else:
        h5.close()
        h5_partial.replace(h5_path)
        if csv_stream is not None:
            csv_stream.close()
            csv_partial.replace(csv_path)

    summary_path = OUT / "condition_seed_summary.csv"
    summary_partial = summary_path.with_suffix(summary_path.suffix + ".partial")
    with summary_partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    summary_partial.replace(summary_path)

    source_paths = set(CHECKPOINT_ROOT / f"seed_{seed}.pt" for seed in HEAD_SEEDS)
    source_paths.update(spec[3] for spec in CONDITIONS.values())
    source_paths.update(
        Path(str(template).format(replicate=replicate))
        for template in METRIC_PATHS.values() for replicate in range(3)
    )
    for path in sorted(source_paths):
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)

    schema = {
        "schema_version": "riskprop3d-upstream-downstream-v1",
        "pairing_key": ["population", "condition", "severity", "head_seed", "row_id"],
        "required_columns": list(REQUIRED_COLUMNS),
        "extra_traceability_columns": list(EXTRA_COLUMNS),
        "definitions": {
            "row_id": "sample_token:instance_token; unique within each population/condition/severity",
            "xy_nll_g": "uncalibrated 2-D diagonal Gaussian XY NLL from the frozen G/M1 head",
            "xy_nll_htc": "uncalibrated 2-D multivariate Student-t XY NLL from the frozen HTC/M2 head",
            "linear90_inside_g": "natural uncalibrated 90% inclusion for the complete 8-D linear state using chi-square(df=8)",
            "linear90_inside_htc": "natural uncalibrated 90% inclusion for the complete 8-D linear state using 8*F(df1=8,df2=nu)",
            "eq90_volume_g": "XY ellipse area in m^2 at empirical 90% coverage; legacy field name retained",
            "eq90_volume_htc": "XY ellipse area in m^2 at empirical 90% coverage; legacy field name retained",
            "eq90_threshold_*": "empirical 0.90 quantile of per-row XY squared standardized residual on the identical condition/seed rows",
            "downstream_es_*": "retained 64-sample R0 trajectory energy score; P2=G and P3=HTC",
        },
        "equal_coverage_geometry": "area = pi * q90 * scale_x * scale_y",
        "upstream_calibration": "none for NLL and natural coverage; empirical rescaling is used only to define the equal-coverage sharpness diagnostic",
        "downstream_contract": "existing frozen CV/MLP, 64 samples, shared process-noise draws, and common random numbers; no replay or retraining",
    }
    write_json_atomic(OUT / "schema.json", schema)

    manifest = {
        "schema_version": schema["schema_version"],
        "rows": total_rows,
        "shards": len(records),
        "conditions": len(CONDITIONS),
        "head_seeds": list(HEAD_SEEDS),
        "records": records,
        "source_sha256": source_hashes,
        "outputs": {
            "paired_rows_h5": {"path": str(h5_path.relative_to(ROOT)), "sha256": sha256(h5_path)},
            "condition_seed_summary_csv": {"path": str(summary_path.relative_to(ROOT)), "sha256": sha256(summary_path)},
            "paired_rows_csv_gz": None if args.skip_csv else {"path": str(csv_path.relative_to(ROOT)), "sha256": sha256(csv_path)},
        },
        "audit": {
            "same_track_rows_asserted": True,
            "same_g_htc_mean_asserted": True,
            "detector_mean_match_asserted": True,
            "required_numeric_values_finite": True,
            "models_retrained": False,
            "conditions_selected_from_results": False,
        },
    }
    write_json_atomic(OUT / "manifest.json", manifest)
    print(json.dumps({
        "complete": True, "rows": total_rows, "shards": len(records),
        "output": str(OUT.relative_to(ROOT)),
    }), flush=True)


if __name__ == "__main__":
    main()
