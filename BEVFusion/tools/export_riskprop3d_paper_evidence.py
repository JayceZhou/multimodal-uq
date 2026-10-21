#!/usr/bin/env python3
"""Export frozen RiskProp3D preservation and attribute evidence for the paper.

This is a read-only re-expression of retained metric archives.  It does not
load a trainable model, sample a trajectory, or access an unread split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import riskprop3d_final_confirmation as final  # noqa: E402
import riskprop3d_r1 as r1  # noqa: E402
import riskprop3d_r2d as r2d  # noqa: E402


OUT = ROOT / "riskprop3d_paper" / "evidence_exports"
R2D_OUT = ROOT / "outputs" / "riskprop3d_r2d"
FINAL_OUT = ROOT / "outputs" / "riskprop3d_final_confirmation"
BOOTSTRAP_SEED = 20260805
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_WEIGHT_CACHE: dict[int, np.ndarray] = {}
EFFECT_MAP_CACHE: dict[tuple, dict[str, float]] = {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.chmod(temporary, 0o644)
    os.replace(temporary, path)


def load_npz_runs(directory: Path) -> list[dict[str, np.ndarray]]:
    runs = []
    for replicate in range(3):
        with np.load(directory / f"replicate_{replicate}.npz", allow_pickle=False) as source:
            runs.append({key: source[key] for key in source.files})
    return runs


def mean_of_map(value: dict[str, float]) -> float | None:
    return float(np.mean(list(value.values()))) if value else None


def scene_metric_maps(
    runs: list[dict[str, np.ndarray]],
    condition: str,
    group: str,
    metric: str,
) -> list[dict[str, float]]:
    return [
        r1.scene_mean_map(
            run[f"{condition}__{group}__{metric}"],
            run[f"{condition}__common__scene_token"],
        )
        for run in runs
    ]


def export_fidelity(r2d_runs: list[dict[str, np.ndarray]], r2d_result: dict) -> tuple[list[dict], dict]:
    definitions = {
        "teacher_energy_distance": {
            "definition": "2*mean(||S-T||_2)-mean(||S-S'||_2)-mean(||T-T'||_2), using independent 64-sample student and teacher trajectory sets flattened over six XY horizons",
            "unit": "m (Euclidean norm of the 12-D trajectory vector)",
        },
        "first_moment_error": {
            "definition": "mean_h ||E_q[Y_h]-mean_m(T_m,h)||_2",
            "unit": "m",
        },
        "second_moment_error": {
            "definition": "mean_h ||Cov_q(Y_h)-Cov_T(Y_h)||_F / max(||Cov_T(Y_h)||_F,1e-6)",
            "unit": "dimensionless relative Frobenius error",
        },
        "coverage90_gap": {
            "definition": "per-track |I(student 90% final ellipse contains GT)-I(teacher 90% final ellipse contains GT)|, averaged by scene; this is coverage-status disagreement, not the absolute difference between two aggregate coverage rates",
            "unit": "proportion",
        },
    }
    candidate = "D3_risk_distilled preservation student"
    baseline = "D2_gt_only matched student"
    rows = []
    checks = {}
    clean = "clean_s0"
    n_tracks = len(r2d_runs[0][f"{clean}__common__scene_token"])
    n_scenes = len(np.unique(r2d_runs[0][f"{clean}__common__scene_token"]))
    for metric, metadata in definitions.items():
        candidate_maps = scene_metric_maps(r2d_runs, clean, "D3_risk_distilled", metric)
        baseline_maps = scene_metric_maps(r2d_runs, clean, "D2_gt_only", metric)
        difference_maps = [
            {key: baseline_map[key] - candidate_map[key] for key in baseline_map}
            for baseline_map, candidate_map in zip(baseline_maps, candidate_maps)
        ]
        report = r1.bootstrap(
            r1.average_maps(difference_maps),
            seed=BOOTSTRAP_SEED,
            repeats=BOOTSTRAP_REPEATS,
        )
        frozen = r2d_result["teacher_fidelity_D3_over_D2"][metric]
        checks[metric] = bool(
            abs(report["mean"] - frozen["mean"]) < 1e-12
            and np.allclose(report["ci95"], frozen["ci95"], rtol=0, atol=1e-12)
            and report["count"] == frozen["count"]
        )
        rows.append(
            {
                "population": "development_clean",
                "population_role": "pre-specified development fidelity endpoint",
                "metric": metric,
                "metric_definition": metadata["definition"],
                "candidate": candidate,
                "baseline": baseline,
                "candidate_value": float(r1.average_maps(candidate_maps).mean()),
                "baseline_value": float(r1.average_maps(baseline_maps).mean()),
                "effect_definition": "baseline - candidate; positive means preservation student is closer to the MC teacher",
                "effect": report["mean"],
                "ci_low": report["ci95"][0],
                "ci_high": report["ci95"][1],
                "seed_0": mean_of_map(difference_maps[0]),
                "seed_1": mean_of_map(difference_maps[1]),
                "seed_2": mean_of_map(difference_maps[2]),
                "unit": metadata["unit"],
                "n_scenes": n_scenes,
                "n_tracks": n_tracks,
                "status": "evaluated",
                "missing_reason": "",
            }
        )
        rows.append(
            {
                "population": "final_validation_clean",
                "population_role": "held-out final validation population",
                "metric": metric,
                "metric_definition": metadata["definition"],
                "candidate": candidate,
                "baseline": baseline,
                "candidate_value": "",
                "baseline_value": "",
                "effect_definition": "baseline - candidate; positive means preservation student is closer to the MC teacher",
                "effect": "",
                "ci_low": "",
                "ci_high": "",
                "seed_0": "",
                "seed_1": "",
                "seed_2": "",
                "unit": metadata["unit"],
                "n_scenes": 150,
                "n_tracks": 68436,
                "status": "not_evaluated",
                "missing_reason": "The frozen official archive retains downstream D1/D2/D3 metrics but not teacher/student sample sets or moment-fidelity arrays; no post-hoc model rerun was performed.",
            }
        )
    fields = [
        "population", "population_role", "metric", "metric_definition", "candidate", "baseline",
        "candidate_value", "baseline_value", "effect_definition", "effect", "ci_low", "ci_high",
        "seed_0", "seed_1", "seed_2", "unit", "n_scenes", "n_tracks", "status", "missing_reason",
    ]
    atomic_csv(OUT / "preservation_fidelity_summary.csv", rows, fields)
    return rows, checks


def averaged_scope_map(run: dict[str, np.ndarray], conditions: tuple[str, ...], prefix, group: str) -> dict[str, float]:
    maps = [
        r1.scene_mean_map(
            run[prefix(condition, group)],
            run[prefix(condition, "common_scene")],
        )
        for condition in conditions
    ]
    return final.combine_condition_maps(maps)


def equivalence_record(
    runs: list[dict[str, np.ndarray]],
    population: str,
    scope: str,
    conditions: tuple[str, ...],
    prefix,
    n_tracks: int,
    confirmatory_role: str,
) -> dict:
    differences = []
    teachers = []
    for run in runs:
        teacher = averaged_scope_map(run, conditions, prefix, "D1_mc_teacher")
        student = averaged_scope_map(run, conditions, prefix, "D3_risk_distilled")
        keys = sorted(set(teacher) & set(student))
        differences.append({key: student[key] - teacher[key] for key in keys})
        teachers.append(teacher)
    difference_values = r1.average_maps(differences)
    teacher_values = r1.average_maps(teachers)
    report = r2d.bootstrap_intervals(difference_values)
    teacher_mean = float(teacher_values.mean())
    margin = 0.02 * teacher_mean
    equivalent = bool(report["ci90"][0] > -margin and report["ci90"][1] < margin)
    superior = bool(report["ci95"][1] < 0)
    if superior:
        decision = "pass_by_direct_superiority"
    elif equivalent:
        decision = "pass_by_equivalence"
    else:
        decision = "fail"
    return {
        "population": population,
        "scope": scope,
        "conditions": list(conditions),
        "candidate": "D3_risk_distilled preservation student",
        "reference": "D1_mc_teacher 64-sample stochastic pushforward",
        "metric": "trajectory_energy",
        "estimand": "scene-weighted, replicate-averaged ES(candidate)-ES(reference); conditions are averaged within scene before replicates are averaged",
        "effect_definition": "candidate - reference; negative favors the preservation student",
        "estimate": report["mean"],
        "ci_level": 0.90,
        "ci_low": report["ci90"][0],
        "ci_high": report["ci90"][1],
        "superiority_ci_level": 0.95,
        "superiority_ci_low": report["ci95"][0],
        "superiority_ci_high": report["ci95"][1],
        "reference_mean_energy": teacher_mean,
        "equivalence_margin_low": -margin,
        "equivalence_margin_high": margin,
        "margin_fraction": 0.02,
        "margin_type": "relative to the scene-weighted mean reference energy, converted to absolute energy-score bounds",
        "tost_rule": "equivalent only if the entire two-sided 90% bootstrap CI lies strictly inside [-0.02*mean(reference ES), +0.02*mean(reference ES)]",
        "tost_equivalence_decision": "pass" if equivalent else "fail",
        "direct_superiority_rule": "direct superiority if the upper bound of the paired 95% CI for candidate-reference is below zero",
        "direct_superiority": superior,
        "original_gate_rule": "pass by either direct superiority or TOST equivalence",
        "decision": decision,
        "pass": bool(superior or equivalent),
        "seed_0": mean_of_map(differences[0]),
        "seed_1": mean_of_map(differences[1]),
        "seed_2": mean_of_map(differences[2]),
        "replicate_combination": "Per replicate: target differences averaged within scene, then across listed conditions. Across replicates: matched scene effects averaged. CI: 10,000 paired scene bootstrap resamples with seed 20260805.",
        "n_scenes": report["count"],
        "n_tracks_per_replicate_across_conditions": n_tracks,
        "confirmatory_role": confirmatory_role,
    }


def export_equivalence(
    r2d_runs: list[dict[str, np.ndarray]],
    official_runs: list[dict[str, np.ndarray]],
    r2d_result: dict,
    official_result: dict,
) -> tuple[list[dict], dict]:
    dev_prefix = lambda condition, group: (
        f"{condition}__common__scene_token" if group == "common_scene" else f"{condition}__{group}__energy"
    )
    official_prefix = lambda condition, group: (
        "official_clean__common__scene_token" if group == "common_scene" else f"official_clean__r2d__{group}__energy"
    )
    track_counts = r2d_result["track_counts"]
    tests = [
        equivalence_record(
            r2d_runs, "development", "clean", ("clean_s0",), dev_prefix,
            track_counts["clean_s0"], "pre-specified confirmatory gate",
        )
    ]
    for family in ("lowlight", "dropout", "beam"):
        full = tuple(f"{family}_s{severity}" for severity in (1, 2, 3))
        tests.append(
            equivalence_record(
                r2d_runs, "development", f"{family}_full_S1-S3", full, dev_prefix,
                sum(track_counts[value] for value in full), "pre-specified confirmatory family gate",
            )
        )
        without_s3 = tuple(f"{family}_s{severity}" for severity in (1, 2))
        tests.append(
            equivalence_record(
                r2d_runs, "development", f"{family}_S1-S2", without_s3, dev_prefix,
                sum(track_counts[value] for value in without_s3), "pre-specified secondary report; not gated",
            )
        )
    tests.append(
        equivalence_record(
            official_runs, "final_validation", "clean", ("official_clean",), official_prefix,
            68436, "post-freeze re-expression of the frozen official per-track energy archive; not used for selection",
        )
    )

    checks = {}
    dev_lookup = {("clean", None): r2d_result["mc_equivalence_D3_vs_D1"]["clean"]}
    for family in ("lowlight", "dropout", "beam"):
        dev_lookup[(f"{family}_full_S1-S3", None)] = r2d_result["mc_equivalence_D3_vs_D1"]["full"][family]
        dev_lookup[(f"{family}_S1-S2", None)] = r2d_result["mc_equivalence_D3_vs_D1"]["without_s3"][family]
    for test in tests[:-1]:
        frozen = dev_lookup[(test["scope"], None)]
        checks[f"development/{test['scope']}"] = bool(
            abs(test["estimate"] - frozen["mean"]) < 1e-12
            and np.allclose([test["ci_low"], test["ci_high"]], frozen["ci90"], rtol=0, atol=1e-12)
            and np.allclose([test["superiority_ci_low"], test["superiority_ci_high"]], frozen["ci95"], rtol=0, atol=1e-12)
            and abs(test["equivalence_margin_high"] - frozen["equivalence_margin"]) < 1e-12
            and test["tost_equivalence_decision"] == ("pass" if frozen["tost_equivalent"] else "fail")
        )
    official = tests[-1]
    frozen_official = official_result["results"]["r2d"]["D1_mc_teacher"]["metrics"]["energy"]
    checks["final_validation/clean"] = bool(
        abs(official["estimate"] + frozen_official["mean"]) < 1e-12
        and np.allclose(
            [official["superiority_ci_low"], official["superiority_ci_high"]],
            [-frozen_official["ci95"][1], -frozen_official["ci95"][0]],
            rtol=0,
            atol=1e-12,
        )
    )
    payload = {
        "status": "FROZEN_ARCHIVE_REEXPRESSION_NO_MODEL_EXECUTION",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "tests": tests,
        "interpretation_note": "TOST equivalence and the original gate are distinct. A test can fail equivalence yet pass the original gate by direct superiority; that is what occurs here.",
    }
    atomic_json(OUT / "preservation_equivalence.json", payload)
    return tests, checks


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def current_cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unavailable"


def current_memory_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return float(line.split()[1]) / 1024.0 / 1024.0
    except OSError:
        pass
    return None


def export_runtime(r2d_result: dict) -> tuple[list[dict], dict]:
    teacher = [float(value) for value in r2d_result["efficiency"]["D1_seconds"]]
    student = [float(value) for value in r2d_result["efficiency"]["D3_seconds"]]
    speedup = [float(value) for value in r2d_result["efficiency"]["speedup_by_replicate"]]
    tracks = int(r2d_result["track_counts"]["clean_s0"])
    batches = math.ceil(tracks / r2d.EVAL_CHUNK)
    seeds = [20260827, 20260828, 20260829]
    rows = []
    for replicate in range(3):
        rows.append(
            {
                "record_type": "replicate",
                "statistic": "observed",
                "replicate": replicate,
                "student_sample_seed": seeds[replicate],
                "n_tracks": tracks,
                "n_timed_batches": batches,
                "max_targets_per_batch": r2d.EVAL_CHUNK,
                "teacher_total_seconds": teacher[replicate],
                "student_total_seconds": student[replicate],
                "teacher_ms_per_target": 1000.0 * teacher[replicate] / tracks,
                "student_ms_per_target": 1000.0 * student[replicate] / tracks,
                "paired_speedup": speedup[replicate],
            }
        )
    statistics_rows = {
        "mean": lambda x: statistics.mean(x),
        "sample_std": lambda x: statistics.stdev(x),
        "median": lambda x: statistics.median(x),
        "q1": lambda x: quantile(x, 0.25),
        "q3": lambda x: quantile(x, 0.75),
    }
    for name, function in statistics_rows.items():
        rows.append(
            {
                "record_type": "aggregate_across_three_replicates",
                "statistic": name,
                "replicate": "",
                "student_sample_seed": "",
                "n_tracks": tracks,
                "n_timed_batches": batches,
                "max_targets_per_batch": r2d.EVAL_CHUNK,
                "teacher_total_seconds": function(teacher),
                "student_total_seconds": function(student),
                "teacher_ms_per_target": 1000.0 * function(teacher) / tracks,
                "student_ms_per_target": 1000.0 * function(student) / tracks,
                "paired_speedup": function(speedup),
            }
        )
    fields = list(rows[0])
    atomic_csv(OUT / "runtime_measurements.csv", rows, fields)

    current_gpu = "unavailable in the export session"
    if torch.cuda.is_available():
        current_gpu = torch.cuda.get_device_name(0)
    memory = current_memory_gib()
    protocol = f"""# Preservation runtime protocol

## Frozen timing evidence

The retained timing endpoint is the clean development population: {tracks:,} targets, processed in {batches} timed chunks per replicate with at most {r2d.EVAL_CHUNK} targets per chunk. Three paired replicates were recorded. The reported `38.55x` value is

```text
mean_r(teacher_total_seconds_r / student_total_seconds_r)
= {r2d_result['efficiency']['speedup_mean']:.12f}x.
```

It is not the ratio of two rounded means. Per-replicate totals and derived per-target latencies are in `runtime_measurements.csv`.

## Timed regions

- **Teacher (`D1_mc_teacher`)**: synchronize CUDA; start `time.perf_counter`; construct NumPy random state draws; sample 64 HTC current states; replace the target current state; execute the frozen MLP on all 64 states; construct the calibrated MLP process noise; synchronize CUDA; stop the clock.
- **Student (`D3_risk_distilled`)**: synchronize CUDA; start `time.perf_counter`; construct the raw probability token; construct and transfer motion/token tensors to the device; execute one student forward pass under `torch.inference_mode`; copy the output arrays to CPU as part of `numpy_output`; synchronize CUDA; stop the clock.
- Downstream mixture sampling, trajectory metrics, teacher-fidelity metrics, and bootstrap inference are excluded from both timed regions.
- The upstream probabilistic-head forward pass is excluded from both methods; timing starts from retained HTC parameters.

## Warm-up and repetition

Before the first timed clean chunk in each replicate, the evaluator runs one unrecorded forward pass for each learned student and one unrecorded teacher chunk. Formal timing then covers every one of the {batches} clean evaluation chunks once. These are full-population accumulated timings, not repeated timing of one synthetic batch. CUDA synchronization occurs immediately before and after every recorded region.

## Precision and execution mode

No autocast or AMP context is used. PyTorch model parameters and transferred model inputs are float32. NumPy Student-t/von-Mises state generation and process-noise construction use float64 intermediates; sampled states are converted to float32 when inserted into the MLP input. The measured student token is float32.

## Hardware and software provenance boundary

The original R2-D runtime archive records assigned CUDA devices (`cuda:0`, `cuda:1`, and `cuda:2`), code/checkpoint hashes, seeds, target counts, and elapsed seconds, but **does not serialize the GPU model, driver, CPU model, RAM, or package lockfile**. Those fields cannot be reconstructed as frozen facts and must not be filled from a later machine snapshot.

For transparency only, the retained `bev` environment at export time reports Python {platform.python_version()}, PyTorch {torch.__version__}, CUDA build {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, CPU `{current_cpu_model()}`, approximately {memory:.1f} GiB RAM, and GPU `{current_gpu}`. These current values are not asserted to be the original timing host. Consequently, `38.55x` is reproducible from the frozen measurements and code scope, but an exact hardware-normalized latency reproduction requires rerunning the unchanged benchmark with hardware metadata capture.
"""
    atomic_text(OUT / "runtime_protocol.md", protocol)
    checks = {
        "mean_paired_speedup_matches_frozen_result": abs(statistics.mean(speedup) - r2d_result["efficiency"]["speedup_mean"]) < 1e-12,
        "replicate_count": 3,
        "timed_batches_per_replicate": batches,
        "original_hardware_metadata_complete": False,
    }
    return rows, checks


def condition_metadata(scope: str, conditions: tuple[str, ...]) -> dict:
    if len(conditions) == 1:
        condition = conditions[0]
        if condition == "clean_s0":
            return {"condition": "clean", "severity": "S0", "scope_type": "individual_condition"}
        family, severity = condition.rsplit("_s", 1)
        return {"condition": family, "severity": f"S{severity}", "scope_type": "individual_condition"}
    if scope == "all_original_degradations":
        return {"condition": "all_original_degradations", "severity": "pooled_S1-S3", "scope_type": "precomputed_pooled_scope"}
    return {"condition": scope.replace("_all", ""), "severity": "pooled_S1-S3", "scope_type": "family_pooled_scope"}


def stratum_definitions(quartiles: list[float]) -> dict[tuple[str, str], str]:
    q1, q2, q3 = quartiles
    return {
        ("range", "lt20"): "detector range < 20 m",
        ("range", "20_to_lt40"): "20 m <= detector range < 40 m",
        ("range", "ge40"): "detector range >= 40 m",
        ("speed", "lt2"): "target speed < 2 m/s",
        ("speed", "2_to_lt5"): "2 m/s <= target speed < 5 m/s",
        ("speed", "ge5"): "target speed >= 5 m/s",
        ("history_length", "2"): "two matched historical frames",
        ("history_length", "3"): "three matched historical frames",
        ("history_length", "4"): "four matched historical frames",
        ("lidar_points", "le5"): "current GT annotation has <= 5 LiDAR points",
        ("lidar_points", "6_to_20"): "current GT annotation has 6-20 LiDAR points",
        ("lidar_points", "gt20"): "current GT annotation has > 20 LiDAR points",
        ("current_xy_residual", "q1"): f"current XY residual < {q1:.12g} m (clean label-cal Q1)",
        ("current_xy_residual", "q2"): f"{q1:.12g} <= current XY residual < {q2:.12g} m",
        ("current_xy_residual", "q3"): f"{q2:.12g} <= current XY residual < {q3:.12g} m",
        ("current_xy_residual", "q4"): f"current XY residual >= {q3:.12g} m (clean label-cal Q4)",
        ("class_group", "vehicle"): "car, truck, construction vehicle, bus, or trailer",
        ("class_group", "pedestrian"): "pedestrian",
        ("class_group", "two_wheeler"): "motorcycle or bicycle",
    }


def materialize_attribute_inputs():
    quartiles = json.loads((FINAL_OUT / "attributes" / "result.json").read_text())["residual_quartiles_from_clean_label_cal"]
    r0_runs = [{} for _ in range(3)]
    r3p_runs = [{} for _ in range(3)]
    selections = {}
    track_rows = {}
    for condition in final.r0.CONDITIONS:
        with np.load(final.r0.OUT / "tracks" / f"{condition}.npz", allow_pickle=False) as source:
            tracks = {name: source[name] for name in ("range", "velocity", "history_length", "class", "sample_token", "instance_token")}
        with np.load(final.r2.OUT / "subsets" / f"{condition}.npz", allow_pickle=False) as source:
            subsets = {name: source[name] for name in ("num_lidar_pts", "current_residual", "sample_token", "instance_token")}
        for name in ("sample_token", "instance_token"):
            if not np.array_equal(tracks[name], subsets[name]):
                raise ValueError(f"attribute row mismatch: {condition}/{name}")
        selections[condition] = final.attribute_selections(condition, tracks, subsets, quartiles)
        track_rows[condition] = len(tracks["class"])
        for replicate in range(3):
            with np.load(final.r0.OUT / "metrics" / f"replicate_{replicate}.npz", allow_pickle=False) as source:
                for name in ("scene_token", "sample_token", "instance_token"):
                    key = f"{condition}__common__{name}"
                    r0_runs[replicate][key] = source[key]
                for predictor in final.r0.PREDICTORS:
                    for group, metrics in {"P1": ("mean_fde",), "P2": ("energy", "log_area90"), "P3": ("energy", "log_area90")}.items():
                        for metric in metrics:
                            key = f"{condition}__{predictor}__{group}__{metric}"
                            r0_runs[replicate][key] = source[key]
            with np.load(final.r3p.OUT / "metrics" / f"replicate_{replicate}.npz", allow_pickle=False) as source:
                for name in ("scene_token", "sample_token", "instance_token"):
                    key = f"{condition}__common__{name}"
                    r3p_runs[replicate][key] = source[key]
                for group, metrics in {"P0_mean_only": ("mean_fde",), "P2_gaussian": ("energy", "log_area90"), "P3_m2_probability": ("energy", "log_area90")}.items():
                    for metric in metrics:
                        key = f"{condition}__{group}__{metric}"
                        r3p_runs[replicate][key] = source[key]
    return quartiles, r0_runs, r3p_runs, selections, track_rows


def condition_effect_map(run, condition: str, selected: np.ndarray, interface: str, probe: str, metric: str) -> dict[str, float]:
    cache_key = (id(run), condition, id(selected), interface, probe, metric)
    if cache_key in EFFECT_MAP_CACHE:
        return EFFECT_MAP_CACHE[cache_key]
    scenes = run[f"{condition}__common__scene_token"]
    if interface == "stochastic_pushforward":
        if metric == "trajectory_energy":
            value = np.asarray(
                run[f"{condition}__{probe}__P2__energy"], dtype=np.float64
            ) - np.asarray(
                run[f"{condition}__{probe}__P3__energy"], dtype=np.float64
            )
            result = r1.scene_mean_map(value, scenes, selected)
            EFFECT_MAP_CACHE[cache_key] = result
            return result
        error = run[f"{condition}__{probe}__P1__mean_fde"]
        baseline = run[f"{condition}__{probe}__P2__log_area90"]
        candidate = run[f"{condition}__{probe}__P3__log_area90"]
    else:
        if metric == "trajectory_energy":
            value = np.asarray(
                run[f"{condition}__P2_gaussian__energy"], dtype=np.float64
            ) - np.asarray(
                run[f"{condition}__P3_m2_probability__energy"], dtype=np.float64
            )
            result = r1.scene_mean_map(value, scenes, selected)
            EFFECT_MAP_CACHE[cache_key] = result
            return result
        error = run[f"{condition}__P0_mean_only__mean_fde"]
        baseline = run[f"{condition}__P2_gaussian__log_area90"]
        candidate = run[f"{condition}__P3_m2_probability__log_area90"]
    result = {}
    for token in np.unique(scenes[selected]):
        keep = selected & (scenes == token)
        if keep.sum() >= 2:
            result[str(token)] = final.r0.risk_coverage(baseline[keep], error[keep]) - final.r0.risk_coverage(candidate[keep], error[keep])
    EFFECT_MAP_CACHE[cache_key] = result
    return result


def attribute_report(runs, conditions, scope_selections, interface, probe, metric):
    replicate_maps = []
    for run in runs:
        maps = [
            condition_effect_map(run, condition, scope_selections[condition], interface, probe, metric)
            for condition in conditions
        ]
        replicate_maps.append(final.combine_condition_maps(maps))
    values = r1.average_maps(replicate_maps)
    values = values[np.isfinite(values)]
    count = len(values)
    if not count:
        report = {"mean": None, "ci95": [None, None], "count": 0}
    else:
        if count not in BOOTSTRAP_WEIGHT_CACHE:
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            indices = rng.integers(
                0, count, size=(BOOTSTRAP_REPEATS, count)
            )
            weights = np.zeros((BOOTSTRAP_REPEATS, count), dtype=np.float64)
            rows = np.repeat(np.arange(BOOTSTRAP_REPEATS), count)
            np.add.at(weights, (rows, indices.reshape(-1)), 1.0 / count)
            BOOTSTRAP_WEIGHT_CACHE[count] = weights
        means = BOOTSTRAP_WEIGHT_CACHE[count] @ values
        report = {
            "mean": float(values.mean()),
            "ci95": [
                float(np.quantile(means, 0.025)),
                float(np.quantile(means, 0.975)),
            ],
            "count": int(count),
        }
    return report, [mean_of_map(value) for value in replicate_maps]


def export_attributes() -> tuple[list[dict], dict]:
    frozen_path = FINAL_OUT / "attributes" / "result.json"
    frozen = json.loads(frozen_path.read_text())
    quartiles, r0_runs, r3p_runs, selections, _ = materialize_attribute_inputs()
    definitions = stratum_definitions(quartiles)
    conditions = tuple(final.r0.CONDITIONS)
    degraded = tuple(value for value in conditions if value != "clean_s0")
    scopes = {condition: (condition,) for condition in conditions}
    scopes.update(
        {
            "lowlight_all": tuple(f"lowlight_s{s}" for s in (1, 2, 3)),
            "dropout_all": tuple(f"dropout_s{s}" for s in (1, 2, 3)),
            "beam_all": tuple(f"beam_s{s}" for s in (1, 2, 3)),
            "all_original_degradations": degraded,
        }
    )
    configurations = [
        ("stochastic_pushforward", "CV", "trajectory_energy", "HTC stochastic pushforward (R0 P3)", "Gaussian stochastic pushforward (R0 P2)", "m (trajectory energy score)"),
        ("stochastic_pushforward", "MLP", "trajectory_energy", "HTC stochastic pushforward (R0 P3)", "Gaussian stochastic pushforward (R0 P2)", "m (trajectory energy score)"),
        ("stochastic_pushforward", "CV", "risk_coverage", "HTC stochastic pushforward (R0 P3)", "Gaussian stochastic pushforward (R0 P2)", "m (area under FDE-risk coverage curve)"),
        ("stochastic_pushforward", "MLP", "risk_coverage", "HTC stochastic pushforward (R0 P3)", "Gaussian stochastic pushforward (R0 P2)", "m (area under FDE-risk coverage curve)"),
        ("direct_probability_conditioning", "R3-P GT-supervised six-mode MLP", "trajectory_energy", "HTC probability token (R3-P P3)", "Gaussian token (R3-P P2)", "m (trajectory energy score)"),
        ("direct_probability_conditioning", "R3-P GT-supervised six-mode MLP", "risk_coverage", "HTC probability token (R3-P P3)", "Gaussian token (R3-P P2)", "m (area under FDE-risk coverage curve)"),
    ]
    rows = []
    matched_checks = []
    suppressed = 0
    for attribute, bins in selections["clean_s0"].items():
        for stratum in bins:
            for scope, scope_conditions in scopes.items():
                scope_selection = {condition: selections[condition][attribute][stratum] for condition in scope_conditions}
                n_tracks = int(sum(value.sum() for value in scope_selection.values()))
                selected_scenes = set()
                for condition in scope_conditions:
                    scene = r0_runs[0][f"{condition}__common__scene_token"]
                    selected_scenes.update(np.asarray(scene)[scope_selection[condition]].astype(str).tolist())
                n_scenes = len(selected_scenes)
                is_suppressed = n_scenes < 10
                metadata = condition_metadata(scope, scope_conditions)
                precomputed_scope = scope in ("clean_s0", "all_original_degradations")
                for interface, probe, metric, candidate, baseline, unit in configurations:
                    runs = r0_runs if interface == "stochastic_pushforward" else r3p_runs
                    if is_suppressed:
                        report = {"mean": None, "ci95": [None, None], "count": 0}
                        seed_values = [None, None, None]
                        status = "suppressed_lt10_selected_scenes"
                        suppressed += 1
                    else:
                        report, seed_values = attribute_report(
                            runs, scope_conditions, scope_selection, interface, probe, metric
                        )
                        status = "precomputed_frozen_scope" if precomputed_scope else "read_only_complete_scope_expansion"
                    row = {
                        "population": "development",
                        "condition": metadata["condition"],
                        "severity": metadata["severity"],
                        "scope_type": metadata["scope_type"],
                        "scope_conditions": "|".join(scope_conditions),
                        "probe": probe,
                        "interface": interface,
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "unit": unit,
                        "attribute": attribute,
                        "stratum": stratum,
                        "stratum_definition": definitions[(attribute, stratum)],
                        "n_scenes": n_scenes,
                        "n_tracks": n_tracks,
                        "n_inference_scenes": report["count"],
                        "effect_definition": "baseline - candidate; positive means lower/better candidate value",
                        "effect": report["mean"] if report["mean"] is not None else "",
                        "ci_low": report["ci95"][0] if report["ci95"][0] is not None else "",
                        "ci_high": report["ci95"][1] if report["ci95"][1] is not None else "",
                        "seed_0": seed_values[0] if seed_values[0] is not None else "",
                        "seed_1": seed_values[1] if seed_values[1] is not None else "",
                        "seed_2": seed_values[2] if seed_values[2] is not None else "",
                        "ci_level": 0.95,
                        "bootstrap_unit": "scene",
                        "bootstrap_repeats": BOOTSTRAP_REPEATS,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "status": status,
                    }
                    rows.append(row)

                    if precomputed_scope and not is_suppressed:
                        frozen_scope = "clean" if scope == "clean_s0" else "degraded"
                        record = frozen["results"][attribute][stratum][frozen_scope]
                        if interface == "stochastic_pushforward" and metric == "trajectory_energy":
                            expected = record["pushforward_energy"][probe]
                        elif interface == "stochastic_pushforward" and metric == "risk_coverage":
                            expected = record["pushforward_risk_coverage"][probe]
                        elif metric == "trajectory_energy":
                            expected = record["conditioning_energy"]
                        else:
                            expected = record["conditioning_risk_coverage"]
                        matched_checks.append(
                            abs(report["mean"] - expected["mean"]) < 1e-12
                            and np.allclose(report["ci95"], expected["ci95"], rtol=0, atol=1e-12)
                            and report["count"] == expected["count"]
                        )
    fields = list(rows[0])
    atomic_csv(OUT / "attribute_strata_complete.csv", rows, fields)
    readme = f"""# Complete descriptive attribute-stratum export

This file expands the frozen development attribute analysis without training, model execution, resampling, condition selection, or significance filtering. It contains every pre-defined stratum for range, speed, history length, LiDAR point-count proxy, current XY residual quartile, and class group.

## Coverage

- Population: development only. No held-out-degradation or final-validation attribute analysis was part of the frozen S4.2 archive, and none is introduced here.
- Conditions: clean S0; every original lowlight/dropout/beam S1, S2, and S3 condition; each full S1-S3 family aggregate; and the original all-degradation aggregate.
- Interfaces: stochastic HTC versus Gaussian pushforward under CV and MLP probes, and direct HTC-token versus Gaussian-token conditioning under the matched R3-P GT-supervised student.
- Metrics: trajectory energy and risk--coverage. Positive `baseline - candidate` favors HTC. All 95% intervals use 10,000 paired scene bootstrap resamples with seed 20260805.
- Rows: {len(rows):,}. Rows are not filtered by effect magnitude, direction, or interval. The status `suppressed_lt10_selected_scenes` appears only when a pre-defined stratum contains fewer than ten selected scenes; such rows are retained with blank estimates rather than deleted. There are {suppressed} such metric rows.

## Provenance and scope status

`clean_s0` and `all_original_degradations` reproduce the two scopes already stored in `outputs/riskprop3d_final_confirmation/attributes/result.json`. Their values are checked exactly against that frozen JSON. Per-severity and per-family rows are a deterministic read-only expansion from the same frozen per-target metric archives, masks, seed pairing, effect direction, scene aggregation, and bootstrap seed. They are descriptive and do not create a new gate or family-wise claim.

`n_scenes` counts scenes containing at least one selected target; `n_inference_scenes` is the number entering the statistic. For risk--coverage, a scene requires at least two selected targets, so these counts can differ. `n_tracks` is the number of selected condition-target rows and is not treated as the inferential sample size.

## Stratum authority

Current-residual boundaries are the frozen clean label-cal quartiles: {quartiles[0]:.12g}, {quartiles[1]:.12g}, and {quartiles[2]:.12g} m. GT point counts and current residuals are evaluation-only descriptors and never model inputs.

## Missingness and validity

The frozen source reports `all_numeric_finite={frozen['all_numeric_finite']}`. No invalid or NaN effect is silently removed. Suppressed rows remain present with an explicit status. All non-suppressed clean/all-degradation rows match the frozen aggregate file.
"""
    atomic_text(OUT / "attribute_strata_readme.md", readme)
    checks = {
        "row_count": len(rows),
        "all_precomputed_scope_values_match": bool(all(matched_checks)),
        "precomputed_values_checked": len(matched_checks),
        "suppressed_rows": suppressed,
        "source_all_numeric_finite": bool(frozen["all_numeric_finite"]),
    }
    return rows, checks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r2d_result_path = R2D_OUT / "result.json"
    official_result_path = FINAL_OUT / "official" / "result.json"
    r2d_result = json.loads(r2d_result_path.read_text())
    official_result = json.loads(official_result_path.read_text())
    r2d_runs = load_npz_runs(R2D_OUT / "metrics")
    official_runs = load_npz_runs(FINAL_OUT / "metrics" / "official")

    fidelity_rows, fidelity_checks = export_fidelity(r2d_runs, r2d_result)
    equivalence_tests, equivalence_checks = export_equivalence(
        r2d_runs, official_runs, r2d_result, official_result
    )
    runtime_rows, runtime_checks = export_runtime(r2d_result)
    attribute_rows, attribute_checks = export_attributes()

    sources = [
        r2d_result_path,
        official_result_path,
        FINAL_OUT / "attributes" / "result.json",
        ROOT / "tools" / "riskprop3d_r2d.py",
        ROOT / "tools" / "riskprop3d_final_confirmation.py",
    ]
    sources.extend(R2D_OUT / "metrics" / f"replicate_{index}.npz" for index in range(3))
    sources.extend(FINAL_OUT / "metrics" / "official" / f"replicate_{index}.npz" for index in range(3))
    sources.extend(final.r0.OUT / "metrics" / f"replicate_{index}.npz" for index in range(3))
    sources.extend(final.r3p.OUT / "metrics" / f"replicate_{index}.npz" for index in range(3))
    outputs = [
        OUT / "preservation_fidelity_summary.csv",
        OUT / "preservation_equivalence.json",
        OUT / "runtime_protocol.md",
        OUT / "runtime_measurements.csv",
        OUT / "attribute_strata_complete.csv",
        OUT / "attribute_strata_readme.md",
    ]
    manifest = {
        "status": "COMPLETE_WITH_EXPLICIT_PROVENANCE_LIMITATIONS",
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "no_model_execution": True,
        "no_new_sampling": True,
        "no_condition_or_result_selection": True,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "output_sha256": {path.name: sha256(path) for path in outputs},
        "checks": {
            "fidelity": fidelity_checks,
            "equivalence": equivalence_checks,
            "runtime": runtime_checks,
            "attributes": attribute_checks,
            "all_checks_pass_except_declared_runtime_hardware_provenance": bool(
                all(fidelity_checks.values())
                and all(equivalence_checks.values())
                and runtime_checks["mean_paired_speedup_matches_frozen_result"]
                and attribute_checks["all_precomputed_scope_values_match"]
                and attribute_checks["source_all_numeric_finite"]
            ),
        },
        "row_counts": {
            "preservation_fidelity_summary.csv": len(fidelity_rows),
            "preservation_equivalence.json/tests": len(equivalence_tests),
            "runtime_measurements.csv": len(runtime_rows),
            "attribute_strata_complete.csv": len(attribute_rows),
        },
        "declared_missing_evidence": {
            "final_validation_teacher_fidelity_rows": 4,
            "reason": "The final archive does not retain the teacher/student sample or moment-fidelity arrays required for the four fidelity definitions; no post-hoc model rerun was performed.",
            "original_runtime_hardware_metadata": "not serialized by the frozen R2-D timing archive",
        },
    }
    atomic_json(OUT / "EXPORT_MANIFEST.json", manifest)
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "manifest": manifest["status"], "rows": manifest["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
