#!/usr/bin/env python3
"""Final confirmation for the perception-to-forecast uncertainty study."""

import argparse
import hashlib
import json
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from probset3d.f1.implementation.probabilistic_query import ProbabilisticQueryHead
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import riskprop3d_r0 as r0
import riskprop3d_r1 as r1
import riskprop3d_r2_core as r2
import riskprop3d_r2d as r2d
import riskprop3d_r3p as r3p


OUT = ROOT / "outputs/riskprop3d_final_confirmation"
PAPER = ROOT / "riskprop3d_paper"
EXPORTER = ROOT / "tools/riskprop3d_confirmation_export.py"
PROTOCOL = PAPER / "FINAL_CONFIRMATION_PROTOCOL.md"
FREEZE = OUT / "freeze_manifest.json"
ACCESS_LOG = OUT / "access_log.jsonl"
EXTERNAL_CONDITIONS = (
    "rotation_s2",
    "rotation_s3",
    "translation_s3",
    "temporal_s2",
    "temporal_s3",
)
EXTERNAL_FAMILIES = {
    "rotation": ("rotation_s2", "rotation_s3"),
    "translation": ("translation_s3",),
    "temporal": ("temporal_s2", "temporal_s3"),
}
OFFICIAL_INFO = ROOT / "outputs/stage2/full/nuscenes_infos_val_6019.pkl"
DEV_TEMPORAL_INFO = {
    2: OUT / "inputs/nuscenes_infos_dev_temporal_s2_valid.pkl",
    3: OUT / "inputs/nuscenes_infos_dev_temporal_s3_valid.pkl",
}
DEV_TEMPORAL_HISTORY = OUT / "inputs/dev_camera_history.json"
COMMON_FIELDS = r2d.COMMON_FIELDS
BOOTSTRAP_SEED = 20260805
SAMPLES = 64
OFFSET_EXTERNAL = 500000
OFFSET_OFFICIAL = 900000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    r0.write_json(Path(path), value)


def append_access(event, **fields):
    if not ACCESS_LOG.exists():
        raise FileNotFoundError("F0 access log has not been activated")
    record = {"event": event, **fields}
    with ACCESS_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def checkpoint_hashes(root):
    return {
        str(index): sha256(root / "checkpoints" / f"replicate_{index}.pt")
        for index in range(3)
    }


def freeze():
    if FREEZE.exists() or ACCESS_LOG.exists():
        raise FileExistsError("final-confirmation freeze already exists")
    authorities = {
        "r0": r0.OUT / "result.json",
        "r15": ROOT / "outputs/riskprop3d_r15/result.json",
        "r2_core": r2.OUT / "result.json",
        "r2d": r2d.OUT / "result.json",
        "r3p": r3p.OUT / "result.json",
    }
    expected = {
        "r0": "GO TO RISKPROP3D METHOD DEVELOPMENT",
        "r15": "GO TO RISKPROP3D-CORE",
        "r2_core": "STOP RISKPROP3D-CORE MECHANISM CLAIM",
        "r2d": "STOP LEARNED AMORTIZED PROPAGATION",
        "r3p": "STOP RISKPROP3D DIRECTION",
    }
    for name, path in authorities.items():
        result = r0.strict_load(path)
        if result["endpoint"] != expected[name]:
            raise ValueError(f"source endpoint changed: {name}")
        if result.get("official_val_read") or result.get("internal_test_read"):
            raise ValueError(f"source seal changed: {name}")
    if not OFFICIAL_INFO.exists():
        raise FileNotFoundError(OFFICIAL_INFO)
    if not DEV_TEMPORAL_HISTORY.exists() or not all(
        path.exists() for path in DEV_TEMPORAL_INFO.values()
    ):
        raise FileNotFoundError("frozen temporal dev metadata is missing")
    scripts = {
        "confirmation": Path(__file__).resolve(),
        "confirmation_exporter": EXPORTER,
        "r0": ROOT / "tools/riskprop3d_r0.py",
        "r1": ROOT / "tools/riskprop3d_r1.py",
        "r2_core": ROOT / "tools/riskprop3d_r2_core.py",
        "r2d": ROOT / "tools/riskprop3d_r2d.py",
        "r3p": ROOT / "tools/riskprop3d_r3p.py",
    }
    documents = {
        name: PAPER / name
        for name in (
            "README.md",
            "EVIDENCE_MATRIX.md",
            "PAPER_MAP.md",
            "FINAL_CONFIRMATION_PROTOCOL.md",
            "PRECONFIRMATION_CLAIM_LEDGER.md",
        )
    }
    documents["confirmation_README.md"] = ROOT / "riskprop3d_final_confirmation/README.md"
    head_hashes = {
        str(seed): sha256(r0.CHECKPOINT_ROOT / f"seed_{seed}.pt")
        for seed in r0.HEAD_SEEDS
    }
    manifest = {
        "status": "F0_IMMUTABLE_EVIDENCE_FREEZE_ACTIVE",
        "authorities": {
            name: {
                "endpoint": expected[name],
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in authorities.items()
        },
        "scripts": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in scripts.items()
        },
        "documents": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in documents.items()
        },
        "checkpoints": {
            "heads": head_hashes,
            "r0_predictor": sha256(r0.OUT / "predictor.pt"),
            "r2_core": checkpoint_hashes(r2.OUT),
            "r2d": checkpoint_hashes(r2d.OUT),
            "r3p": checkpoint_hashes(r3p.OUT),
        },
        "external": {
            "conditions": list(EXTERNAL_CONDITIONS),
            "families": {name: list(value) for name, value in EXTERNAL_FAMILIES.items()},
            "info_path": str((ROOT / r0.strict_load(r0.P0_AUTHORITY)["splits"]["dev"]["info_path"]).relative_to(ROOT)),
            "temporal_info": {
                str(severity): {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "sample_count": len(load_info_map(path)),
                }
                for severity, path in DEV_TEMPORAL_INFO.items()
            },
            "temporal_history": {
                "path": str(DEV_TEMPORAL_HISTORY.relative_to(ROOT)),
                "sha256": sha256(DEV_TEMPORAL_HISTORY),
            },
            "official_val_used": False,
            "excluded_source": "outputs/third_work_r0/feature_shards (official-val-based metadata; not used)",
        },
        "official": {
            "info_path": str(OFFICIAL_INFO.relative_to(ROOT)),
            "info_sha256": sha256(OFFICIAL_INFO),
            "declared_sample_count": 6019,
            "official_val_read": False,
        },
        "seeds": {
            "bootstrap": BOOTSTRAP_SEED,
            "head": list(r0.HEAD_SEEDS),
            "r0_propagation": list(r0.PROPAGATION_SEEDS),
            "r2_core_sampling": list(r2.SAMPLE_SEEDS),
            "r2d_sampling": list(r2d.STUDENT_SAMPLE_SEEDS),
            "r3p_sampling": list(r3p.SAMPLE_SEEDS),
            "external_offset": OFFSET_EXTERNAL,
            "official_offset": OFFSET_OFFICIAL,
        },
        "input_contract": {
            "r0": "frozen 64-d history/current/mask/class state plus M1/M2 head distributions",
            "r2_core": "frozen target-centric R2-Core adapters; no mechanism subset input",
            "r2d": "frozen 64-d state plus 20-d M2 raw probability token",
            "r3p": "frozen 64-d state plus predeclared mean/confidence/M1/M2 20-d token",
            "corruption_identity_input": False,
            "current_gt_residual_input": False,
        },
        "evaluation_contract": {
            "metrics": list(r0.METRICS),
            "primary": "energy",
            "diagnostic": "nll",
            "high_fde": "same-interface deterministic/mean-reference predictive-mean FDE > 2m",
            "risk_ranking": "predicted final log equal-model ellipse area",
            "false_safe": "lowest-risk 80% within scene",
            "equal_coverage_area": "median(pi * empirical q90(final_maha2) * sqrt(final_det))",
            "aggregation": "per-track differences averaged within scene, then paired across three replicates",
            "confidence_interval": "10000 paired-scene bootstrap resamples at seed 20260805",
        },
        "primary_metric": "trajectory_energy_score",
        "bootstrap_unit": "scene",
        "bootstrap_resamples": 10000,
        "internal_test_read": False,
        "method_development_closed": True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    ACCESS_LOG.write_text(
        json.dumps(
            {
                "event": "F0_ACTIVATED",
                "external_result_summary_read": False,
                "official_metric_read": False,
                "internal_test_read": False,
                "note": "Pre-activation inspection was limited to filenames, source metadata, and split compatibility; no external or official metric summary was read.",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(FREEZE, manifest)
    print(json.dumps({"freeze": manifest["status"], "manifest": str(FREEZE)}))


def verify_freeze(allow_claim_freeze=False):
    manifest = r0.strict_load(FREEZE)
    for record in manifest["authorities"].values():
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"authority hash changed: {record['path']}")
    for name, record in manifest["scripts"].items():
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"script hash changed after F0: {name}")
    for name, record in manifest["documents"].items():
        if allow_claim_freeze and name in ("README.md", "EVIDENCE_MATRIX.md", "PAPER_MAP.md"):
            continue
        if sha256(ROOT / record["path"]) != record["sha256"]:
            raise ValueError(f"document hash changed after F0: {name}")
    return manifest


def scene_scope_map(values, scenes, selected):
    return r1.scene_mean_map(np.asarray(values), np.asarray(scenes), selected)


def combine_condition_maps(maps):
    keys = sorted(set().union(*(set(value) for value in maps)))
    return {
        key: float(np.mean([value[key] for value in maps if key in value]))
        for key in keys
    }


def scene_bootstrap(value, seed=BOOTSTRAP_SEED, repeats=10000):
    value = np.asarray(value, dtype=np.float64)
    value = value[np.isfinite(value)]
    if not len(value):
        return {"mean": None, "ci95": [None, None], "count": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(value), size=(repeats, len(value)))
    means = value[indices].mean(axis=1)
    return {
        "mean": float(value.mean()),
        "ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "count": int(len(value)),
    }


def effect_report(runs, conditions, selections, key):
    replicate_maps = []
    for run_index, run in enumerate(runs):
        maps = []
        for condition in conditions:
            difference = key(run, condition)
            maps.append(
                scene_scope_map(
                    difference,
                    run[f"{condition}__common__scene_token"],
                    selections[condition],
                )
            )
        replicate_maps.append(combine_condition_maps(maps))
    values = r1.average_maps(replicate_maps)
    return scene_bootstrap(values)


def risk_report(runs, conditions, selections, stage, predictor=None):
    replicate_maps = []
    for run in runs:
        condition_maps = []
        for condition in conditions:
            selected = np.asarray(selections[condition], dtype=bool)
            scenes = np.asarray(run[f"{condition}__common__scene_token"])
            if stage == "r0":
                prefix = f"{condition}__{predictor}"
                error = np.asarray(run[f"{prefix}__P1__mean_fde"], dtype=np.float64)
                baseline = np.asarray(run[f"{prefix}__P2__log_area90"], dtype=np.float64)
                candidate = np.asarray(run[f"{prefix}__P3__log_area90"], dtype=np.float64)
            else:
                error = np.asarray(run[f"{condition}__P0_mean_only__mean_fde"], dtype=np.float64)
                baseline = np.asarray(run[f"{condition}__P2_gaussian__log_area90"], dtype=np.float64)
                candidate = np.asarray(run[f"{condition}__P3_m2_probability__log_area90"], dtype=np.float64)
            current = {}
            for token in np.unique(scenes[selected]):
                keep = selected & (scenes == token)
                if keep.sum() >= 2:
                    current[str(token)] = r0.risk_coverage(
                        baseline[keep], error[keep]
                    ) - r0.risk_coverage(candidate[keep], error[keep])
            condition_maps.append(current)
        replicate_maps.append(combine_condition_maps(condition_maps))
    return scene_bootstrap(r1.average_maps(replicate_maps))


def attribute_selections(condition, tracks, subsets, residual_quartiles):
    value = {
        "range": np.asarray(tracks["range"], dtype=np.float64),
        "speed": np.asarray(tracks["velocity"], dtype=np.float64),
        "history": np.asarray(tracks["history_length"], dtype=np.int64),
        "lidar_points": np.asarray(subsets["num_lidar_pts"], dtype=np.int64),
        "residual": np.asarray(subsets["current_residual"], dtype=np.float64),
        "class": np.asarray(tracks["class"], dtype=np.int64),
    }
    q1, q2, q3 = residual_quartiles
    return {
        "range": {
            "lt20": value["range"] < 20,
            "20_to_lt40": (value["range"] >= 20) & (value["range"] < 40),
            "ge40": value["range"] >= 40,
        },
        "speed": {
            "lt2": value["speed"] < 2,
            "2_to_lt5": (value["speed"] >= 2) & (value["speed"] < 5),
            "ge5": value["speed"] >= 5,
        },
        "history_length": {
            str(length): value["history"] == length for length in (2, 3, 4)
        },
        "lidar_points": {
            "le5": value["lidar_points"] <= 5,
            "6_to_20": (value["lidar_points"] >= 6) & (value["lidar_points"] <= 20),
            "gt20": value["lidar_points"] > 20,
        },
        "current_xy_residual": {
            "q1": value["residual"] < q1,
            "q2": (value["residual"] >= q1) & (value["residual"] < q2),
            "q3": (value["residual"] >= q2) & (value["residual"] < q3),
            "q4": value["residual"] >= q3,
        },
        "class_group": {
            "vehicle": np.isin(value["class"], (0, 1, 2, 3, 4)),
            "pedestrian": value["class"] == 8,
            "two_wheeler": np.isin(value["class"], (6, 7)),
        },
    }


def attributes():
    verify_freeze()
    path = OUT / "attributes/result.json"
    if path.exists():
        raise FileExistsError(path)
    with np.load(r0.OUT / "tracks/label_cal.npz", allow_pickle=False) as calibration:
        current = r0.HISTORY * r0.STATE_FEATURES
        residual = np.linalg.norm(
            np.asarray(calibration["input"][:, current : current + 2])
            - np.asarray(calibration["gt_input"][:, current : current + 2]),
            axis=-1,
        )
    quartiles = np.quantile(residual, (0.25, 0.50, 0.75)).tolist()
    # Materialize only the columns used below. Repeated lazy reads from the
    # compressed full archives would otherwise decompress the same arrays for
    # every descriptive bin.
    r0_runs = [{} for _ in range(3)]
    r3p_runs = [{} for _ in range(3)]
    condition_selections = {}
    row_counts = {}
    for condition in r0.CONDITIONS:
        with np.load(r0.OUT / "tracks" / f"{condition}.npz", allow_pickle=False) as tracks:
            track_values = {name: tracks[name] for name in ("range", "velocity", "history_length", "class", "sample_token", "instance_token")}
        with np.load(r2.OUT / "subsets" / f"{condition}.npz", allow_pickle=False) as subsets:
            subset_values = {name: subsets[name] for name in ("num_lidar_pts", "current_residual", "sample_token", "instance_token")}
        for name in ("sample_token", "instance_token"):
            if not np.array_equal(track_values[name], subset_values[name]):
                raise ValueError(f"attribute row mismatch: {condition}/{name}")
        condition_selections[condition] = attribute_selections(
            condition, track_values, subset_values, quartiles
        )
        row_counts[condition] = len(track_values["class"])
        for replicate in range(3):
            with np.load(
                r0.OUT / "metrics" / f"replicate_{replicate}.npz",
                allow_pickle=False,
            ) as source:
                for name in ("scene_token", "sample_token", "instance_token"):
                    key = f"{condition}__common__{name}"
                    r0_runs[replicate][key] = source[key]
                for predictor in r0.PREDICTORS:
                    for group, names in {
                        "P1": ("mean_fde",),
                        "P2": ("energy", "log_area90"),
                        "P3": ("energy", "log_area90"),
                    }.items():
                        for name in names:
                            key = f"{condition}__{predictor}__{group}__{name}"
                            r0_runs[replicate][key] = source[key]
            with np.load(
                r3p.OUT / "metrics" / f"replicate_{replicate}.npz",
                allow_pickle=False,
            ) as source:
                for name in ("scene_token", "sample_token", "instance_token"):
                    key = f"{condition}__common__{name}"
                    r3p_runs[replicate][key] = source[key]
                for group, names in {
                    "P0_mean_only": ("mean_fde",),
                    "P2_gaussian": ("energy", "log_area90"),
                    "P3_m2_probability": ("energy", "log_area90"),
                }.items():
                    for name in names:
                        key = f"{condition}__{group}__{name}"
                        r3p_runs[replicate][key] = source[key]
        for name in ("sample_token", "instance_token"):
            if not np.array_equal(
                track_values[name], r0_runs[0][f"{condition}__common__{name}"]
            ):
                raise ValueError(f"metric row mismatch: {condition}/{name}")
    scopes = {
        "clean": ("clean_s0",),
        "degraded": tuple(condition for condition in r0.CONDITIONS if condition != "clean_s0"),
    }
    results = {}
    for attribute, bins in condition_selections["clean_s0"].items():
        results[attribute] = {}
        for label in bins:
            results[attribute][label] = {}
            for scope, conditions in scopes.items():
                selections = {
                    condition: condition_selections[condition][attribute][label]
                    for condition in conditions
                }
                counts = int(sum(selection.sum() for selection in selections.values()))
                scenes = set()
                for condition in conditions:
                    source = r0_runs[0][f"{condition}__common__scene_token"]
                    scenes.update(np.asarray(source)[selections[condition]].astype(str).tolist())
                current = {
                    "track_rows": counts,
                    "scene_count": len(scenes),
                    "suppressed": len(scenes) < 10,
                    "pushforward_energy": {},
                    "pushforward_risk_coverage": {},
                }
                if len(scenes) < 10:
                    current["pushforward_energy"] = None
                    current["pushforward_risk_coverage"] = None
                    current["conditioning_energy"] = None
                    current["conditioning_risk_coverage"] = None
                    results[attribute][label][scope] = current
                    continue
                for predictor in r0.PREDICTORS:
                    current["pushforward_energy"][predictor] = effect_report(
                        r0_runs,
                        conditions,
                        selections,
                        lambda run, condition, predictor=predictor: np.asarray(
                            run[f"{condition}__{predictor}__P2__energy"], dtype=np.float64
                        )
                        - np.asarray(run[f"{condition}__{predictor}__P3__energy"], dtype=np.float64),
                    )
                    current["pushforward_risk_coverage"][predictor] = risk_report(
                        r0_runs, conditions, selections, "r0", predictor
                    )
                current["conditioning_energy"] = effect_report(
                    r3p_runs,
                    conditions,
                    selections,
                    lambda run, condition: np.asarray(
                        run[f"{condition}__P2_gaussian__energy"], dtype=np.float64
                    )
                    - np.asarray(
                        run[f"{condition}__P3_m2_probability__energy"], dtype=np.float64
                    ),
                )
                current["conditioning_risk_coverage"] = risk_report(
                    r3p_runs, conditions, selections, "r3p"
                )
                results[attribute][label][scope] = current
    numeric_count = 0
    finite = True
    for runs in (r0_runs, r3p_runs):
        for run in runs:
            for value in run.values():
                value = np.asarray(value)
                if np.issubdtype(value.dtype, np.number):
                    finite &= bool(np.isfinite(value).all())
                    numeric_count += int(value.size)
    output = {
        "status": "DESCRIPTIVE_ONLY_NO_GATES",
        "question": "Where is M2-over-Gaussian stochastic benefit concentrated, and how does the corresponding learned-conditioning effect contract?",
        "effects": {
            "pushforward": "R0 ES(M1 Gaussian MC) - ES(M2 MC)",
            "conditioning": "R3-P ES(Gaussian token) - ES(M2 token)",
        },
        "residual_quartiles_from_clean_label_cal": quartiles,
        "scopes": {name: list(value) for name, value in scopes.items()},
        "results": results,
        "source_hashes": {
            "freeze": sha256(FREEZE),
            "r0_result": sha256(r0.OUT / "result.json"),
            "r2_core_result": sha256(r2.OUT / "result.json"),
            "r3p_result": sha256(r3p.OUT / "result.json"),
        },
        "row_counts": row_counts,
        "numeric_values_checked": numeric_count,
        "all_numeric_finite": bool(finite),
        "new_hypothesis_generated": False,
        "subset_gate_used": False,
        "official_val_read": False,
        "internal_test_read": False,
    }
    write_json(path, output)
    append_access(
        "F1_ATTRIBUTES_COMPLETE",
        result=str(path.relative_to(ROOT)),
        result_sha256=sha256(path),
        official_metric_read=False,
        internal_test_read=False,
    )
    print(json.dumps({"attributes": "complete", "output": str(path)}))


def export_manifests(phase, condition):
    root = OUT / "exports" / phase / condition
    manifests = []
    for index in range(4):
        path = root / f"shard-{index:02d}-of-04/accepted.json"
        value = r0.strict_load(path)
        if (
            not value["accepted"]
            or value["phase"] != phase
            or value["condition"] != condition
            or value["shard_count"] != 4
            or value["shard_index"] != index
            or value["exporter_sha256"] != sha256(EXPORTER)
        ):
            raise ValueError(f"invalid export manifest: {path}")
        for record in value["parts"]:
            part = ROOT / record["file"]
            if sha256(part) != record["sha256"]:
                raise ValueError(f"export part hash mismatch: {part}")
        manifests.append(value)
    if len({value["full_sample_count"] for value in manifests}) != 1:
        raise ValueError("inconsistent full export count")
    if sum(value["sample_count"] for value in manifests) != manifests[0]["full_sample_count"]:
        raise ValueError("incomplete export coverage")
    return manifests


def export_parts(manifests):
    for manifest in manifests:
        for record in manifest["parts"]:
            with np.load(ROOT / record["file"], allow_pickle=False) as source:
                yield {name: source[name] for name in source.files}


def load_info_map(path):
    with Path(path).open("rb") as stream:
        source = pickle.load(stream)
    values = source["infos"] if isinstance(source, dict) else source
    result = {value["token"]: value for value in values}
    if len(result) != len(values):
        raise ValueError("duplicate info token")
    return result


def prepare_tracks(phase, condition):
    official = phase == "official"
    verify_freeze(allow_claim_freeze=official)
    if official:
        claim_freeze = OUT / "claim_freeze.json"
        if not claim_freeze.exists():
            raise FileNotFoundError("F3 claim freeze is required before official access")
        if condition != "clean":
            raise ValueError("official confirmation is clean-only")
    elif condition not in EXTERNAL_CONDITIONS:
        raise ValueError(condition)
    path = OUT / "tracks" / phase / f"{condition}.npz"
    manifest_path = OUT / "tracks" / phase / f"{condition}.json"
    if path.exists() or manifest_path.exists():
        raise FileExistsError(path)
    manifests = export_manifests(phase, condition)
    if official:
        append_access(
            "OFFICIAL_INFO_OPENED",
            path=str(OFFICIAL_INFO.relative_to(ROOT)),
            sha256=sha256(OFFICIAL_INFO),
            official_metric_read=False,
            internal_test_read=False,
        )
        infos = load_info_map(OFFICIAL_INFO)
    else:
        if condition == "temporal_s2":
            infos = load_info_map(DEV_TEMPORAL_INFO[2])
        elif condition == "temporal_s3":
            infos = load_info_map(DEV_TEMPORAL_INFO[3])
        else:
            infos = r0.load_infos("dev")
    if len(infos) != manifests[0]["full_sample_count"]:
        raise ValueError("export/info sample count mismatch")
    frames = r0.frame_authority(infos)
    nusc = NuScenes(
        version="v1.0-trainval",
        dataroot=str(ROOT / "data/nuscenes"),
        verbose=False,
    )
    order, observations = r0.observation_map(
        r0.source_rows(export_parts(manifests), clean=False),
        infos,
        frames,
        nusc,
        include_head=True,
    )
    tracks = r0.build_tracks(
        order, observations, infos, frames, nusc, include_head=True
    )
    r0.write_npz(path, tracks)
    record = {
        "phase": phase,
        "condition": condition,
        "sample_count": len(infos),
        "track_count": int(len(tracks["input"])),
        "scene_count": int(len(np.unique(tracks["scene_token"]))),
        "tracks_sha256": sha256(path),
        "export_manifests": [
            {
                "path": str(
                    (
                        OUT
                        / "exports"
                        / phase
                        / condition
                        / f"shard-{index:02d}-of-04/accepted.json"
                    ).relative_to(ROOT)
                ),
                "sha256": sha256(
                    OUT
                    / "exports"
                    / phase
                    / condition
                    / f"shard-{index:02d}-of-04/accepted.json"
                ),
            }
            for index in range(4)
        ],
        "official_val_read": official,
        "internal_test_read": False,
    }
    write_json(manifest_path, record)
    append_access(
        "TRACK_AUTHORITY_BUILT",
        phase=phase,
        condition=condition,
        tracks=str(path.relative_to(ROOT)),
        tracks_sha256=record["tracks_sha256"],
        official_metric_read=False,
        internal_test_read=False,
    )
    print(json.dumps(record))


def load_confirmation_tracks(phase, condition):
    path = OUT / "tracks" / phase / f"{condition}.npz"
    manifest = r0.strict_load(path.with_suffix(".json"))
    if sha256(path) != manifest["tracks_sha256"]:
        raise ValueError(f"track hash mismatch: {phase}/{condition}")
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


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
    heads = {
        name: ProbabilisticQueryHead().to(device).eval() for name in ("M1", "M2")
    }
    for name, head in heads.items():
        head.load_state_dict(checkpoint["models"][name])
    return heads


def sliced(values, begin, stop):
    return {name: value[begin:stop] for name, value in values.items()}


def r0_condition_metrics(
    tracks, outputs, predictor_model, parameters, replicate, condition_index, device, offset
):
    storage = {
        predictor: {
            group: {metric: [] for metric in r0.METRICS} for group in ("P1", "P2", "P3")
        }
        for predictor in r0.PREDICTORS
    }
    rng = np.random.default_rng(
        r0.PROPAGATION_SEEDS[replicate] + offset + condition_index * 1009
    )
    for begin in range(0, len(tracks["input"]), r0.EVAL_CHUNK):
        stop = min(begin + r0.EVAL_CHUNK, len(tracks["input"]))
        motion = np.asarray(tracks["input"][begin:stop])
        future = np.asarray(tracks["future"][begin:stop])
        ego = np.asarray(tracks["ego_future"][begin:stop])
        count = stop - begin
        sampled = {"P2": [], "P3": []}
        for _draw in range(2):
            z = rng.standard_normal((count, SAMPLES, 8))
            radial = rng.uniform(1e-7, 1.0 - 1e-7, (count, SAMPLES))
            yaw_normal = rng.standard_normal((count, SAMPLES))
            for group, family, student in (
                ("P2", "M1", False),
                ("P3", "M2", True),
            ):
                state = r0.sample_states(
                    sliced(outputs[family], begin, stop),
                    rng,
                    SAMPLES,
                    student,
                    z,
                    radial,
                    yaw_normal,
                )
                sampled[group].append(r0.replace_current(motion, *state))
        for predictor in r0.PREDICTORS:
            covariance = parameters["covariance"][predictor]
            noise_a = r0.process_noise(rng, covariance, count, SAMPLES)
            noise_b = r0.process_noise(rng, covariance, count, SAMPLES)
            centers = {
                "P1": (
                    r0.predict_centers(
                        predictor, motion[:, None], predictor_model, parameters, device
                    ),
                )
                * 2,
                "P2": tuple(
                    r0.predict_centers(
                        predictor, value, predictor_model, parameters, device
                    )
                    for value in sampled["P2"]
                ),
                "P3": tuple(
                    r0.predict_centers(
                        predictor, value, predictor_model, parameters, device
                    )
                    for value in sampled["P3"]
                ),
            }
            for group, center in centers.items():
                metrics = r0.distribution_metrics(
                    center[0], center[1], future, ego, covariance, noise_a, noise_b
                )
                for name in r0.METRICS:
                    storage[predictor][group][name].append(
                        np.asarray(metrics[name], dtype=np.float32)
                    )
    return {
        f"r0__{predictor}__{group}__{name}": np.concatenate(values)
        for predictor, groups in storage.items()
        for group, metrics in groups.items()
        for name, values in metrics.items()
    }


def learned_condition_metrics(
    tracks, m1, m2, models, checkpoint, replicate, condition_index, device, offset
):
    tokens = r3p.make_tokens(m1, m2, tracks)
    rng = np.random.default_rng(
        r3p.SAMPLE_SEEDS[replicate] + offset + condition_index * 1009
    )
    storage = {
        group: {name: [] for name in r0.METRICS} for group in r3p.GROUPS
    }
    motion = np.asarray(tracks["input"], dtype=np.float32)
    future_all = np.asarray(tracks["future"], dtype=np.float32)
    ego_all = np.asarray(tracks["ego_future"], dtype=np.float32)
    for begin in range(0, len(motion), r3p.EVAL_CHUNK):
        stop = min(begin + r3p.EVAL_CHUNK, len(motion))
        current_motion = torch.from_numpy(motion[begin:stop]).to(device)
        with torch.inference_mode():
            outputs = {
                group: r2d.numpy_output(
                    model(
                        current_motion,
                        torch.from_numpy(tokens[group][begin:stop]).to(device),
                    )
                )
                for group, model in models.items()
            }
        count = stop - begin
        uniforms = (rng.random((count, SAMPLES)), rng.random((count, SAMPLES)))
        normals = (
            rng.standard_normal((count, SAMPLES, r0.HORIZON, 2)),
            rng.standard_normal((count, SAMPLES, r0.HORIZON, 2)),
        )
        for group in r3p.GROUPS:
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
                future_all[begin:stop],
                ego_all[begin:stop],
                temperature,
            )
            for name in r0.METRICS:
                storage[group][name].append(np.asarray(metrics[name], dtype=np.float32))
    return {
        f"r3p__{group}__{name}": np.concatenate(values)
        for group, metrics in storage.items()
        for name, values in metrics.items()
    }


def external_evaluate(replicate, device_index):
    verify_freeze()
    if replicate not in range(3) or device_index not in range(4):
        raise ValueError("replicate must be 0--2 and GPU must be 0--3")
    path = OUT / "metrics/external" / f"replicate_{replicate}.npz"
    if path.exists():
        raise FileExistsError(path)
    device = torch.device(f"cuda:{device_index}")
    set_determinism(r3p.SAMPLE_SEEDS[replicate] + OFFSET_EXTERNAL, device)
    heads = load_heads(replicate, device)
    predictor, parameters = r0.load_predictor(device)
    models, checkpoint = r3p.load_models(replicate, device)
    arrays = {
        "meta__head_seed": np.asarray(r0.HEAD_SEEDS[replicate], dtype=np.int64),
        "meta__r0_seed": np.asarray(
            r0.PROPAGATION_SEEDS[replicate] + OFFSET_EXTERNAL, dtype=np.int64
        ),
        "meta__r3p_seed": np.asarray(
            r3p.SAMPLE_SEEDS[replicate] + OFFSET_EXTERNAL, dtype=np.int64
        ),
    }
    for condition_index, condition in enumerate(EXTERNAL_CONDITIONS):
        tracks = load_confirmation_tracks("external", condition)
        outputs = {
            name: r0.head_outputs(head, tracks, device) for name, head in heads.items()
        }
        current = r0_condition_metrics(
            tracks,
            outputs,
            predictor,
            parameters,
            replicate,
            condition_index,
            device,
            OFFSET_EXTERNAL,
        )
        current.update(
            learned_condition_metrics(
                tracks,
                outputs["M1"],
                outputs["M2"],
                models,
                checkpoint,
                replicate,
                condition_index,
                device,
                OFFSET_EXTERNAL,
            )
        )
        for name in COMMON_FIELDS:
            arrays[f"{condition}__common__{name}"] = np.asarray(tracks[name])
        for name, value in current.items():
            arrays[f"{condition}__{name}"] = value
    r0.write_npz(path, arrays)
    append_access(
        "F2_EXTERNAL_REPLICATE_COMPLETE",
        replicate=replicate,
        archive=str(path.relative_to(ROOT)),
        archive_sha256=sha256(path),
        official_metric_read=False,
        internal_test_read=False,
    )
    print(json.dumps({"external_evaluated": True, "replicate": replicate}))


def metric(run, condition, interface, group, name, predictor=None):
    prefix = f"{condition}__{interface}__"
    if predictor is not None:
        prefix += f"{predictor}__"
    return np.asarray(run[f"{prefix}{group}__{name}"], dtype=np.float64)


def paired_conditions(runs, conditions, difference):
    replicate_maps = []
    for run in runs:
        values = [
            r1.scene_mean_map(
                difference(run, condition),
                run[f"{condition}__common__scene_token"],
            )
            for condition in conditions
        ]
        replicate_maps.append(combine_condition_maps(values))
    return scene_bootstrap(r1.average_maps(replicate_maps))


def seed_directions(runs, conditions, difference):
    result = []
    for run in runs:
        values = [
            r1.scene_mean_map(
                difference(run, condition),
                run[f"{condition}__common__scene_token"],
            )
            for condition in conditions
        ]
        current = combine_condition_maps(values)
        result.append(float(np.mean(list(current.values()))))
    return result


def risk_contrast(runs, conditions, interface, baseline, candidate, reference, predictor=None):
    names = ("high_fde_auprc", "risk_coverage", "false_safe")
    replicate = {name: [] for name in names}
    for run in runs:
        condition_maps = {name: [] for name in names}
        for condition in conditions:
            scene = np.asarray(run[f"{condition}__common__scene_token"])
            error = metric(
                run, condition, interface, reference, "mean_fde", predictor
            )
            event = error > 2.0
            base = metric(run, condition, interface, baseline, "log_area90", predictor)
            cand = metric(run, condition, interface, candidate, "log_area90", predictor)
            current = {name: {} for name in names}
            for token in np.unique(scene):
                selected = scene == token
                if selected.sum() < 2:
                    continue
                key = str(token)
                current_event = event[selected]
                if len(np.unique(current_event)) == 2:
                    current["high_fde_auprc"][key] = float(
                        average_precision_score(current_event, cand[selected])
                        - average_precision_score(current_event, base[selected])
                    )
                current["risk_coverage"][key] = float(
                    r0.risk_coverage(base[selected], error[selected])
                    - r0.risk_coverage(cand[selected], error[selected])
                )
                if current_event.any():
                    base_safe = base[selected] <= np.quantile(base[selected], 0.80)
                    cand_safe = cand[selected] <= np.quantile(cand[selected], 0.80)
                    current["false_safe"][key] = float(
                        (current_event * base_safe).sum() / current_event.sum()
                        - (current_event * cand_safe).sum() / current_event.sum()
                    )
            for name in names:
                condition_maps[name].append(current[name])
        for name in names:
            replicate[name].append(combine_condition_maps(condition_maps[name]))
    return {
        name: scene_bootstrap(r1.average_maps(replicate[name])) for name in names
    }


def equal_coverage_area(run, condition, interface, group, predictor=None):
    mahalanobis = metric(
        run, condition, interface, group, "final_maha2", predictor
    )
    determinant = metric(run, condition, interface, group, "final_det", predictor)
    return float(
        np.median(math.pi * np.quantile(mahalanobis, 0.90) * np.sqrt(determinant))
    )


def comparison_summary(
    runs, conditions, interface, candidate, baselines, reference, predictor=None
):
    result = {}
    for baseline in baselines:
        metrics = {}
        for name in ("energy", "nll", "mean_ade", "mean_fde"):
            metrics[name] = paired_conditions(
                runs,
                conditions,
                lambda run, condition, baseline=baseline, name=name: metric(
                    run, condition, interface, baseline, name, predictor
                )
                - metric(run, condition, interface, candidate, name, predictor),
            )
        ratios = []
        for run in runs:
            current = []
            for condition in conditions:
                current.append(
                    equal_coverage_area(
                        run, condition, interface, candidate, predictor
                    )
                    / equal_coverage_area(
                        run, condition, interface, baseline, predictor
                    )
                )
            ratios.append(float(np.mean(current)))
        result[baseline] = {
            "metrics": metrics,
            "risk": risk_contrast(
                runs,
                conditions,
                interface,
                baseline,
                candidate,
                reference,
                predictor,
            ),
            "equal_coverage_area_ratio_candidate_over_baseline": float(
                np.mean(ratios)
            ),
            "energy_seed_directions": seed_directions(
                runs,
                conditions,
                lambda run, condition, baseline=baseline: metric(
                    run, condition, interface, baseline, "energy", predictor
                )
                - metric(
                    run, condition, interface, candidate, "energy", predictor
                ),
            ),
        }
    return result


def audit_archive_finite(runs):
    count = 0
    finite = True
    for run in runs:
        for name in run.files:
            value = np.asarray(run[name])
            if np.issubdtype(value.dtype, np.number):
                count += int(value.size)
                finite &= bool(np.isfinite(value).all())
    return count, finite


def external_finalize():
    verify_freeze()
    path = OUT / "external/result.json"
    if path.exists():
        raise FileExistsError(path)
    runs = [
        np.load(OUT / "metrics/external" / f"replicate_{index}.npz", allow_pickle=False)
        for index in range(3)
    ]
    for condition in EXTERNAL_CONDITIONS:
        for name in COMMON_FIELDS:
            reference = runs[0][f"{condition}__common__{name}"]
            if not all(
                np.array_equal(reference, run[f"{condition}__common__{name}"])
                for run in runs[1:]
            ):
                raise ValueError(f"external row mismatch: {condition}/{name}")
    scopes = {
        **{condition: (condition,) for condition in EXTERNAL_CONDITIONS},
        **EXTERNAL_FAMILIES,
    }
    results = {}
    for scope, conditions in scopes.items():
        results[scope] = {"r0": {}, "r3p": {}}
        for predictor in r0.PREDICTORS:
            results[scope]["r0"][predictor] = comparison_summary(
                runs,
                conditions,
                "r0",
                "P3",
                ("P1", "P2"),
                "P1",
                predictor,
            )
        results[scope]["r3p"] = comparison_summary(
            runs,
            conditions,
            "r3p",
            "P3_m2_probability",
            r3p.BASELINES,
            "P0_mean_only",
        )
    numeric_count, finite = audit_archive_finite(runs)
    output = {
        "status": "EXTERNAL_TRANSFER_CONFIRMATION_COMPLETE_NO_ENDPOINT",
        "conditions": list(EXTERNAL_CONDITIONS),
        "families": {name: list(value) for name, value in EXTERNAL_FAMILIES.items()},
        "results": results,
        "interpretation_contract": {
            "r0_positive_r3p_null": "central interface boundary transfers",
            "r0_null_or_reversed": "stochastic phenomenon remains bounded to original scope",
            "r3p_positive": "direct conditioning is degradation-type dependent; method development remains closed",
            "both_null_or_reversed": "external corruptions do not confirm the original boundary",
        },
        "source_freeze_sha256": sha256(FREEZE),
        "numeric_values_checked": numeric_count,
        "all_numeric_finite": bool(finite),
        "training_performed": False,
        "official_val_read": False,
        "internal_test_read": False,
        "new_gate_or_endpoint": False,
    }
    write_json(path, output)
    for run in runs:
        run.close()
    append_access(
        "F2_EXTERNAL_RESULTS_OPENED_AND_FINALIZED",
        result=str(path.relative_to(ROOT)),
        result_sha256=sha256(path),
        official_metric_read=False,
        internal_test_read=False,
    )
    print(json.dumps({"external": "complete", "result": str(path)}))


def freeze_claims():
    verify_freeze()
    path = OUT / "claim_freeze.json"
    ledger = PAPER / "FINAL_CLAIM_LEDGER.md"
    external = OUT / "external/result.json"
    attributes_path = OUT / "attributes/result.json"
    if path.exists():
        raise FileExistsError(path)
    for source in (ledger, external, attributes_path):
        if not source.exists():
            raise FileNotFoundError(source)
    value = {
        "status": "F3_FINAL_CLAIM_LEDGER_FROZEN",
        "title": "When Does Perception Uncertainty Help Motion Forecasting? A Controlled Study under Multimodal Sensor Degradation",
        "one_sentence_argument": "Perception distributions improve explicit stochastic pushforward, but this value does not imply that explicit probability representations are necessary under ordinary GT-supervised forecasting; residual value is concentrated in risk ranking.",
        "claim_ledger": {
            "path": str(ledger.relative_to(ROOT)),
            "sha256": sha256(ledger),
        },
        "attributes": {
            "path": str(attributes_path.relative_to(ROOT)),
            "sha256": sha256(attributes_path),
        },
        "external": {
            "path": str(external.relative_to(ROOT)),
            "sha256": sha256(external),
        },
        "freeze_manifest_sha256": sha256(FREEZE),
        "official_comparisons": {
            "r0": ["P1", "P2", "P3"],
            "r2_core": [
                "C0_mean_matched",
                "C2_gaussian_dse",
                "C3_no_tail",
                "C4_no_circular",
                "C5_full_dse",
            ],
            "r2d": [
                "D0_deterministic",
                "D1_mc_teacher",
                "D2_gt_only",
                "D3_risk_distilled",
            ],
            "r3p": list(r3p.GROUPS),
        },
        "metrics": {
            "primary": "trajectory_energy_score",
            "secondary": [
                "high_fde_auprc",
                "risk_coverage",
                "false_safe",
                "equal_coverage_area",
                "mean_ade",
                "mean_fde",
            ],
            "diagnostic": "mixture_nll",
        },
        "bootstrap_unit": "scene",
        "bootstrap_resamples": 10000,
        "official_val_read": False,
        "internal_test_read": False,
        "method_development_closed": True,
    }
    write_json(path, value)
    append_access(
        "F3_FINAL_CLAIMS_FROZEN",
        claim_freeze=str(path.relative_to(ROOT)),
        claim_freeze_sha256=sha256(path),
        official_metric_read=False,
        internal_test_read=False,
    )
    print(json.dumps({"claim_freeze": "complete", "output": str(path)}))


def official_scene(tracks):
    result = {name: np.asarray(value) for name, value in tracks.items()}
    result.update(
        agent_feature=np.asarray(tracks["feature"]),
        agent_logits=np.asarray(tracks["logits"]),
        agent_box=np.asarray(tracks["box"]),
        agent_seed=np.asarray(tracks["seed"]),
        target_agent_index=np.arange(len(tracks["input"]), dtype=np.int64),
    )
    return result


def r2core_condition_metrics(tracks, heads, replicate, device):
    scenes = official_scene(tracks)
    distributions = {
        family: r2.agent_distributions(heads[family], scenes, family, device)
        for family in ("M1", "M2")
    }
    models, checkpoint = r2.load_models(replicate, device)
    groups = (
        "C0_mean_matched",
        "C2_gaussian_dse",
        "C3_no_tail",
        "C4_no_circular",
        "C5_full_dse",
    )
    rng = np.random.default_rng(r2.SAMPLE_SEEDS[replicate] + OFFSET_OFFICIAL)
    storage = {group: {name: [] for name in r0.METRICS} for group in groups}
    for begin in range(0, len(tracks["input"]), r2.EVAL_CHUNK):
        stop = min(begin + r2.EVAL_CHUNK, len(tracks["input"]))
        index = np.arange(begin, stop)
        batch = r2.gather_batch(
            scenes, distributions["M1"], distributions["M2"], index, device
        )
        with torch.inference_mode():
            outputs = {
                group: r1.numpy_output(models[group](*batch)) for group in groups
            }
        count = stop - begin
        uniforms = (rng.random((count, SAMPLES)), rng.random((count, SAMPLES)))
        normals = (
            rng.standard_normal((count, SAMPLES, r0.HORIZON, 2)),
            rng.standard_normal((count, SAMPLES, r0.HORIZON, 2)),
        )
        for group in groups:
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
                np.asarray(tracks["future"])[index],
                np.asarray(tracks["ego_future"])[index],
                temperature,
            )
            for name in r0.METRICS:
                storage[group][name].append(np.asarray(metrics[name], dtype=np.float32))
    return {
        f"r2core__{group}__{name}": np.concatenate(values)
        for group, metrics in storage.items()
        for name, values in metrics.items()
    }


def r2d_condition_metrics(
    tracks, m2, r0_metrics, replicate, device
):
    models, checkpoint = r2d.load_students(replicate, device)
    token = r2d.raw_token(m2)
    rng = np.random.default_rng(
        r2d.STUDENT_SAMPLE_SEEDS[replicate] + OFFSET_OFFICIAL
    )
    storage = {
        group: {name: [] for name in r0.METRICS} for group in r2d.LEARNED
    }
    motion = np.asarray(tracks["input"], dtype=np.float32)
    for begin in range(0, len(motion), r2d.EVAL_CHUNK):
        stop = min(begin + r2d.EVAL_CHUNK, len(motion))
        batch_motion = torch.from_numpy(motion[begin:stop]).to(device)
        batch_token = torch.from_numpy(token[begin:stop]).to(device)
        with torch.inference_mode():
            outputs = {
                group: r2d.numpy_output(model(batch_motion, batch_token))
                for group, model in models.items()
            }
        count = stop - begin
        uniforms = (rng.random((count, SAMPLES)), rng.random((count, SAMPLES)))
        normals = (
            rng.standard_normal((count, SAMPLES, r0.HORIZON, 2)),
            rng.standard_normal((count, SAMPLES, r0.HORIZON, 2)),
        )
        for group in r2d.LEARNED:
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
                np.asarray(tracks["future"])[begin:stop],
                np.asarray(tracks["ego_future"])[begin:stop],
                temperature,
            )
            for name in r0.METRICS:
                storage[group][name].append(np.asarray(metrics[name], dtype=np.float32))
    result = {
        f"r2d__{group}__{name}": np.concatenate(values)
        for group, metrics in storage.items()
        for name, values in metrics.items()
    }
    for group, source in (("D0_deterministic", "P1"), ("D1_mc_teacher", "P3")):
        for name in r0.METRICS:
            result[f"r2d__{group}__{name}"] = np.asarray(
                r0_metrics[f"r0__MLP__{source}__{name}"]
            ).copy()
    return result


def official_evaluate(replicate, device_index):
    verify_freeze(allow_claim_freeze=True)
    claim_freeze = r0.strict_load(OUT / "claim_freeze.json")
    if claim_freeze["status"] != "F3_FINAL_CLAIM_LEDGER_FROZEN":
        raise ValueError("invalid F3 claim freeze")
    if replicate not in range(3) or device_index not in range(4):
        raise ValueError("replicate must be 0--2 and GPU must be 0--3")
    path = OUT / "metrics/official" / f"replicate_{replicate}.npz"
    if path.exists():
        raise FileExistsError(path)
    append_access(
        "OFFICIAL_METRIC_EVALUATION_STARTED",
        replicate=replicate,
        official_metric_read=True,
        internal_test_read=False,
    )
    device = torch.device(f"cuda:{device_index}")
    set_determinism(r3p.SAMPLE_SEEDS[replicate] + OFFSET_OFFICIAL, device)
    tracks = load_confirmation_tracks("official", "clean")
    heads = load_heads(replicate, device)
    head_outputs = {
        name: r0.head_outputs(head, tracks, device) for name, head in heads.items()
    }
    predictor, parameters = r0.load_predictor(device)
    current_r0 = r0_condition_metrics(
        tracks,
        head_outputs,
        predictor,
        parameters,
        replicate,
        0,
        device,
        OFFSET_OFFICIAL,
    )
    r3p_models, r3p_checkpoint = r3p.load_models(replicate, device)
    arrays = {
        "meta__head_seed": np.asarray(r0.HEAD_SEEDS[replicate], dtype=np.int64),
        "meta__r0_seed": np.asarray(
            r0.PROPAGATION_SEEDS[replicate] + OFFSET_OFFICIAL, dtype=np.int64
        ),
        "meta__r2core_seed": np.asarray(
            r2.SAMPLE_SEEDS[replicate] + OFFSET_OFFICIAL, dtype=np.int64
        ),
        "meta__r2d_seed": np.asarray(
            r2d.STUDENT_SAMPLE_SEEDS[replicate] + OFFSET_OFFICIAL, dtype=np.int64
        ),
        "meta__r3p_seed": np.asarray(
            r3p.SAMPLE_SEEDS[replicate] + OFFSET_OFFICIAL, dtype=np.int64
        ),
    }
    for name in COMMON_FIELDS:
        arrays[f"official_clean__common__{name}"] = np.asarray(tracks[name])
    for name, value in current_r0.items():
        arrays[f"official_clean__{name}"] = value
    for name, value in r2core_condition_metrics(
        tracks, heads, replicate, device
    ).items():
        arrays[f"official_clean__{name}"] = value
    for name, value in r2d_condition_metrics(
        tracks, head_outputs["M2"], current_r0, replicate, device
    ).items():
        arrays[f"official_clean__{name}"] = value
    for name, value in learned_condition_metrics(
        tracks,
        head_outputs["M1"],
        head_outputs["M2"],
        r3p_models,
        r3p_checkpoint,
        replicate,
        0,
        device,
        OFFSET_OFFICIAL,
    ).items():
        arrays[f"official_clean__{name}"] = value
    r0.write_npz(path, arrays)
    append_access(
        "OFFICIAL_METRIC_REPLICATE_COMPLETE",
        replicate=replicate,
        archive=str(path.relative_to(ROOT)),
        archive_sha256=sha256(path),
        official_metric_read=True,
        internal_test_read=False,
    )
    print(json.dumps({"official_evaluated": True, "replicate": replicate}))


def official_finalize():
    verify_freeze(allow_claim_freeze=True)
    path = OUT / "official/result.json"
    if path.exists():
        raise FileExistsError(path)
    claim_freeze_path = OUT / "claim_freeze.json"
    claim_freeze = r0.strict_load(claim_freeze_path)
    ledger = ROOT / claim_freeze["claim_ledger"]["path"]
    if sha256(ledger) != claim_freeze["claim_ledger"]["sha256"]:
        raise ValueError("final claim ledger changed after F3")
    runs = [
        np.load(OUT / "metrics/official" / f"replicate_{index}.npz", allow_pickle=False)
        for index in range(3)
    ]
    condition = "official_clean"
    for name in COMMON_FIELDS:
        reference = runs[0][f"{condition}__common__{name}"]
        if not all(
            np.array_equal(reference, run[f"{condition}__common__{name}"])
            for run in runs[1:]
        ):
            raise ValueError(f"official row mismatch: {name}")
    results = {"r0": {}, "r2_core": {}, "r2d": {}, "r3p": {}}
    for predictor in r0.PREDICTORS:
        results["r0"][predictor] = comparison_summary(
            runs,
            (condition,),
            "r0",
            "P3",
            ("P1", "P2"),
            "P1",
            predictor,
        )
    results["r2_core"] = comparison_summary(
        runs,
        (condition,),
        "r2core",
        "C5_full_dse",
        (
            "C0_mean_matched",
            "C2_gaussian_dse",
            "C3_no_tail",
            "C4_no_circular",
        ),
        "C0_mean_matched",
    )
    results["r2d"] = comparison_summary(
        runs,
        (condition,),
        "r2d",
        "D3_risk_distilled",
        ("D0_deterministic", "D1_mc_teacher", "D2_gt_only"),
        "D0_deterministic",
    )
    results["r3p"] = comparison_summary(
        runs,
        (condition,),
        "r3p",
        "P3_m2_probability",
        r3p.BASELINES,
        "P0_mean_only",
    )
    numeric_count, finite = audit_archive_finite(runs)
    output = {
        "status": "OFFICIAL_VALIDATION_FINAL_CONFIRMATION_COMPLETE",
        "role": "confirmation_not_model_selection",
        "results": results,
        "claim_freeze": {
            "path": str(claim_freeze_path.relative_to(ROOT)),
            "sha256": sha256(claim_freeze_path),
        },
        "official_info": {
            "path": str(OFFICIAL_INFO.relative_to(ROOT)),
            "sha256": sha256(OFFICIAL_INFO),
        },
        "track_count": int(len(runs[0][f"{condition}__common__scene_token"])),
        "scene_count": int(
            len(np.unique(runs[0][f"{condition}__common__scene_token"]))
        ),
        "numeric_values_checked": numeric_count,
        "all_numeric_finite": bool(finite),
        "all_three_replicates_present": True,
        "training_performed": False,
        "method_or_metric_selection_performed": False,
        "official_val_read": True,
        "internal_test_read": False,
    }
    write_json(path, output)
    for run in runs:
        run.close()
    append_access(
        "F4_OFFICIAL_RESULTS_FINALIZED",
        result=str(path.relative_to(ROOT)),
        result_sha256=sha256(path),
        official_metric_read=True,
        internal_test_read=False,
    )
    print(json.dumps({"official": "complete", "result": str(path)}))


def self_check():
    assert EXTERNAL_FAMILIES["rotation"] == ("rotation_s2", "rotation_s3")
    assert set(EXTERNAL_CONDITIONS) == set().union(*EXTERNAL_FAMILIES.values())
    assert SAMPLES == r0.SAMPLES == r3p.SAMPLES
    assert BOOTSTRAP_SEED == r0.BOOTSTRAP_SEED
    assert not str(OFFICIAL_INFO).startswith(str(ROOT / "outputs/internal"))
    rng = np.random.default_rng(41)
    scenes = np.asarray(["a", "a", "b", "b"])
    maps = r1.scene_mean_map(rng.normal(size=4), scenes)
    assert set(maps) == {"a", "b"}
    print(
        json.dumps(
            {
                "self_check": "passed",
                "external_conditions": list(EXTERNAL_CONDITIONS),
                "official_groups": {
                    "r0": ["P1", "P2", "P3"],
                    "r2_core": [
                        "C0_mean_matched",
                        "C2_gaussian_dse",
                        "C3_no_tail",
                        "C4_no_circular",
                        "C5_full_dse",
                    ],
                    "r2d": list(r2d.GROUPS),
                    "r3p": list(r3p.GROUPS),
                },
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("attributes")
    subparsers.add_parser("self-check")
    tracks = subparsers.add_parser("prepare-tracks")
    tracks.add_argument("--phase", choices=("external", "official"), required=True)
    tracks.add_argument("--condition", required=True)
    external = subparsers.add_parser("external-evaluate")
    external.add_argument("--replicate", type=int, required=True)
    external.add_argument("--device", type=int, required=True)
    subparsers.add_parser("external-finalize")
    subparsers.add_parser("freeze-claims")
    official = subparsers.add_parser("official-evaluate")
    official.add_argument("--replicate", type=int, required=True)
    official.add_argument("--device", type=int, required=True)
    subparsers.add_parser("official-finalize")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "attributes":
        attributes()
    elif args.command == "prepare-tracks":
        prepare_tracks(args.phase, args.condition)
    elif args.command == "external-evaluate":
        external_evaluate(args.replicate, args.device)
    elif args.command == "external-finalize":
        external_finalize()
    elif args.command == "freeze-claims":
        freeze_claims()
    elif args.command == "official-evaluate":
        official_evaluate(args.replicate, args.device)
    elif args.command == "official-finalize":
        official_finalize()
    else:
        self_check()


if __name__ == "__main__":
    main()
