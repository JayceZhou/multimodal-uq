#!/usr/bin/env python3
"""Read-only DSE transfer audit over frozen RiskProp3D R1 evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tools import riskprop3d_r1 as r1


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/riskprop3d_r15"
R1_RESULT = r1.OUT / "result.json"
FAMILIES = ("lowlight", "dropout", "beam")
RISK_NAMES = ("high_fde_auprc", "risk_coverage", "false_safe")


def metric(run, condition, group, name):
    return np.asarray(run[f"{condition}__{group}__{name}"], dtype=np.float64)


def paired(runs, condition, baseline, candidate, name):
    maps = []
    for run in runs:
        maps.append(
            r1.scene_mean_map(
                metric(run, condition, baseline, name)
                - metric(run, condition, candidate, name),
                run[f"{condition}__common__scene_token"],
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
                metric(run, condition, "B0", "mean_fde"),
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
                metric(run, condition, "B0", "mean_fde"),
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


def condition_area_ratio(run, condition, baseline="B1_MC", candidate="R1_DSE"):
    baseline_maha = metric(run, condition, baseline, "final_maha2")
    candidate_maha = metric(run, condition, candidate, "final_maha2")
    baseline_area = (
        math.pi
        * np.quantile(baseline_maha, 0.90)
        * np.sqrt(metric(run, condition, baseline, "final_det"))
    )
    candidate_area = (
        math.pi
        * np.quantile(candidate_maha, 0.90)
        * np.sqrt(metric(run, condition, candidate, "final_det"))
    )
    return float(np.median(candidate_area) / np.median(baseline_area))


def significant(report):
    return report["ci95"][0] is not None and report["ci95"][0] > 0


def non_significantly_worse(report):
    return report["ci95"][1] is not None and report["ci95"][1] >= 0


def input_validity(runs, source_result):
    metric_hashes = {
        str(index): r1.sha256(r1.OUT / "metrics" / f"replicate_{index}.npz")
        for index in range(3)
    }
    hashes_match = metric_hashes == source_result["input_sha256"]["metrics"]
    seeds_match = True
    common_match = True
    finite = True
    numeric_values = 0
    for index, run in enumerate(runs):
        seeds_match &= (
            int(run["meta__head_seed"]) == r1.HEAD_SEEDS[index]
            and int(run["meta__model_seed"]) == r1.MODEL_SEEDS[index]
            and int(run["meta__sampling_seed"]) == r1.SAMPLE_SEEDS[index]
        )
        for name in run.files:
            value = np.asarray(run[name])
            if np.issubdtype(value.dtype, np.number):
                finite &= bool(np.isfinite(value).all())
                numeric_values += int(value.size)
    for condition in r1.CONDITIONS:
        for name in r1.COMMON_FIELDS:
            if name == "high_uncertainty":
                continue
            reference = runs[0][f"{condition}__common__{name}"]
            common_match &= all(
                np.array_equal(reference, run[f"{condition}__common__{name}"])
                for run in runs[1:]
            )
    checkpoint_sha256 = {}
    checkpoint_valid = True
    for index in range(3):
        path = r1.OUT / "checkpoints" / f"replicate_{index}.pt"
        checkpoint_sha256[str(index)] = r1.sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint_valid &= (
            checkpoint["head_seed"] == r1.HEAD_SEEDS[index]
            and checkpoint["model_seed"] == r1.MODEL_SEEDS[index]
            and checkpoint["sampling_seed"] == r1.SAMPLE_SEEDS[index]
            and "R1_DSE" in checkpoint["models"]
        )
    validity = {
        "V1_source_R1_valid_stop": bool(
            source_result["valid"]
            and source_result["endpoint"] == "STOP RISKPROP3D METHOD ESCALATION"
        ),
        "V2_metric_hashes_match_R1": hashes_match,
        "V3_three_seed_and_common_row_authority": bool(
            seeds_match and common_match and checkpoint_valid
        ),
        "V4_all_numeric_metric_values_finite": bool(finite and numeric_values > 0),
        "V5_forbidden_splits_unread": bool(
            not source_result["official_val_read"]
            and not source_result["internal_test_read"]
        ),
    }
    return validity, metric_hashes, checkpoint_sha256, numeric_values


def audit() -> None:
    path = OUT / "result.json"
    if path.exists():
        raise FileExistsError(path)
    source_result = r1.strict_load(R1_RESULT)
    runs = [
        np.load(r1.OUT / "metrics" / f"replicate_{index}.npz", allow_pickle=False)
        for index in range(3)
    ]
    validity, metric_hashes, checkpoint_hashes, numeric_values = input_validity(
        runs, source_result
    )

    per_condition = {}
    for condition in r1.CONDITIONS:
        per_condition[condition] = {
            comparison: {
                name: paired(runs, condition, baseline, "R1_DSE", name)
                for name in ("energy", "nll")
            }
            for comparison, baseline in (
                ("R1_DSE_over_B2_concat", "B2_concat"),
                ("R1_DSE_over_B1_MC", "B1_MC"),
            )
        }

    transfer = {
        "full": {
            family: family_paired(
                runs,
                family,
                "B2_concat",
                "R1_DSE",
                "energy",
                severities=(1, 2, 3),
            )
            for family in FAMILIES
        },
        "without_s3": {
            family: family_paired(
                runs,
                family,
                "B2_concat",
                "R1_DSE",
                "energy",
                severities=(1, 2),
            )
            for family in FAMILIES
        },
    }

    against_mc = per_condition["clean_s0"]["R1_DSE_over_B1_MC"]["energy"]
    runtime = {}
    for group in ("B1_MC", "R1_DSE"):
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
    speedup = runtime["B1_MC"]["milliseconds_per_track"] / max(
        runtime["R1_DSE"]["milliseconds_per_track"], 1e-12
    )

    condition_sharpness = {
        condition: float(
            np.mean([condition_area_ratio(run, condition) for run in runs])
        )
        for condition in r1.CONDITIONS
    }
    sharpness = {
        "clean": {
            "equal_coverage_area_ratio": condition_sharpness["clean_s0"]
        },
        "full": {
            family: {
                "equal_coverage_area_ratio": float(
                    np.mean(
                        [
                            condition_sharpness[f"{family}_s{severity}"]
                            for severity in (1, 2, 3)
                        ]
                    )
                )
            }
            for family in FAMILIES
        },
        "without_s3": {
            family: {
                "equal_coverage_area_ratio": float(
                    np.mean(
                        [
                            condition_sharpness[f"{family}_s{severity}"]
                            for severity in (1, 2)
                        ]
                    )
                )
            }
            for family in FAMILIES
        },
        "per_condition": condition_sharpness,
    }

    risk = {
        "clean": risk_difference(runs, "clean_s0", "B1_MC", "R1_DSE"),
        "without_s3": {
            family: family_risk(runs, family, "B1_MC", "R1_DSE")
            for family in FAMILIES
        },
    }

    t1 = all(
        significant(per_condition["clean_s0"]["R1_DSE_over_B2_concat"][name])
        for name in ("energy", "nll")
    )
    t2 = all(
        sum(significant(values[family]) for family in FAMILIES) >= 2
        and all(
            values[family]["ci95"][1] is not None
            and values[family]["ci95"][1] >= 0
            for family in FAMILIES
        )
        for values in (transfer["full"], transfer["without_s3"])
    )
    t3 = significant(against_mc) or (
        non_significantly_worse(against_mc) and speedup >= 2.0
    )
    sharpness_values = [sharpness["clean"]["equal_coverage_area_ratio"]]
    sharpness_values.extend(
        sharpness[scope][family]["equal_coverage_area_ratio"]
        for scope in ("full", "without_s3")
        for family in FAMILIES
    )
    t4 = all(value <= 1.10 for value in sharpness_values)
    t5 = any(significant(risk["clean"][name]) for name in RISK_NAMES) or any(
        significant(risk["without_s3"][family][name])
        for family in FAMILIES
        for name in RISK_NAMES
    )
    gates = {
        "T1_clean_replication": t1,
        "T2_cross_degradation": t2,
        "T3_against_monte_carlo": t3,
        "T4_equal_coverage_sharpness": t4,
        "T5_future_risk_utility": t5,
    }
    valid = all(validity.values())
    passed = valid and all(gates.values())
    endpoint = (
        "INVALID R1.5 AUDIT"
        if not valid
        else "GO TO RISKPROP3D-CORE"
        if passed
        else "STOP DSE AS THIRD-WORK CORE"
    )
    result = {
        "against_monte_carlo": {
            "clean_energy": against_mc,
            "runtime": runtime,
            "speedup": speedup,
        },
        "closed_negative_control": {
            "clean_energy": source_result["interaction"]["energy"],
            "replicate_directions": source_result[
                "replicate_clean_energy_directions"
            ]["R1_interaction_over_R1_DSE"],
            "status": "DISTRIBUTIONAL INTERACTION V1 NOT SUPPORTED",
        },
        "endpoint": endpoint,
        "gates": gates,
        "input_sha256": {
            "checkpoints": checkpoint_hashes,
            "code": r1.sha256(Path(__file__)),
            "metrics": metric_hashes,
            "source_r1_result": r1.sha256(R1_RESULT),
        },
        "internal_test_read": False,
        "numeric_metric_value_count": numeric_values,
        "official_val_read": False,
        "passed": passed,
        "per_condition_proper_scores": per_condition,
        "risk": risk,
        "sharpness": sharpness,
        "source_r1_endpoint": source_result["endpoint"],
        "track_counts": source_result["track_counts"],
        "transfer_energy": transfer,
        "valid": valid,
        "validity": validity,
    }
    r1.write_json(path, result)
    for run in runs:
        run.close()
    print(
        json.dumps(
            {"endpoint": endpoint, "gates": gates, "validity": validity},
            indent=2,
            sort_keys=True,
        )
    )


def self_check() -> None:
    scenes = np.asarray(["a", "a", "b", "b"])
    value = np.asarray([1.0, 3.0, 2.0, 4.0])
    mapping = r1.scene_mean_map(value, scenes)
    if mapping != {"a": 2.0, "b": 3.0}:
        raise AssertionError("scene pairing check failed")
    report = r1.bootstrap(r1.average_maps([mapping, mapping]), repeats=100)
    if report["mean"] != 2.5 or report["count"] != 2:
        raise AssertionError("bootstrap aggregation check failed")
    if not significant({"ci95": [0.1, 0.2]}):
        raise AssertionError("gate direction check failed")
    print(json.dumps({"self_check": "passed"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "self-check"))
    args = parser.parse_args()
    if args.command == "audit":
        audit()
    else:
        self_check()


if __name__ == "__main__":
    main()
