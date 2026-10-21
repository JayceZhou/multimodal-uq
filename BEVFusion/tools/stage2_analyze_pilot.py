#!/usr/bin/env python3
"""Analyze completed fixed-subset Stage 2 pilot predictions."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import mmcv
import numpy as np
from scipy.optimize import linear_sum_assignment

from stage2_run_pilot import CLASSES, MODELS, ROOT, matched_records, write_json


BOOTSTRAPS = 1000
PRECISION_TARGETS = (0.70, 0.80, 0.90)
MAIN_PRECISION_TARGET = "0.80"
FIELDS = (
    "tp", "fp", "fn", "recall", "precision", "class_errors",
    "center_error", "size_error", "orientation_error", "velocity_error",
    "prediction_count_score_ge_0_1", "has_prediction_score_ge_0_1",
)


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream]


def prediction_path(run_id):
    return ROOT / "predictions" / f"{run_id}.pkl"


def stage1_prediction_path(model):
    return Path("outputs/stage1/baseline") / f"{model}.pkl"


def token_count(run_id, task=None):
    if task is not None:
        return task["token_set"]
    return run_id.rsplit("-", 1)[-1]


def regenerate_matches(plan):
    runs = {item["run_id"]: item for item in plan["tasks"]}
    for count in ("200", "197"):
        for model in MODELS:
            run_id = f"stage1-clean-{model}-{count}"
            runs[run_id] = {"token_set": count}
    for run_id, item in runs.items():
        infos = sorted(
            mmcv.load(ROOT / f"nuscenes_infos_val_{item['token_set']}.pkl")["infos"],
            key=lambda value: value["timestamp"],
        )
        outputs = mmcv.load(prediction_path(run_id))
        records = matched_records(infos, outputs, run_id)
        path = ROOT / "matched_errors" / f"{run_id}.jsonl"
        path.write_text("".join(json.dumps(value, allow_nan=False) + "\n" for value in records))


def clean_run(item):
    if item["token_set"] == "197":
        return f"stage1-clean-{item['model']}-197"
    if item["model"] == "fusion":
        return "fusion-clean-s0"
    return f"stage1-clean-{item['model']}-200"


def paired_summary(clean, corrupt, fields=FIELDS, seed=20260707):
    rng = np.random.default_rng(seed)
    result = {}
    for field in fields:
        a = np.asarray([np.nan if item.get(field) is None else item[field] for item in clean], dtype=float)
        b = np.asarray([np.nan if item.get(field) is None else item[field] for item in corrupt], dtype=float)
        valid = np.isfinite(a) & np.isfinite(b)
        if not valid.any():
            result[field] = None
            continue
        a, b = a[valid], b[valid]
        delta = b - a
        indices = rng.integers(0, len(delta), size=(BOOTSTRAPS, len(delta)))
        samples = delta[indices].mean(axis=1)
        clean_mean = float(a.mean())
        absolute = float(delta.mean())
        result[field] = {
            "paired_count": int(len(delta)),
            "clean_mean": clean_mean,
            "corrupted_mean": float(b.mean()),
            "absolute_delta": absolute,
            "relative_delta": absolute / abs(clean_mean) if clean_mean else None,
            "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        }
    return result


def bootstrap_all(plan):
    metadata = {item["sample_token"]: item for item in read_jsonl("outputs/stage2/complementarity/scene_index.jsonl")}
    for item in plan["tasks"]:
        run_id = item["run_id"]
        corrupt = read_jsonl(ROOT / "matched_errors" / f"{run_id}.jsonl")
        clean = read_jsonl(ROOT / "matched_errors" / f"{clean_run(item)}.jsonl")
        clean_by_token = {value["sample_token"]: value for value in clean}
        clean = [clean_by_token[value["sample_token"]] for value in corrupt]
        seed = int.from_bytes(hashlib.sha256(run_id.encode()).digest()[:8], "big")
        output = {"run_id": run_id, "bootstrap_samples": BOOTSTRAPS, "overall": paired_summary(clean, corrupt, seed=seed)}
        strata = {}
        for key in ("time_of_day", "city", "weather"):
            for value in sorted({metadata[item["sample_token"]][key] for item in corrupt}):
                indices = [index for index, row in enumerate(corrupt) if metadata[row["sample_token"]][key] == value]
                strata[f"{key}:{value}"] = paired_summary(
                    [clean[index] for index in indices], [corrupt[index] for index in indices], seed=seed,
                )
        output["strata"] = strata
        output["classes"] = {}
        for name in CLASSES:
            class_clean = [value["per_class"][name] for value in clean]
            class_corrupt = [value["per_class"][name] for value in corrupt]
            output["classes"][name] = paired_summary(
                class_clean, class_corrupt,
                fields=("tp", "fp", "fn", "recall", "precision", "center_error"),
                seed=seed,
            )
        write_json(ROOT / "bootstrap" / f"{run_id}.json", output)


def metrics(run_id):
    return json.loads((ROOT / "metrics" / run_id / "pilot_subset_metrics.json").read_text())


def input_proxy(item):
    records = read_jsonl(ROOT / "manifests" / item["run_id"] / "samples.jsonl")
    corruption = item["corruption_type"]
    if item["severity"] == 0:
        return {"name": "clean", "value": 0.0}
    if corruption == "camera_lowlight":
        values = [row["corrupted_brightness"] / row["original_brightness"] for row in records]
        return {"name": "brightness_ratio", "value": float(np.mean(values))}
    if corruption == "lidar_dropout":
        values = [row["retained_point_count"] / row["original_point_count"] for row in records]
        return {"name": "retained_point_ratio", "value": float(np.mean(values))}
    if corruption == "lidar_beam_reduction":
        return {"name": "retained_rings", "value": records[0]["corruption_parameters"]["retained_ring_count"]}
    if corruption.startswith("camera_extrinsic_translation"):
        return {"name": "translation_m", "value": records[0]["corruption_parameters"]["translation_m"]}
    if corruption.startswith("camera_extrinsic_rotation"):
        return {"name": "rotation_deg", "value": records[0]["corruption_parameters"]["rotation_deg"]}
    if corruption == "camera_temporal_offset":
        values = [abs(value) for row in records for value in row["temporal_offset_ms"].values()]
        return {"name": "absolute_offset_ms", "value": float(np.mean(values))}
    return {"name": "feature_branch_missing", "value": 1.0}


def build_curves(plan):
    curves = defaultdict(lambda: defaultdict(list))
    for item in plan["tasks"]:
        if item["family"] == "clean":
            continue
        value = metrics(item["run_id"])
        curves[item["corruption_type"]][item["model"]].append({
            "severity": item["severity"], "input_proxy": input_proxy(item),
            **{key: value[key] for key in value if key.startswith("pilot_subset_")},
        })
    output = {key: dict(value) for key, value in curves.items()}
    write_json(ROOT / "metrics" / "curves.json", output)
    return output


def records_by_token(run_id):
    return {item["sample_token"]: item for item in read_jsonl(ROOT / "matched_errors" / f"{run_id}.jsonl")}


def dominates(left, right):
    return (
        left["tp"] >= right["tp"] and left["fp"] <= right["fp"] and left["fn"] <= right["fn"]
        and (left["tp"] > right["tp"] or left["fp"] < right["fp"] or left["fn"] < right["fn"])
    )


def compare_triplet(fusion_id, camera_id, lidar_id, tokens=None):
    fusion, camera, lidar = map(records_by_token, (fusion_id, camera_id, lidar_id))
    common = sorted(set(fusion) & set(camera) & set(lidar) & (set(tokens) if tokens else set(fusion)))
    better = [dominates(fusion[token], camera[token]) and dominates(fusion[token], lidar[token]) for token in common]
    negative = [dominates(camera[token], fusion[token]) or dominates(lidar[token], fusion[token]) for token in common]
    component_negative = {
        "fusion_tp_below_best_single_ratio": float(np.mean([
            fusion[token]["tp"] < max(camera[token]["tp"], lidar[token]["tp"])
            for token in common
        ])),
        "fusion_fp_above_best_single_ratio": float(np.mean([
            fusion[token]["fp"] > min(camera[token]["fp"], lidar[token]["fp"])
            for token in common
        ])),
        "fusion_fn_above_best_single_ratio": float(np.mean([
            fusion[token]["fn"] > min(camera[token]["fn"], lidar[token]["fn"])
            for token in common
        ])),
    }
    deltas = {}
    for name, source in (("camera", camera), ("lidar", lidar)):
        deltas[name] = {
            field: float(np.mean([fusion[token][field] - source[token][field] for token in common]))
            for field in ("tp", "fp", "fn")
        }
    return {
        "sample_count": len(common),
        "fusion_dominates_both_count": int(sum(better)),
        "fusion_dominates_both_ratio": float(np.mean(better)),
        "fusion_worse_than_a_single_modality_count": int(sum(negative)),
        "negative_fusion_ratio": float(np.mean(negative)),
        "component_negative_ratios": component_negative,
        "mean_fusion_minus_single_modality": deltas,
    }


def fusion_comparisons(plan):
    tasks = {item["run_id"]: item for item in plan["tasks"]}
    output = {
        "definition": "component-wise dominance: TP no lower, FP/FN no higher, with at least one strict inequality",
        "clean": compare_triplet("stage1-clean-fusion-200", "stage1-clean-camera-200", "stage1-clean-lidar-200"),
        "tasks": {},
    }
    for item in plan["tasks"]:
        if item["model"] != "fusion" or item["family"] == "clean":
            continue
        suffix = item["run_id"][len("fusion-"):]
        camera_id, lidar_id = "stage1-clean-camera-200", "stage1-clean-lidar-200"
        if item["token_set"] == "197":
            camera_id, lidar_id = "stage1-clean-camera-197", "stage1-clean-lidar-197"
        if item["corruption_type"].startswith("camera_") and item["family"] != "missing":
            camera_id = f"camera-{suffix}"
        if item["corruption_type"].startswith("lidar_") and item["family"] != "missing":
            lidar_id = f"lidar-{suffix}"
        output["tasks"][item["run_id"]] = {
            "family": item["family"],
            **compare_triplet(item["run_id"], camera_id, lidar_id),
        }
    scenes = read_jsonl("outputs/stage2/complementarity/scene_index.jsonl")
    output["complementarity_control"] = {}
    for label in ("camera_weak_lidar_strong", "lidar_weak_camera_strong", "both_strong", "both_weak"):
        tokens = [row["sample_token"] for row in scenes if row["condition_counts"].get(label, 0)]
        output["complementarity_control"][label] = compare_triplet(
            "stage1-clean-fusion-200", "stage1-clean-camera-200", "stage1-clean-lidar-200", tokens,
        )
    write_json(ROOT / "metrics" / "fusion_comparison.json", output)
    return output


def recognized_gt(info):
    names = np.asarray(info["gt_names"])
    keep = np.isin(names, CLASSES)
    return np.asarray(info["gt_boxes"])[keep], np.asarray([CLASSES.index(name) for name in names[keep]])


def recognized_gt_detail(info):
    names = np.asarray(info["gt_names"])
    keep = np.isin(names, CLASSES)
    return (
        np.asarray(info["gt_boxes"])[keep],
        np.asarray([CLASSES.index(name) for name in names[keep]]),
        np.asarray(info["gt_velocity"])[keep],
    )


def prediction_arrays(prediction):
    return (
        prediction["boxes_3d"].tensor.detach().cpu().numpy(),
        prediction["scores_3d"].detach().cpu().numpy(),
        prediction["labels_3d"].detach().cpu().numpy(),
    )


def class_precision_entries(infos, predictions):
    entries = {index: [] for index in range(len(CLASSES))}
    for info, prediction in zip(infos, predictions):
        gt_boxes, gt_labels, _ = recognized_gt_detail(info)
        boxes, scores, labels = prediction_arrays(prediction)
        for class_index in range(len(CLASSES)):
            pred_indices = np.flatnonzero(labels == class_index)
            gt_indices = np.flatnonzero(gt_labels == class_index)
            matched_predictions = set()
            if len(pred_indices) and len(gt_indices):
                distances = np.linalg.norm(
                    gt_boxes[gt_indices, None, :2] - boxes[pred_indices][None, :, :2],
                    axis=2,
                )
                rows, columns = linear_sum_assignment(distances)
                for row, column in zip(rows, columns):
                    if distances[row, column] <= 2.0:
                        matched_predictions.add(int(pred_indices[column]))
            for pred_index in pred_indices:
                entries[class_index].append((float(scores[pred_index]), pred_index in matched_predictions))
    return entries


def threshold_for_precision(entries, target):
    if not entries:
        return {"threshold": 1.0, "target_met": False, "retained": 0, "precision": None}
    ordered = sorted(entries, key=lambda item: item[0], reverse=True)
    tp = np.cumsum([item[1] for item in ordered], dtype=float)
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    precision = tp / ranks
    valid = np.flatnonzero(precision >= target)
    index = int(valid[-1]) if len(valid) else 0
    return {
        "threshold": float(ordered[index][0]),
        "target_met": bool(len(valid)),
        "retained": int(index + 1),
        "precision": float(precision[index]),
        "total_predictions": len(ordered),
    }


def precision_controlled_thresholds():
    output_path = ROOT / "metrics" / "precision_controlled_score_thresholds.json"
    if output_path.exists():
        return json.loads(output_path.read_text())
    infos = sorted(
        mmcv.load("outputs/stage1/baseline/nuscenes_infos_val.pkl")["infos"],
        key=lambda value: value["timestamp"],
    )
    output = {
        "source": "Stage 1 full clean validation predictions",
        "matching": "same-class GT assignment within 2.0 m; scores used only as detection filters",
        "targets": [f"{value:.2f}" for value in PRECISION_TARGETS],
        "models": {},
    }
    for model in ("camera", "lidar"):
        predictions = mmcv.load(stage1_prediction_path(model))
        entries = class_precision_entries(infos, predictions)
        output["models"][model] = {}
        for target in PRECISION_TARGETS:
            output["models"][model][f"{target:.2f}"] = {
                CLASSES[class_index]: threshold_for_precision(entries[class_index], target)
                for class_index in range(len(CLASSES))
            }
    write_json(output_path, output)
    return output


def class_threshold_array(thresholds, model, target):
    table = thresholds["models"][model][target]
    return np.asarray([table[name]["threshold"] for name in CLASSES], dtype=float)


def associate_thresholded(info, prediction, thresholds):
    gt, gt_labels, gt_velocity = recognized_gt_detail(info)
    boxes, scores, labels = prediction_arrays(prediction)
    keep = np.asarray([
        label < len(thresholds) and score >= thresholds[int(label)]
        for score, label in zip(scores, labels)
    ], dtype=bool)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    result = {}
    if len(gt) and len(boxes):
        distances = np.linalg.norm(gt[:, None, :2] - boxes[None, :, :2], axis=2)
        rows, columns = linear_sum_assignment(distances)
        for row, column in zip(rows, columns):
            if distances[row, column] <= 4.0:
                box = boxes[column]
                center = float(np.linalg.norm(gt[row, :2] - box[:2]))
                yaw = float(abs(np.arctan2(np.sin(box[6] - gt[row, 6]), np.cos(box[6] - gt[row, 6]))))
                velocity = float(np.linalg.norm(box[7:9] - gt_velocity[row]))
                result[int(row)] = {
                    "box": box,
                    "label": int(labels[column]),
                    "score": float(scores[column]),
                    "center_error": center,
                    "yaw_error": yaw,
                    "velocity_error": velocity,
                    "correct": bool(labels[column] == gt_labels[row] and center <= 2.0),
                }
    return gt, gt_labels, result


def disagreement(camera_match, lidar_match):
    if not camera_match or not lidar_match:
        return {
            "dual_detected": False,
            "conflict": False,
            "class_mismatch": False,
            "center_gap": 0.0,
            "yaw_gap": 0.0,
        }
    center_gap = float(np.linalg.norm(camera_match["box"][:2] - lidar_match["box"][:2]))
    yaw_gap = float(abs(np.arctan2(
        np.sin(camera_match["box"][6] - lidar_match["box"][6]),
        np.cos(camera_match["box"][6] - lidar_match["box"][6]),
    )))
    class_mismatch = camera_match["label"] != lidar_match["label"]
    return {
        "dual_detected": True,
        "conflict": bool(class_mismatch or center_gap > 2.0 or yaw_gap > np.pi / 4),
        "class_mismatch": bool(class_mismatch),
        "center_gap": center_gap,
        "yaw_gap": yaw_gap,
    }


def grew(clean, corrupt):
    return (
        (corrupt["class_mismatch"] and not clean["class_mismatch"])
        or corrupt["center_gap"] > clean["center_gap"] + 1e-6
        or corrupt["yaw_gap"] > clean["yaw_gap"] + 1e-6
    )


def summarize_conflict_counts(counts):
    for key in (
        "gt_count", "clean_dual_detected", "corrupt_dual_detected",
        "clean_disagreement", "raw_disagreement", "paired_induced_conflict",
        "conflict_persistence", "conflict_resolution", "both_modalities_wrong",
        "at_least_one_modality_correct",
    ):
        counts.setdefault(key, 0)
    gt = max(counts["gt_count"], 1)
    clean_conflicts = counts["clean_disagreement"]
    induced_base = max(counts["gt_count"] - clean_conflicts, 1)
    induced = counts["paired_induced_conflict"]
    return {
        **dict(counts),
        "raw_disagreement_rate": counts["raw_disagreement"] / gt,
        "clean_disagreement_rate": clean_conflicts / gt,
        "excess_disagreement": (counts["raw_disagreement"] - clean_conflicts) / gt,
        "paired_conflict_induction_rate": induced / induced_base,
        "conflict_persistence_rate": counts["conflict_persistence"] / clean_conflicts if clean_conflicts else 0.0,
        "conflict_resolution_rate": counts["conflict_resolution"] / clean_conflicts if clean_conflicts else 0.0,
        "both_modalities_wrong_ratio": counts["both_modalities_wrong"] / induced if induced else 0.0,
        "at_least_one_modality_correct_ratio": counts["at_least_one_modality_correct"] / induced if induced else 0.0,
        "raw_disagreement_yield_among_dual": counts["raw_disagreement"] / counts["corrupt_dual_detected"] if counts["corrupt_dual_detected"] else 0.0,
        "clean_disagreement_yield_among_dual": clean_conflicts / counts["clean_dual_detected"] if counts["clean_dual_detected"] else 0.0,
    }


def add_regret(counts, camera_match, lidar_match, fusion_match):
    fusion_correct = bool(fusion_match and fusion_match["correct"])
    camera_correct = camera_match["correct"]
    lidar_correct = lidar_match["correct"]
    if camera_correct and not fusion_correct:
        counts["fusion_new_fn_vs_camera"] += 1
    if lidar_correct and not fusion_correct:
        counts["fusion_new_fn_vs_lidar"] += 1
    if fusion_match and not fusion_correct and (camera_correct or lidar_correct):
        counts["fusion_new_fp"] += 1
    for field, output_name in (
        ("center_error", "fusion_center_worse_than_best_single"),
        ("yaw_error", "fusion_orientation_worse_than_best_single"),
        ("velocity_error", "fusion_velocity_worse_than_best_single"),
    ):
        if fusion_match:
            best = min(camera_match[field], lidar_match[field])
            if fusion_match[field] > best + 1e-6:
                counts[output_name] += 1
    if fusion_correct and (not camera_correct or not lidar_correct):
        counts["fusion_repairs_error"] += 1
    if not fusion_correct and (not camera_correct or not lidar_correct):
        counts["fusion_inherits_error"] += 1
    if not fusion_correct and (camera_correct or lidar_correct):
        counts["fusion_extra_error"] += 1


def paired_conflict_counts(camera_id, fusion_id, token_set, thresholds, target):
    infos = sorted(mmcv.load(ROOT / f"nuscenes_infos_val_{token_set}.pkl")["infos"], key=lambda value: value["timestamp"])
    clean_camera = mmcv.load(prediction_path(f"stage1-clean-camera-{token_set}"))
    clean_lidar = mmcv.load(prediction_path(f"stage1-clean-lidar-{token_set}"))
    corrupt_camera = mmcv.load(prediction_path(camera_id))
    corrupt_lidar = clean_lidar
    corrupt_fusion = mmcv.load(prediction_path(fusion_id))
    camera_thresholds = class_threshold_array(thresholds, "camera", target)
    lidar_thresholds = class_threshold_array(thresholds, "lidar", target)
    fusion_thresholds = np.minimum(camera_thresholds, lidar_thresholds)
    counts, regret = Counter(), Counter()
    object_rows = []
    for sample_index, (info, clean_c_pred, clean_l_pred, corrupt_c_pred, corrupt_l_pred, fusion_pred) in enumerate(zip(
        infos, clean_camera, clean_lidar, corrupt_camera, corrupt_lidar, corrupt_fusion
    )):
        gt, labels, clean_c = associate_thresholded(info, clean_c_pred, camera_thresholds)
        _, _, clean_l = associate_thresholded(info, clean_l_pred, lidar_thresholds)
        _, _, corrupt_c = associate_thresholded(info, corrupt_c_pred, camera_thresholds)
        _, _, corrupt_l = associate_thresholded(info, corrupt_l_pred, lidar_thresholds)
        _, _, fusion = associate_thresholded(info, fusion_pred, fusion_thresholds)
        counts["gt_count"] += len(gt)
        for gt_index in range(len(gt)):
            clean_state = disagreement(clean_c.get(gt_index), clean_l.get(gt_index))
            corrupt_state = disagreement(corrupt_c.get(gt_index), corrupt_l.get(gt_index))
            counts["clean_dual_detected"] += int(clean_state["dual_detected"])
            counts["corrupt_dual_detected"] += int(corrupt_state["dual_detected"])
            counts["clean_disagreement"] += int(clean_state["conflict"])
            counts["raw_disagreement"] += int(corrupt_state["conflict"])
            counts["conflict_persistence"] += int(clean_state["conflict"] and corrupt_state["conflict"])
            counts["conflict_resolution"] += int(clean_state["conflict"] and not corrupt_state["conflict"])
            induced = (
                not clean_state["conflict"]
                and corrupt_state["conflict"]
                and corrupt_state["dual_detected"]
                and grew(clean_state, corrupt_state)
            )
            if not induced:
                continue
            counts["paired_induced_conflict"] += 1
            c, l = corrupt_c[gt_index], corrupt_l[gt_index]
            if not c["correct"] and not l["correct"]:
                counts["both_modalities_wrong"] += 1
            if c["correct"] or l["correct"]:
                counts["at_least_one_modality_correct"] += 1
            add_regret(regret, c, l, fusion.get(gt_index))
            object_rows.append({
                "run_id": fusion_id,
                "precision_target": target,
                "sample_token": info["token"],
                "gt_index": gt_index,
                "category": CLASSES[int(labels[gt_index])],
                "clean_center_gap": clean_state["center_gap"],
                "corrupt_center_gap": corrupt_state["center_gap"],
                "clean_yaw_gap": clean_state["yaw_gap"],
                "corrupt_yaw_gap": corrupt_state["yaw_gap"],
                "camera_correct": c["correct"],
                "lidar_correct": l["correct"],
                "fusion_correct": bool(fusion.get(gt_index) and fusion[gt_index]["correct"]),
            })
    summary = summarize_conflict_counts(counts)
    induced = max(summary.get("paired_induced_conflict", 0), 1)
    summary["fusion_regret_on_induced_conflicts"] = {
        **dict(regret),
        "object_count": summary["paired_induced_conflict"],
        "fusion_new_fn_vs_camera_rate": regret["fusion_new_fn_vs_camera"] / induced,
        "fusion_new_fn_vs_lidar_rate": regret["fusion_new_fn_vs_lidar"] / induced,
        "fusion_new_fp_rate": regret["fusion_new_fp"] / induced,
        "fusion_center_worse_than_best_single_rate": regret["fusion_center_worse_than_best_single"] / induced,
        "fusion_orientation_worse_than_best_single_rate": regret["fusion_orientation_worse_than_best_single"] / induced,
        "fusion_velocity_worse_than_best_single_rate": regret["fusion_velocity_worse_than_best_single"] / induced,
        "fusion_inherits_error_rate": regret["fusion_inherits_error"] / induced,
        "fusion_repairs_error_rate": regret["fusion_repairs_error"] / induced,
        "fusion_extra_error_rate": regret["fusion_extra_error"] / induced,
    }
    return summary, object_rows


def associate(info, prediction, threshold):
    gt, gt_labels = recognized_gt(info)
    boxes = prediction["boxes_3d"].tensor.detach().cpu().numpy()
    scores = prediction["scores_3d"].detach().cpu().numpy()
    labels = prediction["labels_3d"].detach().cpu().numpy()
    keep = scores >= threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    result = {}
    if len(gt) and len(boxes):
        distances = np.linalg.norm(gt[:, None, :2] - boxes[None, :, :2], axis=2)
        rows, columns = linear_sum_assignment(distances)
        for row, column in zip(rows, columns):
            if distances[row, column] <= 4.0:
                result[int(row)] = {
                    "box": boxes[column], "label": int(labels[column]), "score": float(scores[column]),
                    "correct": bool(labels[column] == gt_labels[row] and distances[row, column] <= 2.0),
                }
    return gt, gt_labels, result


def conflict_counts(camera_id, lidar_id, fusion_id, token_set):
    infos = sorted(mmcv.load(ROOT / f"nuscenes_infos_val_{token_set}.pkl")["infos"], key=lambda value: value["timestamp"])
    camera, lidar, fusion = map(mmcv.load, map(prediction_path, (camera_id, lidar_id, fusion_id)))
    sensitivities = {}
    for threshold in (0.1, 0.3, 0.5):
        counts = Counter()
        for info, camera_prediction, lidar_prediction, fusion_prediction in zip(infos, camera, lidar, fusion):
            _, _, camera_matches = associate(info, camera_prediction, threshold)
            _, _, lidar_matches = associate(info, lidar_prediction, threshold)
            _, _, fusion_matches = associate(info, fusion_prediction, threshold)
            for index in set(camera_matches) & set(lidar_matches):
                counts["dual_detected"] += 1
                c, l = camera_matches[index], lidar_matches[index]
                center_gap = np.linalg.norm(c["box"][:2] - l["box"][:2])
                yaw_gap = abs(np.arctan2(np.sin(c["box"][6] - l["box"][6]), np.cos(c["box"][6] - l["box"][6])))
                conflict = c["label"] != l["label"] or center_gap > 2.0 or yaw_gap > np.pi / 4
                if conflict:
                    counts["actual_spatial_conflict"] += 1
                    if c["correct"] and not l["correct"]: counts["camera_correct_lidar_wrong"] += 1
                    if l["correct"] and not c["correct"]: counts["lidar_correct_camera_wrong"] += 1
                    if not c["correct"] and not l["correct"]: counts["both_wrong"] += 1
                    f = fusion_matches.get(index)
                    if f and f["correct"]: counts["fusion_repairs_conflict"] += 1
                    elif f: counts["fusion_inherits_conflict"] += 1
                elif c["correct"] and l["correct"]:
                    f = fusion_matches.get(index)
                    if not f or not f["correct"]: counts["fusion_extra_error"] += 1
        data = dict(counts)
        data["high_confidence_dual_detected"] = data.get("dual_detected", 0)
        data["conflict_yield"] = data.get("actual_spatial_conflict", 0) / data.get("dual_detected", 1)
        sensitivities[str(threshold)] = data
    return sensitivities


def complementarity_regret():
    labels = ("camera_weak_lidar_strong", "lidar_weak_camera_strong", "both_strong", "both_weak")
    output = {label: Counter() for label in labels}
    for row in read_jsonl("outputs/stage2/complementarity/object_index.jsonl"):
        label = row["complementarity_label"]
        if label not in output:
            continue
        counts = output[label]
        counts["object_count"] += 1
        camera = row["errors"]["camera"]
        lidar = row["errors"]["lidar"]
        fusion = row["errors"]["fusion"]
        camera_ok = bool(camera["matched"])
        lidar_ok = bool(lidar["matched"])
        fusion_ok = bool(fusion["matched"])
        if camera_ok and not fusion_ok:
            counts["fusion_new_fn_vs_camera"] += 1
        if lidar_ok and not fusion_ok:
            counts["fusion_new_fn_vs_lidar"] += 1
        if fusion_ok:
            for field, out in (
                ("center_error", "fusion_center_worse_than_best_single"),
                ("orientation_error", "fusion_orientation_worse_than_best_single"),
                ("velocity_error", "fusion_velocity_worse_than_best_single"),
            ):
                values = [
                    source[field]
                    for source in (camera, lidar)
                    if source["matched"] and source[field] is not None
                ]
                if values and fusion[field] is not None and fusion[field] > min(values) + 1e-6:
                    counts[out] += 1
        if fusion_ok and (not camera_ok or not lidar_ok):
            counts["fusion_repairs_error"] += 1
        if not fusion_ok and (not camera_ok or not lidar_ok):
            counts["fusion_inherits_error"] += 1
        if not fusion_ok and (camera_ok or lidar_ok):
            counts["fusion_extra_error"] += 1
    return {
        label: {
            **dict(counts),
            "family": "complementarity_control",
            "paired_induced_conflict_count": 0,
            "fusion_extra_error_rate": counts["fusion_extra_error"] / counts["object_count"] if counts["object_count"] else 0.0,
            "fusion_repairs_error_rate": counts["fusion_repairs_error"] / counts["object_count"] if counts["object_count"] else 0.0,
        }
        for label, counts in output.items()
    }


def conflict_analysis(plan, thresholds, token_sets=("200", "197")):
    output = {
        "condition_scope": "conflict-generating conditions; main labels are paired induced conflicts only",
        "association_radius_m": 4.0, "correct_center_radius_m": 2.0,
        "incompatible_center_distance_m": 2.0, "incompatible_orientation_deg": 45.0,
        "main_precision_target": MAIN_PRECISION_TARGET,
        "precision_controlled_score_thresholds": str(ROOT / "metrics" / "precision_controlled_score_thresholds.json"),
        "appendix_score_sensitivity_thresholds": [0.1, 0.3, 0.5],
        "appendix_score_sensitivity_clean_controls": {
            count: conflict_counts(
                f"stage1-clean-camera-{count}", f"stage1-clean-lidar-{count}",
                f"stage1-clean-fusion-{count}", count,
            )
            for count in token_sets
        },
        "precision_controlled_clean_controls": {},
        "tasks": {},
        "complementarity_control_regret": complementarity_regret(),
    }
    object_rows = []
    for count in token_sets:
        output["precision_controlled_clean_controls"][count] = {}
        for target in thresholds["targets"]:
            summary, rows = paired_conflict_counts(
                f"stage1-clean-camera-{count}", f"stage1-clean-fusion-{count}",
                count, thresholds, target,
            )
            summary["note"] = "clean-vs-clean sanity control; paired induction should be zero"
            output["precision_controlled_clean_controls"][count][target] = summary
            object_rows.extend(rows)
    for item in plan["tasks"]:
        if item["model"] != "fusion" or item["family"] != "conflict-generating":
            continue
        suffix = item["run_id"][len("fusion-"):]
        appendix = conflict_counts(
            f"camera-{suffix}", f"stage1-clean-lidar-{item['token_set']}",
            item["run_id"], item["token_set"],
        )
        for threshold, values in appendix.items():
            values["conflict_yield_excess_over_clean"] = (
                values["conflict_yield"]
                - output["appendix_score_sensitivity_clean_controls"][item["token_set"]][threshold]["conflict_yield"]
            )
        precision = {}
        for target in thresholds["targets"]:
            summary, rows = paired_conflict_counts(
                f"camera-{suffix}", item["run_id"], item["token_set"], thresholds, target,
            )
            precision[target] = summary
            object_rows.extend(rows)
        output["tasks"][item["run_id"]] = {
            "precision_controlled": precision,
            "appendix_score_sensitivity": appendix,
        }
    path = ROOT / "metrics" / "paired_induced_conflict_objects.jsonl"
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in object_rows))
    write_json(ROOT / "metrics" / "conflict_analysis.json", output)
    return output


def missing_analysis():
    output = {}
    for missing, reference in (("camera", "stage1-clean-lidar-200"), ("lidar", "stage1-clean-camera-200")):
        run_id = f"fusion-{missing}-missing"
        records = read_jsonl(ROOT / "matched_errors" / f"{run_id}.jsonl")
        reference_records = read_jsonl(ROOT / "matched_errors" / f"{reference}.jsonl")
        predictions = mmcv.load(prediction_path(run_id))
        scores = np.concatenate([value["scores_3d"].detach().cpu().numpy() for value in predictions])
        manifests = read_jsonl(ROOT / "manifests" / run_id / "samples.jsonl")
        feature = [row["corruption_parameters"]["feature_branch_stats"] for row in manifests]
        output[run_id] = {
            "samples_with_prediction_ratio_score_ge_0_1": float(np.mean([row["has_prediction_score_ge_0_1"] for row in records])),
            "mean_detections_per_sample_score_ge_0_1": float(np.mean([row["prediction_count_score_ge_0_1"] for row in records])),
            "score_distribution": {"mean": float(scores.mean()), "median": float(np.median(scores)), "max": float(scores.max()), "p90": float(np.quantile(scores, 0.9))},
            "micro_recall": sum(row["tp"] for row in records) / sum(row["gt_count"] for row in records),
            "reference_micro_recall": sum(row["tp"] for row in reference_records) / sum(row["gt_count"] for row in reference_records),
            "feature_branch_validation": {
                "missing_branch_zeroed_abs_max": max(row["zeroed_abs_max"] for row in feature),
                "camera_input_abs_mean": float(np.mean([row["input_abs_mean"]["camera"] for row in feature])),
                "lidar_input_abs_mean": float(np.mean([row["input_abs_mean"]["lidar"] for row in feature])),
                "fused_abs_mean": float(np.mean([row["fused_abs_mean"] for row in feature])),
            },
        }
    write_json(ROOT / "metrics" / "missing_analysis.json", output)
    return output


def lowlight_news3_summary():
    run_ids = ("camera-camera-lowlight-news3", "fusion-camera-lowlight-news3")
    if not all((ROOT / "metrics" / run_id / "pilot_subset_metrics.json").exists() for run_id in run_ids):
        return None
    output = {
        "candidate": "formal_candidate",
        "severity": 3,
        "parameters": {"exposure_scale": 0.50, "gamma": 1.45},
        "checks": {},
    }
    old = {
        "camera_s2": metrics("camera-camera-lowlight-s2"),
        "camera_old_s3": metrics("camera-camera-lowlight-s3"),
        "fusion_s2": metrics("fusion-camera-lowlight-s2"),
        "fusion_old_s3": metrics("fusion-camera-lowlight-s3"),
    }
    for run_id in run_ids:
        records = read_jsonl(ROOT / "manifests" / run_id / "samples.jsonl")
        matched = read_jsonl(ROOT / "matched_errors" / f"{run_id}.jsonl")
        value = metrics(run_id)
        model = run_id.split("-", 1)[0]
        s2 = old[f"{model}_s2"]
        old_s3 = old[f"{model}_old_s3"]
        output["checks"][run_id] = {
            "mean_brightness_ratio": float(np.mean([
                row["corrupted_brightness"] / row["original_brightness"] for row in records
            ])),
            "black_pixel_ratio": float(np.mean([
                row["corruption_parameters"].get("black_pixel_ratio", 0.0) for row in records
            ])),
            "saturated_pixel_ratio": float(np.mean([
                row["corruption_parameters"].get("saturated_pixel_ratio", 0.0) for row in records
            ])),
            "pilot_subset_mAP": value["pilot_subset_mAP"],
            "pilot_subset_NDS_proxy": value["pilot_subset_NDS_proxy"],
            "between_old_s2_and_old_s3_mAP": (
                old_s3["pilot_subset_mAP"] <= value["pilot_subset_mAP"] <= s2["pilot_subset_mAP"]
            ),
            "between_old_s2_and_old_s3_NDS": (
                old_s3["pilot_subset_NDS_proxy"] <= value["pilot_subset_NDS_proxy"] <= s2["pilot_subset_NDS_proxy"]
            ),
            "samples_with_prediction_ratio_score_ge_0_1": float(np.mean([
                row["has_prediction_score_ge_0_1"] for row in matched
            ])),
        }
    output["accepted_as_formal_s3"] = all(
        row["between_old_s2_and_old_s3_mAP"]
        and row["between_old_s2_and_old_s3_NDS"]
        and row["samples_with_prediction_ratio_score_ge_0_1"] > 0.1
        for row in output["checks"].values()
    )
    write_json(ROOT / "metrics" / "lowlight_news3_summary.json", output)
    return output


def recommendations(plan, conflict):
    decisions = {
        "camera_lowlight": {"1": "keep", "2": "keep", "3": "adjust_pending_new_s3"},
        "lidar_dropout": {"1": "keep", "2": "keep", "3": "keep"},
        "lidar_beam_reduction": {"1": "keep", "2": "keep", "3": "keep"},
        "camera_extrinsic_translation": {"1": "remove", "2": "remove", "3": "keep_as_negative_control"},
        "camera_extrinsic_rotation": {"1": "remove", "2": "keep", "3": "keep"},
        "camera_temporal_offset": {"1": "remove", "2": "keep", "3": "keep"},
        "camera_missing": {"1": "keep"}, "lidar_missing": {"1": "keep"},
    }
    selected = []
    for item in plan["tasks"]:
        if item["family"] == "clean":
            selected.append(item["run_id"])
            continue
        decision = decisions[item["corruption_type"]][str(item["severity"])]
        if decision.startswith("keep"):
            selected.append(item["run_id"])
    status = json.loads((ROOT / "run_status.json").read_text())
    tasks = {item["run_id"]: item for item in plan["tasks"]}
    full_seconds = 0.0
    for run_id in selected:
        item = tasks[run_id]
        scale = 5869 / 197 if item["token_set"] == "197" else 6019 / 200
        full_seconds += status["tasks"][run_id]["resources"]["wall_seconds"] * scale
    return {
        "decisions": decisions,
        "recommended_full_task_count": len(selected),
        "recommended_full_run_ids": selected,
        "estimated_full_single_gpu_hours": full_seconds / 3600,
        "estimated_full_four_gpu_wall_hours": full_seconds / 3600 / 4,
        "note": "Estimates scale pilot wall time linearly and exclude queue contention; no full run was started.",
    }


def main():
    plan = json.loads((ROOT / "run_plan.json").read_text())
    status = json.loads((ROOT / "run_status.json").read_text())
    core_status = Counter(status["tasks"][item["run_id"]]["status"] for item in plan["tasks"])
    if core_status != {"completed": 39}:
        raise RuntimeError("all 39 pilot tasks must complete before analysis")
    regenerate_matches(plan)
    bootstrap_all(plan)
    curves = build_curves(plan)
    fusion = fusion_comparisons(plan)
    thresholds = precision_controlled_thresholds()
    conflict = conflict_analysis(plan, thresholds)
    missing = missing_analysis()
    recommendation = recommendations(plan, conflict)
    clean_stage1, clean_s0 = map(metrics, ("stage1-clean-fusion-200", "fusion-clean-s0"))
    lowlight_supplement = lowlight_news3_summary()
    summary = {
        "status": "pilot_complete",
        "metric_scope": "fixed pilot subset; not official nuScenes leaderboard metrics",
        "task_count": 39,
        "actual_gpu_hours": sum(value["resources"]["wall_seconds"] for value in status["tasks"].values()) / 3600,
        "clean_consistency": {
            "stage1": clean_stage1, "severity_0": clean_s0,
            "mAP_delta": clean_s0["pilot_subset_mAP"] - clean_stage1["pilot_subset_mAP"],
            "NDS_proxy_delta": clean_s0["pilot_subset_NDS_proxy"] - clean_stage1["pilot_subset_NDS_proxy"],
        },
        "curves": curves,
        "precision_controlled_score_thresholds": thresholds,
        "fusion_comparison": fusion,
        "conflict_analysis": conflict,
        "missing_analysis": missing,
        "lowlight_news3_supplement": lowlight_supplement,
        "severity_recommendation": recommendation,
        "limitations": [
            "200/197 sample estimates are pilot proxies, not official nuScenes val metrics.",
            "Post-NMS scores are detection filters, not calibrated probabilities.",
            "Conflict-generating conditions are not automatic object-level conflict labels.",
            "Bootstrap intervals describe pilot sampling uncertainty and are not final significance claims.",
        ],
    }
    write_json(ROOT / "summary.json", summary)
    print(json.dumps({
        "tasks": 39, "gpu_hours": summary["actual_gpu_hours"],
        "clean_mAP_delta": summary["clean_consistency"]["mAP_delta"],
        "clean_NDS_delta": summary["clean_consistency"]["NDS_proxy_delta"],
        "recommended_full_tasks": recommendation["recommended_full_task_count"],
        "main_conflict_target_precision": MAIN_PRECISION_TARGET,
    }, indent=2))


if __name__ == "__main__":
    main()
