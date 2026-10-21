#!/usr/bin/env python3
"""Prepare pilot condition shards and build compact Stage 3 object labels."""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import mmcv
import numpy as np
from nuscenes import NuScenes
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_analyze_pilot import (
    CLASSES,
    class_threshold_array,
    disagreement,
    grew,
)


SEED = 20260708
ROOT = Path("outputs/stage3/labels")
CONDITIONS = [
    *(dict(type="camera_lowlight", severity=value, family="degradation", affected="camera") for value in (1, 2, 3)),
    *(dict(type="lidar_dropout", severity=value, family="degradation", affected="lidar") for value in (1, 2, 3)),
    *(dict(type="lidar_beam_reduction", severity=value, family="degradation", affected="lidar") for value in (1, 2, 3)),
    dict(type="camera_extrinsic_translation", severity=3, family="hard_negative", affected="camera"),
    *(dict(type="camera_extrinsic_rotation", severity=value, family="conflict", affected="camera") for value in (2, 3)),
    *(dict(type="camera_temporal_offset", severity=value, family="conflict", affected="camera") for value in (2, 3)),
    dict(type="camera_missing", severity=1, family="missing", affected="camera"),
    dict(type="lidar_missing", severity=1, family="missing", affected="lidar"),
]


def slug(condition):
    return f"{condition['type']}-s{condition['severity']}"


def strict_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_sha(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest


def temporal_valid(info, history, severity):
    return all(len(history.get(camera["sample_data_token"], [])) >= severity for camera in info["cams"].values())


def prepare(args):
    ROOT.mkdir(parents=True, exist_ok=True)
    source = mmcv.load(args.info)
    infos = sorted(source["infos"], key=lambda item: item["timestamp"])
    history = json.loads(Path(args.temporal_history).read_text())
    weighted = []
    for condition in CONDITIONS:
        weight = 3 if condition["family"] == "conflict" else 2 if condition["family"] == "hard_negative" else 1
        weighted.extend([condition] * weight)
    rng = random.Random(args.seed)
    shuffled = list(infos)
    rng.shuffle(shuffled)
    assignments = {}
    for index, info in enumerate(shuffled):
        choices = weighted[index % len(weighted) :] + weighted[: index % len(weighted)]
        condition = next(
            item for item in choices
            if item["type"] != "camera_temporal_offset" or temporal_valid(info, history, item["severity"])
        )
        assignments[info["token"]] = dict(condition)

    clean = dict(type="clean", severity=0, family="clean", affected=None)
    shards = {"clean": infos}
    for condition in CONDITIONS:
        name = slug(condition)
        shards[name] = [info for info in infos if assignments[info["token"]] == condition]
    info_dir = ROOT / "condition_infos"
    prediction_tasks = []
    for name, values in shards.items():
        path = info_dir / f"{name}.pkl"
        mmcv.dump({"infos": values, "metadata": source["metadata"]}, path)
        if name == "clean":
            for model in ("camera", "lidar"):
                prediction_tasks.append({"run_id": f"pilot-clean-{model}", "model": model, "condition": clean, "info": str(path)})
        elif values:
            condition = assignments[values[0]["token"]]
            if condition["type"] not in ("camera_missing", "lidar_missing"):
                prediction_tasks.append({
                    "run_id": f"pilot-{name}-{condition['affected']}",
                    "model": condition["affected"], "condition": condition, "info": str(path),
                })
    plan = {
        "seed": args.seed,
        "base_sample_count": len(infos),
        "training_entry_count": len(infos) * 2,
        "assignments": assignments,
        "condition_counts": dict(Counter(slug(value) for value in assignments.values())),
        "prediction_tasks": prediction_tasks,
        "temporal_history": args.temporal_history,
    }
    strict_json(ROOT / "condition_plan.json", plan)
    strict_json(ROOT / "schema.json", {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Stage 4/5 preliminary object supervision record",
        "type": "object",
        "required": [
            "split", "condition", "sample_token", "scene_token", "annotation_token",
            "category", "gt_index", "camera", "lidar", "conflict",
            "threshold_target", "thresholds_sha256",
        ],
        "properties": {
            "split": {"type": "string"},
            "condition": {
                "type": "object", "required": ["type", "severity", "family", "affected"],
                "properties": {
                    "type": {"type": "string"}, "severity": {"type": "integer", "minimum": 0, "maximum": 3},
                    "family": {"enum": ["clean", "degradation", "conflict", "hard_negative", "missing"]},
                    "affected": {"type": ["string", "null"]},
                },
            },
            "sample_token": {"type": "string", "minLength": 32, "maxLength": 32},
            "scene_token": {"type": "string", "minLength": 32, "maxLength": 32},
            "annotation_token": {"type": "string", "minLength": 32, "maxLength": 32},
            "category": {"type": "string"}, "gt_index": {"type": "integer", "minimum": 0},
            "camera": {"$ref": "#/$defs/reliability"}, "lidar": {"$ref": "#/$defs/reliability"},
            "conflict": {"type": "boolean"}, "threshold_target": {"enum": ["0.80"]},
            "thresholds_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "$defs": {"reliability": {
            "type": "object",
            "required": ["matched", "score", "score_pass", "center_error", "size_error", "yaw_error", "velocity_error", "reliable_detection"],
            "properties": {
                "matched": {"type": "boolean"}, "score": {"type": ["number", "null"]},
                "score_pass": {"type": "boolean"}, "center_error": {"type": ["number", "null"]},
                "size_error": {"type": ["number", "null"]}, "yaw_error": {"type": ["number", "null"]},
                "velocity_error": {"type": ["number", "null"]}, "reliable_detection": {"type": "boolean"},
            },
        }},
    })
    print(json.dumps({"base_samples": len(infos), "training_entries": len(infos) * 2, "tasks": len(prediction_tasks), "conditions": plan["condition_counts"]}, indent=2))


def prediction_by_token(path, infos):
    predictions = mmcv.load(path)
    if len(predictions) != len(infos):
        raise ValueError(f"prediction length mismatch for {path}")
    return {info["token"]: prediction for info, prediction in zip(sorted(infos, key=lambda item: item["timestamp"]), predictions)}


def modality_record(info, prediction, thresholds):
    gt_boxes = np.asarray(info["gt_boxes"])
    gt_velocity = np.asarray(info["gt_velocity"])
    gt_names = np.asarray(info["gt_names"])
    boxes = prediction["boxes_3d"].tensor.detach().cpu().numpy()
    scores = prediction["scores_3d"].detach().cpu().numpy()
    labels = prediction["labels_3d"].detach().cpu().numpy()
    result = [None] * len(gt_boxes)
    if len(gt_boxes) and len(boxes):
        distances = np.linalg.norm(gt_boxes[:, None, :2] - boxes[None, :, :2], axis=2)
        rows, columns = linear_sum_assignment(distances)
        for row, column in zip(rows, columns):
            if distances[row, column] > 4.0:
                continue
            label = int(labels[column])
            same_class = gt_names[row] in CLASSES and label == CLASSES.index(gt_names[row])
            score_pass = label < len(thresholds) and scores[column] >= thresholds[label]
            gt, pred = gt_boxes[row], boxes[column]
            result[row] = {
                "matched": bool(same_class), "score": float(scores[column]), "score_pass": bool(score_pass),
                "center_error": float(distances[row, column]),
                "size_error": float(1 - np.prod(np.minimum(gt[3:6], pred[3:6])) / np.prod(np.maximum(gt[3:6], pred[3:6]))),
                "yaw_error": float(abs(np.arctan2(np.sin(pred[6] - gt[6]), np.cos(pred[6] - gt[6])))),
                "velocity_error": float(np.linalg.norm(pred[7:9] - gt_velocity[row])),
                "reliable_detection": bool(same_class and score_pass and distances[row, column] <= 2.0),
            }
    empty = {"matched": False, "score": None, "score_pass": False, "center_error": None, "size_error": None, "yaw_error": None, "velocity_error": None, "reliable_detection": False}
    return [value if value is not None else dict(empty) for value in result]


def thresholded_association(info, prediction, thresholds):
    gt = np.asarray(info["gt_boxes"])
    names = np.asarray(info["gt_names"])
    velocities = np.asarray(info["gt_velocity"])
    eligible = np.flatnonzero(np.asarray(info["valid_flag"]) & np.isin(names, CLASSES))
    boxes = prediction["boxes_3d"].tensor.detach().cpu().numpy()
    scores = prediction["scores_3d"].detach().cpu().numpy()
    labels = prediction["labels_3d"].detach().cpu().numpy()
    keep = np.asarray([score >= thresholds[int(label)] for score, label in zip(scores, labels)], dtype=bool)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    result = {}
    if len(eligible) and len(boxes):
        distances = np.linalg.norm(gt[eligible, None, :2] - boxes[None, :, :2], axis=2)
        rows, columns = linear_sum_assignment(distances)
        for compact_row, column in zip(rows, columns):
            if distances[compact_row, column] > 4.0:
                continue
            row = int(eligible[compact_row])
            box = boxes[column]
            center = float(np.linalg.norm(gt[row, :2] - box[:2]))
            result[row] = {
                "box": box, "label": int(labels[column]), "score": float(scores[column]),
                "center_error": center,
                "yaw_error": float(abs(np.arctan2(np.sin(box[6] - gt[row, 6]), np.cos(box[6] - gt[row, 6])))),
                "velocity_error": float(np.linalg.norm(box[7:9] - velocities[row])),
                "correct": bool(labels[column] == CLASSES.index(names[row]) and center <= 2.0),
            }
    return result


def paired_induced(condition_type, clean_state, corrupt_state):
    return bool(
        condition_type in ("camera_extrinsic_rotation", "camera_temporal_offset")
        and not clean_state["conflict"] and corrupt_state["conflict"]
        and corrupt_state["dual_detected"] and grew(clean_state, corrupt_state)
    )


def build(args):
    plan = json.loads((ROOT / "condition_plan.json").read_text())
    source = mmcv.load(args.info)
    infos = sorted(source["infos"], key=lambda item: item["timestamp"])
    info_by_token = {info["token"]: info for info in infos}
    thresholds = json.loads(Path(args.thresholds).read_text())
    camera_thresholds = class_threshold_array(thresholds, "camera", "0.80")
    lidar_thresholds = class_threshold_array(thresholds, "lidar", "0.80")
    prediction_dir = Path(args.predictions)
    clean_camera = prediction_by_token(prediction_dir / "pilot-clean-camera.pkl", infos)
    clean_lidar = prediction_by_token(prediction_dir / "pilot-clean-lidar.pkl", infos)
    current = {}
    for task in plan["prediction_tasks"]:
        if task["condition"]["type"] == "clean":
            continue
        shard_infos = mmcv.load(task["info"])["infos"]
        current[task["run_id"]] = prediction_by_token(prediction_dir / f"{task['run_id']}.pkl", shard_infos)

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataset_root, verbose=False)
    rows, conflicts, hard_negatives, train_infos = [], [], [], []
    checks = Counter()
    for token, info in info_by_token.items():
        for condition in (dict(type="clean", severity=0, family="clean", affected=None), plan["assignments"][token]):
            name = "clean" if condition["type"] == "clean" else slug(condition)
            camera_prediction, lidar_prediction = clean_camera[token], clean_lidar[token]
            if condition["affected"] == "camera" and condition["type"] != "camera_missing":
                camera_prediction = current[f"pilot-{name}-camera"][token]
            if condition["affected"] == "lidar" and condition["type"] != "lidar_missing":
                lidar_prediction = current[f"pilot-{name}-lidar"][token]
            camera_records = modality_record(info, camera_prediction, camera_thresholds)
            lidar_records = modality_record(info, lidar_prediction, lidar_thresholds)
            if condition["type"] == "camera_missing":
                camera_records = [dict(value, matched=False, score=None, score_pass=False, reliable_detection=False) for value in camera_records]
            if condition["type"] == "lidar_missing":
                lidar_records = [dict(value, matched=False, score=None, score_pass=False, reliable_detection=False) for value in lidar_records]

            clean_c = thresholded_association(info, clean_camera[token], camera_thresholds)
            clean_l = thresholded_association(info, clean_lidar[token], lidar_thresholds)
            corrupt_c = thresholded_association(info, camera_prediction, camera_thresholds)
            corrupt_l = thresholded_association(info, lidar_prediction, lidar_thresholds)
            sample = nusc.get("sample", token)
            annotation_tokens = sample["anns"]
            if len(annotation_tokens) != len(info["gt_names"]):
                raise ValueError(f"annotation alignment mismatch for {token}")
            valid_indices = np.flatnonzero(info["valid_flag"])
            targets = []
            for gt_index in valid_indices:
                category = info["gt_names"][gt_index]
                defined = category in CLASSES
                clean_state = disagreement(clean_c.get(int(gt_index)), clean_l.get(int(gt_index)))
                corrupt_state = disagreement(corrupt_c.get(int(gt_index)), corrupt_l.get(int(gt_index)))
                induced = paired_induced(condition["type"], clean_state, corrupt_state)
                target = {
                    "defined": defined, "annotation_token": annotation_tokens[gt_index],
                    "camera": camera_records[gt_index], "lidar": lidar_records[gt_index],
                    "conflict": induced,
                }
                targets.append(target)
                if not defined:
                    continue
                row = {
                    "split": "stage3_train_pilot", "condition": condition, "sample_token": token,
                    "scene_token": sample["scene_token"], "annotation_token": annotation_tokens[gt_index],
                    "category": category, "gt_index": int(gt_index),
                    "camera": camera_records[gt_index], "lidar": lidar_records[gt_index],
                    "conflict": induced, "clean_disagreement": clean_state, "current_disagreement": corrupt_state,
                    "threshold_target": "0.80", "thresholds_sha256": file_sha(args.thresholds),
                }
                rows.append(row)
                conflicts.append(row)
                if not induced and condition["family"] in ("clean", "hard_negative", "conflict", "degradation"):
                    hard_negatives.append(row)
                checks["defined_objects"] += 1
                checks["conflict_positive"] += int(induced)
                checks["positive_dual_high_confidence"] += int(induced and corrupt_state["dual_detected"])
                checks[f"condition__{name}"] += 1
            copied = dict(info)
            copied["stage3_condition"] = dict(condition)
            copied["stage3_targets"] = targets
            copied["stage3_sample_id"] = f"{token}:{name}"
            train_infos.append(copied)

    if checks["conflict_positive"] != checks["positive_dual_high_confidence"]:
        raise ValueError("conflict positive without dual high-confidence predictions")
    if len(train_infos) != plan["training_entry_count"]:
        raise ValueError("training entry count mismatch")
    train_path = ROOT / "nuscenes_infos_stage3_pilot_train.pkl"
    mmcv.dump({"infos": train_infos, "metadata": source["metadata"]}, train_path)
    for path, values in (
        (ROOT / "reliability_objects.jsonl", rows),
        (ROOT / "paired_conflict_records.jsonl", conflicts),
        (ROOT / "hard_negative_records.jsonl", hard_negatives),
    ):
        path.write_text("".join(json.dumps(value, allow_nan=False, sort_keys=True) + "\n" for value in values))
    integrity = {
        "training_entries": len(train_infos), "object_records": len(rows),
        "hard_negative_records": len(hard_negatives), "checks": dict(checks),
        "clean_vs_clean_paired_induction": sum(row["conflict"] for row in rows if row["condition"]["family"] == "clean"),
        "translation_positive_count": sum(row["conflict"] for row in rows if row["condition"]["type"] == "camera_extrinsic_translation"),
        "thresholds_sha256": file_sha(args.thresholds),
    }
    if integrity["clean_vs_clean_paired_induction"] or integrity["translation_positive_count"]:
        raise ValueError("negative control was labeled as conflict")
    strict_json(ROOT / "integrity_summary.json", integrity)
    print(json.dumps(integrity, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "build"))
    parser.add_argument("--info", default="outputs/stage3/splits/nuscenes_infos_stage3_pilot_base.pkl")
    parser.add_argument("--temporal-history", default="outputs/stage3/splits/pilot_camera_history.json")
    parser.add_argument("--thresholds", default="outputs/stage3/labels/precision_controlled_thresholds.json")
    parser.add_argument("--predictions", default="outputs/stage3/labels/predictions")
    parser.add_argument("--dataset-root", default="data/nuscenes")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else build)(args)


if __name__ == "__main__":
    main()
