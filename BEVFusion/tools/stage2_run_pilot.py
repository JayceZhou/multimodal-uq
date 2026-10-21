#!/usr/bin/env python3
"""Prepare, run, resume, and evaluate the fixed Stage 2 pilot."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mmcv
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


ROOT = Path("outputs/stage2/pilot")
PYTHON = sys.executable
SEED = 20260707
MODELS = {
    "fusion": (
        "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml",
        "pretrained/bevfusion-det.pth",
    ),
    "camera": (
        "configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml",
        "pretrained/camera-only-det.pth",
    ),
    "lidar": (
        "configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml",
        "pretrained/lidar-only-det.pth",
    ),
}
CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            allow_nan=False,
            default=lambda item: item.tolist(),
        )
        + "\n"
    )
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task(run_id, model, corruption, severity, family, tokens="200"):
    config, checkpoint = MODELS[model]
    return {
        "run_id": run_id,
        "model": model,
        "config": config,
        "checkpoint": checkpoint,
        "corruption_type": corruption,
        "severity": severity,
        "family": family,
        "token_set": tokens,
    }


def build_plan():
    tasks = [task("fusion-clean-s0", "fusion", "camera_lowlight", 0, "clean")]
    specs = (
        ("camera-lowlight", "camera_lowlight", "degradation", ("camera", "fusion")),
        ("lidar-dropout", "lidar_dropout", "degradation", ("lidar", "fusion")),
        ("lidar-beams", "lidar_beam_reduction", "degradation", ("lidar", "fusion")),
        ("camera-translation", "camera_extrinsic_translation", "conflict-generating", ("camera", "fusion")),
        ("camera-rotation", "camera_extrinsic_rotation", "conflict-generating", ("camera", "fusion")),
        ("camera-temporal", "camera_temporal_offset", "conflict-generating", ("camera", "fusion")),
    )
    for prefix, corruption, family, models in specs:
        token_set = "197" if corruption == "camera_temporal_offset" else "200"
        for severity in (1, 2, 3):
            for model in models:
                tasks.append(task(
                    f"{model}-{prefix}-s{severity}", model, corruption, severity,
                    family, token_set,
                ))
    tasks.extend((
        task("fusion-camera-missing", "fusion", "camera_missing", 1, "missing"),
        task("fusion-lidar-missing", "fusion", "lidar_missing", 1, "missing"),
    ))
    assert len(tasks) == 39
    return {"created_utc": now(), "seed": SEED, "task_count": 39, "tasks": tasks}


def supplement_tasks():
    return [
        {
            **task("camera-camera-lowlight-news3", "camera", "camera_lowlight", 3, "degradation"),
            "lowlight_candidate": "formal_candidate",
        },
        {
            **task("fusion-camera-lowlight-news3", "fusion", "camera_lowlight", 3, "degradation"),
            "lowlight_candidate": "formal_candidate",
        },
    ]


def task_map(plan, include_supplement=True):
    tasks = list(plan["tasks"])
    if include_supplement:
        tasks.extend(supplement_tasks())
    return {item["run_id"]: item for item in tasks}


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("predictions", "manifests", "metrics", "matched_errors", "bootstrap", "logs"):
        (ROOT / name).mkdir(exist_ok=True)
    frozen = json.loads((ROOT / "sample_tokens.json").read_text())
    tokens200 = frozen["sample_tokens"]
    if len(tokens200) != 200 or len(set(tokens200)) != 200:
        raise ValueError("frozen pilot list must contain exactly 200 unique tokens")
    temporal = json.loads(Path("outputs/stage2/metrics/temporal_validity.json").read_text())["samples"]
    tokens197 = [
        token for token in tokens200
        if all(temporal[token][f"valid_severity_{severity}"] for severity in (1, 2, 3))
    ]
    if len(tokens197) != 197:
        raise ValueError(f"expected 197 common temporal tokens, got {len(tokens197)}")
    temporal_path = ROOT / "sample_tokens_temporal_197.json"
    write_json(temporal_path, {"seed": SEED, "sample_tokens": tokens197})

    source = mmcv.load("outputs/stage1/baseline/nuscenes_infos_val.pkl")
    all_infos = sorted(source["infos"], key=lambda item: item["timestamp"])
    info_by_token = {item["token"]: item for item in all_infos}
    for count, tokens in ((200, tokens200), (197, tokens197)):
        missing = set(tokens).difference(info_by_token)
        if missing:
            raise ValueError(f"{len(missing)} frozen tokens are absent from validation infos")
        subset = [item for item in all_infos if item["token"] in set(tokens)]
        mmcv.dump(
            {"infos": subset, "metadata": source["metadata"]},
            ROOT / f"nuscenes_infos_val_{count}.pkl",
        )

    for model in MODELS:
        full = mmcv.load(f"outputs/stage1/baseline/{model}.pkl")
        if len(full) != len(all_infos):
            raise ValueError(f"Stage 1 {model} predictions are not validation-aligned")
        by_token = {info["token"]: prediction for info, prediction in zip(all_infos, full)}
        for count, tokens in ((200, tokens200), (197, tokens197)):
            ordered = [item["token"] for item in mmcv.load(ROOT / f"nuscenes_infos_val_{count}.pkl")["infos"]]
            mmcv.dump([by_token[token] for token in ordered], ROOT / "predictions" / f"stage1-clean-{model}-{count}.pkl")

    plan = build_plan()
    plan["token_lists"] = {
        "200": {"path": str(ROOT / "sample_tokens.json"), "sha256": sha256(ROOT / "sample_tokens.json")},
        "197": {"path": str(temporal_path), "sha256": sha256(temporal_path)},
    }
    write_json(ROOT / "run_plan.json", plan)
    status_path = ROOT / "run_status.json"
    if not status_path.exists():
        write_json(status_path, {
            "updated_utc": now(),
            "tasks": {item["run_id"]: {"status": "pending"} for item in plan["tasks"]},
        })
    return plan


def load_plan():
    path = ROOT / "run_plan.json"
    return json.loads(path.read_text()) if path.exists() else prepare()


def monitor(command, log_path, gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.time()
    maximum_memory = maximum_util = read_bytes = write_bytes = 0
    with open(log_path, "w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        while process.poll() is None:
            try:
                output = subprocess.check_output([
                    "nvidia-smi", f"--id={gpu}",
                    "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits",
                ], text=True, timeout=5)
                memory, utilization = (int(value.strip()) for value in output.split(","))
                maximum_memory, maximum_util = max(maximum_memory, memory), max(maximum_util, utilization)
                io = Path(f"/proc/{process.pid}/io")
                if io.exists():
                    values = dict(line.split(": ") for line in io.read_text().splitlines())
                    read_bytes = int(values.get("read_bytes", 0))
                    write_bytes = int(values.get("write_bytes", 0))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            time.sleep(2)
    return process.returncode, {
        "wall_seconds": time.time() - started,
        "gpu": gpu,
        "max_gpu_memory_mib": maximum_memory,
        "max_gpu_utilization_percent": maximum_util,
        "process_read_bytes": read_bytes,
        "process_write_bytes": write_bytes,
    }


def command_for(item, plan):
    token_set = item["token_set"]
    command = [
        PYTHON, "tools/test.py", item["config"], item["checkpoint"],
        "--out", str(ROOT / "predictions" / f"{item['run_id']}.pkl"),
        "--run-id", item["run_id"],
        "--corruption-type", item["corruption_type"],
        "--severity", str(item["severity"]),
        "--corruption-seed", str(SEED),
        "--lowlight-candidate", item.get("lowlight_candidate", "calibrated_candidate"),
        "--stage2-manifest-dir", str(ROOT / "manifests" / item["run_id"]),
        "--stage2-token-list-sha256", plan["token_lists"][token_set]["sha256"],
        "--seed", str(SEED), "--deterministic",
    ]
    if item["corruption_type"] == "camera_temporal_offset":
        command.extend(("--temporal-history-file", "outputs/stage2/configs/camera_history.json"))
    command.extend((
        "--cfg-options", f"data.test.ann_file={ROOT / ('nuscenes_infos_val_' + token_set + '.pkl')}",
        "data.workers_per_gpu=4",
    ))
    return command


def finite_prediction(prediction):
    return all(torch.isfinite(
        value.tensor if hasattr(value, "tensor") else value
    ).all().item() for value in (prediction["boxes_3d"], prediction["scores_3d"]))


def matched_records(infos, outputs, run_id, score_threshold=0.1, distance_threshold=2.0):
    records = []
    for info, prediction in zip(infos, outputs):
        boxes = prediction["boxes_3d"].tensor.detach().cpu().numpy()
        scores = prediction["scores_3d"].detach().cpu().numpy()
        labels = prediction["labels_3d"].detach().cpu().numpy()
        keep = scores >= score_threshold
        filtered_boxes, filtered_scores, filtered_labels = boxes[keep], scores[keep], labels[keep]
        gt_names = np.asarray(info["gt_names"])
        gt_keep = np.isin(gt_names, CLASSES)
        gt_boxes = np.asarray(info["gt_boxes"])[gt_keep]
        gt_names = gt_names[gt_keep]
        gt_velocity = np.asarray(info["gt_velocity"])[gt_keep]
        gt_labels = np.asarray([CLASSES.index(name) for name in gt_names])
        pairs = []
        if len(gt_boxes) and len(filtered_boxes):
            distances = np.linalg.norm(gt_boxes[:, None, :2] - filtered_boxes[None, :, :2], axis=2)
            rows, columns = linear_sum_assignment(distances)
            pairs = [(int(row), int(column)) for row, column in zip(rows, columns) if distances[row, column] <= distance_threshold]
        correct = [(row, column) for row, column in pairs if gt_labels[row] == filtered_labels[column]]
        class_errors = len(pairs) - len(correct)
        tp = len(correct)
        fp = len(filtered_boxes) - tp
        fn = len(gt_boxes) - tp
        center, size, orientation, velocity = [], [], [], []
        for row, column in correct:
            gt, pred = gt_boxes[row], filtered_boxes[column]
            center.append(float(np.linalg.norm(gt[:2] - pred[:2])))
            size.append(float(1 - np.prod(np.minimum(gt[3:6], pred[3:6])) / np.prod(np.maximum(gt[3:6], pred[3:6]))))
            orientation.append(float(abs(np.arctan2(np.sin(pred[6] - gt[6]), np.cos(pred[6] - gt[6])))))
            velocity.append(float(np.linalg.norm(pred[7:9] - gt_velocity[row])))
        mean = lambda values: float(np.mean([value for value in values if np.isfinite(value)])) if any(np.isfinite(value) for value in values) else None
        per_class = {}
        for class_index, name in enumerate(CLASSES):
            class_pairs = [pair for pair in correct if gt_labels[pair[0]] == class_index]
            class_gt = int((gt_labels == class_index).sum())
            class_predictions = int((filtered_labels == class_index).sum())
            class_center = [
                float(np.linalg.norm(gt_boxes[row, :2] - filtered_boxes[column, :2]))
                for row, column in class_pairs
            ]
            per_class[name] = {
                "gt_count": class_gt,
                "tp": len(class_pairs),
                "fp": class_predictions - len(class_pairs),
                "fn": class_gt - len(class_pairs),
                "recall": len(class_pairs) / class_gt if class_gt else None,
                "precision": len(class_pairs) / class_predictions if class_predictions else None,
                "center_error": mean(class_center),
            }
        records.append({
            "run_id": run_id,
            "sample_token": info["token"],
            "gt_count": int(len(gt_boxes)), "tp": tp, "fp": fp, "fn": fn,
            "recall": tp / len(gt_boxes) if len(gt_boxes) else None,
            "precision": tp / len(filtered_boxes) if len(filtered_boxes) else None,
            "class_errors": class_errors,
            "center_error": mean(center), "size_error": mean(size),
            "orientation_error": mean(orientation), "velocity_error": mean(velocity),
            "prediction_count": int(len(boxes)),
            "prediction_count_score_ge_0_1": int(keep.sum()),
            "has_prediction_score_ge_0_1": bool(keep.any()),
            "max_detection_score": float(scores.max()) if len(scores) else None,
            "mean_detection_score": float(scores.mean()) if len(scores) else None,
            "per_class": per_class,
        })
    return records


def evaluate(run_id):
    plan = load_plan()
    tasks = {item["run_id"]: item for item in plan["tasks"]}
    tasks.update(task_map(plan))
    if run_id.startswith("stage1-clean-"):
        _, _, model, token_set = run_id.split("-")
        item = {"run_id": run_id, "model": model, "config": MODELS[model][0], "token_set": token_set}
        prediction_path = ROOT / "predictions" / f"{run_id}.pkl"
    else:
        item = tasks[run_id]
        prediction_path = ROOT / "predictions" / f"{run_id}.pkl"
    info_path = ROOT / f"nuscenes_infos_val_{item['token_set']}.pkl"
    data = mmcv.load(info_path)
    infos = sorted(data["infos"], key=lambda value: value["timestamp"])
    outputs = mmcv.load(prediction_path)
    if len(outputs) != len(infos) or not all(finite_prediction(value) for value in outputs):
        raise ValueError(f"{run_id}: incomplete or non-finite predictions")

    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet3d.datasets import build_dataset
    from mmdet3d.utils import recursive_eval
    from test import remove_unused_radar
    configs.load(item["config"], recursive=True)
    cfg = Config(recursive_eval(configs), filename=item["config"])
    cfg.data.test.ann_file = str(info_path)
    cfg.data.test.test_mode = True
    remove_unused_radar(cfg)
    dataset = build_dataset(cfg.data.test)
    metric_dir = ROOT / "metrics" / run_id
    result_path, temporary = dataset.format_results(outputs, str(metric_dir / "nusc"))

    from nuscenes import NuScenes
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.loaders import load_gt as official_load_gt
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.detection import evaluate as detection_evaluate
    target_tokens = {info["token"] for info in infos}
    cache_path = ROOT / "metrics" / "gt_val_frozen_200.pkl"

    def subset_gt(nusc, eval_set, box_cls, verbose=False):
        if cache_path.exists():
            frozen = EvalBoxes.deserialize(mmcv.load(cache_path), DetectionBox)
        else:
            full = official_load_gt(nusc, eval_set, box_cls, verbose=verbose)
            frozen = EvalBoxes()
            allowed = set(json.loads((ROOT / "sample_tokens.json").read_text())["sample_tokens"])
            for token in full.sample_tokens:
                if token in allowed:
                    frozen.add_boxes(token, full[token])
            mmcv.dump(frozen.serialize(), cache_path)
        subset = EvalBoxes()
        for token in frozen.sample_tokens:
            if token in target_tokens:
                subset.add_boxes(token, frozen[token])
        return subset

    detection_evaluate.load_gt = subset_gt
    nusc = NuScenes(version=dataset.version, dataroot=dataset.dataset_root, verbose=False)
    evaluator = detection_evaluate.DetectionEval(
        nusc, dataset.eval_detection_configs, result_path, "val",
        output_dir=str(metric_dir), verbose=False,
    )
    evaluator.main(render_curves=False)
    raw = json.loads((metric_dir / "metrics_summary.json").read_text())
    proxy = {
        "metric_scope": f"fixed_{item['token_set']}_sample_pilot_subset_not_official_nuscenes",
        "pilot_subset_mAP": raw["mean_ap"],
        "pilot_subset_NDS_proxy": raw["nd_score"],
        "pilot_subset_mATE": raw["tp_errors"]["trans_err"],
        "pilot_subset_mASE": raw["tp_errors"]["scale_err"],
        "pilot_subset_mAOE": raw["tp_errors"]["orient_err"],
        "pilot_subset_mAVE": raw["tp_errors"]["vel_err"],
        "pilot_subset_mAAE": raw["tp_errors"]["attr_err"],
        "per_class_AP": {name: float(np.mean(list(values.values()))) for name, values in raw["label_aps"].items()},
        "sample_count": len(infos),
    }
    write_json(metric_dir / "pilot_subset_metrics.json", proxy)
    records = matched_records(infos, outputs, run_id)
    matched_path = ROOT / "matched_errors" / f"{run_id}.jsonl"
    matched_path.write_text("".join(json.dumps(value, allow_nan=False) + "\n" for value in records))
    aggregate = {
        "sample_count": len(records),
        "gt_count": sum(value["gt_count"] for value in records),
        "tp": sum(value["tp"] for value in records),
        "fp": sum(value["fp"] for value in records),
        "fn": sum(value["fn"] for value in records),
        "class_errors": sum(value["class_errors"] for value in records),
        "samples_with_prediction_ratio": float(np.mean([value["has_prediction_score_ge_0_1"] for value in records])),
    }
    write_json(metric_dir / "matched_summary.json", aggregate)
    if temporary is not None:
        temporary.cleanup()
    print(json.dumps(proxy, indent=2))
    return proxy


def validate_manifest(item, plan):
    directory = ROOT / "manifests" / item["run_id"]
    manifest = json.loads((directory / "manifest.json").read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    records = [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))) for line in (directory / "samples.jsonl").open()]
    expected = int(item["token_set"])
    if manifest["sample_count"] != expected or len(records) != expected:
        raise ValueError(f"{item['run_id']}: manifest token count mismatch")
    if manifest["token_list_sha256"] != plan["token_lists"][item["token_set"]]["sha256"]:
        raise ValueError(f"{item['run_id']}: token SHA256 mismatch")
    if any(not record["valid"] for record in records):
        raise ValueError(f"{item['run_id']}: invalid sample present")
    if item["corruption_type"] == "camera_lowlight" and item["severity"]:
        if not all(record["corrupted_brightness"] < record["original_brightness"] for record in records):
            raise ValueError("lowlight proxy did not decrease")
    if item["corruption_type"] == "lidar_dropout" and item["severity"]:
        ratio = 1 - (0.25, 0.5, 0.75)[item["severity"] - 1]
        actual = np.mean([record["retained_point_count"] / record["original_point_count"] for record in records])
        if abs(actual - ratio) > 0.01:
            raise ValueError("dropout proxy does not match configured severity")


def run_group(run_ids, gpu, retry_failed=False):
    plan = load_plan()
    tasks = task_map(plan)
    status_path = ROOT / "run_status.json"
    status = json.loads(status_path.read_text())
    for item in supplement_tasks():
        status["tasks"].setdefault(item["run_id"], {"status": "pending"})
    for run_id in run_ids:
        item = tasks[run_id]
        state = status["tasks"][run_id]["status"]
        if state == "completed":
            print(f"SKIP completed {run_id}", flush=True)
            continue
        if state in ("failed", "running") and not retry_failed:
            raise RuntimeError(f"{run_id} is {state}; use --retry-failed explicitly")
        prediction = ROOT / "predictions" / f"{run_id}.pkl"
        manifest = ROOT / "manifests" / run_id
        log_path = ROOT / "logs" / f"{run_id}.log"
        if state in ("failed", "running") and retry_failed:
            suffix = datetime.now(timezone.utc).strftime(".failed-%Y%m%dT%H%M%SZ")
            for path in (prediction, manifest, log_path):
                if path.exists():
                    shutil.move(str(path), str(path) + suffix)
        if (prediction.exists() or manifest.exists()) and state == "pending":
            raise FileExistsError(f"refusing to overwrite existing output for {run_id}")
        command = command_for(item, plan)
        status["tasks"][run_id] = {"status": "running", "started_utc": now(), "command": command}
        status["updated_utc"] = now()
        write_json(status_path, status)
        print(f"START {run_id}", flush=True)
        exit_code, resources = monitor(command, log_path, gpu)
        status = json.loads(status_path.read_text())
        status["tasks"][run_id].update(exit_code=exit_code, resources=resources, finished_utc=now())
        if exit_code:
            status["tasks"][run_id]["status"] = "failed"
            write_json(status_path, status)
            raise RuntimeError(f"{run_id} failed with exit code {exit_code}")
        validate_manifest(item, plan)
        evaluation = subprocess.run([PYTHON, __file__, "--phase", "evaluate", "--run-id", run_id])
        if evaluation.returncode:
            status["tasks"][run_id].update(status="failed", evaluation_exit_code=evaluation.returncode)
            write_json(status_path, status)
            raise RuntimeError(f"{run_id} subset evaluation failed")
        status["tasks"][run_id].update(status="completed", evaluation_exit_code=0)
        completed = [value for value in status["tasks"].values() if value["status"] == "completed"]
        if completed:
            average = np.mean([value["resources"]["wall_seconds"] for value in completed])
            status["estimated_total_gpu_hours"] = float(average * len(plan["tasks"]) / 3600)
        status["updated_utc"] = now()
        write_json(status_path, status)
        print(f"DONE {run_id} {resources['wall_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "probes", "remaining", "all", "supplement-news3", "evaluate"), default="prepare")
    parser.add_argument("--run-id")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        plan = prepare()
        evaluate("stage1-clean-fusion-200")
        print(f"prepared {plan['task_count']} tasks")
        return
    if args.phase == "evaluate":
        evaluate(args.run_id)
        return
    plan = prepare()
    probes = ("fusion-clean-s0", "fusion-camera-lowlight-s2", "fusion-lidar-dropout-s2")
    if args.phase == "probes":
        run_group(probes, args.gpu, args.retry_failed)
    elif args.phase == "remaining":
        run_group([item["run_id"] for item in plan["tasks"] if item["run_id"] not in probes], args.gpu, args.retry_failed)
    elif args.phase == "supplement-news3":
        run_group([item["run_id"] for item in supplement_tasks()], args.gpu, args.retry_failed)
    else:
        run_group([item["run_id"] for item in plan["tasks"]], args.gpu, args.retry_failed)


if __name__ == "__main__":
    main()
