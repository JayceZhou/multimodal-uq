import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import mmcv
import numpy as np
import torch
from nuscenes import NuScenes
from PIL import Image
from scipy.optimize import linear_sum_assignment

from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmdet3d.datasets.pipelines.stage2_corruptions import LOWLIGHT_CANDIDATES


def parse_args():
    parser = argparse.ArgumentParser(description="Build Stage 2 pre-pilot metadata")
    parser.add_argument(
        "--info-file", default="outputs/stage1/baseline/nuscenes_infos_val.pkl"
    )
    parser.add_argument("--dataset-root", default="data/nuscenes")
    parser.add_argument("--output-dir", default="outputs/stage2")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--calibration-samples", type=int, default=24)
    parser.add_argument("--pilot-size", type=int, default=200)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(json_safe(value), allow_nan=False) + "\n" for value in values)
    )


def scene_metadata(nusc, sample_token, cache):
    sample = nusc.get("sample", sample_token)
    scene = nusc.get("scene", sample["scene_token"])
    if scene["token"] not in cache:
        location = nusc.get("log", scene["log_token"])["location"]
        description = scene["description"].lower()
        cache[scene["token"]] = {
            "scene_token": scene["token"],
            "scene_name": scene["name"],
            "description": scene["description"],
            "city": "boston" if "boston" in location else "singapore",
            "location": location,
            "time_of_day": "night" if "night" in description else "day",
            "weather": "rain" if "rain" in description else "not_rain",
        }
    return sample, cache[scene["token"]]


def lidar2image(camera):
    rotation = np.linalg.inv(camera["sensor2lidar_rotation"])
    translation = camera["sensor2lidar_translation"] @ rotation.T
    lidar2camera = np.eye(4)
    lidar2camera[:3, :3] = rotation.T
    lidar2camera[3, :3] = -translation
    intrinsic = np.eye(4)
    intrinsic[:3, :3] = camera["cam_intrinsic"]
    return intrinsic @ lidar2camera.T


def box_corners(box):
    center = np.asarray(box[:3], dtype=np.float64)
    width, length, height = np.asarray(box[3:6], dtype=np.float64)
    x = np.asarray([-width, -width, width, width, -width, -width, width, width]) / 2
    y = np.asarray([-length, length, length, -length, -length, length, length, -length]) / 2
    z = np.asarray([-height, -height, -height, -height, height, height, height, height]) / 2
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.asarray([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    return np.stack([x, y, z], axis=1) @ rotation.T + center


def projected_bbox(box, camera, image_size):
    corners = box_corners(box)
    projected = lidar2image(camera) @ np.column_stack([corners, np.ones(8)]).T
    valid = projected[2] > 0.5
    if valid.sum() < 2:
        return None
    xy = (projected[:2, valid] / projected[2:3, valid]).T
    raw = [xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()]
    width, height = image_size
    clipped = [
        max(0.0, raw[0]),
        max(0.0, raw[1]),
        min(float(width), raw[2]),
        min(float(height), raw[3]),
    ]
    area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
    if area <= 0:
        return None
    return {
        "bbox": clipped,
        "area": area,
        "truncated": raw != clipped,
    }


def crop_brightness(image, bbox):
    left, top, right, bottom = [int(round(value)) for value in bbox]
    crop = np.asarray(image)[top:bottom, left:right]
    return float(crop.mean() / 255.0) if crop.size else None


def lowlight_array(image, parameters):
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    dark = np.power(
        np.clip(pixels * parameters["exposure_scale"], 0.0, 1.0),
        parameters["gamma"],
    )
    return np.rint(dark * 255).astype(np.uint8)


def aggregate(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def build_camera_history(nusc, infos):
    history = {}
    sample_validity = {}
    for info in infos:
        per_camera = {}
        for channel, camera in info["cams"].items():
            current = nusc.get("sample_data", camera["sample_data_token"])
            entries = []
            previous = current["prev"]
            while previous and len(entries) < 3:
                record = nusc.get("sample_data", previous)
                entries.append(
                    {
                        "sample_data_token": previous,
                        "timestamp": record["timestamp"],
                        "data_path": str(Path(nusc.dataroot) / record["filename"]),
                    }
                )
                previous = record["prev"]
            history[current["token"]] = entries
            per_camera[channel] = len(entries)
        sample_validity[info["token"]] = {
            "available_previous_frames": per_camera,
            "valid_severity_1": all(value >= 1 for value in per_camera.values()),
            "valid_severity_2": all(value >= 2 for value in per_camera.values()),
            "valid_severity_3": all(value >= 3 for value in per_camera.values()),
        }
    return history, sample_validity


def temporal_summary(history, infos, sample_validity):
    summary = {}
    for severity in (1, 2, 3):
        offsets, spreads = [], []
        for info in infos:
            if not sample_validity[info["token"]][f"valid_severity_{severity}"]:
                continue
            sample_offsets = []
            for camera in info["cams"].values():
                previous = history[camera["sample_data_token"]][severity - 1]
                sample_offsets.append(
                    (previous["timestamp"] - camera["timestamp"]) / 1000.0
                )
            offsets.extend(sample_offsets)
            spreads.append(max(sample_offsets) - min(sample_offsets))
        valid = sum(
            value[f"valid_severity_{severity}"] for value in sample_validity.values()
        )
        summary[str(severity)] = {
            "valid_sample_count": valid,
            "invalid_sample_count": len(infos) - valid,
            "valid_rate": valid / len(infos),
            "camera_offset_ms": aggregate(offsets),
            "six_camera_offset_spread_ms": aggregate(spreads),
        }
    return summary


def calibration_samples(infos, metadata, count, seed):
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for info in infos:
        meta = metadata[info["token"]]
        buckets[(meta["time_of_day"], meta["city"])].append(info)
    selected = []
    keys = sorted(buckets)
    while len(selected) < min(count, len(infos)):
        changed = False
        for key in keys:
            if not buckets[key]:
                continue
            index = int(rng.integers(len(buckets[key])))
            selected.append(buckets[key].pop(index))
            changed = True
            if len(selected) == min(count, len(infos)):
                break
        if not changed:
            break
    return selected


def build_lowlight_calibration(infos, metadata, count, seed):
    selected = calibration_samples(infos, metadata, count, seed)
    raw = defaultdict(lambda: defaultdict(list))
    sample_records = []
    for info in selected:
        images = {name: Image.open(camera["data_path"]).convert("RGB") for name, camera in info["cams"].items()}
        boxes = {
            name: [
                projection
                for box in info["gt_boxes"]
                if (projection := projected_bbox(box, camera, images[name].size))
            ]
            for name, camera in info["cams"].items()
        }
        sample_record = {
            "sample_token": info["token"],
            "time_of_day": metadata[info["token"]]["time_of_day"],
            "city": metadata[info["token"]]["city"],
            "weather": metadata[info["token"]]["weather"],
            "candidates": {},
        }
        for candidate, severities in LOWLIGHT_CANDIDATES.items():
            sample_record["candidates"][candidate] = {}
            for severity, parameters in severities.items():
                metrics = defaultdict(list)
                for channel, image in images.items():
                    clean = np.asarray(image)
                    dark = lowlight_array(image, parameters)
                    clean_gray, dark_gray = clean.mean(2), dark.mean(2)
                    metrics["full_mean_ratio"].append(dark.mean() / max(clean.mean(), 1e-6))
                    metrics["median_ratio"].append(
                        np.median(dark_gray) / max(np.median(clean_gray), 1e-6)
                    )
                    metrics["black_pixel_ratio"].append(float((dark_gray <= 5).mean()))
                    metrics["saturated_pixel_ratio"].append(float((dark_gray >= 250).mean()))
                    for projection in boxes[channel]:
                        clean_value = crop_brightness(clean, projection["bbox"])
                        dark_value = crop_brightness(dark, projection["bbox"])
                        if clean_value and dark_value is not None:
                            metrics["target_region_ratio"].append(dark_value / clean_value)
                compact = {key: float(np.mean(values)) for key, values in metrics.items()}
                sample_record["candidates"][candidate][str(severity)] = compact
                for key, values in metrics.items():
                    raw[(candidate, severity)][key].extend(values)
        sample_records.append(sample_record)
        for image in images.values():
            image.close()
    summary = {
        "seed": seed,
        "sample_count": len(selected),
        "sample_tokens": [info["token"] for info in selected],
        "strata": Counter(
            (metadata[info["token"]]["time_of_day"], metadata[info["token"]]["city"])
            for info in selected
        ),
        "black_threshold_uint8": 5,
        "saturated_threshold_uint8": 250,
        "candidates": {},
        "sample_records": sample_records,
    }
    summary["strata"] = {"/".join(key): value for key, value in summary["strata"].items()}
    for (candidate, severity), metrics in raw.items():
        summary["candidates"].setdefault(candidate, {})[str(severity)] = {
            key: aggregate(values) for key, values in metrics.items()
        }
    return summary


def prediction_matches(info, prediction, score_threshold=0.1, distance_threshold=2.0):
    boxes = prediction["boxes_3d"].tensor.detach().cpu().numpy()
    scores = prediction["scores_3d"].detach().cpu().numpy()
    labels = prediction["labels_3d"].detach().cpu().numpy()
    keep = scores >= score_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    matches = [None] * len(info["gt_boxes"])
    matched_predictions = 0
    for class_index, class_name in enumerate(NuScenesDataset.CLASSES):
        gt_indices = np.flatnonzero(np.asarray(info["gt_names"]) == class_name)
        pred_indices = np.flatnonzero(labels == class_index)
        if not len(gt_indices) or not len(pred_indices):
            continue
        distances = np.linalg.norm(
            info["gt_boxes"][gt_indices, None, :2] - boxes[pred_indices][None, :, :2],
            axis=2,
        )
        rows, columns = linear_sum_assignment(distances)
        for row, column in zip(rows, columns):
            distance = float(distances[row, column])
            if distance > distance_threshold:
                continue
            gt_index, pred_index = int(gt_indices[row]), int(pred_indices[column])
            gt = info["gt_boxes"][gt_index]
            pred = boxes[pred_index]
            yaw_error = abs(float(np.arctan2(np.sin(pred[6] - gt[6]), np.cos(pred[6] - gt[6]))))
            velocity_error = float(np.linalg.norm(pred[7:9] - info["gt_velocity"][gt_index]))
            matches[gt_index] = {
                "matched": True,
                "score": float(scores[pred_index]),
                "center_error": distance,
                "orientation_error": yaw_error,
                "velocity_error": velocity_error,
            }
            matched_predictions += 1
    for index, match in enumerate(matches):
        if match is None:
            matches[index] = {"matched": False, "score": None, "center_error": None, "orientation_error": None, "velocity_error": None}
    return matches, int(len(boxes) - matched_predictions)


def object_condition(record):
    camera_error = record["errors"]["camera"]
    lidar_error = record["errors"]["lidar"]
    camera_strong = (
        bool(record["visible_cameras"])
        and record["target_brightness"] is not None
        and record["target_brightness"] >= 0.12
        and int(record["visibility_token"]) >= 3
        and camera_error["matched"]
        and camera_error["center_error"] <= 2.0
    )
    lidar_strong = (
        record["num_lidar_pts"] >= 5
        and lidar_error["matched"]
        and lidar_error["center_error"] <= 2.0
    )
    if camera_strong and lidar_strong:
        return "both_strong"
    if camera_strong:
        return "lidar_weak_camera_strong"
    if lidar_strong:
        return "camera_weak_lidar_strong"
    return "both_weak"


def build_complementarity(nusc, infos, metadata, predictions, temporal_validity):
    objects, scenes = [], []
    visibility = {item["token"]: item["description"] for item in nusc.visibility}
    for sample_index, info in enumerate(infos):
        sample = nusc.get("sample", info["token"])
        if len(sample["anns"]) != len(info["gt_boxes"]):
            raise ValueError(f"annotation alignment failed for {info['token']}")
        model_matches = {}
        false_positives = {}
        for modality in ("camera", "lidar", "fusion"):
            model_matches[modality], false_positives[modality] = prediction_matches(
                info, predictions[modality][sample_index]
            )
        images = {name: Image.open(camera["data_path"]).convert("RGB") for name, camera in info["cams"].items()}
        condition_counts = Counter()
        categories = Counter()
        dynamic_count = 0
        for object_index, annotation_token in enumerate(sample["anns"]):
            annotation = nusc.get("sample_annotation", annotation_token)
            projections = []
            for channel, camera in info["cams"].items():
                projection = projected_bbox(info["gt_boxes"][object_index], camera, images[channel].size)
                if projection:
                    projection.update(
                        channel=channel,
                        brightness=crop_brightness(images[channel], projection["bbox"]),
                    )
                    projections.append(projection)
            velocity = info["gt_velocity"][object_index]
            if np.linalg.norm(velocity) > 0.5:
                dynamic_count += 1
            brightness_values = [
                item["brightness"] for item in projections if item["brightness"] is not None
            ]
            record = {
                "family": "complementarity_control",
                "sample_token": info["token"],
                "annotation_token": annotation_token,
                "category": info["gt_names"][object_index],
                "distance": float(np.linalg.norm(info["gt_boxes"][object_index, :2])),
                "visibility_token": annotation["visibility_token"],
                "visibility": visibility.get(annotation["visibility_token"]),
                "num_lidar_pts": int(info["num_lidar_pts"][object_index]),
                "num_radar_pts": int(info["num_radar_pts"][object_index]),
                "visible_cameras": [item["channel"] for item in projections],
                "camera_projections": projections,
                "projection_box_area": max([item["area"] for item in projections], default=0.0),
                "target_brightness": (
                    float(np.mean(brightness_values)) if brightness_values else None
                ),
                "truncated": any(item["truncated"] for item in projections),
                "time_of_day": metadata[info["token"]]["time_of_day"],
                "city": metadata[info["token"]]["city"],
                "weather": metadata[info["token"]]["weather"],
                "errors": {modality: model_matches[modality][object_index] for modality in model_matches},
            }
            record["complementarity_label"] = object_condition(record)
            condition_counts[record["complementarity_label"]] += 1
            categories[record["category"]] += 1
            objects.append(record)
        for image in images.values():
            image.close()
        scenes.append(
            {
                "family": "complementarity_control",
                "sample_token": info["token"],
                **metadata[info["token"]],
                "object_count": int(len(sample["anns"])),
                "dynamic_object_count": dynamic_count,
                "condition_counts": dict(condition_counts),
                "category_counts": dict(categories),
                "false_positive_counts": false_positives,
                **temporal_validity[info["token"]],
            }
        )
    summary = {
        "family": "complementarity_control",
        "sample_count": len(scenes),
        "object_count": len(objects),
        "condition_counts": dict(Counter(item["complementarity_label"] for item in objects)),
        "city_counts": dict(Counter(item["city"] for item in scenes)),
        "time_of_day_counts": dict(Counter(item["time_of_day"] for item in scenes)),
        "weather_counts": dict(Counter(item["weather"] for item in scenes)),
        "matching": {"score_threshold": 0.1, "center_distance_threshold_m": 2.0},
        "label_rule": {"camera_brightness_min": 0.12, "visibility_token_min": 3, "lidar_points_min": 5},
    }
    return objects, scenes, summary


def sample_features(scene):
    features = {
        f"city:{scene['city']}",
        f"time:{scene['time_of_day']}",
        f"weather:{scene['weather']}",
    }
    features.update(f"condition:{key}" for key, value in scene["condition_counts"].items() if value)
    features.update(f"category:{key}" for key, value in scene["category_counts"].items() if value)
    if scene["dynamic_object_count"]:
        features.add("dynamic")
    if scene.get("valid_severity_3", False):
        features.add("temporal_s3_valid")
    return features


def build_pilot(scenes, size, seed):
    size = min(size, len(scenes))
    targets = {
        "city:boston": 80,
        "city:singapore": 80,
        "time:night": 35,
        "time:day": 100,
        "weather:rain": 35,
        "weather:not_rain": 100,
        "dynamic": 100,
        "temporal_s3_valid": 180,
    }
    targets.update({f"condition:{name}": 30 for name in ("camera_weak_lidar_strong", "lidar_weak_camera_strong", "both_strong", "both_weak")})
    targets.update({f"category:{name}": 10 for name in NuScenesDataset.CLASSES})
    rng = np.random.default_rng(seed)
    jitter = {scene["sample_token"]: float(rng.random()) for scene in scenes}
    features = {scene["sample_token"]: sample_features(scene) for scene in scenes}
    selected, counts = [], Counter()
    remaining = {scene["sample_token"]: scene for scene in scenes}
    for _ in range(size):
        def score(scene):
            token = scene["sample_token"]
            coverage = sum(
                1.0 / targets[feature]
                for feature in features[token]
                if feature in targets and counts[feature] < targets[feature]
            )
            return coverage + jitter[token] * 1e-6

        chosen = max(remaining.values(), key=score)
        token = chosen["sample_token"]
        selected.append(token)
        counts.update(features[token])
        remaining.pop(token)
    selected_scenes = [next(scene for scene in scenes if scene["sample_token"] == token) for token in selected]
    summary = {
        "seed": seed,
        "sample_count": len(selected),
        "targets": targets,
        "achieved_feature_counts": {key: counts[key] for key in sorted(targets)},
        "city_counts": dict(Counter(scene["city"] for scene in selected_scenes)),
        "time_of_day_counts": dict(Counter(scene["time_of_day"] for scene in selected_scenes)),
        "weather_counts": dict(Counter(scene["weather"] for scene in selected_scenes)),
        "temporal_validity": {
            f"severity_{severity}": sum(
                scene.get(f"valid_severity_{severity}", False)
                for scene in selected_scenes
            )
            for severity in (1, 2, 3)
        },
        "condition_sample_counts": {
            name: sum(scene["condition_counts"].get(name, 0) > 0 for scene in selected_scenes)
            for name in ("camera_weak_lidar_strong", "lidar_weak_camera_strong", "both_strong", "both_weak")
        },
        "category_sample_counts": {
            name: sum(scene["category_counts"].get(name, 0) > 0 for scene in selected_scenes)
            for name in NuScenesDataset.CLASSES
        },
    }
    return selected, summary


def main():
    args = parse_args()
    output = Path(args.output_dir)
    infos = sorted(mmcv.load(args.info_file)["infos"], key=lambda item: item["timestamp"])
    if args.max_samples:
        infos = infos[: args.max_samples]
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataset_root, verbose=False)
    scene_cache = {}
    metadata = {}
    for info in infos:
        _, metadata[info["token"]] = scene_metadata(nusc, info["token"], scene_cache)

    history, temporal_validity = build_camera_history(nusc, infos)
    write_json(output / "configs/camera_history.json", history)
    write_json(
        output / "metrics/temporal_validity.json",
        {
            "summary": temporal_summary(history, infos, temporal_validity),
            "samples": temporal_validity,
        },
    )

    calibration = build_lowlight_calibration(
        infos, metadata, args.calibration_samples, args.seed
    )
    write_json(output / "metrics/lowlight_calibration.json", calibration)

    prediction_paths = {
        modality: Path("outputs/stage1/baseline") / f"{modality}.pkl"
        for modality in ("camera", "lidar", "fusion")
    }
    predictions = {modality: mmcv.load(path)[: len(infos)] for modality, path in prediction_paths.items()}
    objects, scenes, summary = build_complementarity(
        nusc, infos, metadata, predictions, temporal_validity
    )
    write_jsonl(output / "complementarity/object_index.jsonl", objects)
    write_jsonl(output / "complementarity/scene_index.jsonl", scenes)
    write_json(output / "complementarity/summary.json", summary)

    tokens, pilot_summary = build_pilot(scenes, args.pilot_size, args.seed)
    write_json(
        output / "pilot/sample_tokens.json",
        {"seed": args.seed, "sample_tokens": tokens, "summary": pilot_summary},
    )
    (output / "pilot/sample_tokens.txt").write_text("\n".join(tokens) + "\n")
    write_json(output / "pilot/stratification_summary.json", pilot_summary)


if __name__ == "__main__":
    main()
