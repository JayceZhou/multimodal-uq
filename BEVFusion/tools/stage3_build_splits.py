#!/usr/bin/env python3
"""Build deterministic, scene-disjoint Stage 3 train/calibration/dev splits."""

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import mmcv
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


SEED = 20260708
ROOT = Path("outputs/stage3/splits")


def strict_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_stratum(scene, log):
    description = scene["description"].lower()
    location = log["location"].lower()
    city = "boston" if "boston" in location else "singapore"
    return city, "night" if "night" in description else "day", "rain" if "rain" in description else "not_rain"


def stratified_take(items, count, seed, key=lambda item: item["stratum"]):
    """Take an exact count using largest-remainder stratum quotas."""
    if not 0 <= count <= len(items):
        raise ValueError(f"invalid take count {count} for {len(items)} items")
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    rng = random.Random(seed)
    for values in groups.values():
        values.sort(key=lambda item: item["token"])
        rng.shuffle(values)
    raw = {name: len(values) * count / len(items) for name, values in groups.items()}
    quota = {name: int(math.floor(value)) for name, value in raw.items()}
    remaining = count - sum(quota.values())
    order = sorted(groups, key=lambda name: (-(raw[name] - quota[name]), str(name)))
    for name in order[:remaining]:
        quota[name] += 1
    selected = [item for name, values in groups.items() for item in values[: quota[name]]]
    selected_tokens = {item["token"] for item in selected}
    rest = [item for item in items if item["token"] not in selected_tokens]
    assert len(selected) == count and len(rest) + count == len(items)
    return selected, rest


def samples_for_scene(nusc, scene):
    tokens = []
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        tokens.append(token)
        token = sample["next"]
    if len(tokens) != scene["nbr_samples"]:
        raise ValueError(f"scene {scene['token']} sample-chain mismatch")
    return tokens


def summary(records):
    return {
        "scene_count": len(records),
        "sample_count": sum(len(item["sample_tokens"]) for item in records),
        "strata": {"|".join(key): value for key, value in sorted(Counter(item["stratum"] for item in records).items())},
    }


def build(args):
    ROOT.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataset_root, verbose=False)
    official = create_splits_scenes()
    scene_by_name = {scene["name"]: scene for scene in nusc.scene}
    train_names, val_names = set(official["train"]), set(official["val"])
    if len(train_names) != 700:
        raise ValueError(f"expected 700 official train scenes, got {len(train_names)}")

    records = []
    for name in sorted(train_names):
        scene = scene_by_name[name]
        log = nusc.get("log", scene["log_token"])
        records.append({
            "name": name,
            "token": scene["token"],
            "description": scene["description"],
            "stratum": scene_stratum(scene, log),
            "sample_tokens": samples_for_scene(nusc, scene),
        })

    label_cal, remaining = stratified_take(records, 70, args.seed + 1)
    dev, train = stratified_take(remaining, 70, args.seed + 2)
    splits = {"stage3_train": train, "stage3_label_cal": label_cal, "stage3_dev": dev}
    if {name: len(value) for name, value in splits.items()} != {
        "stage3_train": 560, "stage3_label_cal": 70, "stage3_dev": 70
    }:
        raise AssertionError("unexpected scene counts")

    source = mmcv.load(args.train_info)
    info_by_token = {item["token"]: item for item in source["infos"]}
    val_source = mmcv.load(args.val_info)
    val_info_tokens = {item["token"] for item in val_source["infos"]}
    official_val_scene_tokens = {scene_by_name[name]["token"] for name in val_names}
    official_val_sample_tokens = {
        token for name in val_names for token in samples_for_scene(nusc, scene_by_name[name])
    }
    if official_val_sample_tokens != val_info_tokens:
        raise ValueError("official val sample tokens do not match the frozen Stage 1 val info")

    paths = []
    token_sets = {}
    for split_name, split_records in splits.items():
        scene_tokens = sorted(item["token"] for item in split_records)
        sample_tokens = sorted(token for item in split_records for token in item["sample_tokens"])
        if len(sample_tokens) != len(set(sample_tokens)):
            raise ValueError(f"duplicate samples in {split_name}")
        missing = set(sample_tokens) - set(info_by_token)
        if missing:
            raise ValueError(f"{split_name} has {len(missing)} samples absent from train info")
        scene_path, sample_path = ROOT / f"{split_name}_scenes.json", ROOT / f"{split_name}_samples.json"
        strict_json(scene_path, {"seed": args.seed, "scene_tokens": scene_tokens})
        strict_json(sample_path, {"seed": args.seed, "sample_tokens": sample_tokens})
        info_path = ROOT / f"nuscenes_infos_{split_name}.pkl"
        ordered = [item for item in source["infos"] if item["token"] in set(sample_tokens)]
        mmcv.dump({"infos": ordered, "metadata": source["metadata"]}, info_path)
        paths.extend((scene_path, sample_path, info_path))
        token_sets[split_name] = {"scenes": set(scene_tokens), "samples": set(sample_tokens)}

    split_names = list(splits)
    intersections = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            intersections[f"{left}__{right}"] = {
                "scene_count": len(token_sets[left]["scenes"] & token_sets[right]["scenes"]),
                "sample_count": len(token_sets[left]["samples"] & token_sets[right]["samples"]),
            }
    val_overlap = {
        name: {
            "scene_count": len(tokens["scenes"] & official_val_scene_tokens),
            "sample_count": len(tokens["samples"] & official_val_sample_tokens),
        }
        for name, tokens in token_sets.items()
    }
    if any(value[dimension] for value in intersections.values() for dimension in ("scene_count", "sample_count")):
        raise ValueError("Stage 3 split overlap detected")
    if any(value[dimension] for value in val_overlap.values() for dimension in ("scene_count", "sample_count")):
        raise ValueError("official val leaked into Stage 3 development splits")

    # Scene-level pilot: start near 2,000 samples and include whole scenes only.
    average = sum(len(item["sample_tokens"]) for item in train) / len(train)
    pilot_scene_count = max(1, math.ceil(args.pilot_min_samples / average))
    while True:
        pilot, _ = stratified_take(train, pilot_scene_count, args.seed + 3)
        pilot_samples = [token for item in pilot for token in item["sample_tokens"]]
        if len(pilot_samples) >= args.pilot_min_samples:
            break
        pilot_scene_count += 1
    if len(pilot_samples) > args.pilot_max_samples:
        raise ValueError("scene-level pilot exceeds requested maximum")
    pilot_scene_path, pilot_sample_path = ROOT / "pilot_scenes.json", ROOT / "pilot_samples.json"
    strict_json(pilot_scene_path, {"seed": args.seed, "scene_tokens": sorted(item["token"] for item in pilot)})
    strict_json(pilot_sample_path, {"seed": args.seed, "sample_tokens": sorted(pilot_samples)})
    pilot_info_path = ROOT / "nuscenes_infos_stage3_pilot_base.pkl"
    pilot_set = set(pilot_samples)
    pilot_infos = [item for item in source["infos"] if item["token"] in pilot_set]
    mmcv.dump({"infos": pilot_infos, "metadata": source["metadata"]}, pilot_info_path)
    history = {}
    for info in pilot_infos:
        for camera in info["cams"].values():
            current = camera["sample_data_token"]
            previous = nusc.get("sample_data", current)["prev"]
            rows = []
            while previous and len(rows) < 3:
                sample_data = nusc.get("sample_data", previous)
                rows.append({
                    "sample_data_token": previous,
                    "timestamp": sample_data["timestamp"],
                    "data_path": str(Path("data/nuscenes") / sample_data["filename"]),
                })
                previous = sample_data["prev"]
            history[current] = rows
    history_path = ROOT / "pilot_camera_history.json"
    strict_json(history_path, history)
    paths.extend((pilot_scene_path, pilot_sample_path, pilot_info_path, history_path))

    manifest = {
        "seed": args.seed,
        "source": {"train_info": args.train_info, "val_info": args.val_info},
        "official_scene_counts": {"train": len(train_names), "val": len(val_names)},
        "splits": {name: summary(value) for name, value in splits.items()},
        "pilot": summary(pilot),
        "intersections": intersections,
        "official_val_overlap": val_overlap,
        "checks": {
            "all_stage3_intersections_zero": True,
            "official_val_scene_and_sample_overlap_zero": True,
            "all_train_info_samples_accounted_for": sum(len(value["samples"]) for value in token_sets.values()) == len(source["infos"]),
        },
        "files": {},
    }
    manifest_path = ROOT / "manifest.json"
    for path in paths:
        manifest["files"][str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    strict_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "splits": manifest["splits"], "pilot": manifest["pilot"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/nuscenes")
    parser.add_argument("--train-info", default="outputs/stage1/baseline/nuscenes_infos_train.pkl")
    parser.add_argument("--val-info", default="outputs/stage1/baseline/nuscenes_infos_val.pkl")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pilot-min-samples", type=int, default=2000)
    parser.add_argument("--pilot-max-samples", type=int, default=5000)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
