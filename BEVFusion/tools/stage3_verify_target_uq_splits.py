#!/usr/bin/env python3
"""Verify Stage 3 target-UQ split isolation and class distributions."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import mmcv


ROOT = Path("outputs/stage3/target_uq")


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


def split_distribution(info_path):
    data = mmcv.load(info_path)
    classes = Counter()
    valid_classes = Counter()
    velocity_valid = 0
    object_count = 0
    for info in data["infos"]:
        names = info.get("gt_names", [])
        valid = info.get("valid_flag")
        velocities = info.get("gt_velocity")
        for index, name in enumerate(names):
            classes[str(name)] += 1
            object_count += 1
            if valid is None or bool(valid[index]):
                valid_classes[str(name)] += 1
            if velocities is not None:
                vx, vy = velocities[index]
                if abs(float(vx)) < 1e6 and abs(float(vy)) < 1e6:
                    velocity_valid += 1
    return {
        "sample_count": len(data["infos"]),
        "object_count": object_count,
        "class_distribution": dict(sorted(classes.items())),
        "valid_class_distribution": dict(sorted(valid_classes.items())),
        "velocity_valid_count": velocity_valid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/stage3/splits/manifest.json")
    parser.add_argument("--output", default=str(ROOT / "splits" / "verification.json"))
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    if not manifest["checks"]["all_stage3_intersections_zero"]:
        raise ValueError("Stage 3 split intersection check failed")
    if not manifest["checks"]["official_val_scene_and_sample_overlap_zero"]:
        raise ValueError("official val leakage check failed")

    split_files = {
        "stage3_train": "outputs/stage3/splits/nuscenes_infos_stage3_train.pkl",
        "stage3_label_cal": "outputs/stage3/splits/nuscenes_infos_stage3_label_cal.pkl",
        "stage3_dev": "outputs/stage3/splits/nuscenes_infos_stage3_dev.pkl",
    }
    output = {
        "seed": manifest["seed"],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "checks": manifest["checks"],
        "intersections": manifest["intersections"],
        "official_val_overlap": manifest["official_val_overlap"],
        "split_strata": {name: data["strata"] for name, data in manifest["splits"].items()},
        "splits": {},
    }
    for name, path in split_files.items():
        output["splits"][name] = {
            **split_distribution(path),
            "info_path": path,
            "info_sha256": sha256(path),
        }
    strict_json(args.output, output)
    print(json.dumps({"output": args.output, "checks": output["checks"]}, indent=2))


if __name__ == "__main__":
    main()
