#!/usr/bin/env python3
"""Export frozen query authority for RiskProp3D final confirmation."""

import argparse
import hashlib
import json
import os
import pickle
import resource
import time
from pathlib import Path

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes

from tools import probset3d_f0_export as export
from tools import stage3_run_real_target_uq as stage3
from tools.stage3d_b_native_assignment import (
    assignment_from_cost,
    cost_components,
    detection_gt,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/riskprop3d_final_confirmation/exports"
DEV_INFO = ROOT / "outputs/stage3/splits/nuscenes_infos_stage3_dev.pkl"
DEV_TEMPORAL_INFO = {
    2: ROOT / "outputs/riskprop3d_final_confirmation/inputs/nuscenes_infos_dev_temporal_s2_valid.pkl",
    3: ROOT / "outputs/riskprop3d_final_confirmation/inputs/nuscenes_infos_dev_temporal_s3_valid.pkl",
}
TEMPORAL_HISTORY = ROOT / "outputs/riskprop3d_final_confirmation/inputs/dev_camera_history.json"
OFFICIAL_INFO = ROOT / "outputs/stage2/full/nuscenes_infos_val_6019.pkl"
CONDITIONS = {
    "rotation_s2": ("camera_extrinsic_rotation", 2),
    "rotation_s3": ("camera_extrinsic_rotation", 3),
    "translation_s3": ("camera_extrinsic_translation", 3),
    "temporal_s2": ("camera_temporal_offset", 2),
    "temporal_s3": ("camera_temporal_offset", 3),
}
FIELDS = (
    "sample_token",
    "scene_token",
    "query_features",
    "query_logits",
    "query_boxes",
    "seed_probabilities",
    "native_matched_gt",
    "gt_boxes",
    "gt_labels",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class Writer:
    def __init__(self, root, part_size):
        self.root = Path(root)
        self.part_size = part_size
        self.rows = []
        self.parts = []

    @staticmethod
    def variable(rows, name, width, dtype):
        values = [np.asarray(row[name], dtype=dtype) for row in rows]
        offsets = np.zeros(len(values) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(value) for value in values])
        joined = (
            np.concatenate(values)
            if offsets[-1]
            else np.empty((0, width), dtype=dtype)
            if width is not None
            else np.empty(0, dtype=dtype)
        )
        return joined, offsets

    def append(self, row):
        self.rows.append(row)
        if len(self.rows) == self.part_size:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        path = self.root / f"part-{len(self.parts):05d}.npz"
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.rows
        arrays = {
            "sample_tokens": np.asarray(
                [row["sample_token"] for row in rows], dtype="U32"
            ),
            "scene_tokens": np.asarray(
                [row["scene_token"] for row in rows], dtype="U32"
            ),
            "query_features": np.stack(
                [row["query_features"] for row in rows]
            ).astype(np.float32),
            "query_logits": np.stack(
                [row["query_logits"] for row in rows]
            ).astype(np.float32),
            "query_boxes": np.stack(
                [row["query_boxes"] for row in rows]
            ).astype(np.float32),
            "seed_probabilities": np.stack(
                [row["seed_probabilities"] for row in rows]
            ).astype(np.float32),
            "native_matched_gt": np.stack(
                [row["native_matched_gt"] for row in rows]
            ).astype(np.int16),
        }
        for name, width, dtype in (
            ("gt_boxes", 9, np.float32),
            ("gt_labels", None, np.int16),
        ):
            arrays[name], arrays[f"{name}_offsets"] = self.variable(
                rows, name, width, dtype
            )
        temporary = path.with_suffix(".npz.partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        self.parts.append(
            {
                "file": str(path.relative_to(ROOT)),
                "sample_count": len(rows),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
        self.rows = []

    def close(self):
        self.flush()
        return self.parts


def condition(args):
    if args.phase == "official":
        if args.condition != "clean":
            raise ValueError("official confirmation is clean-only")
        return "clean", 0, OFFICIAL_INFO, True
    corruption, severity = CONDITIONS[args.condition]
    info = DEV_TEMPORAL_INFO[severity] if corruption == "camera_temporal_offset" else DEV_INFO
    return corruption, severity, info, False


def prepare_temporal_info():
    with DEV_INFO.open("rb") as stream:
        source = pickle.load(stream)
    infos = source["infos"] if isinstance(source, dict) else source
    if TEMPORAL_HISTORY.exists():
        history = json.loads(TEMPORAL_HISTORY.read_text(encoding="utf-8"))
    else:
        nusc = NuScenes(
            version="v1.0-trainval",
            dataroot=str(ROOT / "data/nuscenes"),
            verbose=False,
        )
        history = {}
        for info in infos:
            for camera in info["cams"].values():
                current = nusc.get("sample_data", camera["sample_data_token"])
                previous, entries = current["prev"], []
                while previous and len(entries) < 3:
                    record = nusc.get("sample_data", previous)
                    entries.append(
                        {
                            "sample_data_token": previous,
                            "timestamp": record["timestamp"],
                            "data_path": str(
                                ROOT / "data/nuscenes" / record["filename"]
                            ),
                        }
                    )
                    previous = record["prev"]
                history[current["token"]] = entries
        write_json(TEMPORAL_HISTORY, history)
    expected_camera_tokens = {
        camera["sample_data_token"]
        for info in infos
        for camera in info["cams"].values()
    }
    if set(history) != expected_camera_tokens:
        raise ValueError("temporal history does not exactly cover dev camera tokens")
    records = {}
    for severity, path in DEV_TEMPORAL_INFO.items():
        if path.exists():
            raise FileExistsError(path)
        selected = [
            info
            for info in infos
            if all(
                len(history.get(camera["sample_data_token"], [])) >= severity
                for camera in info["cams"].values()
            )
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        value = {"infos": selected}
        if isinstance(source, dict) and "metadata" in source:
            value["metadata"] = source["metadata"]
        with temporary.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        records[str(severity)] = {
            "path": str(path.relative_to(ROOT)),
            "sample_count": len(selected),
            "sha256": sha256(path),
        }
    print(
        json.dumps(
            {
                "temporal_history": {
                    "path": str(TEMPORAL_HISTORY.relative_to(ROOT)),
                    "sha256": sha256(TEMPORAL_HISTORY),
                },
                "temporal_info": records,
            }
        )
    )


def run(args):
    if args.device not in range(4) or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("GPU must be 0--3 and shard index must be valid")
    corruption, severity, info_path, official = condition(args)
    root = OUT / args.phase / args.condition / f"shard-{args.shard_index:02d}-of-{args.shard_count:02d}"
    accepted = root / "accepted.json"
    if accepted.exists() or root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    if sha256(export.CONFIG) != export.EXPECTED_CONFIG_SHA256:
        raise ValueError("detector config hash mismatch")
    if sha256(export.CHECKPOINT) != export.EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("detector checkpoint hash mismatch")
    with info_path.open("rb") as stream:
        source = pickle.load(stream)
    infos = source["infos"] if isinstance(source, dict) else source
    info_by_token = {item["token"]: item for item in infos}
    if len(info_by_token) != len(infos):
        raise ValueError("duplicate info token")
    scenes = export.scene_map(infos)

    cfg = stage3.load_cfg(str(export.CONFIG))
    if corruption != "clean":
        export.inject_stage2_corruption(
            cfg,
            corruption,
            severity,
            export.SEED,
            None,
            "calibrated_candidate",
            str(TEMPORAL_HISTORY.relative_to(ROOT)),
        )
    dataset, loader, full_count = export.select_dataset(
        cfg, info_path, args.shard_index, args.shard_count
    )
    expected = {item["token"] for item in dataset.data_infos}
    torch.cuda.set_device(args.device)
    torch.manual_seed(export.SEED)
    np.random.seed(export.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = stage3.build_frozen_model(cfg, str(export.CHECKPOINT), args.device)
    module = model.module if hasattr(model, "module") else model
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ValueError("detector is not frozen")
    head = module.heads["object"]
    captured = []

    def capture_dense(_module, _inputs, output):
        if captured:
            raise RuntimeError("dense hook called twice")
        captured.append(output.detach())

    handle = head.heatmap_head.register_forward_hook(capture_dense)
    writer = Writer(root / "parts", args.part_size)
    seen = []
    started = time.time()
    try:
        for index, data in enumerate(loader):
            with torch.inference_mode():
                output = model(
                    return_loss=False,
                    rescale=True,
                    return_stage3_target_uq=True,
                    stage3_branch="fusion",
                    stage3_num_proposals=export.PROPOSALS,
                    **data,
                )[0]
            if len(captured) != 1:
                raise RuntimeError("dense hook did not produce one tensor")
            dense = captured.pop()
            item = output["stage3_target_uq"]
            token = item["sample_token"]
            if token not in expected or token in seen:
                raise ValueError("unexpected or duplicate sample token")
            _, gt_boxes, gt_labels = detection_gt(info_by_token[token])
            boxes = item["decoded_boxes"].float()
            logits = item["class_logits"].float()
            components = cost_components(
                "fusion",
                model,
                logits,
                boxes,
                gt_boxes,
                gt_labels,
                torch.device(f"cuda:{args.device}"),
            )
            matched, _, _, _ = assignment_from_cost(
                components["total"], boxes[:, :2].numpy(), gt_boxes[:, :2]
            )
            _, _, seed_indices, seed_labels, seed_probability = export.exact_local_max(
                head, dense
            )
            if not torch.equal(
                seed_indices[0], item["query_indices"].long().to(seed_indices.device)
            ) or not torch.equal(
                seed_labels[0], item["query_labels"].long().to(seed_labels.device)
            ):
                raise ValueError("pre-topK/query identity mismatch")
            row = {
                "sample_token": token,
                "scene_token": scenes[token],
                "query_features": item["features"].numpy(),
                "query_logits": logits.numpy(),
                "query_boxes": boxes.numpy(),
                "seed_probabilities": seed_probability[0].cpu().numpy(),
                "native_matched_gt": matched,
                "gt_boxes": gt_boxes,
                "gt_labels": gt_labels,
            }
            if not all(
                np.isfinite(value).all()
                for name, value in row.items()
                if name not in ("sample_token", "scene_token")
            ):
                raise ValueError("non-finite export row")
            writer.append(row)
            seen.append(token)
            if index == 0 or (index + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "condition": args.condition,
                            "phase": args.phase,
                            "samples": index + 1,
                            "shard": args.shard_index,
                        }
                    ),
                    flush=True,
                )
    finally:
        handle.remove()
    parts = writer.close()
    if set(seen) != expected or len(seen) != len(expected):
        raise ValueError("shard token coverage mismatch")
    if any(parameter.grad is not None for parameter in module.parameters()):
        raise ValueError("frozen detector accumulated gradients")
    manifest = {
        "accepted": True,
        "condition": args.condition,
        "corruption": corruption,
        "severity": severity,
        "phase": args.phase,
        "official_val_used": official,
        "internal_test_used": False,
        "training_performed": False,
        "fields": list(FIELDS),
        "full_sample_count": full_count,
        "sample_count": len(seen),
        "sample_token_sha256": hashlib.sha256(
            "\n".join(seen).encode("ascii")
        ).hexdigest(),
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "parts": parts,
        "info_path": str(info_path.relative_to(ROOT)),
        "info_sha256": sha256(info_path),
        "config_sha256": sha256(export.CONFIG),
        "checkpoint_sha256": sha256(export.CHECKPOINT),
        "exporter_sha256": sha256(Path(__file__)),
        "source_exporter_sha256": sha256(Path(export.__file__)),
        "temporal_history_sha256": (
            sha256(TEMPORAL_HISTORY)
            if corruption == "camera_temporal_offset"
            else None
        ),
        "seed": export.SEED,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(args.device),
        "wall_time_seconds": time.time() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    write_json(accepted, manifest)
    print(json.dumps({"accepted": str(accepted), "samples": len(seen)}))


def self_check():
    assert set(CONDITIONS) == {
        "rotation_s2",
        "rotation_s3",
        "translation_s3",
        "temporal_s2",
        "temporal_s3",
    }
    assert FIELDS[-3:] == ("native_matched_gt", "gt_boxes", "gt_labels")
    assert export.PROPOSALS == 200 and export.SEED == 20260708
    assert DEV_INFO.exists()
    print(json.dumps({"self_check": "passed", "conditions": CONDITIONS}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("external", "official"))
    parser.add_argument("--condition")
    parser.add_argument("--device", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--part-size", type=int, default=32)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--prepare-temporal-info", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.prepare_temporal_info:
        prepare_temporal_info()
        return
    if args.phase is None or args.condition is None or args.device is None or args.shard_index is None:
        parser.error("phase, condition, device, and shard-index are required")
    if args.phase == "external" and args.condition not in CONDITIONS:
        parser.error("invalid external condition")
    run(args)


if __name__ == "__main__":
    main()
