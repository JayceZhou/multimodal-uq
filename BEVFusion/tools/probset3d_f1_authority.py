#!/usr/bin/env python3
"""Export and mechanically validate clean F1 query authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import random
import resource
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/probset3d-f1-matplotlib")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models.crc_target_uq import ClassConditionalRealNVPDensity
from tools import stage3_run_real_target_uq as stage3
from tools.probset3d_f0_export import exact_local_max
from tools.stage3d_b_native_assignment import (
    cost_components,
    detection_gt,
    deterministic_assignment_from_cost,
)


ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "probset3d/P0_DATA_SPLIT_AUTHORITY.json"
SCHEMA = ROOT / "probset3d/f1/pilot/query_authority_schema.json"
NUSCENES_SAMPLES = ROOT / "data/nuscenes/v1.0-trainval/sample.json"
CONFIG = ROOT / "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
CHECKPOINT = ROOT / "pretrained/bevfusion-det.pth"
ASSIGNMENT_SOURCE = ROOT / "tools/stage3d_b_native_assignment.py"
DENSITY_CHECKPOINT = (
    ROOT
    / "outputs/transactions_extension/completion_run/query2u_density/fusion/checkpoint.pt"
)
OLD_AUTHORITY_OUT = ROOT / "outputs/probset3d/f1/authority/clean_query"
R1_ROOT = ROOT / "outputs/probset3d/f1/authority_r1"
A0_ROOT = ROOT / "outputs/probset3d/f1/authority_a0"
RUN_ROOTS = {name: A0_ROOT / name for name in ("run_a", "run_b", "run_c")}
R1_ACCEPTANCE = R1_ROOT / "authority_r1_acceptance.json"
R1_REPORT = R1_ROOT / "authority_r1_acceptance.md"
EXPECTED_CONFIG_SHA256 = "cb7bbbc25e8fdf67c2e914b11661828b684312300deb2afc410793192713e04b"
EXPECTED_CHECKPOINT_SHA256 = "ee0a389213922343508db40dff0ec04767b36d8f512da34b093939c37aaf2122"
EXPECTED_DENSITY_SHA256 = "8fa810e35d2b21b4c8b0ace724024a79e4cb146829ec1282815bc582546a10d7"
EXPECTED_NUSCENES_SAMPLES_SHA256 = "6035ac58b6e971622be2bb1be15b917e7cb4e05ae984d6b339c27c1699c4ad9d"
SPLITS = ("train", "label_cal", "dev")
SAMPLE_COUNTS = {"train": 22508, "label_cal": 2814, "dev": 2808}
SHARD_COUNTS = {"train": 6, "label_cal": 1, "dev": 1}
QUERY_COUNT = 200
FEATURE_DIM = 128
CLASS_COUNT = 10
SEED = 20260730


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def load_sample_scene_map():
    if sha256(NUSCENES_SAMPLES) != EXPECTED_NUSCENES_SAMPLES_SHA256:
        raise ValueError("nuScenes sample.json identity mismatch")
    records = strict_load(NUSCENES_SAMPLES)
    result = {record["token"]: record["scene_token"] for record in records}
    if len(result) != len(records):
        raise ValueError("duplicate sample token in nuScenes sample.json")
    return result


def strict_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def variable_arrays(rows, field: str, width=None, dtype=None):
    values = [np.asarray(row[field], dtype=dtype) for row in rows]
    offsets = np.zeros(len(values) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(value) for value in values])
    if offsets[-1]:
        joined = np.concatenate(values, axis=0)
    elif width is None:
        joined = np.empty(0, dtype=dtype)
    else:
        joined = np.empty((0, width), dtype=dtype)
    return joined, offsets


class PartWriter:
    def __init__(
        self,
        root: Path,
        split: str,
        part_size: int,
        density: ClassConditionalRealNVPDensity,
        device: torch.device,
    ) -> None:
        self.root = root
        self.split = split
        self.part_size = part_size
        self.density = density
        self.device = device
        self.rows = []
        self.parts = []

    def append(self, row) -> None:
        self.rows.append(row)

    def flush(self) -> None:
        if not self.rows:
            return
        rows = self.rows
        features = np.stack([row["query_features"] for row in rows]).astype(
            np.float32, copy=False
        )
        seed_class = np.stack([row["seed_class"] for row in rows]).astype(
            np.int64, copy=False
        )
        with torch.inference_mode():
            density_value = self.density(
                torch.from_numpy(features).to(self.device),
                torch.from_numpy(seed_class).to(self.device),
            ).cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(density_value).all() or not (
            (density_value >= 0).all() and (density_value <= 1).all()
        ):
            raise ValueError("query density is outside [0,1] or non-finite")

        part_index = len(self.parts)
        path = self.root / f"part-{part_index:05d}.npz"
        temporary = path.with_suffix(".npz.partial")
        if path.exists() or temporary.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        self.root.mkdir(parents=True, exist_ok=True)
        arrays = {
            "split_ids": np.asarray([self.split] * len(rows), dtype="U10"),
            "sample_tokens": np.asarray(
                [row["sample_token"] for row in rows], dtype="U32"
            ),
            "scene_tokens": np.asarray(
                [row["scene_token"] for row in rows], dtype="U32"
            ),
            "query_id": np.stack([row["query_id"] for row in rows]),
            "seed_flat_index": np.stack(
                [row["seed_flat_index"] for row in rows]
            ),
            "seed_position_xy": np.stack(
                [row["seed_position_xy"] for row in rows]
            ),
            "seed_class": np.stack([row["seed_class"] for row in rows]),
            "seed_score": np.stack([row["seed_score"] for row in rows]),
            "query_features": features,
            "class_logits": np.stack(
                [row["class_logits"] for row in rows]
            ).astype(np.float32, copy=False),
            "decoded_box_mean": np.stack(
                [row["decoded_box_mean"] for row in rows]
            ).astype(np.float32, copy=False),
            "native_matched_gt": np.stack(
                [row["native_matched_gt"] for row in rows]
            ),
            "matched_gt_class": np.stack(
                [row["matched_gt_class"] for row in rows]
            ),
            "matched_gt_box": np.stack(
                [row["matched_gt_box"] for row in rows]
            ).astype(np.float32, copy=False),
            "query_density": density_value,
        }
        for field, width, dtype in (
            ("gt_boxes", 9, np.float32),
            ("gt_labels", None, np.int16),
            ("duplicate_edge_query_id", None, np.int16),
            ("duplicate_edge_gt_index", None, np.int16),
        ):
            joined, offsets = variable_arrays(rows, field, width, dtype)
            arrays[field] = joined
            arrays[field + "_offsets"] = offsets
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        temporary.replace(path)
        self.parts.append(
            {
                "path": display(path),
                "sample_count": len(rows),
                "query_record_count": len(rows) * QUERY_COUNT,
                "sha256": sha256(path),
                "compressed_sha256": sha256(path),
                "uncompressed_sha256": arrays_sha256(arrays),
                "size_bytes": path.stat().st_size,
            }
        )
        self.rows = []

    def close(self):
        rows = sorted(
            self.rows,
            key=lambda row: (
                row["scene_token"],
                row["sample_token"],
            ),
        )
        self.rows = []
        for begin in range(0, len(rows), self.part_size):
            self.rows = rows[begin : begin + self.part_size]
            self.flush()
        return self.parts


def load_density(device: torch.device) -> ClassConditionalRealNVPDensity:
    if sha256(DENSITY_CHECKPOINT) != EXPECTED_DENSITY_SHA256:
        raise ValueError("query-density checkpoint hash mismatch")
    payload = torch.load(
        DENSITY_CHECKPOINT, map_location=device, weights_only=True
    )
    if int(payload["feature_dim"]) != FEATURE_DIM:
        raise ValueError("query-density feature dimension mismatch")
    model = ClassConditionalRealNVPDensity(
        CLASS_COUNT,
        FEATURE_DIM,
        hidden_dim=int(payload["hidden_dim"]),
        coupling_blocks=int(payload["coupling_blocks"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def arrays_sha256(arrays) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def duplicate_edges(query_labels, query_boxes, gt_labels, gt_boxes):
    """Canonical query-to-GT duplicate-neighborhood memberships.

    F1 freezes these as typed (query_id, gt_index) edges, not unordered
    query-to-query pairs. The 2.0 m boundary is inclusive.
    """
    if not len(gt_boxes):
        return np.empty(0, np.int16), np.empty(0, np.int16)
    delta = (
        np.asarray(query_boxes[:, None, :2], dtype=np.float64)
        - np.asarray(gt_boxes[None, :, :2], dtype=np.float64)
    )
    squared_distance = np.sum(delta * delta, axis=-1)
    mask = (
        query_labels[:, None] == gt_labels[None, :]
    ) & (squared_distance <= 4.0)
    query_id, gt_index = np.nonzero(mask)
    order = np.lexsort((gt_index, query_id))
    return query_id[order].astype(np.int16), gt_index[order].astype(np.int16)


def select_dataset_a0(
    cfg,
    info_path: Path,
    shard_index: int,
    shard_count: int,
    traversal: str,
    workers: int,
    sample_scene,
):
    dataset_cfg, samples_per_gpu = stage3.dataset_cfg_for(cfg, str(info_path))
    for transform in dataset_cfg.pipeline:
        if transform["type"] == "LoadPointsFromMultiSweeps":
            transform["test_mode"] = True
    dataset = build_dataset(dataset_cfg)
    full_count = len(dataset.data_infos)
    canonical = sorted(
        range(full_count),
        key=lambda index: (
            sample_scene[dataset.data_infos[index]["token"]],
            dataset.data_infos[index]["token"],
        ),
    )
    selected = np.asarray(
        np.array_split(np.asarray(canonical, dtype=np.int64), shard_count)[
            shard_index
        ],
        dtype=np.int64,
    )
    if traversal == "reverse":
        selected = selected[::-1]
    elif traversal == "rotate" and len(selected):
        selected = np.roll(selected, len(selected) // 3)
    dataset.data_infos = [dataset.data_infos[int(index)] for index in selected]
    if hasattr(dataset, "flag"):
        dataset.flag = np.asarray(dataset.flag)[selected]
    loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers,
        dist=False,
        shuffle=False,
        seed=SEED,
    )
    return dataset, loader, full_count


def export(args) -> None:
    if args.split not in SPLITS:
        raise ValueError("F1 authority is limited to clean train/label_cal/dev")
    if args.shard_count != SHARD_COUNTS[args.split]:
        raise ValueError(
            f"{args.split} shard_count is frozen to {SHARD_COUNTS[args.split]}"
        )
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")
    if sha256(CONFIG) != EXPECTED_CONFIG_SHA256:
        raise ValueError("frozen detector config hash mismatch")
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen detector checkpoint hash mismatch")

    incomplete = RUN_ROOTS[args.run] / "INCOMPLETE_EXPORT.json"
    incomplete.parent.mkdir(parents=True, exist_ok=True)
    try:
        with incomplete.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "status": "INCOMPLETE_EXPORT",
                        "run": args.run,
                        "schema_version": "probset3d_f1_a0_incomplete_export_v1",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
    except FileExistsError:
        pass

    authority = strict_load(P0)
    split = authority["splits"][args.split]
    if split["sample_count"] != SAMPLE_COUNTS[args.split]:
        raise ValueError("P0 split sample count changed")
    info_path = ROOT / split["info_path"]
    token_path = ROOT / split["sample_tokens_path"]
    scene_path = ROOT / split["scene_tokens_path"]
    if (
        sha256(info_path) != split["info_sha256"]
        or sha256(token_path) != split["sample_tokens_sha256"]
        or sha256(scene_path) != split["scene_tokens_sha256"]
    ):
        raise ValueError("P0 split identity mismatch")
    allowed_tokens = strict_load(token_path)["sample_tokens"]

    output_root = (RUN_ROOTS[args.run] / "clean_query").resolve()
    run_root = (
        output_root
        / args.split
        / f"shard-{args.shard_index:02d}-of-{args.shard_count:02d}"
    )
    accepted_path = run_root / "accepted.json"
    if accepted_path.exists() or (run_root.exists() and any(run_root.iterdir())):
        raise FileExistsError(f"refusing to overwrite authority shard {run_root}")

    with info_path.open("rb") as stream:
        payload = pickle.load(stream)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    if len(infos) != SAMPLE_COUNTS[args.split]:
        raise ValueError("split info count mismatch")
    info_by_token = {item["token"]: item for item in infos}
    if set(info_by_token) != set(allowed_tokens):
        raise ValueError("split info/token authority mismatch")
    sample_scene = load_sample_scene_map()
    if not set(allowed_tokens) <= set(sample_scene):
        raise ValueError("split sample absent from nuScenes sample.json")
    scenes = {token: sample_scene[token] for token in allowed_tokens}
    allowed_scenes = set(strict_load(scene_path)["scene_tokens"])
    if set(scenes.values()) != allowed_scenes:
        raise ValueError("sample.json scene authority does not match P0")

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    cfg = stage3.load_cfg(str(CONFIG))
    dataset, loader, full_count = select_dataset_a0(
        cfg,
        info_path,
        args.shard_index,
        args.shard_count,
        args.traversal,
        args.workers,
        sample_scene,
    )
    expected_tokens = [item["token"] for item in dataset.data_infos]
    model = stage3.build_frozen_model(
        cfg, str(CHECKPOINT), args.device
    )
    module = model.module if hasattr(model, "module") else model
    detector_trainable = [
        name for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    if detector_trainable:
        raise ValueError("frozen BEVFusion unexpectedly exposes trainable parameters")
    detector_state_before = state_sha256(module)
    checkpoint_before = sha256(CHECKPOINT)
    density = load_density(device)
    head = module.heads["object"]
    captured = []

    def capture_dense(_module, _inputs, output):
        if captured:
            raise RuntimeError("dense hook called twice before consumption")
        captured.append(output.detach())

    handle = head.heatmap_head.register_forward_hook(capture_dense)
    writer = PartWriter(
        run_root / "parts", args.split, args.part_size, density, device
    )
    seen = []
    topk_identity_failures = 0
    query_identity_failures = 0
    assignment_failures = 0
    finite_failures = 0
    duplicate_edge_count = 0
    matched_query_count = 0
    started = time.time()
    try:
        for sample_index, data in enumerate(loader):
            if args.max_samples and sample_index >= args.max_samples:
                break
            with torch.inference_mode():
                output = model(
                    return_loss=False,
                    rescale=True,
                    return_stage3_target_uq=True,
                    stage3_branch="fusion",
                    stage3_num_proposals=QUERY_COUNT,
                    **data,
                )[0]
            if len(captured) != 1:
                raise RuntimeError("dense heatmap hook did not return one tensor")
            dense = captured.pop()
            item = output["stage3_target_uq"]
            token = item["sample_token"]
            if token != expected_tokens[sample_index] or token in seen:
                raise ValueError("sample token order/uniqueness failure")
            info = info_by_token[token]
            _, gt_boxes, gt_labels = detection_gt(info)
            query_boxes = item["decoded_boxes"].float().numpy()
            query_logits = item["class_logits"].float().numpy()
            query_features = item["features"].float().numpy()
            query_id = item["proposal_ids"].numpy().astype(np.int16)
            components = cost_components(
                "fusion",
                model,
                item["class_logits"].float(),
                item["decoded_boxes"].float(),
                gt_boxes,
                gt_labels,
                device,
            )
            matched, _, _, _ = deterministic_assignment_from_cost(
                components["total"],
                query_boxes[:, :2],
                gt_boxes[:, :2],
                query_ids=query_id,
                gt_ids=np.arange(len(gt_boxes), dtype=np.int64),
            )
            _, local_mask, seed_index, seed_class, seed_score = exact_local_max(
                head, dense
            )
            seed_index = seed_index[0].cpu().numpy().astype(np.int32)
            seed_class = seed_class[0].cpu().numpy().astype(np.int16)
            seed_score = seed_score[0].cpu().float().numpy()
            exported_index = item["query_indices"].numpy().astype(np.int32)
            exported_class = item["query_labels"].numpy().astype(np.int16)
            topk_ok = np.array_equal(seed_index, exported_index) and np.array_equal(
                seed_class, exported_class
            )
            query_id_ok = np.array_equal(
                query_id, np.arange(QUERY_COUNT, dtype=np.int16)
            )
            positive_gt = matched[matched >= 0]
            assignment_ok = (
                len(positive_gt) == len(set(positive_gt.tolist()))
                and np.all(matched < len(gt_boxes))
            )
            finite_ok = all(
                np.isfinite(value).all()
                for value in (
                    query_boxes,
                    query_logits,
                    query_features,
                    seed_score,
                    gt_boxes,
                )
            ) and bool(torch.isfinite(dense).all())
            topk_identity_failures += int(not topk_ok)
            query_identity_failures += int(not query_id_ok)
            assignment_failures += int(not assignment_ok)
            finite_failures += int(not finite_ok)
            if not (topk_ok and query_id_ok and assignment_ok and finite_ok):
                raise ValueError(f"query authority invariant failure for {token}")

            matched_class = np.full(QUERY_COUNT, -1, dtype=np.int16)
            matched_box = np.zeros((QUERY_COUNT, 9), dtype=np.float32)
            positive = matched >= 0
            if positive.any():
                matched_class[positive] = gt_labels[matched[positive]].astype(
                    np.int16
                )
                matched_box[positive] = gt_boxes[matched[positive]]
            edge_query, edge_gt = duplicate_edges(
                seed_class, query_boxes, gt_labels, gt_boxes
            )
            grid_width = int(dense.shape[-1])
            seed_position = np.column_stack(
                (seed_index % grid_width, seed_index // grid_width)
            ).astype(np.int16)
            writer.append(
                {
                    "sample_token": token,
                    "scene_token": scenes[token],
                    "query_id": query_id,
                    "seed_flat_index": seed_index,
                    "seed_position_xy": seed_position,
                    "seed_class": seed_class,
                    "seed_score": seed_score,
                    "query_features": query_features,
                    "class_logits": query_logits,
                    "decoded_box_mean": query_boxes,
                    "native_matched_gt": matched.astype(np.int16),
                    "matched_gt_class": matched_class,
                    "matched_gt_box": matched_box,
                    "gt_boxes": gt_boxes,
                    "gt_labels": gt_labels,
                    "duplicate_edge_query_id": edge_query,
                    "duplicate_edge_gt_index": edge_gt,
                }
            )
            seen.append(token)
            matched_query_count += int(positive.sum())
            duplicate_edge_count += len(edge_query)
            if sample_index == 0 or (sample_index + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "split": args.split,
                            "shard": args.shard_index,
                            "collected": sample_index + 1,
                            "expected": len(dataset),
                            "elapsed_seconds": time.time() - started,
                        }
                    ),
                    flush=True,
                )
    finally:
        handle.remove()
    parts = writer.close()
    expected_count = (
        len(dataset)
        if args.max_samples is None
        else min(len(dataset), args.max_samples)
    )
    if len(seen) != expected_count or len(seen) != len(set(seen)):
        raise ValueError("authority shard sample conservation failure")
    if args.max_samples is None and seen != expected_tokens:
        raise ValueError("authority shard token coverage/order failure")
    if any(parameter.grad is not None for parameter in module.parameters()):
        raise ValueError("frozen BEVFusion accumulated a gradient")
    detector_state_after = state_sha256(module)
    checkpoint_after = sha256(CHECKPOINT)
    if (
        detector_state_before != detector_state_after
        or checkpoint_before != checkpoint_after
    ):
        raise ValueError("frozen BEVFusion identity changed during export")

    manifest = {
        "accepted": True,
        "schema_version": "probset3d_f1_clean_query_raw_export_a0_v1",
        "run": args.run,
        "split": args.split,
        "condition": "clean",
        "sample_count": len(seen),
        "query_record_count": len(seen) * QUERY_COUNT,
        "full_split_sample_count": full_count,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "max_samples": args.max_samples,
        "sample_tokens_sha256": hashlib.sha256(
            "\n".join(seen).encode("ascii")
        ).hexdigest(),
        "canonical_sample_tokens_sha256": hashlib.sha256(
            "\n".join(sorted(seen, key=lambda token: (scenes[token], token))).encode(
                "ascii"
            )
        ).hexdigest(),
        "config": display(CONFIG),
        "config_sha256": sha256(CONFIG),
        "checkpoint": display(CHECKPOINT),
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "detector_state_sha256_before": detector_state_before,
        "detector_state_sha256_after": detector_state_after,
        "density_checkpoint": display(DENSITY_CHECKPOINT),
        "density_checkpoint_sha256": sha256(DENSITY_CHECKPOINT),
        "density_checkpoint_deserialized_for_isolated_column": True,
        "query_density_primary_read_authorized": False,
        "split_info": split["info_path"],
        "split_info_sha256": sha256(info_path),
        "split_tokens": split["sample_tokens_path"],
        "split_tokens_sha256": sha256(token_path),
        "split_scenes": split["scene_tokens_path"],
        "split_scenes_sha256": sha256(scene_path),
        "scene_authority": display(NUSCENES_SAMPLES),
        "scene_authority_sha256": sha256(NUSCENES_SAMPLES),
        "scene_authority_semantics": "sample.json[sample_token].scene_token",
        "query_count": QUERY_COUNT,
        "feature_dimension": FEATURE_DIM,
        "class_count": CLASS_COUNT,
        "grid_shape": list(dense.shape[-3:]),
        "part_count": len(parts),
        "parts": parts,
        "matched_query_count": matched_query_count,
        "unmatched_query_count": len(seen) * QUERY_COUNT - matched_query_count,
        "duplicate_edge_count": duplicate_edge_count,
        "checks": {
            "topk_identity_failures": topk_identity_failures,
            "query_identity_failures": query_identity_failures,
            "assignment_failures": assignment_failures,
            "finite_failures": finite_failures,
            "detector_trainable_parameter_count": len(detector_trainable),
            "detector_parameters_with_gradients": 0,
            "degradation_data_read": False,
            "official_val_read": False,
            "internal_test_read": False,
            "scientific_metric_read": False,
        },
        "resources": {
            "device_index": args.device,
            "gpu_name": torch.cuda.get_device_name(device),
            "peak_cuda_allocation_bytes": torch.cuda.max_memory_allocated(device),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "wall_time_seconds": time.time() - started,
        },
        "seed": SEED,
        "worker_count": args.workers,
        "input_traversal": args.traversal,
        "sample_sort_key_after_canonicalization": [
            "split",
            "scene_token",
            "sample_token",
            "query_id",
        ],
        "assignment_tie_break": ["original_cost", "gt_index", "query_id"],
        "assignment_wrapper": display(ASSIGNMENT_SOURCE),
        "assignment_wrapper_sha256": sha256(ASSIGNMENT_SOURCE),
        "duplicate_membership_rule": {
            "edge_semantics": ["query_id", "gt_index"],
            "class_equal": True,
            "distance_xy_m": "<=2.0",
            "sort_key": ["sample_token", "query_id", "gt_index"],
        },
        "dependencies": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "exporter": display(Path(__file__)),
        "exporter_sha256": sha256(Path(__file__)),
    }
    strict_write(accepted_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def resolve_part(relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else ROOT / path


def validate_part(path: Path, expected_split: str, sample_scene):
    with np.load(path, allow_pickle=False) as part:
        arrays = {name: part[name] for name in part.files}
    sample_count = len(arrays["sample_tokens"])
    expected_query_id = np.arange(QUERY_COUNT, dtype=np.int16)
    if arrays["query_id"].shape != (sample_count, QUERY_COUNT):
        raise ValueError(f"query_id shape failure: {path}")
    if not np.all(arrays["query_id"] == expected_query_id[None, :]):
        raise ValueError(f"query ID missing/duplicate/reordered: {path}")
    required_shapes = {
        "seed_flat_index": (sample_count, QUERY_COUNT),
        "seed_position_xy": (sample_count, QUERY_COUNT, 2),
        "seed_class": (sample_count, QUERY_COUNT),
        "seed_score": (sample_count, QUERY_COUNT),
        "query_features": (sample_count, QUERY_COUNT, FEATURE_DIM),
        "class_logits": (sample_count, QUERY_COUNT, CLASS_COUNT),
        "decoded_box_mean": (sample_count, QUERY_COUNT, 9),
        "native_matched_gt": (sample_count, QUERY_COUNT),
        "matched_gt_class": (sample_count, QUERY_COUNT),
        "matched_gt_box": (sample_count, QUERY_COUNT, 9),
        "query_density": (sample_count, QUERY_COUNT),
    }
    for name, shape in required_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} shape failure in {path}")
    if not np.all(arrays["split_ids"] == expected_split):
        raise ValueError(f"split column mismatch: {path}")
    expected_scenes = np.asarray(
        [sample_scene[str(token)] for token in arrays["sample_tokens"]],
        dtype="U32",
    )
    if not np.array_equal(arrays["scene_tokens"], expected_scenes):
        raise ValueError(f"sample.json scene inheritance failure: {path}")
    for name in (
        "seed_score",
        "query_features",
        "class_logits",
        "decoded_box_mean",
        "matched_gt_box",
        "query_density",
    ):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"non-finite {name}: {path}")
    if not (
        (arrays["query_density"] >= 0).all()
        and (arrays["query_density"] <= 1).all()
    ):
        raise ValueError(f"query density range failure: {path}")
    if not (
        (arrays["seed_class"] >= 0).all()
        and (arrays["seed_class"] < CLASS_COUNT).all()
    ):
        raise ValueError(f"seed class range failure: {path}")

    matched_total = 0
    duplicate_total = 0
    for row in range(sample_count):
        def variable(name):
            offsets = arrays[name + "_offsets"]
            return arrays[name][offsets[row] : offsets[row + 1]]

        gt_boxes = variable("gt_boxes")
        gt_labels = variable("gt_labels")
        edge_query = variable("duplicate_edge_query_id")
        edge_gt = variable("duplicate_edge_gt_index")
        matched = arrays["native_matched_gt"][row].astype(np.int64)
        positive = matched >= 0
        positive_gt = matched[positive]
        if (
            len(positive_gt) != len(set(positive_gt.tolist()))
            or np.any(positive_gt >= len(gt_boxes))
        ):
            raise ValueError(f"assignment one-to-one failure: {path}/{row}")
        if int(positive.sum()) + int((~positive).sum()) != QUERY_COUNT:
            raise ValueError(f"matched/unmatched conservation failure: {path}/{row}")
        expected_class = np.full(QUERY_COUNT, -1, dtype=np.int16)
        expected_box = np.zeros((QUERY_COUNT, 9), dtype=np.float32)
        if positive.any():
            expected_class[positive] = gt_labels[positive_gt]
            expected_box[positive] = gt_boxes[positive_gt]
        if not np.array_equal(
            arrays["matched_gt_class"][row], expected_class
        ) or not np.array_equal(arrays["matched_gt_box"][row], expected_box):
            raise ValueError(f"matched GT materialization failure: {path}/{row}")
        expected_query, expected_gt = duplicate_edges(
            arrays["seed_class"][row],
            arrays["decoded_box_mean"][row],
            gt_labels,
            gt_boxes,
        )
        if not np.array_equal(edge_query, expected_query) or not np.array_equal(
            edge_gt, expected_gt
        ):
            raise ValueError(f"duplicate membership rule failure: {path}/{row}")
        matched_total += int(positive.sum())
        duplicate_total += len(edge_query)
    return {
        "sample_tokens": arrays["sample_tokens"].tolist(),
        "scene_tokens": arrays["scene_tokens"].tolist(),
        "sample_count": sample_count,
        "query_record_count": sample_count * QUERY_COUNT,
        "matched_query_count": matched_total,
        "unmatched_query_count": sample_count * QUERY_COUNT - matched_total,
        "duplicate_edge_count": duplicate_total,
    }


def validate(args) -> None:
    run_root = RUN_ROOTS[args.run]
    output_root = run_root / "clean_query"
    global_manifest_path = run_root / "query_authority_manifest.json"
    validation_path = run_root / "query_authority_validation.json"
    incomplete_path = run_root / "INCOMPLETE_EXPORT.json"
    if not incomplete_path.is_file():
        raise ValueError("missing INCOMPLETE_EXPORT marker")
    if global_manifest_path.exists() or validation_path.exists():
        raise FileExistsError("refusing to overwrite frozen authority validation")
    p0 = strict_load(P0)
    sample_scene = load_sample_scene_map()
    split_results = {}
    all_scenes = {}
    authority_digest = hashlib.sha256()
    for split_name in SPLITS:
        split = p0["splits"][split_name]
        for path_key, hash_key in (
            ("sample_tokens_path", "sample_tokens_sha256"),
            ("scene_tokens_path", "scene_tokens_sha256"),
        ):
            if sha256(ROOT / split[path_key]) != split[hash_key]:
                raise ValueError(f"P0 split identity mismatch: {split_name}")
        allowed_tokens = strict_load(
            ROOT / split["sample_tokens_path"]
        )["sample_tokens"]
        allowed_scenes = strict_load(
            ROOT / split["scene_tokens_path"]
        )["scene_tokens"]
        manifests = sorted(
            (output_root / split_name).glob("shard-*/accepted.json")
        )
        if len(manifests) != SHARD_COUNTS[split_name]:
            raise ValueError(f"incomplete shard count for {split_name}")
        seen_tokens = []
        seen_scenes = []
        content_files = []
        matched = unmatched = duplicate = 0
        for shard_manifest_path in manifests:
            manifest = strict_load(shard_manifest_path)
            if (
                not manifest["accepted"]
                or manifest["run"] != args.run
                or manifest["split"] != split_name
                or manifest["condition"] != "clean"
                or manifest["max_samples"] is not None
            ):
                raise ValueError(f"unaccepted/non-complete shard {manifest_path}")
            if manifest["exporter_sha256"] != sha256(Path(__file__)):
                raise ValueError("exporter source changed across shards")
            if manifest["assignment_wrapper_sha256"] != sha256(
                ASSIGNMENT_SOURCE
            ):
                raise ValueError("assignment wrapper changed across shards")
            for check, value in manifest["checks"].items():
                if check.endswith("_failures") and value != 0:
                    raise ValueError(f"shard check failed: {check}")
            for part in manifest["parts"]:
                part_path = resolve_part(part["path"])
                actual_hash = sha256(part_path)
                if actual_hash != part["sha256"]:
                    raise ValueError(f"content SHA mismatch: {part_path}")
                with np.load(part_path, allow_pickle=False) as archive:
                    actual_uncompressed = arrays_sha256(
                        {name: archive[name] for name in archive.files}
                    )
                if actual_uncompressed != part["uncompressed_sha256"]:
                    raise ValueError(
                        f"uncompressed content SHA mismatch: {part_path}"
                    )
                validated = validate_part(part_path, split_name, sample_scene)
                if validated["sample_count"] != part["sample_count"]:
                    raise ValueError(f"part manifest count mismatch: {part_path}")
                seen_tokens.extend(validated["sample_tokens"])
                seen_scenes.extend(validated["scene_tokens"])
                matched += validated["matched_query_count"]
                unmatched += validated["unmatched_query_count"]
                duplicate += validated["duplicate_edge_count"]
                content_files.append(
                    {
                        "path": display(part_path),
                        "sha256": actual_hash,
                        "compressed_sha256": actual_hash,
                        "uncompressed_sha256": actual_uncompressed,
                        "size_bytes": part_path.stat().st_size,
                    }
                )
                authority_digest.update(split_name.encode("ascii"))
                authority_digest.update(actual_uncompressed.encode("ascii"))
        if (
            len(seen_tokens) != SAMPLE_COUNTS[split_name]
            or len(seen_tokens) != len(set(seen_tokens))
            or set(seen_tokens) != set(allowed_tokens)
        ):
            raise ValueError(f"sample conservation/authority failure: {split_name}")
        if set(seen_scenes) != set(allowed_scenes):
            raise ValueError(f"scene authority failure: {split_name}")
        if any(
            scene != sample_scene[token]
            for token, scene in zip(seen_tokens, seen_scenes)
        ):
            raise ValueError(f"query scene inheritance failure: {split_name}")
        if list(zip(seen_scenes, seen_tokens)) != sorted(
            zip(seen_scenes, seen_tokens)
        ):
            raise ValueError(f"canonical sample sort failure: {split_name}")
        query_records = len(seen_tokens) * QUERY_COUNT
        if matched + unmatched != query_records:
            raise ValueError(f"query conservation failure: {split_name}")
        split_results[split_name] = {
            "sample_count": len(seen_tokens),
            "query_record_count": query_records,
            "matched_query_count": matched,
            "unmatched_query_count": unmatched,
            "duplicate_edge_count": duplicate,
            "scene_count": len(set(seen_scenes)),
            "sample_token_set_sha256": hashlib.sha256(
                "\n".join(sorted(seen_tokens)).encode("ascii")
            ).hexdigest(),
            "content_file_count": len(content_files),
            "content_files": content_files,
        }
        all_scenes[split_name] = set(seen_scenes)
    overlaps = {
        "train_label_cal": len(all_scenes["train"] & all_scenes["label_cal"]),
        "train_dev": len(all_scenes["train"] & all_scenes["dev"]),
        "label_cal_dev": len(all_scenes["label_cal"] & all_scenes["dev"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"scene split overlap: {overlaps}")
    validation = {
        "accepted": True,
        "schema_version": "probset3d_f1_query_authority_validation_a0_v1",
        "run": args.run,
        "splits": {
            name: {
                key: value
                for key, value in result.items()
                if key != "content_files"
            }
            for name, result in split_results.items()
        },
        "scene_overlaps": overlaps,
        "mechanical_checks": {
            "every_sample_exactly_200_queries": True,
            "query_id_exact_0_to_199": True,
            "no_query_loss_duplicate_or_reorder": True,
            "assignment_one_to_one": True,
            "matched_unmatched_conservation": True,
            "duplicate_membership_exact_frozen_rule": True,
            "split_scenes_disjoint": True,
            "scene_tokens_equal_p0": True,
            "query_scene_inherits_sample_json": True,
            "all_content_sha256_verified": True,
            "query_density_isolated_column_finite": True,
            "scientific_metric_read": False,
            "canonical_sort_verified": True,
            "compressed_and_uncompressed_sha256_verified": True,
        },
    }
    manifest = {
        "accepted": True,
        "schema_version": "probset3d_f1_query_authority_manifest_a0_v1",
        "run": args.run,
        "condition": "clean",
        "authority_digest": authority_digest.hexdigest(),
        "schema": display(SCHEMA),
        "schema_sha256": sha256(SCHEMA),
        "p0_authority": display(P0),
        "p0_authority_sha256": sha256(P0),
        "scene_authority": display(NUSCENES_SAMPLES),
        "scene_authority_sha256": sha256(NUSCENES_SAMPLES),
        "scene_authority_semantics": "sample.json[sample_token].scene_token",
        "config": display(CONFIG),
        "config_sha256": sha256(CONFIG),
        "checkpoint": display(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "density_checkpoint": display(DENSITY_CHECKPOINT),
        "density_checkpoint_sha256": sha256(DENSITY_CHECKPOINT),
        "query_density_primary_M1_M2_M3_read": False,
        "input_query_files": [],
        "input_query_source": (
            "frozen BEVFusion forward from P0 split info; no pre-existing query "
            "file was used"
        ),
        "canonical_sort_key": [
            "split",
            "scene_token",
            "sample_token",
            "query_id",
        ],
        "assignment_tie_break": ["original_cost", "gt_index", "query_id"],
        "assignment_wrapper": display(ASSIGNMENT_SOURCE),
        "assignment_wrapper_sha256": sha256(ASSIGNMENT_SOURCE),
        "duplicate_membership_rule": {
            "edge_semantics": ["query_id", "gt_index"],
            "class_equal": True,
            "distance_xy_m": "<=2.0",
            "sort_key": ["sample_token", "query_id", "gt_index"],
        },
        "seed": SEED,
        "splits": split_results,
        "scene_overlaps": overlaps,
        "scientific_metrics_generated": False,
        "degradation_data_read": False,
        "official_val_read": False,
        "internal_test_read": False,
        "exporter": display(Path(__file__)),
        "exporter_sha256": sha256(Path(__file__)),
    }
    shard_manifests = [
        strict_load(path)
        for split_name in SPLITS
        for path in sorted(
            (output_root / split_name).glob("shard-*/accepted.json")
        )
    ]
    manifest["worker_counts"] = sorted(
        {value["worker_count"] for value in shard_manifests}
    )
    manifest["input_traversals"] = sorted(
        {value["input_traversal"] for value in shard_manifests}
    )
    manifest["dependencies"] = shard_manifests[0]["dependencies"]

    validation_bytes = (
        json.dumps(validation, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    manifest["validation"] = display(validation_path)
    manifest["validation_sha256"] = hashlib.sha256(validation_bytes).hexdigest()
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    validation_partial = validation_path.with_suffix(".json.partial")
    manifest_partial = global_manifest_path.with_suffix(".json.partial")
    if validation_partial.exists() or manifest_partial.exists():
        raise FileExistsError("stale authority finalization partial")
    validation_partial.write_bytes(validation_bytes)
    manifest_partial.write_bytes(manifest_bytes)
    validation_partial.replace(validation_path)
    manifest_partial.replace(global_manifest_path)
    if sha256(validation_path) != manifest["validation_sha256"]:
        raise ValueError("final validation SHA mismatch")
    incomplete_path.unlink()
    print(json.dumps(validation, indent=2, sort_keys=True))


FIXED_SAMPLE_FIELDS = (
    "split_ids",
    "sample_tokens",
    "query_id",
    "seed_flat_index",
    "seed_position_xy",
    "seed_class",
    "seed_score",
    "query_features",
    "class_logits",
    "decoded_box_mean",
    "native_matched_gt",
    "matched_gt_class",
    "matched_gt_box",
    "query_density",
)
VARIABLE_SAMPLE_FIELDS = (
    "gt_boxes",
    "gt_labels",
    "duplicate_edge_query_id",
    "duplicate_edge_gt_index",
)


def update_array_digest(digest, name: str, value) -> None:
    value = np.ascontiguousarray(value)
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())


def canonical_sample_digests(root: Path, split: str, include_scene: bool):
    results = {}
    manifests = sorted((root / split).glob("shard-*/accepted.json"))
    for manifest_path in manifests:
        manifest = strict_load(manifest_path)
        for part in manifest["parts"]:
            path = resolve_part(part["path"])
            with np.load(path, allow_pickle=False) as arrays:
                for row, raw_token in enumerate(arrays["sample_tokens"]):
                    token = str(raw_token)
                    if token in results:
                        raise ValueError(f"duplicate canonical sample: {token}")
                    digest = hashlib.sha256()
                    for name in FIXED_SAMPLE_FIELDS:
                        update_array_digest(digest, name, arrays[name][row])
                    if include_scene:
                        update_array_digest(
                            digest, "scene_tokens", arrays["scene_tokens"][row]
                        )
                    for name in VARIABLE_SAMPLE_FIELDS:
                        offsets = arrays[name + "_offsets"]
                        update_array_digest(
                            digest,
                            name,
                            arrays[name][offsets[row] : offsets[row + 1]],
                        )
                    results[token] = digest.hexdigest()
    aggregate = hashlib.sha256()
    for token in sorted(results):
        aggregate.update(token.encode("ascii"))
        aggregate.update(results[token].encode("ascii"))
    return aggregate.hexdigest(), results


def scientific_manifest_projection(root: Path):
    projected = []
    for split in SPLITS:
        for path in sorted((root / split).glob("shard-*/accepted.json")):
            value = strict_load(path)
            value.pop("run", None)
            value.pop("resources", None)
            value["parts"] = [
                {
                    "sample_count": part["sample_count"],
                    "query_record_count": part["query_record_count"],
                }
                for part in value["parts"]
            ]
            projected.append(value)
    encoded = json.dumps(
        projected, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_runs() -> None:
    if R1_ACCEPTANCE.exists() or R1_REPORT.exists():
        raise FileExistsError("refusing to overwrite frozen R1 acceptance")
    for run, root in RUN_ROOTS.items():
        validation = strict_load(root / "query_authority_validation.json")
        manifest = strict_load(root / "query_authority_manifest.json")
        if not validation["accepted"] or not manifest["accepted"]:
            raise ValueError(f"R1 authority run is not accepted: {run}")

    split_results = {}
    all_passed = True
    for split in SPLITS:
        old_non_scene, old_samples = canonical_sample_digests(
            OLD_AUTHORITY_OUT, split, include_scene=False
        )
        a_non_scene, a_samples = canonical_sample_digests(
            RUN_ROOTS["run_a"] / "clean_query", split, include_scene=False
        )
        b_non_scene, b_samples = canonical_sample_digests(
            RUN_ROOTS["run_b"] / "clean_query", split, include_scene=False
        )
        a_full, a_full_samples = canonical_sample_digests(
            RUN_ROOTS["run_a"] / "clean_query", split, include_scene=True
        )
        b_full, b_full_samples = canonical_sample_digests(
            RUN_ROOTS["run_b"] / "clean_query", split, include_scene=True
        )
        non_scene_equal_old = old_samples == a_samples == b_samples
        double_run_equal = a_full_samples == b_full_samples
        passed = non_scene_equal_old and double_run_equal
        all_passed &= passed
        split_results[split] = {
            "accepted": passed,
            "sample_count": len(a_samples),
            "old_invalid_non_scene_sha256": old_non_scene,
            "run_a_non_scene_sha256": a_non_scene,
            "run_b_non_scene_sha256": b_non_scene,
            "non_scene_fields_byte_identical_to_old_invalid": non_scene_equal_old,
            "run_a_uncompressed_content_sha256": a_full,
            "run_b_uncompressed_content_sha256": b_full,
            "double_run_query_arrays_identical": double_run_equal,
        }

    projection_a = scientific_manifest_projection(
        RUN_ROOTS["run_a"] / "clean_query"
    )
    projection_b = scientific_manifest_projection(
        RUN_ROOTS["run_b"] / "clean_query"
    )
    scientific_json_equal = projection_a == projection_b
    all_passed &= scientific_json_equal
    acceptance = {
        "accepted": all_passed,
        "schema_version": "probset3d_f1_authority_r1_acceptance_v1",
        "endpoint": (
            "R1 AUTHORITY ACCEPTED — GO TO FULL LEGALITY PILOT"
            if all_passed
            else "INVALID — STOP F1-R1 PILOT"
        ),
        "scene_authority": display(NUSCENES_SAMPLES),
        "scene_authority_sha256": sha256(NUSCENES_SAMPLES),
        "scene_authority_semantics": "sample.json[sample_token].scene_token",
        "old_invalid_authority_used_as_export_source": False,
        "old_invalid_authority_read_only_comparison": True,
        "scientific_json_projection_scope": (
            "accepted shard manifests excluding run namespace, resources, "
            "container path/hash/size"
        ),
        "run_a_scientific_json_sha256": projection_a,
        "run_b_scientific_json_sha256": projection_b,
        "scientific_json_identical": scientific_json_equal,
        "splits": split_results,
        "scientific_metrics_generated_or_read": False,
    }
    rows = [
        "# ProbSet3D F1-R1 Authority Acceptance",
        "",
        f"Decision: **{acceptance['endpoint']}**",
        "",
        "| Split | Samples | Non-scene fields equal old INVALID | Run A/B uncompressed arrays equal |",
        "| --- | ---: | --- | --- |",
    ]
    for split, value in split_results.items():
        rows.append(
            f"| {split} | {value['sample_count']:,} | "
            f"{'PASS' if value['non_scene_fields_byte_identical_to_old_invalid'] else 'FAIL'} | "
            f"{'PASS' if value['double_run_query_arrays_identical'] else 'FAIL'} |"
        )
    rows.extend(
        [
            "",
            "Both runs were regenerated from the frozen detector and source data.",
            "The old INVALID authority was read only for the required non-scene",
            "byte-identity comparison and was never an export source.",
            "",
        ]
    )
    strict_write(R1_ACCEPTANCE, acceptance)
    R1_REPORT.write_text("\n".join(rows), encoding="utf-8")
    print(json.dumps(acceptance, indent=2, sort_keys=True))
    if not all_passed:
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    exporter = subparsers.add_parser("export")
    exporter.add_argument("--run", required=True, choices=tuple(RUN_ROOTS))
    exporter.add_argument("--split", required=True, choices=SPLITS)
    exporter.add_argument("--device", required=True, type=int)
    exporter.add_argument("--shard-index", required=True, type=int)
    exporter.add_argument("--shard-count", required=True, type=int)
    exporter.add_argument("--part-size", type=int, default=32)
    exporter.add_argument("--max-samples", type=int)
    exporter.add_argument(
        "--traversal", choices=("canonical", "reverse", "rotate"), required=True
    )
    exporter.add_argument("--workers", type=int, required=True)
    validator = subparsers.add_parser("validate")
    validator.add_argument("--run", required=True, choices=tuple(RUN_ROOTS))
    subparsers.add_parser("compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "export":
        export(args)
    elif args.command == "validate":
        validate(args)
    else:
        compare_runs()


if __name__ == "__main__":
    main()
