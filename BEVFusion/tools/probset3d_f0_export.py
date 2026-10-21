#!/usr/bin/env python3
"""Export frozen Fusion query/pre-topK authority for ProbSet3D F0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import resource
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/probset3d-matplotlib")

import numpy as np
import torch
import torch.nn.functional as F

from mmdet3d.apis.stage2_manifest import inject_stage2_corruption
from mmdet3d.datasets import build_dataloader, build_dataset
from tools import stage3_run_real_target_uq as stage3
from tools.stage3d_b_collect_labels import scene_map
from tools.stage3d_b_native_assignment import (
    assignment_from_cost,
    cost_components,
    detection_gt,
)


ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "probset3d/P0_DATA_SPLIT_AUTHORITY.json"
DEFAULT_OUT = ROOT / "outputs/probset3d/f0/exports"
CONFIG = ROOT / "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
CHECKPOINT = ROOT / "pretrained/bevfusion-det.pth"
EXPECTED_CONFIG_SHA256 = "cb7bbbc25e8fdf67c2e914b11661828b684312300deb2afc410793192713e04b"
EXPECTED_CHECKPOINT_SHA256 = "ee0a389213922343508db40dff0ec04767b36d8f512da34b093939c37aaf2122"
SEED = 20260708
PROPOSALS = 200
CONDITIONS = {
    "clean": ("clean", 0),
    "lowlight_s3": ("camera_lowlight", 3),
    "dropout_s3": ("lidar_dropout", 3),
    "beam_s3": ("lidar_beam_reduction", 3),
}


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


def strict_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def condition_id(split: str, condition: str) -> str:
    suffix = {
        "clean": "clean-s0",
        "lowlight_s3": "camera-lowlight-s3",
        "dropout_s3": "lidar-dropout-s3",
        "beam_s3": "lidar-beam-reduction-s3",
    }[condition]
    return f"fusion-{split}-{suffix}"


def exact_local_max(head, dense_logits: torch.Tensor):
    probability = dense_logits.sigmoid()
    padding = head.nms_kernel_size // 2
    local_max = torch.zeros_like(probability)
    inner = F.max_pool2d(
        probability,
        kernel_size=head.nms_kernel_size,
        stride=1,
        padding=0,
    )
    local_max[:, :, padding:-padding, padding:-padding] = inner
    if head.test_cfg["dataset"] == "nuScenes":
        for class_index in (8, 9):
            local_max[:, class_index] = F.max_pool2d(
                probability[:, class_index], kernel_size=1, stride=1, padding=0
            )
    mask = probability == local_max
    gated = probability * mask
    flattened = gated.flatten(1)
    top = flattened.argsort(dim=-1, descending=True)[..., :PROPOSALS]
    spatial_size = probability.shape[-2] * probability.shape[-1]
    return (
        probability,
        mask,
        top % spatial_size,
        top // spatial_size,
        flattened.gather(1, top),
    )


class ShardWriter:
    def __init__(self, root: Path, shard_size: int, include_dense: bool, include_features: bool):
        self.root = root
        self.shard_size = shard_size
        self.include_dense = include_dense
        self.include_features = include_features
        self.rows = []
        self.parts = []

    @staticmethod
    def _concat_variable(rows, field, width=None, dtype=None):
        values = [np.asarray(row[field], dtype=dtype) for row in rows]
        offsets = np.zeros(len(values) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(value) for value in values])
        if values and offsets[-1]:
            joined = np.concatenate(values, axis=0)
        elif width is None:
            joined = np.empty(0, dtype=dtype)
        else:
            joined = np.empty((0, width), dtype=dtype)
        return joined, offsets

    def append(self, row) -> None:
        self.rows.append(row)
        if len(self.rows) == self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        part_index = len(self.parts)
        path = self.root / f"part-{part_index:05d}.npz"
        temporary = path.with_suffix(".npz.partial")
        if path.exists() or temporary.exists():
            raise FileExistsError(f"refusing to overwrite export shard: {path}")
        self.root.mkdir(parents=True, exist_ok=True)
        rows = self.rows
        arrays = {
            "sample_tokens": np.asarray([row["sample_token"] for row in rows], dtype="U32"),
            "scene_tokens": np.asarray([row["scene_token"] for row in rows], dtype="U32"),
            "query_indices": np.stack([row["query_indices"] for row in rows]),
            "query_labels": np.stack([row["query_labels"] for row in rows]),
            "query_logits": np.stack([row["query_logits"] for row in rows]),
            "query_boxes": np.stack([row["query_boxes"] for row in rows]),
            "query_scores": np.stack([row["query_scores"] for row in rows]),
            "query_kept": np.stack([row["query_kept"] for row in rows]),
            "native_matched_gt": np.stack([row["native_matched_gt"] for row in rows]),
            "native_ignore": np.stack([row["native_ignore"] for row in rows]),
        }
        if self.include_features:
            arrays["query_features"] = np.stack([row["query_features"] for row in rows])
        if self.include_dense:
            arrays.update(
                dense_logits=np.stack([row["dense_logits"] for row in rows]),
                local_max_packbits=np.stack([row["local_max_packbits"] for row in rows]),
                seed_probabilities=np.stack([row["seed_probabilities"] for row in rows]),
            )
        for field, width, dtype in (
            ("gt_boxes", 9, np.float32),
            ("gt_labels", None, np.int16),
            ("final_boxes", 9, np.float32),
            ("final_scores", None, np.float32),
            ("final_labels", None, np.int16),
        ):
            joined, offsets = self._concat_variable(rows, field, width, dtype)
            arrays[field] = joined
            arrays[field + "_offsets"] = offsets
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        temporary.replace(path)
        self.parts.append(
            {
                "path": display_path(path),
                "sample_count": len(rows),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
        self.rows = []

    def close(self):
        self.flush()
        return self.parts


def select_dataset(cfg, info_path: Path, shard_index: int, shard_count: int):
    dataset_cfg, samples_per_gpu = stage3.dataset_cfg_for(cfg, str(info_path))
    dataset = build_dataset(dataset_cfg)
    full_count = len(dataset.data_infos)
    selected = np.arange(shard_index, full_count, shard_count, dtype=np.int64)
    dataset.data_infos = [dataset.data_infos[int(index)] for index in selected]
    if hasattr(dataset, "flag"):
        dataset.flag = np.asarray(dataset.flag)[selected]
    loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=4,
        dist=False,
        shuffle=False,
    )
    return dataset, loader, full_count


def run(args) -> None:
    torch.cuda.set_device(args.device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    authority = strict_load(P0)
    split = authority["splits"][args.split]
    info_path = ROOT / split["info_path"]
    token_path = ROOT / split["sample_tokens_path"]
    if sha256(info_path) != split["info_sha256"] or sha256(token_path) != split["sample_tokens_sha256"]:
        raise ValueError("P0 split authority hash mismatch")
    if sha256(CONFIG) != EXPECTED_CONFIG_SHA256 or sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("frozen Fusion config/checkpoint hash mismatch")
    if args.split == "train" and args.condition != "clean":
        raise ValueError("train reference export is clean-only")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")

    output_root = Path(args.output_root).resolve()
    run_id = condition_id(args.split, args.condition)
    run_root = output_root / run_id / f"shard-{args.shard_index:02d}-of-{args.shard_count:02d}"
    accepted_path = run_root / "accepted.json"
    if accepted_path.exists() or run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite export run: {run_root}")

    with info_path.open("rb") as stream:
        value = pickle.load(stream)
    infos = value["infos"] if isinstance(value, dict) else value
    info_by_token = {item["token"]: item for item in infos}
    scenes = scene_map(infos)
    allowed_tokens = set(strict_load(token_path)["sample_tokens"])
    if len(infos) != split["sample_count"] or set(info_by_token) != allowed_tokens:
        raise ValueError("split info/token coverage mismatch")

    cfg = stage3.load_cfg(str(CONFIG))
    corruption_type, severity = CONDITIONS[args.condition]
    if corruption_type != "clean":
        inject_stage2_corruption(
            cfg,
            corruption_type,
            severity,
            SEED,
            None,
            "calibrated_candidate",
            "outputs/stage2/configs/camera_history.json",
        )
    dataset, loader, full_count = select_dataset(
        cfg, info_path, args.shard_index, args.shard_count
    )
    expected_selected_tokens = {item["token"] for item in dataset.data_infos}

    model = stage3.build_frozen_model(cfg, str(CHECKPOINT), args.device)
    module = model.module if hasattr(model, "module") else model
    trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    if trainable:
        raise ValueError(f"frozen detector unexpectedly trainable: {trainable[:5]}")
    head = module.heads["object"]
    captured = []

    def capture_dense(_module, _inputs, output):
        if captured:
            raise RuntimeError("dense heatmap hook called more than once before consumption")
        captured.append(output.detach())

    handle = head.heatmap_head.register_forward_hook(capture_dense)
    include_dense = args.split == "dev"
    include_features = args.split == "dev"
    writer = ShardWriter(run_root / "parts", args.part_size, include_dense, include_features)
    started = time.time()
    checkpoint_sha_before = sha256(CHECKPOINT)
    topk_identity_failures = 0
    dense_finite_failures = 0
    seen_tokens = []
    peak_cuda_before = torch.cuda.max_memory_allocated(args.device)
    try:
        for sample_index, data in enumerate(loader):
            if args.max_samples and sample_index >= args.max_samples:
                break
            with torch.no_grad():
                output = model(
                    return_loss=False,
                    rescale=True,
                    return_stage3_target_uq=True,
                    stage3_branch="fusion",
                    stage3_num_proposals=PROPOSALS,
                    **data,
                )[0]
            if len(captured) != 1:
                raise RuntimeError("dense heatmap hook did not produce exactly one tensor")
            dense = captured.pop()
            item = output["stage3_target_uq"]
            token = item["sample_token"]
            if token not in expected_selected_tokens or token in seen_tokens:
                raise ValueError("export token is outside the selected unique shard")
            info = info_by_token[token]
            _, gt_boxes, gt_labels = detection_gt(info)
            query_boxes = item["decoded_boxes"].float()
            query_logits = item["class_logits"].float()
            components = cost_components(
                "fusion",
                model,
                query_logits,
                query_boxes,
                gt_boxes,
                gt_labels,
                torch.device(f"cuda:{args.device}"),
            )
            matched, _, ignore, _ = assignment_from_cost(
                components["total"], query_boxes[:, :2].numpy(), gt_boxes[:, :2]
            )
            probability, local_mask, seed_indices, seed_labels, seed_probability = exact_local_max(
                head, dense
            )
            exported_indices = item["query_indices"].long().to(seed_indices.device)
            exported_labels = item["query_labels"].long().to(seed_labels.device)
            identity = torch.equal(seed_indices[0], exported_indices) and torch.equal(
                seed_labels[0], exported_labels
            )
            topk_identity_failures += int(not identity)
            dense_finite_failures += int(
                not bool(torch.isfinite(dense).all() and torch.isfinite(probability).all())
            )
            if not identity:
                raise ValueError(f"pre-topK/query seed identity mismatch for {token}")

            final_boxes = output["boxes_3d"].tensor.detach().cpu().float().numpy()
            final_scores = output["scores_3d"].detach().cpu().float().numpy()
            final_labels = output["labels_3d"].detach().cpu().numpy().astype(np.int16)
            row = {
                "sample_token": token,
                "scene_token": scenes[token],
                "query_indices": item["query_indices"].numpy().astype(np.int32),
                "query_labels": item["query_labels"].numpy().astype(np.int16),
                "query_logits": query_logits.numpy(),
                "query_boxes": query_boxes.numpy(),
                "query_scores": item["post_score_for_mapping"].float().numpy(),
                "query_kept": item["kept_post_nms_mask"].numpy(),
                "native_matched_gt": matched.astype(np.int16),
                "native_ignore": ignore,
                "query_features": item["features"].numpy(),
                "dense_logits": dense[0].detach().cpu().numpy(),
                "local_max_packbits": np.packbits(
                    local_mask[0].detach().cpu().numpy().reshape(-1), bitorder="little"
                ),
                "seed_probabilities": seed_probability[0].detach().cpu().float().numpy(),
                "gt_boxes": gt_boxes,
                "gt_labels": gt_labels,
                "final_boxes": final_boxes,
                "final_scores": final_scores,
                "final_labels": final_labels,
            }
            if not include_dense:
                row.pop("dense_logits")
                row.pop("local_max_packbits")
                row.pop("seed_probabilities")
            if not include_features:
                row.pop("query_features")
            writer.append(row)
            seen_tokens.append(token)
            if sample_index == 0 or (sample_index + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "run_id": run_id,
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

    if any(parameter.grad is not None for parameter in module.parameters()):
        raise ValueError("frozen detector accumulated gradients")
    checkpoint_sha_after = sha256(CHECKPOINT)
    if checkpoint_sha_before != checkpoint_sha_after:
        raise ValueError("frozen detector checkpoint changed")
    expected_count = len(dataset) if not args.max_samples else min(len(dataset), args.max_samples)
    if len(seen_tokens) != expected_count or len(seen_tokens) != len(set(seen_tokens)):
        raise ValueError("export sample coverage mismatch")
    if not args.max_samples and set(seen_tokens) != expected_selected_tokens:
        raise ValueError("complete shard token coverage mismatch")

    bbox_coder = head.bbox_coder
    manifest = {
        "accepted": True,
        "schema_version": "probset3d_f0_query_dense_export_v1",
        "run_id": run_id,
        "split": args.split,
        "condition": {
            "name": args.condition,
            "corruption_type": corruption_type,
            "severity": severity,
            "lowlight_candidate": "calibrated_candidate" if corruption_type == "camera_lowlight" else None,
        },
        "sample_count": len(seen_tokens),
        "full_split_sample_count": full_count,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "max_samples": args.max_samples,
        "sample_tokens_sha256": hashlib.sha256(
            "\n".join(seen_tokens).encode("ascii")
        ).hexdigest(),
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "split_info": split["info_path"],
        "split_info_sha256": sha256(info_path),
        "split_token_authority": split["sample_tokens_path"],
        "split_token_authority_sha256": sha256(token_path),
        "query_count": PROPOSALS,
        "query_feature_exported": include_features,
        "pre_topk_dense_exported": include_dense,
        "final_detection_exported": True,
        "historical_hungarian_role": "comparison_only_not_set_likelihood_authority",
        "grid": {
            "dense_shape": list(dense.shape[-3:]),
            "local_max_mask_bitorder": "little",
            "nms_kernel_size": int(head.nms_kernel_size),
            "point_cloud_range": [float(value) for value in bbox_coder.pc_range],
            "voxel_size": [float(value) for value in bbox_coder.voxel_size],
            "out_size_factor": int(bbox_coder.out_size_factor),
        },
        "checks": {
            "all_tokens_authorized": set(seen_tokens).issubset(allowed_tokens),
            "unique_tokens": len(seen_tokens) == len(set(seen_tokens)),
            "topk_identity_failures": topk_identity_failures,
            "dense_finite_failures": dense_finite_failures,
            "detector_trainable_parameter_count": len(trainable),
            "detector_parameters_with_gradients": 0,
            "official_val_used": False,
            "internal_test_used": False,
            "dc0_artifact_used": False,
            "training_performed": False,
        },
        "parts": parts,
        "part_count": len(parts),
        "wall_time_seconds": time.time() - started,
        "peak_cuda_allocation_bytes": max(
            torch.cuda.max_memory_allocated(args.device), peak_cuda_before
        ),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "device_index": args.device,
        "gpu_name": torch.cuda.get_device_name(args.device),
        "seed": SEED,
        "exporter_sha256": sha256(Path(__file__)),
    }
    strict_write(accepted_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("train", "dev"))
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--part-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
