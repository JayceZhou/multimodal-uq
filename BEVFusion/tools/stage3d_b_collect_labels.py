#!/usr/bin/env python3
"""Collect frozen proposals and Stage 3D-B A/B/C labels for one branch."""

import argparse
import gzip
import json
import pickle
from pathlib import Path
import time

import numpy as np
import torch

from mmdet3d.models.target_uq import match_stage3_proposals_to_gt
from tools import stage3_run_real_target_uq as stage3
from tools.stage3d_b_native_assignment import (
    BRANCHES,
    CLASSES,
    assignment_from_cost,
    cost_components,
    detection_gt,
    sha256,
    strict_dump,
)


OUT = Path("outputs/stage3d/stage3d_b")
SPLITS = {
    "pilot": "outputs/stage3/splits/nuscenes_infos_stage3_pilot_base.pkl",
    "label_cal": "outputs/stage3/splits/nuscenes_infos_stage3_label_cal.pkl",
    "dev": "outputs/stage3/splits/nuscenes_infos_stage3_dev.pkl",
}
SCHEMES = ("legacy", "hungarian", "hungarian_ignore")


def scene_map(infos):
    by_token = {item["token"]: item for item in infos}
    result = {}
    for token in by_token:
        root = token
        while by_token[root].get("prev_token") in by_token:
            root = by_token[root]["prev_token"]
        result[token] = root
    return result


def repaired_targets(decoded, gt_boxes, gt_labels, matched, ignore):
    proposals = len(decoded)
    class_target = torch.full((proposals,), len(CLASSES), dtype=torch.long)
    positive = torch.from_numpy(matched >= 0)
    classification_mask = ~torch.from_numpy(ignore)
    matched_tensor = torch.from_numpy(matched.astype(np.int64))
    distance = torch.full((proposals,), -1.0, dtype=torch.float32)
    targets = {
        "center_xy": torch.zeros(proposals, 2),
        "z": torch.zeros(proposals, 1),
        "log_size": torch.zeros(proposals, 3),
        "yaw": torch.zeros(proposals),
        "velocity": torch.zeros(proposals, 2),
    }
    velocity_mask = torch.zeros(proposals, dtype=torch.bool)
    if bool(positive.any()):
        indices = matched_tensor[positive]
        selected = torch.as_tensor(gt_boxes, dtype=torch.float32)[indices]
        class_target[positive] = torch.as_tensor(gt_labels, dtype=torch.long)[indices]
        targets["center_xy"][positive] = selected[:, :2]
        targets["z"][positive] = selected[:, 2:3]
        targets["log_size"][positive] = selected[:, 3:6].clamp_min(1e-4).log()
        targets["yaw"][positive] = selected[:, 6]
        if selected.shape[1] >= 9:
            velocity = selected[:, 7:9]
            valid = torch.isfinite(velocity).all(-1) & (velocity.abs() < 100).all(-1)
            targets["velocity"][positive] = torch.where(valid[:, None], velocity, torch.zeros_like(velocity))
            positive_indices = positive.nonzero(as_tuple=False).squeeze(-1)
            velocity_mask[positive_indices[valid]] = True
        distance[positive] = (
            decoded[positive, :2] - targets["center_xy"][positive]
        ).norm(dim=-1)
    return {
        "class_target": class_target,
        "positive_mask": positive,
        "classification_mask": classification_mask,
        "ignore_mask": ~classification_mask,
        "velocity_mask": velocity_mask,
        "matched_gt_index": matched_tensor,
        "match_distance_xy": distance,
        **targets,
    }


def legacy_targets(decoded, gt_box, gt_label):
    value = match_stage3_proposals_to_gt(
        decoded.unsqueeze(0), [gt_box], [gt_label],
        num_classes=len(CLASSES), positive_distance_threshold=2.0,
    )
    result = {name: tensor.squeeze(0).cpu() for name, tensor in value.items()}
    result["classification_mask"] = torch.ones(len(decoded), dtype=torch.bool)
    result["ignore_mask"] = torch.zeros(len(decoded), dtype=torch.bool)
    return result


def label_record(sample, scheme):
    target = sample["targets"][scheme]
    positive = target["positive_mask"]
    matched = target["matched_gt_index"]
    return {
        "sample_token": sample["sample_token"],
        "scene_token": sample["scene_token"],
        "branch": sample["branch"],
        "split": sample["split"],
        "scheme": scheme,
        "proposal_count": int(len(positive)),
        "positive_count": int(positive.sum()),
        "background_count": int((target["classification_mask"] & ~positive).sum()),
        "ignore_count": int(target["ignore_mask"].sum()),
        "unique_matched_gt_count": len(set(int(value) for value in matched[positive].tolist())),
        "class_target": target["class_target"].tolist(),
        "matched_gt_index": matched.tolist(),
        "positive_mask": positive.tolist(),
        "ignore_mask": target["ignore_mask"].tolist(),
        "classification_mask": target["classification_mask"].tolist(),
        "match_distance_xy_m": [
            float(value) if float(value) >= 0 else None
            for value in target["match_distance_xy"].tolist()
        ],
        "target_uq_probability_used_for_assignment": False,
        "post_nms_score_used_for_assignment": False,
    }


def collect_split(model, cfg, branch, split, ann_file, device_index):
    with Path(ann_file).open("rb") as stream:
        value = pickle.load(stream)
    infos = value["infos"] if isinstance(value, dict) else value
    info_by_token = {item["token"]: item for item in infos}
    scenes = scene_map(infos)
    _, loader = stage3.build_loader(cfg, ann_file)
    samples = []
    for scheme in SCHEMES:
        (OUT / "labels" / branch / scheme).mkdir(parents=True, exist_ok=True)
    labels = {
        scheme: gzip.open(
            OUT / "labels" / branch / scheme / f"{split}.jsonl.gz", "wt", compresslevel=4,
        )
        for scheme in SCHEMES
    }
    device = torch.device(f"cuda:{device_index}")
    start = time.time()
    try:
        for sample_index, data in enumerate(loader):
            with torch.no_grad():
                output = model(
                    return_loss=False, rescale=True, return_stage3_target_uq=True,
                    stage3_branch=branch, stage3_num_proposals=200, **data,
                )
            item = output[0]["stage3_target_uq"]
            token = item["sample_token"]
            info = info_by_token[token]
            original_indices, gt_boxes, gt_labels = detection_gt(info)
            decoded = item["decoded_boxes"].float()
            components = cost_components(
                branch, model, item["class_logits"].float(), decoded,
                gt_boxes, gt_labels, device,
            )
            matched, _, ignore, _ = assignment_from_cost(
                components["total"], decoded[:, :2].numpy(), gt_boxes[:, :2],
            )
            gt_box = stage3.dc_first(data, "gt_bboxes_3d")
            gt_label = stage3.dc_first(data, "gt_labels_3d")
            legacy = legacy_targets(decoded, gt_box, gt_label)
            repaired = repaired_targets(decoded, gt_boxes, gt_labels, matched, np.zeros(len(decoded), dtype=bool))
            repaired_ignore = repaired_targets(decoded, gt_boxes, gt_labels, matched, ignore)
            if not torch.equal(repaired["positive_mask"], repaired_ignore["positive_mask"]):
                raise ValueError("B/C positive mismatch")
            matched_positive = repaired["matched_gt_index"][repaired["positive_mask"]]
            if len(set(matched_positive.tolist())) != len(matched_positive):
                raise ValueError("duplicate positive in repaired labels")
            sample = {
                "sample_token": token,
                "scene_token": scenes[token],
                "branch": branch,
                "split": split,
                "class_logits": item["class_logits"].float(),
                "features": item["features"].float(),
                "means": {name: tensor.float() for name, tensor in item["means"].items()},
                "decoded_boxes": decoded,
                "proposal_ids": item["proposal_ids"].long(),
                "query_indices": item["query_indices"].long(),
                "query_labels": item["query_labels"].long(),
                "post_nms_ids": item["post_nms_ids"].long(),
                "post_score_for_mapping": item["post_score_for_mapping"].float(),
                "kept_post_nms_mask": item["kept_post_nms_mask"].bool(),
                "legacy_gt_count": int(gt_label.numel()),
                "native_gt_count": int(len(gt_boxes)),
                "native_original_gt_indices": original_indices.tolist(),
                "targets": {
                    "legacy": legacy,
                    "hungarian": repaired,
                    "hungarian_ignore": repaired_ignore,
                },
            }
            samples.append(sample)
            for scheme in SCHEMES:
                labels[scheme].write(json.dumps(label_record(sample, scheme), sort_keys=True, allow_nan=False) + "\n")
            if (sample_index + 1) == 1 or (sample_index + 1) % 100 == 0 or sample_index + 1 == len(infos):
                print(json.dumps({"branch": branch, "split": split, "collected": sample_index + 1, "total": len(infos)}), flush=True)
    finally:
        for stream in labels.values():
            stream.close()
    cache = OUT / "work" / f"{branch}_{split}.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"branch": branch, "split": split, "samples": samples}, cache)
    return {
        "sample_count": len(samples),
        "proposal_count": len(samples) * 200,
        "wall_time_seconds": time.time() - start,
        "cache": str(cache),
        "cache_sha256": sha256(cache),
        "labels": {
            scheme: {
                "path": str(OUT / "labels" / branch / scheme / f"{split}.jsonl.gz"),
                "sha256": sha256(OUT / "labels" / branch / scheme / f"{split}.jsonl.gz"),
            }
            for scheme in SCHEMES
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True, choices=sorted(BRANCHES))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--splits", nargs="+", choices=sorted(SPLITS), default=list(SPLITS))
    parser.add_argument("--pilot-ann", default=SPLITS["pilot"])
    args = parser.parse_args()
    torch.manual_seed(20260708)
    np.random.seed(20260708)
    config, checkpoint = BRANCHES[args.branch]
    cfg = stage3.load_cfg(config)
    model = stage3.build_frozen_model(cfg, checkpoint, args.device)
    trainable = [name for name, parameter in model.module.named_parameters() if parameter.requires_grad]
    if trainable:
        raise ValueError(f"frozen detector unexpectedly trainable: {trainable[:5]}")
    checkpoint_sha_before = sha256(checkpoint)
    torch.cuda.reset_peak_memory_stats(args.device)
    results = {}
    selected_splits = {name: (args.pilot_ann if name == "pilot" else SPLITS[name]) for name in args.splits}
    for split, ann_file in selected_splits.items():
        results[split] = collect_split(model, cfg, args.branch, split, ann_file, args.device)
    if any(parameter.grad is not None for parameter in model.module.parameters()):
        raise ValueError("frozen detector accumulated gradients")
    checkpoint_sha_after = sha256(checkpoint)
    if checkpoint_sha_before != checkpoint_sha_after:
        raise ValueError("frozen detector checkpoint changed")
    manifest = {
        "stage": "stage3d_b_collection",
        "branch": args.branch,
        "seed": 20260708,
        "config": config,
        "config_sha256": sha256(config),
        "checkpoint": checkpoint,
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "detector_trainable_parameter_count": 0,
        "detector_parameter_with_gradient_count": 0,
        "assignment_spec": "outputs/stage3d/stage3d_b/assignment_spec.json",
        "official_val_used": False,
        "gpu": {
            "visible_index": args.device,
            "name": torch.cuda.get_device_name(args.device),
            "uuid": args.gpu_uuid,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(args.device),
        },
        "splits": results,
    }
    strict_dump(OUT / f"collection_{args.branch}.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
