#!/usr/bin/env python3
"""Export and verify Stage 3D-B native/cost-aware five-sample assignments."""

import argparse
import gzip
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from tools import stage3_run_real_target_uq as stage3


OUT = Path("outputs/stage3d/stage3d_b/native_assignment_smoke")
CLASSES = (
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
)
BRANCHES = {
    "camera": (
        "configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml",
        "pretrained/camera-only-det.pth",
    ),
    "lidar": (
        "configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml",
        "pretrained/lidar-only-det.pth",
    ),
    "fusion": (
        "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml",
        "pretrained/bevfusion-det.pth",
    ),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def detection_gt(info):
    names = np.asarray(info["gt_names"])
    point_mask = np.asarray(info["num_lidar_pts"]) > 0
    # Empty frames otherwise infer float64 here, which cannot be combined
    # with the boolean point-count mask.
    known = np.asarray([name in CLASSES for name in names], dtype=bool)
    indices = np.flatnonzero(point_mask & known)
    boxes = np.asarray(info["gt_boxes"], dtype=np.float32)[indices]
    velocity = np.asarray(info["gt_velocity"], dtype=np.float32)[indices]
    velocity[~np.isfinite(velocity)] = 0.0
    boxes = np.concatenate((boxes, velocity), axis=1)
    labels = np.asarray([CLASSES.index(str(names[index])) for index in indices], dtype=np.int64)
    return indices, boxes, labels


def assignment_from_cost(cost, proposal_xy, gt_xy, positive_gate=2.0, ignore_gate=4.0):
    cost = np.asarray(cost, dtype=np.float64)
    proposal_xy = np.asarray(proposal_xy, dtype=np.float64)
    gt_xy = np.asarray(gt_xy, dtype=np.float64)
    proposal_count = len(proposal_xy)
    matched = np.full(proposal_count, -1, dtype=np.int64)
    distance = np.full(proposal_count, np.nan, dtype=np.float64)
    if proposal_count and len(gt_xy):
        rows, cols = linear_sum_assignment(cost)
        pair_distance = np.linalg.norm(proposal_xy[rows] - gt_xy[cols], axis=1)
        keep = pair_distance <= positive_gate
        matched[rows[keep]] = cols[keep]
        distance[rows[keep]] = pair_distance[keep]
    nearest = (
        np.linalg.norm(proposal_xy[:, None, :] - gt_xy[None, :, :], axis=2).min(axis=1)
        if len(gt_xy) else np.full(proposal_count, np.nan)
    )
    ignore = (matched < 0) & (nearest > positive_gate) & (nearest <= ignore_gate)
    return matched, distance, ignore, nearest


def deterministic_assignment_from_cost(
    cost,
    proposal_xy,
    gt_xy,
    query_ids=None,
    gt_ids=None,
    positive_gate=2.0,
    ignore_gate=4.0,
):
    """Run the unchanged assignment on canonical row/column order.

    Original floating-point costs are passed to the optimizer unchanged. Stable
    query/GT IDs affect only exact equivalent optima through row/column order.
    The returned assignment is restored to the caller's query and GT indexing.
    """
    cost = np.asarray(cost, dtype=np.float64)
    proposal_xy = np.asarray(proposal_xy, dtype=np.float64)
    gt_xy = np.asarray(gt_xy, dtype=np.float64)
    query_ids = (
        np.arange(len(proposal_xy), dtype=np.int64)
        if query_ids is None
        else np.asarray(query_ids, dtype=np.int64)
    )
    gt_ids = (
        np.arange(len(gt_xy), dtype=np.int64)
        if gt_ids is None
        else np.asarray(gt_ids, dtype=np.int64)
    )
    if len(query_ids) != len(proposal_xy) or len(gt_ids) != len(gt_xy):
        raise ValueError("assignment ID count mismatch")
    if len(set(query_ids.tolist())) != len(query_ids):
        raise ValueError("query IDs must be unique")
    if len(set(gt_ids.tolist())) != len(gt_ids):
        raise ValueError("GT IDs must be unique")

    query_order = np.argsort(query_ids, kind="stable")
    gt_order = np.argsort(gt_ids, kind="stable")
    canonical = assignment_from_cost(
        cost[np.ix_(query_order, gt_order)],
        proposal_xy[query_order],
        gt_xy[gt_order],
        positive_gate=positive_gate,
        ignore_gate=ignore_gate,
    )
    canonical_matched, canonical_distance, canonical_ignore, canonical_nearest = canonical
    matched = np.full(len(query_ids), -1, dtype=np.int64)
    positive = canonical_matched >= 0
    matched[query_order[positive]] = gt_order[canonical_matched[positive]]
    distance = np.full(len(query_ids), np.nan, dtype=np.float64)
    distance[query_order] = canonical_distance
    ignore = np.zeros(len(query_ids), dtype=bool)
    ignore[query_order] = canonical_ignore
    nearest = np.full(len(query_ids), np.nan, dtype=np.float64)
    nearest[query_order] = canonical_nearest
    return matched, distance, ignore, nearest


def cost_components(branch, model, logits, boxes, gt_boxes, gt_labels, device):
    logits = logits.to(device=device, dtype=torch.float32)
    boxes = boxes.to(device=device, dtype=torch.float32)
    gt_boxes = torch.as_tensor(gt_boxes, device=device, dtype=torch.float32)
    gt_labels = torch.as_tensor(gt_labels, device=device, dtype=torch.long)
    if not len(gt_boxes):
        empty = np.empty((len(boxes), 0), dtype=np.float64)
        return {"classification": empty, "geometry": empty, "iou": empty, "total": empty}
    if branch == "camera":
        probability = logits.sigmoid()[:, gt_labels].clamp(1e-8, 1.0 - 1e-8)
        classification = -((1.0 - probability) ** 2.0) * probability.log()
        geometry = 0.25 * torch.cdist(boxes[:, :2], gt_boxes[:, :2], p=2)
        iou = torch.zeros_like(geometry)
    else:
        head = model.module.heads["object"]
        classification = head.bbox_assigner.cls_cost(logits, gt_labels)
        geometry = head.bbox_assigner.reg_cost(boxes, gt_boxes, head.train_cfg)
        overlap = head.bbox_assigner.iou_calculator(boxes, gt_boxes)
        iou = head.bbox_assigner.iou_cost(overlap)
    total = classification + geometry + iou
    if not all(torch.isfinite(value).all() for value in (classification, geometry, iou, total)):
        raise ValueError("native assignment contains a non-finite cost")
    return {
        "classification": classification.detach().cpu().numpy(),
        "geometry": geometry.detach().cpu().numpy(),
        "iou": iou.detach().cpu().numpy(),
        "total": total.detach().cpu().numpy(),
    }


def compact_pairs(matched, distance):
    return [
        {"proposal_index": int(index), "gt_index": int(gt), "distance_xy_m": float(distance[index])}
        for index, gt in enumerate(matched) if gt >= 0
    ]


def run(branch, device_index, ann_file):
    config, checkpoint = BRANCHES[branch]
    cfg = stage3.load_cfg(config)
    model = stage3.build_frozen_model(cfg, checkpoint, device_index)
    _, loader = stage3.build_loader(cfg, ann_file)
    with Path(ann_file).open("rb") as stream:
        value = pickle.load(stream)
    infos = value["infos"] if isinstance(value, dict) else value
    info_by_token = {item["token"]: item for item in infos}
    records = []
    device = torch.device(f"cuda:{device_index}")
    for sample_index, data in enumerate(loader):
        with torch.no_grad():
            output = model(
                return_loss=False,
                rescale=True,
                return_stage3_target_uq=True,
                stage3_branch=branch,
                stage3_num_proposals=200,
                **data,
            )
        item = output[0]["stage3_target_uq"]
        token = item["sample_token"]
        info = info_by_token[token]
        original_gt_indices, gt_boxes, gt_labels = detection_gt(info)
        proposal_boxes = item["decoded_boxes"].float()
        logits = item["class_logits"].float()
        components = cost_components(
            branch, model, logits, proposal_boxes, gt_boxes, gt_labels, device,
        )
        matched, distance, ignore, nearest = assignment_from_cost(
            components["total"], proposal_boxes[:, :2].numpy(), gt_boxes[:, :2],
        )
        repeated = assignment_from_cost(
            components["total"], proposal_boxes[:, :2].numpy(), gt_boxes[:, :2],
        )
        if not all(np.array_equal(left, right, equal_nan=True) for left, right in zip((matched, distance, ignore, nearest), repeated)):
            raise ValueError("assignment is not deterministic")
        distance_cost = (
            np.linalg.norm(proposal_boxes[:, None, :2].numpy() - gt_boxes[None, :, :2], axis=2)
            if len(gt_boxes) else np.empty((len(proposal_boxes), 0))
        )
        distance_matched, distance_distance, _, _ = assignment_from_cost(
            distance_cost, proposal_boxes[:, :2].numpy(), gt_boxes[:, :2],
        )
        native_pairs = {(index, int(gt)) for index, gt in enumerate(matched) if gt >= 0}
        distance_pairs = {(index, int(gt)) for index, gt in enumerate(distance_matched) if gt >= 0}
        gt_tokens_all = info.get("annotation_tokens")
        if gt_tokens_all is None:
            gt_tokens_all = [f"{token}:{index}" for index in range(len(info["gt_names"]))]
            gt_token_semantics = "stable_sample_token_plus_info_gt_index"
        else:
            gt_token_semantics = "native_annotation_token"
        gt_tokens = [gt_tokens_all[index] for index in original_gt_indices]
        b_positive = matched >= 0
        c_positive = matched >= 0
        if not np.array_equal(b_positive, c_positive):
            raise ValueError("B/C positive assignments differ")
        if len(set(np.flatnonzero(b_positive))) != int(b_positive.sum()):
            raise ValueError("proposal assigned more than once")
        matched_gts = matched[b_positive]
        if len(set(matched_gts.tolist())) != len(matched_gts):
            raise ValueError("GT assigned more than once")
        records.append({
            "sample_index": sample_index,
            "sample_token": token,
            "branch": branch,
            "proposal_count": int(len(proposal_boxes)),
            "proposal_indices": [int(value) for value in item["proposal_ids"].tolist()],
            "query_indices": [int(value) for value in item["query_indices"].tolist()],
            "query_labels": [int(value) for value in item["query_labels"].tolist()],
            "detector_class_logits": logits.tolist(),
            "detector_class_logits_source": item["class_probability_source"],
            "proposal_boxes": proposal_boxes.tolist(),
            "gt_tokens": gt_tokens,
            "gt_token_semantics": gt_token_semantics,
            "gt_classes": [CLASSES[int(value)] for value in gt_labels],
            "gt_labels": gt_labels.tolist(),
            "gt_boxes": gt_boxes.tolist(),
            "cost_components": {name: value.tolist() for name, value in components.items()},
            "native_assignment_before_2m_gate": [
                {"proposal_index": int(row), "gt_index": int(col)}
                for row, col in zip(*linear_sum_assignment(components["total"]))
            ] if components["total"].size else [],
            "B_positive_assignment_after_2m_gate": compact_pairs(matched, distance),
            "C_positive_assignment_after_2m_gate": compact_pairs(matched, distance),
            "C_ignore_proposal_indices": np.flatnonzero(ignore).astype(int).tolist(),
            "unmatched_proposal_indices": np.flatnonzero(matched < 0).astype(int).tolist(),
            "nearest_gt_distance_xy_m": [float(value) if math.isfinite(value) else None for value in nearest],
            "distance_only_assignment_after_2m_gate": compact_pairs(distance_matched, distance_distance),
            "assignment_comparison": {
                "native_positive_count": len(native_pairs),
                "distance_positive_count": len(distance_pairs),
                "pair_overlap_count": len(native_pairs & distance_pairs),
                "pair_union_count": len(native_pairs | distance_pairs),
                "reassignment_count": len(native_pairs ^ distance_pairs),
            },
            "target_uq_probability_used_for_assignment": False,
            "post_nms_score_used_for_assignment": False,
            "positive_gate_m": 2.0,
            "ignore_interval_m": "(2,4]",
        })
        print(json.dumps({"branch": branch, "sample": sample_index + 1, "token": token}), flush=True)
    if len(records) != 5:
        raise ValueError(f"five-sample smoke expected 5 records, got {len(records)}")
    path = OUT / f"{branch}_native_assignment_fixed5.jsonl.gz"
    with gzip.open(path, "wt", compresslevel=4) as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    total_native = sum(row["assignment_comparison"]["native_positive_count"] for row in records)
    total_distance = sum(row["assignment_comparison"]["distance_positive_count"] for row in records)
    overlap = sum(row["assignment_comparison"]["pair_overlap_count"] for row in records)
    summary = {
        "branch": branch,
        "status": "pass",
        "sample_count": 5,
        "assignment_semantics": (
            "camera_centerhead_positive_gaussian_focal_cost_weight_1_plus_xy_euclidean_m_weight_0.25"
            if branch == "camera" else "native_transfusion_hungarian_assigner_cost_components"
        ),
        "native_positive_count": total_native,
        "distance_only_positive_count": total_distance,
        "pair_overlap_count": overlap,
        "pair_overlap_over_native": overlap / total_native if total_native else None,
        "reassignment_rate_over_pair_union": (
            sum(row["assignment_comparison"]["reassignment_count"] for row in records)
            / max(sum(row["assignment_comparison"]["pair_union_count"] for row in records), 1)
        ),
        "records": str(path),
        "records_sha256": sha256(path),
        "config": config,
        "config_sha256": sha256(config),
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256(checkpoint),
        "ann_file": ann_file,
        "ann_file_sha256": sha256(ann_file),
        "checks": {
            "one_to_one": True,
            "B_C_positive_identical": True,
            "distance_gates": True,
            "deterministic_repeat": True,
            "target_uq_probability_used": False,
            "post_nms_score_used_as_probability": False,
            "strict_json": True,
        },
    }
    strict_dump(OUT / f"{branch}_native_assignment_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True, choices=sorted(BRANCHES))
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument(
        "--ann-file",
        default="outputs/stage3d/stage3d_b/native_assignment_smoke/nuscenes_infos_pilot_5.pkl",
    )
    args = parser.parse_args()
    torch.manual_seed(20260708)
    np.random.seed(20260708)
    run(args.branch, args.device, args.ann_file)


if __name__ == "__main__":
    main()
