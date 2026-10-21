#!/usr/bin/env python3
"""Real Fusion pre-NMS Stage 3A target-UQ pilot.

This keeps BEVFusion frozen, extracts compact TransFusion proposal tensors, and
trains only the Stage 3 target-UQ adapter. It never uses post-NMS score as a
class probability and never reads the official val split.
"""

import argparse
import copy
import gzip
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.apis.stage2_manifest import inject_stage2_corruption
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models.fusion_models.stage2_missing_hook import install_stage2_missing_hook
from mmdet3d.models import build_model
from mmdet3d.models.target_uq import (
    StructuredTargetUQAdapter,
    gaussian_nll_diag,
    gaussian_nll_full,
    log_i0,
    match_stage3_proposals_to_gt,
    target_uq_loss,
    wrap_angle,
)
from mmdet3d.utils import recursive_eval
from torch.nn import functional as F
from torchpack.utils.config import configs


ROOT = Path("outputs/stage3/target_uq")
CLASS_COUNT = 10
CHI2_2D = {"50": 1.38629436112, "80": 3.21887582487, "90": 4.60517018599, "95": 5.99146454711}


def strict_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def metric_path(args, name):
    return ROOT / "metrics" / f"{args.output_prefix}{name}.json"


def manifest_path(args, name):
    return ROOT / "manifests" / f"{args.output_prefix}{name}.json"


def as_float(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def remove_unused_radar(cfg):
    if cfg.model.get("encoders", {}).get("radar"):
        return
    datasets = cfg.data.test if isinstance(cfg.data.test, list) else [cfg.data.test]
    for dataset in datasets:
        dataset.pipeline = [step for step in dataset.pipeline if step.type != "LoadRadarPointsMultiSweeps"]
        for step in dataset.pipeline:
            if step.type == "Collect3D":
                step["keys"] = [key for key in step["keys"] if key != "radar"]


def load_cfg(config_path):
    configs.load(config_path, recursive=True)
    cfg = Config(recursive_eval(configs), filename=config_path)
    cfg.model.pretrained = None
    if cfg.model.get("encoders", {}).get("camera"):
        cfg.model.encoders.camera.backbone.init_cfg = None
    cfg.model.train_cfg = None
    cfg.data.workers_per_gpu = 0
    remove_unused_radar(cfg)
    return cfg


def dataset_cfg_for(cfg, ann_file):
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.ann_file = ann_file
    dataset_cfg.test_mode = True
    samples_per_gpu = dataset_cfg.pop("samples_per_gpu", 1)
    if samples_per_gpu > 1:
        dataset_cfg.pipeline = replace_ImageToTensor(dataset_cfg.pipeline)
    return dataset_cfg, samples_per_gpu


def build_loader(cfg, ann_file, start_index=0):
    dataset_cfg, samples_per_gpu = dataset_cfg_for(cfg, ann_file)
    dataset = build_dataset(dataset_cfg)
    if start_index:
        dataset.data_infos = dataset.data_infos[start_index:]
        if hasattr(dataset, "flag"):
            dataset.flag = dataset.flag[start_index:]
    loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )
    return dataset, loader


def build_frozen_model(cfg, checkpoint, device):
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16", None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, checkpoint, map_location="cpu", strict=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = MMDataParallel(model.cuda(device), device_ids=[device])
    model.eval()
    return model


def dc_first(data, key):
    value = data[key].data[0]
    return value[0] if isinstance(value, list) else value


def collect_samples(model, loader, max_samples, branch, match_threshold):
    samples = []
    start = time.time()
    for data in loader:
        if len(samples) >= max_samples:
            break
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
        gt_box = dc_first(data, "gt_bboxes_3d")
        gt_label = dc_first(data, "gt_labels_3d")
        decoded = item["decoded_boxes"].unsqueeze(0)
        targets = match_stage3_proposals_to_gt(
            decoded,
            [gt_box],
            [gt_label],
            num_classes=CLASS_COUNT,
            positive_distance_threshold=match_threshold,
        )
        token = item.get("sample_token") or dc_first(data, "metas").get("token") or f"sample_{len(samples)}"
        samples.append(
            {
                "sample_token": token,
                "class_logits": item["class_logits"],
                "features": item["features"],
                "means": item["means"],
                "decoded_boxes": item["decoded_boxes"],
                "proposal_ids": item["proposal_ids"],
                "post_nms_ids": item["post_nms_ids"],
                "post_score_for_mapping": item["post_score_for_mapping"],
                "kept_post_nms_mask": item["kept_post_nms_mask"],
                "targets": {key: value.squeeze(0).cpu() for key, value in targets.items()},
                "gt_count": int(gt_label.numel()),
            }
        )
        if len(samples) == 1 or len(samples) % 100 == 0 or len(samples) == max_samples:
            print(f"[stage3a] collected {len(samples)}/{max_samples} {branch} samples", flush=True)
    return {"samples": samples, "wall_time_seconds": time.time() - start}


def stack_samples(samples, device):
    means = {}
    for key in samples[0]["means"]:
        means[key] = torch.stack([sample["means"][key] for sample in samples]).to(device)
    targets = {}
    for key in samples[0]["targets"]:
        targets[key] = torch.stack([sample["targets"][key] for sample in samples]).to(device)
    return (
        torch.stack([sample["features"] for sample in samples]).to(device),
        torch.stack([sample["class_logits"] for sample in samples]).to(device),
        means,
        targets,
    )


def total_loss(losses):
    return losses["loss/stage3_target_uq/total"]


def finite_distribution_checks(distribution):
    prob_sum_ok = bool(torch.allclose(distribution["p_all"].sum(-1), torch.ones_like(distribution["p_exist"]), atol=1e-5))
    xy_spd = bool((torch.linalg.eigvalsh(distribution["center_xy_cov"]) > 0).all())
    vel_spd = bool((torch.linalg.eigvalsh(distribution["velocity_cov"]) > 0).all())
    finite = all(torch.isfinite(value).all() for value in distribution.values() if torch.is_tensor(value))
    return {"prob_sum_ok": prob_sum_ok, "xy_spd": xy_spd, "velocity_spd": vel_spd, "finite": bool(finite)}


def new_adapter(variant, in_channels, args, classifier_mode=None):
    return StructuredTargetUQAdapter(
        num_classes=CLASS_COUNT,
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        variant=variant,
        fixed_log_var=args.fixed_log_var,
        classifier_mode=classifier_mode or args.classifier_modes[0],
    )


def run_key(variant, classifier_mode):
    return f"{variant}_{classifier_mode}"


def smoke_and_overfit(variant, classifier_mode, samples, device, args):
    features, logits, means, targets = stack_samples(samples[:5], device)
    adapter = new_adapter(variant, features.shape[-1], args, classifier_mode).to(device)
    before = {name: param.detach().clone() for name, param in adapter.named_parameters()}
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    distribution = adapter(logits, means, features)
    losses = target_uq_loss(distribution, targets)
    first_loss = float(total_loss(losses).detach().cpu())
    total_loss(losses).backward()
    grads_finite = all(param.grad is None or bool(torch.isfinite(param.grad).all()) for param in adapter.parameters())
    optimizer.step()
    changed = [name for name, param in adapter.named_parameters() if not torch.equal(before[name], param.detach())]
    checks = finite_distribution_checks(distribution)

    overfit = new_adapter(variant, features.shape[-1], args, classifier_mode).to(device)
    optimizer = torch.optim.AdamW(overfit.parameters(), lr=args.lr)
    history = []
    for step in range(args.overfit_iterations):
        optimizer.zero_grad(set_to_none=True)
        dist = overfit(logits, means, features)
        losses = target_uq_loss(dist, targets)
        loss = total_loss(losses)
        loss.backward()
        optimizer.step()
        if step in (0, args.overfit_iterations - 1):
            history.append(float(loss.detach().cpu()))

    ckpt = ROOT / "checkpoints" / args.checkpoint_subdir / f"{args.output_prefix}real_{args.branch}_overfit_{run_key(variant, classifier_mode)}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": overfit.state_dict(), "variant": variant}, ckpt)
    restored = new_adapter(variant, features.shape[-1], args, classifier_mode).to(device)
    restored.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["model"])
    with torch.no_grad():
        left = overfit(logits, means, features)["p_exist"]
        right = restored(logits, means, features)["p_exist"]
    return {
        "forward_backward": {
            "sample_count": len(samples[:5]),
            "loss": first_loss,
            "grads_finite": bool(grads_finite),
            "adapter_parameters_changed": changed,
            **checks,
        },
        "overfit_100_iteration": {
            "iterations": args.overfit_iterations,
            "first_total": history[0],
            "last_total": history[-1],
            "loss_decreased": history[-1] < history[0],
        },
        "checkpoint": {
            "path": str(ckpt),
            "restore_max_abs_diff": float((left - right).abs().max().detach().cpu()),
        },
    }


def train_adapter(variant, classifier_mode, samples, device, args):
    features, _, _, _ = stack_samples(samples[:1], device)
    adapter = new_adapter(variant, features.shape[-1], args, classifier_mode).to(device)
    before = {name: param.detach().clone() for name, param in adapter.named_parameters()}
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    history = []
    start = time.time()
    for step in range(args.train_iterations):
        begin = (step * args.batch_size) % len(samples)
        batch = [samples[(begin + offset) % len(samples)] for offset in range(args.batch_size)]
        features, logits, means, targets = stack_samples(batch, device)
        optimizer.zero_grad(set_to_none=True)
        losses = target_uq_loss(adapter(logits, means, features), targets)
        loss = total_loss(losses)
        loss.backward()
        optimizer.step()
        if step in (0, args.train_iterations - 1) or (step + 1) % args.history_interval == 0:
            history.append({"step": step + 1, "total": float(loss.detach().cpu())})
    ckpt = ROOT / "checkpoints" / args.checkpoint_subdir / f"{args.output_prefix}real_{args.branch}_pilot_{run_key(variant, classifier_mode)}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    changed = [name for name, param in adapter.named_parameters() if not torch.equal(before[name], param.detach())]
    torch.save({
        "model": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "variant": variant,
        "classifier_mode": classifier_mode,
        "branch": args.branch,
        "stage": args.stage_name,
        "iterations": args.train_iterations,
        "batch_size": args.batch_size,
        "history": history,
    }, ckpt)
    return adapter, {
        "iterations": args.train_iterations,
        "batch_size": args.batch_size,
        "history": history,
        "loss_decreased": history[-1]["total"] < history[0]["total"],
        "wall_time_seconds": time.time() - start,
        "checkpoint": str(ckpt),
        "optimizer_state_saved": True,
        "scheduler_state_saved": False,
        "adapter_parameters_changed": changed,
    }


def calibrated_probabilities(logits_k1, calibration=None):
    calibration = calibration or {"method": "identity"}
    method = calibration.get("method", "identity")
    if method == "identity":
        return logits_k1.softmax(dim=-1)
    if method == "global_k1_temperature":
        return (logits_k1 / float(calibration.get("temperature", 1.0))).softmax(dim=-1)
    if method == "vector_scaling":
        bias = torch.as_tensor(calibration.get("bias", [0.0] * (CLASS_COUNT + 1)), device=logits_k1.device, dtype=logits_k1.dtype)
        return (logits_k1 / float(calibration.get("temperature", 1.0)) + bias).softmax(dim=-1)
    class_temperature = calibration.get("classification_temperature", 1.0)
    existence_temperature = calibration.get("existence_temperature", 1.0)
    existence_bias = calibration.get("existence_bias", 0.0)
    foreground = logits_k1[..., :-1] / float(class_temperature)
    p_class = foreground.softmax(dim=-1)
    exist_logit = torch.logsumexp(logits_k1[..., :-1], dim=-1) - logits_k1[..., -1] + float(existence_bias)
    p_exist = torch.sigmoid(exist_logit / float(existence_temperature))
    return torch.cat((p_class * p_exist.unsqueeze(-1), 1.0 - p_exist.unsqueeze(-1)), dim=-1)


def simple_ece(confidence, correct, bins=15):
    if confidence.numel() == 0:
        return None
    total = float(confidence.numel())
    value = confidence.new_zeros(())
    for lower in torch.linspace(0, 1, bins + 1, device=confidence.device)[:-1]:
        upper = lower + 1.0 / bins
        mask = (confidence > lower) & (confidence <= upper)
        if bool(mask.any()):
            value += mask.float().mean() * (confidence[mask].mean() - correct[mask].float().mean()).abs()
    return float(value.detach().cpu())


def reliability_bins(confidence, correct, bins=15):
    rows = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (confidence > lower) & (confidence <= upper)
        rows.append({
            "bin": index,
            "lower": lower,
            "upper": upper,
            "count": int(mask.sum().detach().cpu()),
            "confidence": mean_or_none(confidence[mask]) if bool(mask.any()) else None,
            "accuracy": mean_or_none(correct[mask].float()) if bool(mask.any()) else None,
        })
    return rows


def confidence_histogram(confidence, bins=15):
    hist = torch.histc(confidence.detach().float().cpu(), bins=bins, min=0.0, max=1.0)
    return [int(value) for value in hist.tolist()]


def auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.bool_)
    pos, neg = int(labels.sum()), int((~labels).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def aupr(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.bool_)
    if int(labels.sum()) == 0:
        return None
    order = np.argsort(-scores)
    y = labels[order].astype(np.float64)
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall_step = y / max(float(labels.sum()), 1.0)
    return float(np.sum(precision * recall_step))


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def mean_or_none(tensor):
    if tensor.numel() == 0:
        return None
    return float(tensor.mean().detach().cpu())


def mahalanobis(value, mean, chol, mask):
    diff = (value - mean).unsqueeze(-1)
    solve = torch.cholesky_solve(diff, chol)
    maha = (diff.transpose(-1, -2) @ solve).squeeze(-1).squeeze(-1)
    return maha[mask]


def evaluate_adapter(adapter, samples, device, calibration=None, batch_size=8):
    calibration = calibration or {}
    cov_scale = calibration.get("covariance_scale", 1.0)
    xy_scale = calibration.get("xy_covariance_scale", cov_scale)
    z_scale = calibration.get("z_variance_scale", cov_scale)
    size_scale = calibration.get("size_variance_scale", cov_scale)
    vel_scale = calibration.get("velocity_covariance_scale", cov_scale)
    yaw_scale = calibration.get("yaw_kappa_scale", 1.0)
    adapter.eval()
    cls_nlls, exist_nlls, briers, confidences, corrects = [], [], [], [], []
    probability_batches, label_batches = [], []
    error_scores, error_labels = [], []
    xy_nlls, z_nlls, size_nlls, yaw_nlls, yaw_gaussian_nlls, vel_nlls = [], [], [], [], [], []
    xy_maha, vel_maha, xy_trace, xy_det, vel_trace, vel_det = [], [], [], [], [], []
    xy_errors, xy_uncertainty = [], []
    positive_count = velocity_count = gt_count = unique_gt_matches = post_kept = 0

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            features, logits, means, targets = stack_samples(batch, device)
            dist = adapter(logits, means, features)
            p_all = calibrated_probabilities(dist["logits_k1"], calibration)
            labels = targets["class_target"].long()
            flat_prob = p_all.reshape(-1, p_all.shape[-1])
            flat_labels = labels.reshape(-1)
            picked = flat_prob.gather(1, flat_labels[:, None]).squeeze(1).clamp_min(1e-12)
            cls_nlls.append(-picked.log())
            positive = targets["positive_mask"].bool()
            p_exist = p_all[..., :-1].sum(-1).clamp(1e-8, 1 - 1e-8)
            exist_target = positive.float()
            exist_nlls.append(F.binary_cross_entropy(p_exist, exist_target, reduction="none").reshape(-1))
            one_hot = F.one_hot(flat_labels, num_classes=CLASS_COUNT + 1).float()
            briers.append((flat_prob - one_hot).pow(2).sum(-1))
            probability_batches.append(flat_prob.detach().cpu())
            label_batches.append(flat_labels.detach().cpu())
            conf, pred = flat_prob.max(dim=-1)
            ok = pred == flat_labels
            confidences.append(conf)
            corrects.append(ok)
            error_scores.extend((1.0 - conf).detach().cpu().tolist())
            error_labels.extend((~ok).detach().cpu().tolist())

            positive_count += int(positive.sum().item())
            velocity_mask = positive & targets["velocity_mask"].bool()
            velocity_count += int(velocity_mask.sum().item())
            post_kept += int(sum(sample["kept_post_nms_mask"].sum().item() for sample in batch))
            gt_count += int(sum(sample["gt_count"] for sample in batch))
            for sample in batch:
                matched = sample["targets"]["matched_gt_index"][sample["targets"]["positive_mask"]]
                unique_gt_matches += len(set(int(v) for v in matched.tolist() if int(v) >= 0))

            xy_chol = dist["center_xy_cholesky"] * math.sqrt(float(xy_scale))
            vel_chol = dist["velocity_cholesky"] * math.sqrt(float(vel_scale))
            z_log_var = dist["z_log_var"] + math.log(float(z_scale))
            size_log_var = dist["log_size_log_var"] + math.log(float(size_scale))
            if bool(positive.any()):
                xy_nlls.append(gaussian_nll_full(targets["center_xy"], dist["center_xy_mean"], xy_chol, positive))
                z_nlls.append(gaussian_nll_diag(targets["z"], dist["z_mean"], z_log_var, positive))
                size_nlls.append(gaussian_nll_diag(targets["log_size"], dist["log_size_mean"], size_log_var, positive))
                kappa = (dist["yaw_kappa"] * float(yaw_scale)).clamp_min(1e-8)
                yaw_nll = math.log(2.0 * math.pi) + log_i0(kappa)
                yaw_nll = yaw_nll - kappa * torch.cos(wrap_angle(targets["yaw"] - dist["yaw_mean"]))
                yaw_nlls.append(yaw_nll[positive])
                yaw_var = (1.0 / kappa.clamp_min(1e-4)).clamp(1e-4, 1e4)
                yaw_error = wrap_angle(targets["yaw"] - dist["yaw_mean"])
                yaw_gaussian = 0.5 * (math.log(2.0 * math.pi) + yaw_var.log() + yaw_error.pow(2) / yaw_var)
                yaw_gaussian_nlls.append(yaw_gaussian[positive])
                maha = mahalanobis(targets["center_xy"], dist["center_xy_mean"], xy_chol, positive)
                xy_maha.append(maha)
                cov = dist["center_xy_cov"] * float(xy_scale)
                xy_trace.append(cov.diagonal(dim1=-2, dim2=-1).sum(-1)[positive])
                xy_det.append(torch.linalg.det(cov)[positive])
                xy_errors.extend((targets["center_xy"] - dist["center_xy_mean"]).norm(dim=-1)[positive].detach().cpu().tolist())
                xy_uncertainty.extend(cov.diagonal(dim1=-2, dim2=-1).sum(-1)[positive].detach().cpu().tolist())
            if bool(velocity_mask.any()):
                vel_nlls.append(gaussian_nll_full(targets["velocity"], dist["velocity_mean"], vel_chol, velocity_mask))
                vel_maha.append(mahalanobis(targets["velocity"], dist["velocity_mean"], vel_chol, velocity_mask))
                vcov = dist["velocity_cov"] * float(vel_scale)
                vel_trace.append(vcov.diagonal(dim1=-2, dim2=-1).sum(-1)[velocity_mask])
                vel_det.append(torch.linalg.det(vcov)[velocity_mask])

    cls_nll = torch.cat(cls_nlls)
    exist_nll = torch.cat(exist_nlls)
    brier = torch.cat(briers)
    confidence = torch.cat(confidences)
    correct = torch.cat(corrects)
    all_prob = torch.cat(probability_batches)
    all_labels = torch.cat(label_batches)
    p_exist_all = all_prob[:, :-1].sum(-1)
    exist_target_all = (all_labels != CLASS_COUNT).float()
    positive_mask_all = all_labels != CLASS_COUNT
    background_mask_all = ~positive_mask_all
    classwise_ece = [
        simple_ece(all_prob[:, class_index], all_labels == class_index)
        for class_index in range(CLASS_COUNT + 1)
    ]
    coverage = {}
    xy_maha_all = torch.cat(xy_maha) if xy_maha else torch.empty(0)
    for name, threshold in CHI2_2D.items():
        coverage[name] = mean_or_none((xy_maha_all <= threshold).float())
    return {
        "sample_count": len(samples),
        "proposal_count": len(samples) * int(samples[0]["class_logits"].shape[0]) if samples else 0,
        "positive_count": positive_count,
        "background_count": len(samples) * int(samples[0]["class_logits"].shape[0]) - positive_count if samples else 0,
        "velocity_count": velocity_count,
        "classification": {
            "categorical_nll": as_float(cls_nll.mean()),
            "existence_nll": as_float(exist_nll.mean()),
            "brier": as_float(brier.mean()),
            "existence_brier": as_float((p_exist_all - exist_target_all).pow(2).mean()),
            "ece": as_float(simple_ece(confidence, correct)),
            "existence_ece": as_float(simple_ece(p_exist_all, exist_target_all.bool())),
            "positive_ece": as_float(simple_ece(confidence[positive_mask_all], correct[positive_mask_all])) if bool(positive_mask_all.any()) else None,
            "background_ece": as_float(simple_ece(confidence[background_mask_all], correct[background_mask_all])) if bool(background_mask_all.any()) else None,
            "classwise_ece": [as_float(value) for value in classwise_ece],
            "classwise_ece_mean": as_float(np.mean([value for value in classwise_ece if value is not None])),
            "reliability_bins": reliability_bins(confidence, correct),
            "existence_reliability_bins": reliability_bins(p_exist_all, exist_target_all.bool()),
            "confidence_histogram": confidence_histogram(confidence),
            "error_detection_auroc": as_float(auroc(error_scores, error_labels)),
            "error_detection_aupr": as_float(aupr(error_scores, error_labels)),
        },
        "regression": {
            "xy_nll": as_float(torch.cat(xy_nlls).mean() if xy_nlls else None),
            "z_nll": as_float(torch.cat(z_nlls).mean() if z_nlls else None),
            "log_size_nll": as_float(torch.cat(size_nlls).mean() if size_nlls else None),
            "yaw_von_mises_nll": as_float(torch.cat(yaw_nlls).mean() if yaw_nlls else None),
            "yaw_wrapped_gaussian_nll": as_float(torch.cat(yaw_gaussian_nlls).mean() if yaw_gaussian_nlls else None),
            "yaw_von_mises_better_than_wrapped_gaussian": (
                as_float(torch.cat(yaw_nlls).mean()) < as_float(torch.cat(yaw_gaussian_nlls).mean())
                if yaw_nlls and yaw_gaussian_nlls else None
            ),
            "velocity_nll": as_float(torch.cat(vel_nlls).mean() if vel_nlls else None),
            "xy_mahalanobis_mean": as_float(torch.cat(xy_maha).mean() if xy_maha else None),
            "velocity_mahalanobis_mean": as_float(torch.cat(vel_maha).mean() if vel_maha else None),
            "xy_coverage": coverage,
            "xy_coverage_error_mean": as_float(np.mean([abs((coverage[k] or 0.0) - float(k) / 100.0) for k in coverage])),
            "xy_trace_sharpness": as_float(torch.cat(xy_trace).mean() if xy_trace else None),
            "xy_det_sharpness": as_float(torch.cat(xy_det).mean() if xy_det else None),
            "velocity_trace_sharpness": as_float(torch.cat(vel_trace).mean() if vel_trace else None),
            "velocity_det_sharpness": as_float(torch.cat(vel_det).mean() if vel_det else None),
            "uncertainty_error_spearman": as_float(spearman(xy_uncertainty, xy_errors)),
        },
        "detection_proxy": {
            "scope": "proposal center-distance proxy on Stage 3 split, not official nuScenes mAP/NDS",
            "gt_count": gt_count,
            "unique_gt_matched": unique_gt_matches,
            "proposal_recall": as_float(unique_gt_matches / gt_count if gt_count else None),
            "post_filter_kept_count": post_kept,
            "post_filter_kept_mean": as_float(post_kept / len(samples) if samples else None),
        },
    }


def metric_total_regression(metrics):
    values = [
        metrics["regression"]["xy_nll"],
        metrics["regression"]["z_nll"],
        metrics["regression"]["log_size_nll"],
        metrics["regression"]["velocity_nll"],
    ]
    finite = [value for value in values if value is not None]
    return float(sum(finite)) if finite else float("inf")


def logits_and_labels(adapter, samples, device, batch_size):
    logits, labels = [], []
    adapter.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            features, raw_logits, means, targets = stack_samples(batch, device)
            logits.append(adapter(raw_logits, means, features)["logits_k1"].reshape(-1, CLASS_COUNT + 1).detach())
            labels.append(targets["class_target"].reshape(-1).long())
    return torch.cat(logits), torch.cat(labels)


def calibration_nll(logits, labels, calibration):
    probs = calibrated_probabilities(logits, calibration).reshape(-1, CLASS_COUNT + 1)
    return float(-probs.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12).log().mean().detach().cpu())


def calibration_brier(logits, labels, calibration):
    probs = calibrated_probabilities(logits, calibration).reshape(-1, CLASS_COUNT + 1)
    one_hot = F.one_hot(labels, num_classes=CLASS_COUNT + 1).float()
    return float((probs - one_hot).pow(2).sum(-1).mean().detach().cpu())


def fit_vector_scaling(logits, labels, l2=1e-3):
    temperature = torch.nn.Parameter(torch.ones((), device=logits.device))
    bias = torch.nn.Parameter(torch.zeros(CLASS_COUNT + 1, device=logits.device))
    optimizer = torch.optim.Adam([temperature, bias], lr=0.05)
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        scaled = logits / temperature.clamp_min(0.05) + bias
        loss = F.cross_entropy(scaled, labels) + float(l2) * bias.pow(2).mean()
        loss.backward()
        optimizer.step()
    return {
        "method": "vector_scaling",
        "temperature": float(temperature.detach().clamp_min(0.05).cpu()),
        "bias": [float(value) for value in bias.detach().cpu().tolist()],
        "l2": float(l2),
    }


def fit_classification_calibrators(adapter, samples, device, args):
    logits, labels = logits_and_labels(adapter, samples, device, args.eval_batch_size)
    calibrators = {"identity": {"method": "identity"}}
    grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    calibrators["global_k1_temperature"] = min(
        ({"method": "global_k1_temperature", "temperature": value} for value in grid),
        key=lambda item: calibration_nll(logits, labels, item),
    )
    sep = []
    for class_t in grid:
        for exist_t in grid:
            for bias in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                sep.append({
                    "method": "separated",
                    "classification_temperature": class_t,
                    "existence_temperature": exist_t,
                    "existence_bias": bias,
                })
    calibrators["separated"] = min(sep, key=lambda item: calibration_nll(logits, labels, item))
    calibrators["vector_scaling"] = fit_vector_scaling(logits, labels)
    calibrators["brier_temperature"] = min(
        ({"method": "global_k1_temperature", "temperature": value, "optimized_for": "brier"} for value in grid),
        key=lambda item: calibration_brier(logits, labels, item),
    )
    rows = {}
    for name, calibration in calibrators.items():
        rows[name] = {
            "parameters": calibration,
            "label_cal_categorical_nll": calibration_nll(logits, labels, calibration),
            "label_cal_brier": calibration_brier(logits, labels, calibration),
        }
    best_name = min(
        (name for name in calibrators if name != "brier_temperature"),
        key=lambda name: rows[name]["label_cal_categorical_nll"],
    )
    return calibrators[best_name], {"selected": best_name, "methods": rows}


def fit_regression_calibration(adapter, samples, device, args, base):
    result, details = dict(base), {}
    for field, metric_path, candidates in [
        ("xy_covariance_scale", ("regression", "xy_nll"), [0.25, 0.5, 1.0, 2.0, 4.0]),
        ("z_variance_scale", ("regression", "z_nll"), [0.25, 0.5, 1.0, 2.0, 4.0]),
        ("size_variance_scale", ("regression", "log_size_nll"), [0.25, 0.5, 1.0, 2.0, 4.0]),
        ("velocity_covariance_scale", ("regression", "velocity_nll"), [0.25, 0.5, 1.0, 2.0, 4.0]),
        ("yaw_kappa_scale", ("regression", "yaw_von_mises_nll"), [0.25, 0.5, 1.0, 2.0, 4.0]),
    ]:
        best_scale, best_value = 1.0, float("inf")
        for scale in candidates:
            metrics = evaluate_adapter(adapter, samples, device, {**result, field: scale}, args.eval_batch_size)
            value = metrics[metric_path[0]][metric_path[1]]
            if value is not None and value < best_value:
                best_scale, best_value = scale, value
        result[field] = best_scale
        details[field] = {"selected": best_scale, "label_cal_nll": best_value}
    result["covariance_scale"] = 1.0
    return result, details


def calibrate(adapter, samples, device, args):
    cls_cal, cls_ablation = fit_classification_calibrators(adapter, samples, device, args)
    full_cal, reg_details = fit_regression_calibration(adapter, samples, device, args, cls_cal)
    return full_cal, cls_ablation, reg_details


def stage3_degradation_conditions(branch, suite):
    if suite == "stage3c":
        if branch == "camera":
            return [
                ("clean", 0),
                ("camera_lowlight", 1),
                ("camera_lowlight", 2),
                ("camera_lowlight", 3),
                ("camera_extrinsic_rotation", 2),
                ("camera_extrinsic_rotation", 3),
                ("camera_temporal_offset", 2),
                ("camera_temporal_offset", 3),
                ("camera_missing", 1),
            ]
        if branch == "lidar":
            return [
                ("clean", 0),
                ("lidar_dropout", 1),
                ("lidar_dropout", 2),
                ("lidar_dropout", 3),
                ("lidar_beam_reduction", 1),
                ("lidar_beam_reduction", 2),
                ("lidar_beam_reduction", 3),
                ("camera_extrinsic_rotation", 2),
                ("camera_extrinsic_rotation", 3),
                ("camera_temporal_offset", 2),
                ("camera_temporal_offset", 3),
                ("lidar_missing", 1),
            ]
        return [
            ("clean", 0),
            ("camera_lowlight", 1),
            ("camera_lowlight", 2),
            ("camera_lowlight", 3),
            ("lidar_dropout", 1),
            ("lidar_dropout", 2),
            ("lidar_dropout", 3),
            ("lidar_beam_reduction", 1),
            ("lidar_beam_reduction", 2),
            ("lidar_beam_reduction", 3),
            ("camera_extrinsic_rotation", 2),
            ("camera_extrinsic_rotation", 3),
            ("camera_temporal_offset", 2),
            ("camera_temporal_offset", 3),
            ("camera_missing", 1),
            ("lidar_missing", 1),
        ]
    if suite == "stage3b":
        return stage3b_degradation_conditions(branch)
    return [
        ("clean", 0),
        ("camera_lowlight", 3),
        ("lidar_dropout", 3),
        ("lidar_beam_reduction", 3),
        ("camera_extrinsic_rotation", 2),
        ("camera_extrinsic_rotation", 3),
        ("camera_temporal_offset", 2),
        ("camera_temporal_offset", 3),
    ]


def stage3b_degradation_conditions(branch):
    if branch == "camera":
        return [
            ("clean", 0),
            ("camera_lowlight", 3),
            ("camera_extrinsic_rotation", 2),
            ("camera_extrinsic_rotation", 3),
            ("camera_temporal_offset", 2),
            ("camera_temporal_offset", 3),
            ("camera_missing", 1),
        ]
    if branch == "lidar":
        return [
            ("clean", 0),
            ("lidar_dropout", 3),
            ("lidar_beam_reduction", 3),
            ("camera_extrinsic_rotation", 2),
            ("camera_extrinsic_rotation", 3),
            ("camera_temporal_offset", 2),
            ("camera_temporal_offset", 3),
            ("lidar_missing", 1),
        ]
    return [
        ("clean", 0),
        ("camera_lowlight", 3),
        ("lidar_dropout", 3),
        ("lidar_beam_reduction", 3),
        ("camera_extrinsic_rotation", 2),
        ("camera_extrinsic_rotation", 3),
        ("camera_temporal_offset", 2),
        ("camera_temporal_offset", 3),
        ("camera_missing", 1),
        ("lidar_missing", 1),
    ]


def install_missing_if_needed(model, branch, corruption):
    if corruption not in ("camera_missing", "lidar_missing"):
        return None, None
    module = model.module if hasattr(model, "module") else model
    if module.fuser is None:
        return None, {
            "status": "diagnostic_not_applicable_single_modality_model",
            "regression_nll_excluded": True,
            "branch": branch,
            "condition": corruption,
        }
    missing = install_stage2_missing_hook(module, corruption)
    return missing, None


def not_applicable_condition(branch, corruption):
    if branch == "lidar" and corruption.startswith("camera_"):
        return {
            "status": "diagnostic_not_applicable_no_camera_encoder",
            "branch": branch,
            "condition": corruption,
        }
    if branch == "camera" and corruption.startswith("lidar_"):
        return {
            "status": "diagnostic_not_applicable_no_lidar_encoder",
            "branch": branch,
            "condition": corruption,
        }
    return None


def remove_missing_hook(missing):
    if missing is None:
        return
    missing["handle"].remove()
    missing["fused_handle"].remove()


def export_distribution_jsonl(
    adapter,
    samples,
    branch,
    calibration,
    device,
    batch_size,
    path,
    split=None,
    calibration_id=None,
    checkpoint_id=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cov_scale = calibration.get("covariance_scale", 1.0)
    xy_scale = float(calibration.get("xy_covariance_scale", cov_scale))
    z_scale = float(calibration.get("z_variance_scale", cov_scale))
    size_scale = float(calibration.get("size_variance_scale", cov_scale))
    vel_scale = float(calibration.get("velocity_covariance_scale", cov_scale))
    yaw_scale = float(calibration.get("yaw_kappa_scale", 1.0))
    count = 0
    adapter.eval()
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as stream, torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            features, logits, means, targets = stack_samples(batch, device)
            dist = adapter(logits, means, features)
            p_all = calibrated_probabilities(dist["logits_k1"], calibration)
            p_exist = p_all[..., :-1].sum(-1)
            p_class = p_all[..., :-1] / p_exist[..., None].clamp_min(1e-8)
            xy_cov = dist["center_xy_cov"] * xy_scale
            z_var = torch.exp(dist["z_log_var"]) * z_scale
            size_var = torch.exp(dist["log_size_log_var"]) * size_scale
            vel_cov = dist["velocity_cov"] * vel_scale
            yaw_kappa = dist["yaw_kappa"] * yaw_scale
            for batch_index, sample in enumerate(batch):
                proposal_total = int(sample["proposal_ids"].numel())
                for proposal_index in range(proposal_total):
                    record = {
                        "sample_token": sample["sample_token"],
                        "branch": branch,
                        "split": split,
                        "proposal_id": int(sample["proposal_ids"][proposal_index]),
                        "post_detection_id": int(sample["post_nms_ids"][proposal_index]),
                        "post_score_for_mapping": float(sample["post_score_for_mapping"][proposal_index]),
                        "class_probability_source": dist["class_probability_source"],
                        "score_is_probability": False,
                        "calibrated_class_probability": [float(v) for v in p_class[batch_index, proposal_index].detach().cpu().tolist()],
                        "calibrated_p_exist": float(p_exist[batch_index, proposal_index].detach().cpu()),
                        "calibrated_kplus1_probability": [float(v) for v in p_all[batch_index, proposal_index].detach().cpu().tolist()],
                        "box_mean": [float(v) for v in sample["decoded_boxes"][proposal_index].detach().cpu().tolist()],
                        "center_xy_covariance": [[float(v) for v in row] for row in xy_cov[batch_index, proposal_index].detach().cpu().tolist()],
                        "z_variance": float(z_var[batch_index, proposal_index].detach().cpu()),
                        "log_size_mean": [float(v) for v in dist["log_size_mean"][batch_index, proposal_index].detach().cpu().tolist()],
                        "log_size_variance": [float(v) for v in size_var[batch_index, proposal_index].detach().cpu().tolist()],
                        "yaw_mean": float(dist["yaw_mean"][batch_index, proposal_index].detach().cpu()),
                        "yaw_kappa": float(yaw_kappa[batch_index, proposal_index].detach().cpu()),
                        "velocity_mean": [float(v) for v in dist["velocity_mean"][batch_index, proposal_index].detach().cpu().tolist()],
                        "velocity_covariance": [[float(v) for v in row] for row in vel_cov[batch_index, proposal_index].detach().cpu().tolist()],
                        "matched_gt_id": int(sample["targets"]["matched_gt_index"][proposal_index]),
                        "positive_mask": bool(sample["targets"]["positive_mask"][proposal_index]),
                        "velocity_mask": bool(sample["targets"]["velocity_mask"][proposal_index]),
                        "effective_mask": bool(sample["kept_post_nms_mask"][proposal_index]),
                        "calibration_id": calibration_id,
                        "checkpoint_id": checkpoint_id,
                    }
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    count += 1
    return {"path": str(path), "record_count": count, "schema": "stage3_target_uq_distribution_jsonl_gzip" if str(path).endswith(".gz") else "stage3_target_uq_distribution_jsonl"}


def degradation_cfg(cfg, args, corruption_type, severity):
    copied = copy.deepcopy(cfg)
    inject_stage2_corruption(
        copied,
        corruption_type,
        severity,
        args.seed,
        None,
        "calibrated_candidate",
        args.temporal_history_file,
    )
    return copied


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml")
    parser.add_argument("--checkpoint", default="pretrained/bevfusion-det.pth")
    parser.add_argument("--train-ann", default="outputs/stage3/splits/nuscenes_infos_stage3_train.pkl")
    parser.add_argument("--label-cal-ann", default="outputs/stage3/splits/nuscenes_infos_stage3_label_cal.pkl")
    parser.add_argument("--dev-ann", default="outputs/stage3/splits/nuscenes_infos_stage3_dev.pkl")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--branch", default="fusion")
    parser.add_argument("--variants", nargs="+", default=["fixed", "diagonal", "structured"])
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--classifier-modes", nargs="+", default=["heatmap"], choices=["heatmap", "feature_k1", "separated"])
    parser.add_argument("--max-cal-samples", type=int, default=2814)
    parser.add_argument("--max-dev-samples", type=int, default=2808)
    parser.add_argument("--max-degradation-samples", type=int, default=256)
    parser.add_argument("--train-iterations", type=int, default=500)
    parser.add_argument("--overfit-iterations", type=int, default=100)
    parser.add_argument("--history-interval", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--fixed-log-var", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--positive-distance-threshold", type=float, default=2.0)
    parser.add_argument("--run-degradation", action="store_true")
    parser.add_argument("--degradation-suite", default="stage3a", choices=["stage3a", "stage3b", "stage3c"])
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--checkpoint-subdir", default="")
    parser.add_argument("--stage-name", default="stage3a_real_pre_nms_target_uq")
    parser.add_argument("--export-distributions", action="store_true")
    parser.add_argument("--distribution-key", default="structured_feature_k1")
    parser.add_argument("--distribution-root", default="distributions")
    parser.add_argument("--export-splits", nargs="+", default=["dev"], choices=["train", "label_cal", "dev"])
    parser.add_argument("--export-compressed", action="store_true")
    parser.add_argument("--temporal-history-file", default="outputs/stage3/splits/pilot_camera_history.json")
    return parser.parse_args()


def main():
    args = parse_args()
    run_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.cuda.set_device(args.device)
    cfg = load_cfg(args.config)
    model = build_frozen_model(cfg, args.checkpoint, args.device)
    _, train_loader = build_loader(cfg, args.train_ann)
    _, cal_loader = build_loader(cfg, args.label_cal_ann)
    _, dev_loader = build_loader(cfg, args.dev_ann)

    train_pack = collect_samples(model, train_loader, args.max_train_samples, args.branch, args.positive_distance_threshold)
    cal_pack = collect_samples(model, cal_loader, args.max_cal_samples, args.branch, args.positive_distance_threshold)
    dev_pack = collect_samples(model, dev_loader, args.max_dev_samples, args.branch, args.positive_distance_threshold)
    train_samples, cal_samples, dev_samples = train_pack["samples"], cal_pack["samples"], dev_pack["samples"]
    if len(train_samples) < 5:
        raise ValueError("real smoke requires at least 5 Stage 3 train samples")

    device = torch.device(f"cuda:{args.device}")
    in_channels = int(train_samples[0]["features"].shape[-1])
    adapter_parameters_by_classifier_mode = {}
    for classifier_mode in args.classifier_modes:
        adapter_probe = new_adapter(args.variants[0], in_channels, args, classifier_mode)
        adapter_parameters_by_classifier_mode[classifier_mode] = {
            "count": sum(parameter.numel() for parameter in adapter_probe.parameters()),
            "names": [name for name, _ in adapter_probe.named_parameters()],
        }
    first_mode = args.classifier_modes[0]
    manifest_file = manifest_path(args, f"real_{args.branch}_manifest")
    manifest = {
        "stage": args.stage_name,
        "seed": args.seed,
        "branch": args.branch,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "splits": {
            "train": args.train_ann,
            "label_cal": args.label_cal_ann,
            "dev": args.dev_ann,
            "official_val_used": False,
        },
        "matching_rule": "class-agnostic nearest xy center distance",
        "positive_distance_threshold_m": args.positive_distance_threshold,
        "negative_rule": "unmatched proposals are K+1 background",
        "ignored_proposal_rule": "none in Stage 3A minimal pilot",
        "probability_source": "pre-NMS heatmap logits or pre-NMS query feature classifier; never post-NMS score",
        "post_score_usage": "post_score_for_mapping only; never calibrated probability",
        "frozen_mean_path": True,
        "base_trainable_parameter_count": 0,
        "adapter_in_channels": in_channels,
        "adapter_trainable_parameter_count_per_variant": adapter_parameters_by_classifier_mode[first_mode]["count"],
        "adapter_trainable_parameter_names": adapter_parameters_by_classifier_mode[first_mode]["names"],
        "adapter_trainable_parameters_by_classifier_mode": adapter_parameters_by_classifier_mode,
        "adapter_variants": args.variants,
        "classifier_modes": args.classifier_modes,
        "train_iterations": args.train_iterations,
        "effective_batch_size": args.batch_size,
        "train_coverage": "one_or_more_full_internal_train_passes" if args.train_iterations * args.batch_size >= len(train_samples) else "partial_train_pass",
        "history_interval": args.history_interval,
        "dirty_diff_patch": "outputs/stage3/target_uq/manifests/stage3c_start_dirty.patch",
        "loss_weights": {"cls": 1.0, "xy": 1.0, "z": 1.0, "size": 1.0, "yaw": 1.0, "vel": 1.0},
        "numerical": {"eps": 1e-4, "fixed_log_var": args.fixed_log_var, "covariance_scale_grid": [0.5, 1.0, 2.0, 4.0]},
        "sample_counts": {
            "train": len(train_samples),
            "label_cal": len(cal_samples),
            "dev": len(dev_samples),
        },
        "preliminary_stage4_stage5_baseline": "retained_default_off_not_trained",
    }
    strict_json(manifest_file, manifest)

    clean_results, calibration_results, dev_results = {}, {}, {}
    calibrator_ablation, regression_calibration, classifier_head_ablation = {}, {}, {}
    trained = {}
    for classifier_mode in args.classifier_modes:
        for variant in args.variants:
            key = run_key(variant, classifier_mode)
            smoke = smoke_and_overfit(variant, classifier_mode, train_samples, device, args)
            adapter, training = train_adapter(variant, classifier_mode, train_samples, device, args)
            trained[key] = adapter
            raw_cal = evaluate_adapter(adapter, cal_samples, device, None, args.eval_batch_size)
            calibration, cls_ablation, reg_details = calibrate(adapter, cal_samples, device, args)
            calibrated_cal = evaluate_adapter(adapter, cal_samples, device, calibration, args.eval_batch_size)
            raw_dev = evaluate_adapter(adapter, dev_samples, device, None, args.eval_batch_size)
            calibrated_dev = evaluate_adapter(adapter, dev_samples, device, calibration, args.eval_batch_size)
            clean_results[key] = {
                "variant": variant,
                "classifier_mode": classifier_mode,
                "smoke": smoke,
                "training": training,
                "label_cal_raw": raw_cal,
                "dev_raw": raw_dev,
            }
            calibration_results[key] = {
                "variant": variant,
                "classifier_mode": classifier_mode,
                "parameters": calibration,
                "label_cal_raw": raw_cal,
                "label_cal_calibrated": calibrated_cal,
            }
            strict_json(ROOT / "calibrators" / "stage3c" / f"{args.output_prefix}{key}.json", {
                "stage": args.stage_name,
                "branch": args.branch,
                "variant": variant,
                "classifier_mode": classifier_mode,
                "split": "stage3_label_cal",
                "official_val_used": False,
                "dev_used_for_fitting": False,
                "parameters": calibration,
                "classification_ablation": cls_ablation,
                "regression_calibration": reg_details,
            })
            calibrator_ablation[key] = cls_ablation
            regression_calibration[key] = reg_details
            dev_results[key] = {
                "variant": variant,
                "classifier_mode": classifier_mode,
                "raw": raw_dev,
                "calibrated": calibrated_dev,
                "calibration_parameters": calibration,
            }
            if variant == "structured":
                classifier_head_ablation[classifier_mode] = {
                    "label_cal_nll": calibrated_cal["classification"]["categorical_nll"],
                    "label_cal_ece": calibrated_cal["classification"]["ece"],
                    "label_cal_brier": calibrated_cal["classification"]["brier"],
                    "dev_nll": calibrated_dev["classification"]["categorical_nll"],
                    "dev_ece": calibrated_dev["classification"]["ece"],
                    "dev_brier": calibrated_dev["classification"]["brier"],
                    "probability_source": adapter(torch.zeros(1, 1, CLASS_COUNT, device=device), {
                        "center_xy": torch.zeros(1, 1, 2, device=device),
                        "z": torch.zeros(1, 1, 1, device=device),
                        "log_size": torch.zeros(1, 1, 3, device=device),
                        "yaw": torch.zeros(1, 1, device=device),
                        "velocity": torch.zeros(1, 1, 2, device=device),
                    }, torch.zeros(1, 1, in_channels, device=device))["class_probability_source"],
                }

    strict_json(metric_path(args, "real_clean_pilot"), {
        "scope": f"real {args.branch} pre-NMS proposal pilot on Stage 3 train split",
        "collection_wall_time_seconds": {
            "train": train_pack["wall_time_seconds"],
            "label_cal": cal_pack["wall_time_seconds"],
            "dev": dev_pack["wall_time_seconds"],
        },
        "manifest": str(manifest_file),
        "results": clean_results,
    })
    strict_json(metric_path(args, "calibration"), {
        "scope": "calibration parameters fitted only on stage3_label_cal",
        "split": "stage3_label_cal",
        "official_val_used": False,
        "results": calibration_results,
    })
    strict_json(metric_path(args, "calibration_full_label_cal"), {
        "scope": "full label_cal calibration evaluation; calibrator selection uses only stage3_label_cal",
        "split": "stage3_label_cal",
        "official_val_used": False,
        "results": calibration_results,
    })
    strict_json(metric_path(args, "stage3_dev_eval"), {
        "scope": "Stage 3 dev evaluation; raw and label_cal-calibrated",
        "split": "stage3_dev",
        "official_val_used": False,
        "results": dev_results,
    })
    strict_json(metric_path(args, "stage3_dev_full_eval"), {
        "scope": "full Stage 3 dev evaluation; no calibrator fitting on dev",
        "split": "stage3_dev",
        "official_val_used": False,
        "results": dev_results,
    })
    strict_json(metric_path(args, "calibrator_ablation"), {
        "scope": "identity/global/separated/vector calibrators fitted on label_cal only",
        "results": calibrator_ablation,
    })
    strict_json(metric_path(args, "classifier_head_ablation"), {
        "scope": "structured classifier-mode ablation; heatmap vs feature K+1 vs separated",
        "results": classifier_head_ablation,
    })
    strict_json(metric_path(args, "regression_calibration"), {
        "scope": "separate xy/z/size/velocity/yaw regression calibration scales fitted on label_cal only",
        "results": regression_calibration,
    })
    strict_json(manifest_path(args, "calibration_manifest"), {
        "seed": args.seed,
        "branch": args.branch,
        "train_split": args.train_ann,
        "calibration_split": args.label_cal_ann,
        "dev_split": args.dev_ann,
        "official_val_used": False,
        "calibrator_selection_split": "stage3_label_cal",
        "dev_used_for_fitting": False,
        "post_nms_score_used_for_probability_or_calibration": False,
        "classifier_modes": args.classifier_modes,
        "variants": args.variants,
    })

    degradation = {"status": "not_requested", "conditions": {}}
    if args.run_degradation:
        structured_keys = [key for key, item in calibration_results.items() if item["variant"] == "structured"]
        degradation_key = min(
            structured_keys,
            key=lambda key: calibration_results[key]["label_cal_calibrated"]["classification"]["categorical_nll"],
        )
        degradation["status"] = "completed_or_recorded_per_condition"
        degradation["selected_key"] = degradation_key
        conditions = stage3_degradation_conditions(args.branch, args.degradation_suite)
        for corruption, severity in conditions:
            name = f"{corruption}_s{severity}"
            skipped = not_applicable_condition(args.branch, corruption)
            if skipped is not None:
                degradation["conditions"][name] = skipped
                continue
            missing, skipped = install_missing_if_needed(model, args.branch, corruption)
            if skipped is not None:
                degradation["conditions"][name] = skipped
                continue
            try:
                cond_cfg = cfg if corruption == "clean" else degradation_cfg(cfg, args, corruption, severity)
                _, loader = build_loader(cond_cfg, args.dev_ann)
                pack = collect_samples(model, loader, args.max_degradation_samples, args.branch, args.positive_distance_threshold)
                degradation["conditions"][name] = {
                    degradation_key: evaluate_adapter(
                        trained[degradation_key],
                        pack["samples"],
                        device,
                        dev_results[degradation_key]["calibration_parameters"],
                        args.eval_batch_size,
                    )
                }
                if corruption in ("camera_missing", "lidar_missing"):
                    degradation["conditions"][name][degradation_key]["regression_nll_excluded_from_main"] = True
                    degradation["conditions"][name][degradation_key]["missing_hook_stats"] = {
                        key: value for key, value in getattr(model.module, "stage2_missing", {}).items()
                        if key not in ("hook", "handle", "fused_handle")
                    }
            except Exception as exc:
                degradation["conditions"][name] = {"status": "failed", "error": str(exc)}
            finally:
                remove_missing_hook(missing)
    strict_json(metric_path(args, "degradation_pilot"), degradation)
    strict_json(metric_path(args, "degradation_calibrated_recheck"), degradation)

    semantic_classifier_modes = {"feature_k1", "separated"}
    probability_pass_keys = []
    for key, result in dev_results.items():
        if result["classifier_mode"] not in semantic_classifier_modes:
            continue
        raw_cls = result["raw"]["classification"]
        cal_cls = result["calibrated"]["classification"]
        if (
            cal_cls["categorical_nll"] < raw_cls["categorical_nll"]
            and cal_cls["ece"] < raw_cls["ece"]
            and cal_cls["brier"] < raw_cls["brier"]
        ):
            probability_pass_keys.append(key)
    preferred_key = None
    for candidate in [
        "structured_feature_k1",
        "structured_separated",
        "diagonal_feature_k1",
        "diagonal_separated",
        "fixed_feature_k1",
        "fixed_separated",
    ]:
        if candidate in probability_pass_keys:
            preferred_key = candidate
            break
    if preferred_key is None and probability_pass_keys:
        preferred_key = min(
            probability_pass_keys,
            key=lambda key: dev_results[key]["calibrated"]["classification"]["categorical_nll"],
        )
    stage3b_recommended = preferred_key is not None
    if "structured_feature_k1" in dev_results and "diagonal_feature_k1" in dev_results:
        structure_status = (
            "partial_pass_structured_feature_k1_xy_nll_better_than_diagonal"
            if dev_results["structured_feature_k1"]["calibrated"]["regression"]["xy_nll"]
            < dev_results["diagonal_feature_k1"]["calibrated"]["regression"]["xy_nll"]
            else "fail_structured_feature_k1_not_better_than_diagonal"
        )
    else:
        structure_status = "not_evaluable"

    distribution_export = None
    if args.export_distributions:
        if args.distribution_key not in trained:
            raise ValueError(f"distribution key {args.distribution_key} was not trained")
        split_samples = {"train": train_samples, "label_cal": cal_samples, "dev": dev_samples}
        split_names = {"train": "stage3_train", "label_cal": "stage3_label_cal", "dev": "stage3_dev"}
        distribution_export = {}
        for split in args.export_splits:
            suffix = ".jsonl.gz" if args.export_compressed else ".jsonl"
            path = ROOT / args.distribution_root / args.branch / f"{args.output_prefix}{split_names[split]}_{args.distribution_key}{suffix}"
            distribution_export[split] = export_distribution_jsonl(
                trained[args.distribution_key],
                split_samples[split],
                args.branch,
                dev_results[args.distribution_key]["calibration_parameters"],
                device,
                args.eval_batch_size,
                path,
                split=split_names[split],
                calibration_id=str(ROOT / "calibrators" / "stage3c" / f"{args.output_prefix}{args.distribution_key}.json"),
                checkpoint_id=clean_results[args.distribution_key]["training"]["checkpoint"],
            )

    summary = {
        "stage": args.stage_name,
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "real_pre_nms_wired": True,
        "post_nms_probability_leakage_risk": "guarded: class_probability_source is pre_nms_heatmap_or_query_feature; post scores used only for mapping",
        "official_val_used": False,
        "branch_completed": args.branch,
        "sample_counts": manifest["sample_counts"],
        "variants": args.variants,
        "classifier_modes": args.classifier_modes,
        "selected_stage3a_cal_key": preferred_key,
        "probability_pass_keys": probability_pass_keys,
        "heatmap_classifier_status": "diagnostic_only_not_primary; feature_k1_or_separated is required for calibrated probability semantics",
        "key_dev_metrics": {
            key: {
                "variant": dev_results[key]["variant"],
                "classifier_mode": dev_results[key]["classifier_mode"],
                "categorical_nll_raw": dev_results[key]["raw"]["classification"]["categorical_nll"],
                "categorical_nll_calibrated": dev_results[key]["calibrated"]["classification"]["categorical_nll"],
                "ece_raw": dev_results[key]["raw"]["classification"]["ece"],
                "ece_calibrated": dev_results[key]["calibrated"]["classification"]["ece"],
                "brier_raw": dev_results[key]["raw"]["classification"]["brier"],
                "brier_calibrated": dev_results[key]["calibrated"]["classification"]["brier"],
                "xy_nll_raw": dev_results[key]["raw"]["regression"]["xy_nll"],
                "xy_nll_calibrated": dev_results[key]["calibrated"]["regression"]["xy_nll"],
                "xy_coverage_error_calibrated": dev_results[key]["calibrated"]["regression"]["xy_coverage_error_mean"],
                "xy_trace_sharpness_calibrated": dev_results[key]["calibrated"]["regression"]["xy_trace_sharpness"],
                "yaw_von_mises_nll_calibrated": dev_results[key]["calibrated"]["regression"]["yaw_von_mises_nll"],
                "yaw_wrapped_gaussian_nll_calibrated": dev_results[key]["calibrated"]["regression"]["yaw_wrapped_gaussian_nll"],
                "uncertainty_error_spearman_calibrated": dev_results[key]["calibrated"]["regression"]["uncertainty_error_spearman"],
            }
            for key in dev_results
        },
        "gate_decision": {
            "engineering": "partial_pass_fusion_only",
            "probability": "pass_for_feature_k1_or_separated" if stage3b_recommended else "fail_dev_ece_or_brier_not_improved",
            "structure": structure_status,
            "stage3b_three_branch": "recommended_narrow_pilot_with_feature_k1_or_separated_only"
            if stage3b_recommended
            else "not_recommended_until_calibration_is_fixed",
            "stage4_stage5": "not_recommended",
        },
        "resource_use": {
            "successful_run_wall_time_seconds": time.time() - run_start,
            "successful_run_gpu_hours_single_h20": (time.time() - run_start) / 3600.0,
            "torch_cuda_max_memory_gib": torch.cuda.max_memory_allocated(args.device) / (1024.0 ** 3),
        },
        "recommend_stage3b_three_branch_pilot": stage3b_recommended,
        "recommend_stage4_stage5": False,
        "outputs": {
            "manifest": str(manifest_file),
            "real_clean_pilot": str(metric_path(args, "real_clean_pilot")),
            "calibration": str(metric_path(args, "calibration")),
            "stage3_dev_eval": str(metric_path(args, "stage3_dev_eval")),
            "degradation_pilot": str(metric_path(args, "degradation_pilot")),
            "calibration_full_label_cal": str(metric_path(args, "calibration_full_label_cal")),
            "stage3_dev_full_eval": str(metric_path(args, "stage3_dev_full_eval")),
            "calibrator_ablation": str(metric_path(args, "calibrator_ablation")),
            "classifier_head_ablation": str(metric_path(args, "classifier_head_ablation")),
            "regression_calibration": str(metric_path(args, "regression_calibration")),
            "degradation_calibrated_recheck": str(metric_path(args, "degradation_calibrated_recheck")),
            "distribution_export": distribution_export,
        },
    }
    if distribution_export is not None:
        summary["distribution_export"] = distribution_export
    strict_json(ROOT / f"{args.output_prefix}summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
