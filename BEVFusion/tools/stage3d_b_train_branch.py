#!/usr/bin/env python3
"""Train and evaluate one Stage 3D-B structured A/B/C branch group."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from mmdet3d.models.target_uq import (
    StructuredTargetUQAdapter,
    gaussian_nll_diag,
    gaussian_nll_full,
    log_i0,
    target_uq_loss,
    wrap_angle,
)
from tools import stage3_calibrate_camera as calibration_tool
from tools import stage3_run_real_target_uq as stage3
from tools.stage3d_b_collect_labels import SCHEMES
from tools.stage3d_b_native_assignment import BRANCHES, sha256, strict_dump


OUT = Path("outputs/stage3d/stage3d_b")
ARM_SCHEME = {
    "A_legacy_nearest_2m": "legacy",
    "B_hungarian_1to1": "hungarian",
    "C_hungarian_1to1_ignore_2to4m": "hungarian_ignore",
}
CHI2 = {"50": 1.38629436112, "80": 3.21887582487, "90": 4.60517018599, "95": 5.99146454711}
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


def state_sha256(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        tensor = state[name].detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def adapter(in_channels):
    return StructuredTargetUQAdapter(
        num_classes=10, in_channels=in_channels, hidden_channels=32,
        variant="structured", fixed_log_var=0.0, classifier_mode="feature_k1",
    )


def stack(samples, scheme, device):
    features = torch.stack([sample["features"] for sample in samples]).to(device)
    logits = torch.stack([sample["class_logits"] for sample in samples]).to(device)
    means = {
        name: torch.stack([sample["means"][name] for sample in samples]).to(device)
        for name in samples[0]["means"]
    }
    targets = {
        name: torch.stack([sample["targets"][scheme][name] for sample in samples]).to(device)
        for name in samples[0]["targets"][scheme]
    }
    return features, logits, means, targets


def loss_with_ignore(distribution, targets):
    valid = targets["classification_mask"].bool()
    positive = targets["positive_mask"].bool()
    velocity = positive & targets["velocity_mask"].bool()
    logits = distribution["logits_k1"][valid]
    labels = targets["class_target"][valid].long()
    if not logits.numel():
        raise ValueError("classification denominator is empty")
    losses = {
        "cls": F.cross_entropy(logits, labels),
        "xy": gaussian_nll_full(
            targets["center_xy"], distribution["center_xy_mean"],
            distribution["center_xy_cholesky"], positive,
        ).mean() if bool(positive.any()) else logits.new_zeros(()),
        "z": gaussian_nll_diag(
            targets["z"], distribution["z_mean"], distribution["z_log_var"], positive,
        ).mean() if bool(positive.any()) else logits.new_zeros(()),
        "size": gaussian_nll_diag(
            targets["log_size"], distribution["log_size_mean"],
            distribution["log_size_log_var"], positive,
        ).mean() if bool(positive.any()) else logits.new_zeros(()),
        "yaw": (
            math.log(2 * math.pi) + log_i0(distribution["yaw_kappa"])
            - distribution["yaw_kappa"] * torch.cos(wrap_angle(targets["yaw"] - distribution["yaw_mean"]))
        )[positive].mean() if bool(positive.any()) else logits.new_zeros(()),
        "vel": gaussian_nll_full(
            targets["velocity"], distribution["velocity_mean"],
            distribution["velocity_cholesky"], velocity,
        ).mean() if bool(velocity.any()) else logits.new_zeros(()),
    }
    total = sum(losses.values())
    return total, losses


def parameter_gradient_norm(model):
    values = [parameter.grad.detach().norm() ** 2 for parameter in model.parameters() if parameter.grad is not None]
    return float(torch.stack(values).sum().sqrt().cpu()) if values else 0.0


def smoke_overfit(samples, scheme, init_state, device, lr):
    subset = samples[:5]
    features, logits, means, targets = stack(subset, scheme, device)
    model = adapter(features.shape[-1]).to(device)
    model.load_state_dict(init_state)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    distribution = model(logits, means, features)
    loss, _ = loss_with_ignore(distribution, targets)
    if not torch.isfinite(loss):
        raise ValueError("non-finite forward/backward smoke loss")
    loss.backward()
    grad_norm = parameter_gradient_norm(model)
    if not math.isfinite(grad_norm):
        raise ValueError("non-finite smoke gradient")
    optimizer.step()
    changed = [name for name, value in model.named_parameters() if not torch.equal(before[name], value.detach())]
    eig_xy = torch.linalg.eigvalsh(distribution["center_xy_cov"])
    eig_vel = torch.linalg.eigvalsh(distribution["velocity_cov"])
    if not changed or not bool((eig_xy > 0).all()) or not bool((eig_vel > 0).all()):
        raise ValueError("smoke parameter/SPD check failed")
    if not torch.allclose(distribution["p_all"].sum(-1), torch.ones_like(distribution["p_exist"]), atol=1e-5):
        raise ValueError("smoke probability normalization failed")
    model = adapter(features.shape[-1]).to(device)
    model.load_state_dict(init_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        current, _ = loss_with_ignore(model(logits, means, features), targets)
        if not torch.isfinite(current):
            raise ValueError(f"non-finite overfit loss at {step}")
        current.backward()
        optimizer.step()
        if step in (0, 99):
            history.append(float(current.detach().cpu()))
    if history[-1] >= history[0]:
        raise ValueError(f"100-iteration overfit did not decrease: {history}")
    return {
        "sample_count": 5,
        "forward_backward_loss": float(loss.detach().cpu()),
        "gradient_norm": grad_norm,
        "trainable_parameter_names": [name for name, _ in model.named_parameters()],
        "parameter_change_count": len(changed),
        "probability_normalization": "pass",
        "xy_and_velocity_spd": "pass",
        "overfit_first_loss": history[0],
        "overfit_last_loss": history[-1],
        "overfit_loss_decreased": True,
    }


def train_arm(samples, scheme, init_state, device, iterations, batch_size, lr, branch, arm):
    in_channels = samples[0]["features"].shape[-1]
    model = adapter(in_channels).to(device)
    model.load_state_dict(init_state)
    initial_sha = state_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    finite_steps = 0
    start = time.time()
    for step in range(iterations):
        begin = (step * batch_size) % len(samples)
        current_samples = [samples[(begin + offset) % len(samples)] for offset in range(batch_size)]
        features, logits, means, targets = stack(current_samples, scheme, device)
        optimizer.zero_grad(set_to_none=True)
        loss, pieces = loss_with_ignore(model(logits, means, features), targets)
        if not torch.isfinite(loss):
            raise ValueError(f"{arm}: non-finite loss at iteration {step + 1}")
        loss.backward()
        grad_norm = parameter_gradient_norm(model)
        if not math.isfinite(grad_norm):
            raise ValueError(f"{arm}: non-finite gradient at iteration {step + 1}")
        optimizer.step()
        finite_steps += 1
        if step == 0 or step + 1 == iterations or (step + 1) % 100 == 0:
            history.append({
                "iteration": step + 1,
                "total": float(loss.detach().cpu()),
                "gradient_norm": grad_norm,
                "components": {name: float(value.detach().cpu()) for name, value in pieces.items()},
            })
            print(json.dumps({"branch": branch, "arm": arm, **history[-1]}), flush=True)
    final_sha = state_sha256(model.state_dict())
    if final_sha == initial_sha:
        raise ValueError(f"{arm}: Target-UQ parameters did not change")
    checkpoint = OUT / "runs" / branch / arm / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": None,
        "branch": branch, "arm": arm, "label_scheme": scheme,
        "variant": "structured", "classifier_mode": "feature_k1", "seed": 20260708,
        "iterations": iterations, "batch_size": batch_size, "learning_rate": lr,
        "initialization_sha256": initial_sha, "final_parameter_sha256": final_sha,
        "history": history,
    }, checkpoint)
    restored = adapter(in_channels).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    restored.load_state_dict(payload["model"])
    probe = samples[:1]
    features, logits, means, _ = stack(probe, scheme, device)
    with torch.no_grad():
        difference = (model(logits, means, features)["p_all"] - restored(logits, means, features)["p_all"]).abs().max()
    if float(difference.cpu()) != 0.0:
        raise ValueError(f"{arm}: checkpoint restore mismatch")
    return model, {
        "iterations": iterations, "batch_size": batch_size, "learning_rate": lr,
        "scheduler": None, "finite_loss_rate": finite_steps / iterations,
        "history": history, "first_loss": history[0]["total"], "last_loss": history[-1]["total"],
        "loss_decreased": history[-1]["total"] < history[0]["total"],
        "wall_time_seconds": time.time() - start,
        "initialization_sha256": initial_sha, "final_parameter_sha256": final_sha,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "restore_max_abs_diff": 0.0,
        "trainable_parameter_names": [name for name, _ in model.named_parameters()],
    }


def predict(model, samples, device, batch_size=8):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            features, logits, means, _ = stack(batch, "legacy", device)
            distribution = model(logits, means, features)
            for index in range(len(batch)):
                output.append({
                    name: value[index].detach().cpu()
                    for name, value in distribution.items() if torch.is_tensor(value)
                })
    return output


def classification_arrays(predictions, samples, scheme):
    logits, labels, scenes = [], [], []
    for prediction, sample in zip(predictions, samples):
        target = sample["targets"][scheme]
        valid = target["classification_mask"].bool()
        logits.append(prediction["logits_k1"][valid])
        labels.append(target["class_target"][valid].long())
        scenes.extend([sample["scene_token"]] * int(valid.sum()))
    return torch.cat(logits), torch.cat(labels), np.asarray(scenes)


def fit_oof_calibrator(predictions, samples, scheme, device):
    logits_cpu, labels_cpu, record_scenes = classification_arrays(predictions, samples, scheme)
    scene_by_sample = {sample["sample_token"]: sample["scene_token"] for sample in samples}
    folds = calibration_tool.build_scene_folds(scene_by_sample, seed=20260708, fold_count=5)
    scene_fold = {
        scene: fold_index
        for fold_index, fold in enumerate(folds)
        for scene in fold["scenes"]
    }
    record_fold = np.asarray([scene_fold[scene] for scene in record_scenes], dtype=np.int64)
    logits = logits_cpu.to(device)
    labels = labels_cpu.to(device)
    candidate_names = ["identity", "scalar"] + [f"vector_l2_{value:g}" for value in calibration_tool.L2_GRID]
    candidate_names += ["separated_scalar"] + [f"separated_vector_l2_{value:g}" for value in calibration_tool.L2_GRID]
    oof = {name: np.empty((len(labels_cpu), 11), dtype=np.float32) for name in candidate_names}
    fold_parameters = {name: [] for name in candidate_names}
    for fold_index in range(5):
        train_index = torch.as_tensor(np.flatnonzero(record_fold != fold_index), device=device)
        hold_np = np.flatnonzero(record_fold == fold_index)
        hold_index = torch.as_tensor(hold_np, device=device)
        fitted = calibration_tool.fit_candidates(logits.index_select(0, train_index), labels.index_select(0, train_index))
        for name in candidate_names:
            oof[name][hold_np] = calibration_tool.apply_calibration_torch(
                logits.index_select(0, hold_index), fitted[name],
            ).detach().cpu().numpy()
            fold_parameters[name].append({"fold": fold_index, "parameters": fitted[name]})
    candidate_metrics = {}
    for name in candidate_names:
        candidate_metrics[name] = calibration_tool.metrics(oof[name], labels_cpu.numpy())
        candidate_metrics[name]["complexity"] = fold_parameters[name][0]["parameters"]["complexity"]
    selected, audit = calibration_tool.select_candidate(candidate_metrics)
    full = calibration_tool.fit_candidates(logits, labels)
    return full[selected], {
        "selection": selected,
        "candidate_metrics": candidate_metrics,
        "selection_audit": audit,
        "fold_parameters": fold_parameters,
        "folds": [{"fold": index, "scene_tokens": fold["scenes"], "sample_count": fold["sample_count"]} for index, fold in enumerate(folds)],
        "record_count": len(labels_cpu),
        "scene_count": len(set(record_scenes.tolist())),
    }


def dev_calibration_gate(parameters, predictions, samples, scheme):
    logits, labels, _ = classification_arrays(predictions, samples, scheme)
    raw_probability = logits.softmax(-1).numpy()
    calibrated = calibration_tool.apply_calibration_torch(logits, parameters).numpy()
    raw_metrics = calibration_tool.metrics(raw_probability, labels.numpy())
    calibrated_metrics = calibration_tool.metrics(calibrated, labels.numpy())
    passed, violations, decision = calibration_tool.dev_gate(raw_metrics, calibrated_metrics)
    canonical = parameters if passed else {"method": "identity", "complexity": 0}
    return canonical, {
        "raw": raw_metrics, "selected_candidate": calibrated_metrics,
        "passed": passed, "violations": violations, "decision": decision,
        "dev_evaluation_count": 1,
    }


def regression_base_arrays(predictions, samples, scheme):
    result = {name: [] for name in ("xy_maha", "xy_logdet", "z_sq", "z_logvar", "size_sq", "size_logvar", "vel_maha", "vel_logdet", "yaw_error", "yaw_kappa")}
    for prediction, sample in zip(predictions, samples):
        target = sample["targets"][scheme]
        positive = target["positive_mask"].bool()
        velocity = positive & target["velocity_mask"].bool()
        if bool(positive.any()):
            covariance = prediction["center_xy_cov"][positive].double()
            difference = (target["center_xy"][positive] - prediction["center_xy_mean"][positive]).double()
            result["xy_maha"].append(torch.einsum("ni,nij,nj->n", difference, torch.linalg.inv(covariance), difference))
            result["xy_logdet"].append(torch.linalg.slogdet(covariance).logabsdet)
            z_logvar = prediction["z_log_var"][positive].double()
            result["z_sq"].append((target["z"][positive] - prediction["z_mean"][positive]).double().pow(2))
            result["z_logvar"].append(z_logvar)
            size_logvar = prediction["log_size_log_var"][positive].double()
            result["size_sq"].append((target["log_size"][positive] - prediction["log_size_mean"][positive]).double().pow(2))
            result["size_logvar"].append(size_logvar)
            result["yaw_error"].append(wrap_angle(target["yaw"][positive] - prediction["yaw_mean"][positive]).double())
            result["yaw_kappa"].append(prediction["yaw_kappa"][positive].double())
        if bool(velocity.any()):
            covariance = prediction["velocity_cov"][velocity].double()
            difference = (target["velocity"][velocity] - prediction["velocity_mean"][velocity]).double()
            result["vel_maha"].append(torch.einsum("ni,nij,nj->n", difference, torch.linalg.inv(covariance), difference))
            result["vel_logdet"].append(torch.linalg.slogdet(covariance).logabsdet)
    return {name: torch.cat(values) if values else torch.empty(0, dtype=torch.double) for name, values in result.items()}


def fit_regression_scales(predictions, samples, scheme):
    arrays = regression_base_arrays(predictions, samples, scheme)
    def full_nll(maha, logdet, scale):
        return (0.5 * (2 * math.log(2 * math.pi) + logdet + 2 * math.log(scale) + maha / scale)).mean().item()
    def diag_nll(square, logvar, scale):
        return (0.5 * (math.log(2 * math.pi) + logvar + math.log(scale) + square / (logvar.exp() * scale))).sum(-1).mean().item()
    result, audit = {}, {}
    for name, objective in (
        ("xy_covariance_scale", lambda scale: full_nll(arrays["xy_maha"], arrays["xy_logdet"], scale)),
        ("z_variance_scale", lambda scale: diag_nll(arrays["z_sq"], arrays["z_logvar"], scale)),
        ("size_variance_scale", lambda scale: diag_nll(arrays["size_sq"], arrays["size_logvar"], scale)),
        ("velocity_covariance_scale", lambda scale: full_nll(arrays["vel_maha"], arrays["vel_logdet"], scale)),
        ("yaw_kappa_scale", lambda scale: (math.log(2 * math.pi) + log_i0(arrays["yaw_kappa"] * scale) - arrays["yaw_kappa"] * scale * torch.cos(arrays["yaw_error"])).mean().item()),
    ):
        values = {str(scale): objective(scale) for scale in SCALES}
        selected = min(SCALES, key=lambda scale: values[str(scale)])
        result[name] = selected
        audit[name] = {"selected": selected, "label_cal_nll_by_scale": values}
    return result, audit


def ece(probability, target, adaptive=False):
    return calibration_tool.calibration_error(np.asarray(probability), np.asarray(target), adaptive=adaptive)[0]


def evaluate(predictions, samples, scheme, calibration):
    logits, labels, _ = classification_arrays(predictions, samples, scheme)
    probability = calibration_tool.apply_calibration_torch(logits, calibration).numpy()
    cls = calibration_tool.metrics(probability, labels.numpy())
    p_exist = probability[:, :-1].sum(1)
    exists = labels.numpy() != 10
    cls["existence_auroc"] = stage3.auroc(p_exist.tolist(), exists.tolist())
    cls["existence_aupr"] = stage3.aupr(p_exist.tolist(), exists.tolist())
    picked = np.clip(probability[np.arange(len(labels)), labels.numpy()], 1e-30, 1.0)
    cls["positive_categorical_nll"] = float(-np.log(picked[exists]).mean()) if exists.any() else None
    cls["background_categorical_nll"] = float(-np.log(picked[~exists]).mean()) if (~exists).any() else None

    xy_nll = []; z_nll = []; size_nll = []; vel_nll = []; yaw_nll = []
    maha_all = []; trace = []; determinant = []; max_eigenvalue = []; errors = []; uncertainty = []; axis_angle = []
    positive_count = background_count = ignore_count = unique_gt = gt_count = 0
    scene_rows = {}
    xy_scale = float(calibration.get("xy_covariance_scale", 1.0))
    z_scale = float(calibration.get("z_variance_scale", 1.0))
    size_scale = float(calibration.get("size_variance_scale", 1.0))
    vel_scale = float(calibration.get("velocity_covariance_scale", 1.0))
    yaw_scale = float(calibration.get("yaw_kappa_scale", 1.0))
    proposal_offset = 0
    for prediction, sample in zip(predictions, samples):
        target = sample["targets"][scheme]
        valid = target["classification_mask"].bool()
        positive = target["positive_mask"].bool()
        velocity = positive & target["velocity_mask"].bool()
        count = int(valid.sum())
        local_probability = probability[proposal_offset:proposal_offset + count]
        local_labels = labels.numpy()[proposal_offset:proposal_offset + count]
        proposal_offset += count
        positive_count += int(positive.sum()); ignore_count += int(target["ignore_mask"].sum())
        background_count += int((valid & ~positive).sum())
        matched = target["matched_gt_index"][positive]
        unique_gt += len(set(int(value) for value in matched.tolist()))
        gt_count += sample["legacy_gt_count"] if scheme == "legacy" else sample["native_gt_count"]
        local_xy = []
        if bool(positive.any()):
            covariance = prediction["center_xy_cov"][positive].double() * xy_scale
            difference = (target["center_xy"][positive] - prediction["center_xy_mean"][positive]).double()
            inverse = torch.linalg.inv(covariance)
            maha = torch.einsum("ni,nij,nj->n", difference, inverse, difference)
            logdet = torch.linalg.slogdet(covariance).logabsdet
            current_xy = 0.5 * (2 * math.log(2 * math.pi) + logdet + maha)
            xy_nll.extend(current_xy.tolist()); local_xy = current_xy.tolist(); maha_all.extend(maha.tolist())
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            trace.extend(covariance.diagonal(dim1=-2, dim2=-1).sum(-1).tolist())
            determinant.extend(torch.linalg.det(covariance).tolist()); max_eigenvalue.extend(eigenvalues[:, -1].tolist())
            norm = difference.norm(dim=-1); errors.extend(norm.tolist()); uncertainty.extend(eigenvalues[:, -1].tolist())
            valid_error = norm > 1e-8
            if bool(valid_error.any()):
                major = eigenvectors[valid_error, :, -1]
                direction = difference[valid_error] / norm[valid_error, None]
                cosine = (major * direction).sum(-1).abs().clamp(0, 1)
                axis_angle.extend(torch.rad2deg(torch.acos(cosine)).tolist())
            z_logvar = prediction["z_log_var"][positive] + math.log(z_scale)
            z_nll.extend(gaussian_nll_diag(target["z"][positive], prediction["z_mean"][positive], z_logvar).tolist())
            size_logvar = prediction["log_size_log_var"][positive] + math.log(size_scale)
            size_nll.extend(gaussian_nll_diag(target["log_size"][positive], prediction["log_size_mean"][positive], size_logvar).tolist())
            kappa = prediction["yaw_kappa"][positive] * yaw_scale
            yaw_nll.extend((math.log(2 * math.pi) + log_i0(kappa) - kappa * torch.cos(wrap_angle(target["yaw"][positive] - prediction["yaw_mean"][positive]))).tolist())
        if bool(velocity.any()):
            chol = prediction["velocity_cholesky"] * math.sqrt(vel_scale)
            vel_nll.extend(gaussian_nll_full(target["velocity"], prediction["velocity_mean"], chol, velocity).tolist())
        scene = sample["scene_token"]
        scene_rows.setdefault(scene, []).append({
            "existence_nll": float(-(exists[proposal_offset-count:proposal_offset] * np.log(np.clip(p_exist[proposal_offset-count:proposal_offset], 1e-30, 1)) + (~exists[proposal_offset-count:proposal_offset]) * np.log(np.clip(1-p_exist[proposal_offset-count:proposal_offset], 1e-30, 1))).mean()),
            "brier": float(np.square(local_probability - np.eye(11)[local_labels]).sum(1).mean()),
            "ece": ece(local_probability.max(1), local_probability.argmax(1) == local_labels),
            "adaptive_ece": ece(local_probability.max(1), local_probability.argmax(1) == local_labels, adaptive=True),
            "conditional_class_nll": float(-np.log(np.clip(local_probability[local_labels != 10, :-1] / np.clip(local_probability[local_labels != 10, :-1].sum(1, keepdims=True), 1e-30, None), 1e-30, 1))[np.arange(int((local_labels != 10).sum())), local_labels[local_labels != 10]].mean()) if (local_labels != 10).any() else None,
            "xy_nll": float(np.mean(local_xy)) if local_xy else None,
            "coverage_error": float(np.mean([abs(np.mean(np.asarray(maha.tolist()) <= threshold) - int(name)/100) for name, threshold in CHI2.items()])) if bool(positive.any()) else None,
            "spearman": stage3.spearman(eigenvalues[:, -1].tolist(), norm.tolist()) if bool(positive.any()) else None,
            "recall": len(set(int(value) for value in matched.tolist())) / max(sample["legacy_gt_count"] if scheme == "legacy" else sample["native_gt_count"], 1),
        })
    coverage = {name: float(np.mean(np.asarray(maha_all) <= threshold)) if maha_all else None for name, threshold in CHI2.items()}
    scene_metrics = {}
    for scene, rows in scene_rows.items():
        scene_metrics[scene] = {
            name: float(np.mean([row[name] for row in rows if row[name] is not None])) if any(row[name] is not None for row in rows) else None
            for name in rows[0]
        }
    return {
        "classification": cls,
        "regression": {
            "xy_nll": float(np.mean(xy_nll)) if xy_nll else None,
            "z_nll": float(np.mean(z_nll)) if z_nll else None,
            "log_size_nll": float(np.mean(size_nll)) if size_nll else None,
            "velocity_nll": float(np.mean(vel_nll)) if vel_nll else None,
            "yaw_von_mises_nll": float(np.mean(yaw_nll)) if yaw_nll else None,
            "ellipse_coverage": coverage,
            "ellipse_coverage_error_mean": float(np.mean([abs(coverage[name] - int(name)/100) for name in CHI2])) if maha_all else None,
            "covariance_trace": float(np.mean(trace)) if trace else None,
            "covariance_determinant": float(np.mean(determinant)) if determinant else None,
            "covariance_max_eigenvalue": float(np.mean(max_eigenvalue)) if max_eigenvalue else None,
            "sharpness": float(np.mean(trace)) if trace else None,
            "error_uncertainty_spearman": stage3.spearman(uncertainty, errors),
            "principal_axis_error_angle_degrees": float(np.mean(axis_angle)) if axis_angle else None,
            "spd_and_finite": bool(all(np.isfinite(value) for value in xy_nll + trace + determinant) and all(value > 0 for value in determinant)),
        },
        "labels": {
            "proposal_count": len(samples) * 200,
            "positive_count": positive_count, "background_count": background_count, "ignore_count": ignore_count,
            "positive_rate": positive_count / (len(samples) * 200),
            "background_rate": background_count / (len(samples) * 200),
            "ignore_rate": ignore_count / (len(samples) * 200),
            "unique_matched_gt": unique_gt, "gt_count": gt_count,
            "unique_proposal_recall": unique_gt / gt_count if gt_count else None,
            "duplicate_positive_ratio": 1.0 - unique_gt / positive_count if positive_count else None,
        },
        "probability_scale": {
            "mean_p_exist": float(p_exist.mean()),
            "positive_mean_p_exist": float(p_exist[exists].mean()) if exists.any() else None,
            "background_mean_p_exist": float(p_exist[~exists].mean()) if (~exists).any() else None,
        },
        "denominators": {
            "scenes": len(scene_metrics), "samples": len(samples),
            "effective_proposals": len(labels), "positive_proposals": positive_count,
        },
        "scene_metrics": scene_metrics,
    }


def bootstrap_delta(left, right, metric, resamples=1000, seed=20260708):
    scenes = sorted(set(left["scene_metrics"]) & set(right["scene_metrics"]))
    pairs = [(left["scene_metrics"][scene][metric], right["scene_metrics"][scene][metric]) for scene in scenes]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    values = np.asarray([b - a for a, b in pairs], dtype=np.float64)
    if not len(values):
        return {"estimate": None, "ci95": [None, None], "effective_scenes": 0}
    draws = np.random.default_rng(seed).integers(0, len(values), size=(resamples, len(values)))
    distribution = values[draws].mean(1)
    return {
        "estimate": float(values.mean()),
        "ci95": [float(value) for value in np.quantile(distribution, [0.025, 0.975])],
        "effective_scenes": len(values), "resamples": resamples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True, choices=sorted(BRANCHES))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if (args.iterations, args.batch_size, args.lr) != (2000, 4, 1e-3):
        raise ValueError("preregistered optimizer controls changed")
    random.seed(20260708); np.random.seed(20260708); torch.manual_seed(20260708)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    caches = {
        split: torch.load(OUT / "work" / f"{args.branch}_{split}.pt", map_location="cpu", weights_only=False)["samples"]
        for split in ("pilot", "label_cal", "dev")
    }
    if [len(caches[name]) for name in ("pilot", "label_cal", "dev")] != [2012, 2814, 2808]:
        raise ValueError("cache sample count mismatch")
    in_channels = caches["pilot"][0]["features"].shape[-1]
    initial_model = adapter(in_channels)
    init_state = copy.deepcopy(initial_model.state_dict())
    init_sha = state_sha256(init_state)
    init_path = OUT / "initializations" / f"{args.branch}_structured_feature_k1.pt"
    init_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": init_state, "sha256": init_sha, "seed": 20260708, "branch": args.branch}, init_path)
    results = {"arms": {}}
    trained = {}
    # Establish the CUDA context before resetting counters.  This avoids the
    # driver's "Invalid device argument" on a fresh, single-visible-GPU process.
    torch.empty(0, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    for arm, scheme in ARM_SCHEME.items():
        smoke = smoke_overfit(caches["pilot"], scheme, init_state, device, args.lr)
        model, training = train_arm(
            caches["pilot"], scheme, init_state, device, args.iterations,
            args.batch_size, args.lr, args.branch, arm,
        )
        trained[arm] = model
        results["arms"][arm] = {"scheme": scheme, "smoke": smoke, "training": training}
    if len({item["training"]["initialization_sha256"] for item in results["arms"].values()}) != 1:
        raise ValueError("A/B/C initialization mismatch")

    metrics = {}; calibrators = {}; bootstrap = {}
    for arm, scheme in ARM_SCHEME.items():
        model = trained[arm]
        label_prediction = predict(model, caches["label_cal"], device)
        selected, oof = fit_oof_calibrator(label_prediction, caches["label_cal"], scheme, device)
        dev_prediction = predict(model, caches["dev"], device)
        canonical, dev_gate = dev_calibration_gate(selected, dev_prediction, caches["dev"], scheme)
        regression, regression_audit = fit_regression_scales(label_prediction, caches["label_cal"], "hungarian" if scheme == "hungarian_ignore" else scheme)
        canonical = {**canonical, **regression}
        calibrators[arm] = {
            "selection_split": "stage3_label_cal_scene_level_5_fold_oof",
            "dev_used_for_selection": False,
            "oof": oof, "dev_transfer_gate": dev_gate,
            "canonical": canonical, "regression_calibration": regression_audit,
        }
        metrics[arm] = {}
        for reference in SCHEMES:
            metrics[arm][reference] = {
                "raw": evaluate(dev_prediction, caches["dev"], reference, {"method": "identity"}),
                "calibrated": evaluate(dev_prediction, caches["dev"], reference, canonical),
            }
        del label_prediction, dev_prediction

    metric_names = (
        "existence_nll", "brier", "ece", "adaptive_ece", "conditional_class_nll",
        "xy_nll", "coverage_error", "spearman", "recall",
    )
    for reference in ("hungarian", "hungarian_ignore"):
        bootstrap[reference] = {}
        baseline = metrics["A_legacy_nearest_2m"][reference]["calibrated"]
        for arm in ("B_hungarian_1to1", "C_hungarian_1to1_ignore_2to4m"):
            candidate = metrics[arm][reference]["calibrated"]
            bootstrap[reference][arm] = {
                name: bootstrap_delta(baseline, candidate, name)
                for name in metric_names
            }

    config, checkpoint = BRANCHES[args.branch]
    manifest = {
        "stage": "stage3d_b_structured_pilot_branch",
        "branch": args.branch, "seed": 20260708,
        "variant": "structured", "classifier_mode": "feature_k1",
        "iterations": 2000, "batch_size": 4, "learning_rate": 1e-3, "scheduler": None,
        "initialization": str(init_path), "initialization_sha256": init_sha,
        "detector_checkpoint": checkpoint, "detector_checkpoint_sha256": sha256(checkpoint),
        "config": config, "config_sha256": sha256(config),
        "data_order": "deterministic cyclic order begin=(iteration*batch_size)%2012",
        "official_val_used": False, "dev_early_stopping": False,
        "gpu": {
            "visible_index": args.device, "name": torch.cuda.get_device_name(args.device),
            "uuid": args.gpu_uuid,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        },
        **results,
    }
    strict_dump(OUT / "runs" / args.branch / "group_manifest.json", manifest)
    strict_dump(OUT / "calibrators" / f"{args.branch}.json", calibrators)
    strict_dump(OUT / "evaluation" / f"{args.branch}.json", metrics)
    strict_dump(OUT / "bootstrap" / f"{args.branch}.json", bootstrap)
    print(json.dumps({"branch": args.branch, "status": "complete", "arms": list(ARM_SCHEME)}, indent=2))


if __name__ == "__main__":
    main()
