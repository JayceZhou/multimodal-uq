"""Target-level structured uncertainty utilities for Stage 3.

This module is deliberately independent from the Stage 4/5 reliability and
conflict baseline. It consumes pre-NMS class logits and deterministic box means,
then adds only K+1 existence/class probabilities and structured covariance
parameters.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F


NUSCENES_DETECTION_CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)


@dataclass(frozen=True)
class StructuredDetectionDistributionSchema:
    version: str = "stage3_target_uq_v1"
    class_count: int = 10
    background_index: int = 10
    class_probability_source: str = "pre_nms_logits"
    score_is_probability: bool = False


def _finite(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value


def require_pre_nms_logits(source: str) -> None:
    if source != "pre_nms_logits":
        raise ValueError("Stage 3 class probabilities must come from pre-NMS logits")


def softplus_positive(value: torch.Tensor, eps: float = 1e-4, max_value: float = 1e4) -> torch.Tensor:
    return F.softplus(value).clamp_max(max_value) + eps


def wrap_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def angular_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return wrap_angle(pred - target).abs()


def log_i0(value: torch.Tensor) -> torch.Tensor:
    """Numerically stable log modified Bessel I0."""
    x = value.abs()
    special = getattr(torch, "special", None)
    if special is not None and hasattr(special, "i0e"):
        return torch.log(special.i0e(x).clamp_min(1e-30)) + x

    y_small = (x / 3.75) ** 2
    small = 1.0 + y_small * (
        3.5156229
        + y_small * (
            3.0899424
            + y_small
            * (1.2067492 + y_small * (0.2659732 + y_small * (0.0360768 + y_small * 0.0045813)))
        )
    )
    y_large = 3.75 / x.clamp_min(1e-12)
    large_poly = (
        0.39894228
        + y_large
        * (
            0.01328592
            + y_large
            * (
                0.00225319
                + y_large
                * (
                    -0.00157565
                    + y_large
                    * (
                        0.00916281
                        + y_large
                        * (
                            -0.02057706
                            + y_large
                            * (0.02635537 + y_large * (-0.01647633 + y_large * 0.00392377))
                        )
                    )
                )
            )
        )
    )
    large = x + torch.log(large_poly.clamp_min(1e-30)) - 0.5 * torch.log(x.clamp_min(1e-12))
    return torch.where(x < 3.75, torch.log(small.clamp_min(1e-30)), large)


def kplus1_probabilities(
    class_logits: torch.Tensor,
    background_logit: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    require_pre_nms_logits("pre_nms_logits")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if background_logit is None:
        background_logit = class_logits.new_zeros(*class_logits.shape[:-1], 1)
    if background_logit.shape[-1] != 1:
        raise ValueError("background_logit must have last dimension 1")
    logits = torch.cat((class_logits, background_logit), dim=-1) / float(temperature)
    probs = logits.softmax(dim=-1)
    p_background = probs[..., -1:]
    p_exist = 1.0 - p_background
    p_class = probs[..., :-1] / p_exist.clamp_min(1e-8)
    return {
        "logits_k1": _finite("logits_k1", logits),
        "p_all": _finite("p_all", probs),
        "p_exist": _finite("p_exist", p_exist.squeeze(-1)),
        "p_class_given_exist": _finite("p_class_given_exist", p_class),
    }


def cholesky_from_params(params: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if params.shape[-1] != 3:
        raise ValueError("2D Cholesky params must have last dimension 3")
    l11 = softplus_positive(params[..., 0], eps=eps)
    l21 = params[..., 1]
    l22 = softplus_positive(params[..., 2], eps=eps)
    zeros = torch.zeros_like(l11)
    row0 = torch.stack((l11, zeros), dim=-1)
    row1 = torch.stack((l21, l22), dim=-1)
    return _finite("cholesky", torch.stack((row0, row1), dim=-2))


def covariance_from_cholesky(cholesky: torch.Tensor) -> torch.Tensor:
    cov = cholesky @ cholesky.transpose(-1, -2)
    return _finite("covariance", cov)


def gaussian_nll_diag(
    value: torch.Tensor,
    mean: torch.Tensor,
    log_var: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    log_var = log_var.clamp(-20.0, 20.0)
    nll = 0.5 * (math.log(2.0 * math.pi) + log_var + (value - mean).pow(2) * torch.exp(-log_var))
    nll = nll.sum(dim=-1)
    if mask is not None:
        nll = nll[mask]
    return _finite("gaussian_nll_diag", nll)


def gaussian_nll_full(
    value: torch.Tensor,
    mean: torch.Tensor,
    cholesky: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    diff = (value - mean).unsqueeze(-1)
    solve = torch.cholesky_solve(diff, cholesky)
    maha = (diff.transpose(-1, -2) @ solve).squeeze(-1).squeeze(-1)
    log_det = 2.0 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(dim=-1)
    nll = 0.5 * (2.0 * math.log(2.0 * math.pi) + log_det + maha)
    if mask is not None:
        nll = nll[mask]
    return _finite("gaussian_nll_full", nll)


def von_mises_nll(
    target_yaw: torch.Tensor,
    mean_yaw: torch.Tensor,
    kappa_raw: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    kappa = softplus_positive(kappa_raw, eps=0.0)
    nll = math.log(2.0 * math.pi) + log_i0(kappa) - kappa * torch.cos(wrap_angle(target_yaw - mean_yaw))
    if mask is not None:
        nll = nll[mask]
    return _finite("von_mises_nll", nll)


def _mean_or_zero(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return value.new_zeros(())
    return value.mean()


class StructuredTargetUQAdapter(nn.Module):
    """Small adapter from pre-NMS logits and proposal features to distribution params."""

    VARIANTS = {"deterministic", "fixed", "diagonal", "structured"}
    CLASSIFIER_MODES = {"heatmap", "feature_k1", "separated"}

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: Optional[int] = None,
        hidden_channels: int = 32,
        variant: str = "structured",
        fixed_log_var: float = 0.0,
        classifier_mode: str = "heatmap",
    ) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unsupported target UQ variant: {variant}")
        if classifier_mode not in self.CLASSIFIER_MODES:
            raise ValueError(f"unsupported classifier mode: {classifier_mode}")
        if classifier_mode != "heatmap" and in_channels is None:
            raise ValueError("feature classifier modes require in_channels")
        self.num_classes = int(num_classes)
        self.variant = variant
        self.fixed_log_var = float(fixed_log_var)
        self.classifier_mode = classifier_mode
        self.background_logit = nn.Parameter(torch.zeros(1))
        output_dim = 1 + 3 + 1 + 3 + 1 + 3
        self.net = None
        self.class_net = None
        self.exist_net = None
        if in_channels is not None:
            self.net = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, output_dim),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            if classifier_mode == "feature_k1":
                self.class_net = nn.Linear(in_channels, self.num_classes + 1)
            elif classifier_mode == "separated":
                self.class_net = nn.Linear(in_channels, self.num_classes)
                self.exist_net = nn.Linear(in_channels, 1)
        else:
            self.params = nn.Parameter(torch.zeros(output_dim))

    def _raw_params(self, class_logits: torch.Tensor, features: Optional[torch.Tensor]) -> torch.Tensor:
        shape = class_logits.shape[:-1]
        if self.net is not None:
            if features is None:
                raise ValueError("features are required when in_channels is set")
            raw = self.net(features)
        else:
            raw = self.params.view(*([1] * len(shape)), -1).expand(*shape, -1)
        return _finite("target_uq_raw_params", raw)

    def forward(
        self,
        class_logits: torch.Tensor,
        means: Dict[str, torch.Tensor],
        features: Optional[torch.Tensor] = None,
        source: str = "pre_nms_logits",
    ) -> Dict[str, torch.Tensor]:
        require_pre_nms_logits(source)
        if class_logits.shape[-1] != self.num_classes:
            raise ValueError("class_logits last dimension must equal num_classes")
        raw = self._raw_params(class_logits, features)
        if self.classifier_mode == "heatmap":
            bg = (self.background_logit + raw[..., 0:1]).expand(*class_logits.shape[:-1], 1)
            probs = kplus1_probabilities(class_logits, bg)
            class_probability_source = "pre_nms_heatmap_logits"
        elif self.classifier_mode == "feature_k1":
            logits_k1 = self.class_net(features)
            p_all = logits_k1.softmax(dim=-1)
            p_exist = 1.0 - p_all[..., -1]
            probs = {
                "logits_k1": _finite("logits_k1", logits_k1),
                "p_all": _finite("p_all", p_all),
                "p_exist": _finite("p_exist", p_exist),
                "p_class_given_exist": _finite("p_class_given_exist", p_all[..., :-1] / p_exist[..., None].clamp_min(1e-8)),
            }
            class_probability_source = "pre_nms_query_feature_k1_classifier"
        else:
            class_cond_logits = self.class_net(features)
            exist_logit = self.exist_net(features).squeeze(-1)
            p_exist = torch.sigmoid(exist_logit)
            p_class = class_cond_logits.softmax(dim=-1)
            p_all = torch.cat((p_exist[..., None] * p_class, 1.0 - p_exist[..., None]), dim=-1)
            probs = {
                "logits_k1": _finite("logits_k1", p_all.clamp_min(1e-12).log()),
                "p_all": _finite("p_all", p_all),
                "p_exist": _finite("p_exist", p_exist),
                "p_class_given_exist": _finite("p_class_given_exist", p_class),
            }
            class_probability_source = "pre_nms_query_feature_separated_classifier"

        xy_raw, z_raw, size_raw, yaw_raw, vel_raw = raw[..., 1:4], raw[..., 4:5], raw[..., 5:8], raw[..., 8], raw[..., 9:12]
        fixed = class_logits.new_full(class_logits.shape[:-1] + (1,), self.fixed_log_var)
        if self.variant in ("deterministic", "fixed"):
            xy_chol = torch.diag_embed(torch.exp(0.5 * fixed.expand(*class_logits.shape[:-1], 2)))
            vel_chol = torch.diag_embed(torch.exp(0.5 * fixed.expand(*class_logits.shape[:-1], 2)))
            z_log_var = fixed
            size_log_var = fixed.expand(*class_logits.shape[:-1], 3)
            kappa_raw = torch.ones_like(yaw_raw)
        elif self.variant == "diagonal":
            xy_chol = torch.diag_embed(softplus_positive(xy_raw[..., [0, 2]]).sqrt())
            vel_chol = torch.diag_embed(softplus_positive(vel_raw[..., [0, 2]]).sqrt())
            z_log_var = torch.log(softplus_positive(z_raw))
            size_log_var = torch.log(softplus_positive(size_raw))
            kappa_raw = yaw_raw
        else:
            xy_chol = cholesky_from_params(xy_raw)
            vel_chol = cholesky_from_params(vel_raw)
            z_log_var = torch.log(softplus_positive(z_raw))
            size_log_var = torch.log(softplus_positive(size_raw))
            kappa_raw = yaw_raw

        output = {
            **probs,
            "center_xy_mean": means["center_xy"],
            "center_xy_cholesky": xy_chol,
            "center_xy_cov": covariance_from_cholesky(xy_chol),
            "z_mean": means["z"],
            "z_log_var": z_log_var,
            "log_size_mean": means["log_size"],
            "log_size_log_var": size_log_var,
            "yaw_mean": means["yaw"],
            "yaw_kappa_raw": kappa_raw,
            "yaw_kappa": softplus_positive(kappa_raw, eps=0.0),
            "velocity_mean": means["velocity"],
            "velocity_cholesky": vel_chol,
            "velocity_cov": covariance_from_cholesky(vel_chol),
            "variant": self.variant,
            "classifier_mode": self.classifier_mode,
            "class_probability_source": class_probability_source,
        }
        for key, value in output.items():
            if torch.is_tensor(value):
                _finite(key, value)
        return output


def target_uq_loss(
    distribution: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    lambdas: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    weights = {
        "cls": 1.0,
        "xy": 1.0,
        "z": 1.0,
        "size": 1.0,
        "yaw": 1.0,
        "vel": 1.0,
    }
    if lambdas:
        weights.update(lambdas)
    positive = targets["positive_mask"].bool()
    velocity_mask = positive & targets.get("velocity_mask", torch.ones_like(positive)).bool()
    logits = distribution["logits_k1"].reshape(-1, distribution["logits_k1"].shape[-1])
    labels = targets["class_target"].reshape(-1).long()
    cls = F.cross_entropy(logits, labels)
    xy = _mean_or_zero(
        gaussian_nll_full(
            targets["center_xy"],
            distribution["center_xy_mean"],
            distribution["center_xy_cholesky"],
            positive,
        )
    )
    z = _mean_or_zero(
        gaussian_nll_diag(targets["z"], distribution["z_mean"], distribution["z_log_var"], positive)
    )
    size = _mean_or_zero(
        gaussian_nll_diag(
            targets["log_size"],
            distribution["log_size_mean"],
            distribution["log_size_log_var"],
            positive,
        )
    )
    yaw = _mean_or_zero(
        von_mises_nll(targets["yaw"], distribution["yaw_mean"], distribution["yaw_kappa_raw"], positive)
    )
    vel = _mean_or_zero(
        gaussian_nll_full(
            targets["velocity"],
            distribution["velocity_mean"],
            distribution["velocity_cholesky"],
            velocity_mask,
        )
    )
    losses = {
        "loss/stage3_target_uq/cls": cls * weights["cls"],
        "loss/stage3_target_uq/xy": xy * weights["xy"],
        "loss/stage3_target_uq/z": z * weights["z"],
        "loss/stage3_target_uq/size": size * weights["size"],
        "loss/stage3_target_uq/yaw": yaw * weights["yaw"],
        "loss/stage3_target_uq/vel": vel * weights["vel"],
    }
    losses["loss/stage3_target_uq/total"] = sum(losses.values())
    losses["stats/stage3_target_uq/positive_count"] = positive.float().sum().detach()
    losses["stats/stage3_target_uq/velocity_count"] = velocity_mask.float().sum().detach()
    return losses


def _tensor_list(value: torch.Tensor) -> List:
    return value.detach().cpu().tolist()


def distribution_records(
    distribution: Dict[str, torch.Tensor],
    sample_tokens: Iterable[str],
    branch: str,
    proposal_ids: Optional[torch.Tensor] = None,
    post_nms_ids: Optional[torch.Tensor] = None,
) -> List[Dict]:
    schema = StructuredDetectionDistributionSchema(
        class_count=distribution["p_class_given_exist"].shape[-1],
        background_index=distribution["p_class_given_exist"].shape[-1],
    )
    tokens = list(sample_tokens)
    p_class = distribution["p_class_given_exist"].detach().cpu()
    if len(tokens) != p_class.shape[0]:
        raise ValueError("sample token count must match batch")
    if proposal_ids is None:
        proposal_ids = torch.arange(p_class.shape[1]).view(1, -1).expand(p_class.shape[0], -1)
    if post_nms_ids is None:
        post_nms_ids = torch.full_like(proposal_ids, -1)
    records = []
    for batch_index, token in enumerate(tokens):
        for item_index in range(p_class.shape[1]):
            records.append(
                {
                    "schema_version": schema.version,
                    "sample_token": token,
                    "branch": branch,
                    "pre_nms_proposal_id": int(proposal_ids[batch_index, item_index]),
                    "post_nms_detection_id": int(post_nms_ids[batch_index, item_index]),
                    "class_probability_source": schema.class_probability_source,
                    "score_is_probability": schema.score_is_probability,
                    "p_exist": float(distribution["p_exist"][batch_index, item_index].detach().cpu()),
                    "p_class_given_exist": _tensor_list(distribution["p_class_given_exist"][batch_index, item_index]),
                    "center_mean": _tensor_list(
                        torch.cat(
                            (
                                distribution["center_xy_mean"][batch_index, item_index],
                                distribution["z_mean"][batch_index, item_index],
                            )
                        )
                    ),
                    "Sigma_xy": _tensor_list(distribution["center_xy_cov"][batch_index, item_index]),
                    "z_variance": float(torch.exp(distribution["z_log_var"][batch_index, item_index]).detach().cpu()),
                    "log_size_mean": _tensor_list(distribution["log_size_mean"][batch_index, item_index]),
                    "log_size_diag_variance": _tensor_list(
                        torch.exp(distribution["log_size_log_var"][batch_index, item_index])
                    ),
                    "yaw_mean": float(distribution["yaw_mean"][batch_index, item_index].detach().cpu()),
                    "yaw_kappa": float(distribution["yaw_kappa"][batch_index, item_index].detach().cpu()),
                    "velocity_mean": _tensor_list(distribution["velocity_mean"][batch_index, item_index]),
                    "Sigma_v": _tensor_list(distribution["velocity_cov"][batch_index, item_index]),
                    "valid_fields": {
                        "velocity": True,
                        "full_xy_covariance": distribution["variant"] == "structured",
                        "full_velocity_covariance": distribution["variant"] == "structured",
                    },
                    "match": {"status": "unmatched", "gt_index": None},
                }
            )
    return records


def assert_post_nms_mapping_complete(records: Iterable[Dict]) -> None:
    for record in records:
        if "pre_nms_proposal_id" not in record or "post_nms_detection_id" not in record:
            raise ValueError("pre-NMS to post-NMS mapping fields are required")
        require_pre_nms_logits(record.get("class_probability_source"))


def _last_transfusion_prediction(preds_dicts: Any) -> Dict[str, torch.Tensor]:
    """Return the final TransFusion prediction dict from BEVFusion's nested output."""
    value = preds_dicts
    while isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("empty TransFusion prediction container")
        value = value[0]
    if not isinstance(value, dict):
        raise TypeError("expected a TransFusion prediction dict")
    return value


def transfusion_stage3_proposals(
    preds_dicts: Any,
    bbox_coder: Any,
    num_classes: int = 10,
    num_proposals: Optional[int] = None,
    nms_type: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Extract real pre-NMS TransFusion proposals for Stage 3 target-UQ.

    Class probabilities must be computed later from ``class_logits``. The
    decoded post score is returned only to reproduce the post-filter mapping.
    """
    if nms_type is not None:
        raise ValueError("Stage 3A pre/post mapping currently supports nms_type=None only")
    pred = _last_transfusion_prediction(preds_dicts)
    for key in ("heatmap", "center", "height", "dim", "rot", "stage3_query_features"):
        if key not in pred:
            raise KeyError(f"TransFusion prediction is missing {key}")

    features = pred["stage3_query_features"].float()
    proposals = int(num_proposals or features.shape[1])
    class_logits_chw = pred["heatmap"][..., -proposals:].float()
    class_logits = class_logits_chw.permute(0, 2, 1).contiguous()
    if class_logits.shape[-1] != num_classes:
        raise ValueError("pre-NMS class logits last dimension must equal num_classes")

    query_labels = pred.get("stage3_query_labels")
    if query_labels is None:
        query_labels = class_logits.argmax(dim=-1)
    query_labels = query_labels[..., -proposals:].long()
    one_hot = F.one_hot(query_labels, num_classes=num_classes).permute(0, 2, 1).float()
    query_score = pred.get("query_heatmap_score")
    if query_score is None:
        query_score = torch.ones_like(class_logits_chw[:, :1])
    post_score_for_mapping = class_logits_chw.sigmoid() * query_score.float() * one_hot

    center = pred["center"][..., -proposals:].float()
    height = pred["height"][..., -proposals:].float()
    dim = pred["dim"][..., -proposals:].float()
    rot = pred["rot"][..., -proposals:].float()
    vel = pred.get("vel")
    vel = None if vel is None else vel[..., -proposals:].float()

    decoded = bbox_coder.decode(
        post_score_for_mapping.clone(),
        rot.clone(),
        dim.clone(),
        center.clone(),
        height.clone(),
        None if vel is None else vel.clone(),
        filter=False,
    )
    decoded_boxes = torch.stack([item["bboxes"] for item in decoded], dim=0)
    mapping_scores = torch.stack([item["scores"] for item in decoded], dim=0)

    batch_size = class_logits.shape[0]
    device = class_logits.device
    proposal_ids = torch.arange(proposals, device=device).view(1, -1).expand(batch_size, -1)
    post_nms_ids = torch.full((batch_size, proposals), -1, dtype=torch.long, device=device)
    keep = torch.ones((batch_size, proposals), dtype=torch.bool, device=device)
    if bbox_coder.post_center_range is not None:
        post_range = torch.as_tensor(bbox_coder.post_center_range, device=device, dtype=decoded_boxes.dtype)
        keep &= (decoded_boxes[..., :3] >= post_range[:3]).all(dim=-1)
        keep &= (decoded_boxes[..., :3] <= post_range[3:]).all(dim=-1)
    if bbox_coder.score_threshold:
        keep &= mapping_scores > float(bbox_coder.score_threshold)
    for batch_index in range(batch_size):
        count = int(keep[batch_index].sum().item())
        post_nms_ids[batch_index, keep[batch_index]] = torch.arange(count, device=device)

    velocity = decoded_boxes.new_zeros(batch_size, proposals, 2)
    if decoded_boxes.shape[-1] >= 9:
        velocity = decoded_boxes[..., 7:9]
    query_indices = pred.get("stage3_query_indices")
    if query_indices is None:
        query_indices = proposal_ids
    else:
        query_indices = query_indices[..., -proposals:].long()
    decoder_query_ids = pred.get("stage3_decoder_query_ids")
    if decoder_query_ids is None:
        decoder_query_ids = proposal_ids
    else:
        decoder_query_ids = decoder_query_ids[..., -proposals:].long()

    return {
        "class_probability_source": "pre_nms_logits",
        "class_logits": _finite("stage3_class_logits", class_logits),
        "features": _finite("stage3_query_features", features),
        "proposal_ids": proposal_ids,
        "query_indices": query_indices,
        "decoder_query_ids": decoder_query_ids,
        "query_labels": query_labels,
        "post_nms_ids": post_nms_ids,
        "post_score_for_mapping": _finite("post_score_for_mapping", mapping_scores),
        "decoded_boxes": _finite("decoded_boxes", decoded_boxes),
        "means": {
            "center_xy": decoded_boxes[..., 0:2],
            "z": decoded_boxes[..., 2:3],
            "log_size": dim.permute(0, 2, 1).contiguous(),
            "yaw": decoded_boxes[..., 6],
            "velocity": velocity,
        },
        "kept_post_nms_mask": keep,
    }


def _gather_proposals(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if value.dim() == 2:
        return value.gather(1, index)
    return value.gather(1, index.unsqueeze(-1).expand(-1, -1, value.shape[-1]))


def centerhead_stage3_proposals(
    preds_dicts: Any,
    head: Any,
    num_classes: int = 10,
    num_proposals: int = 200,
) -> Dict[str, torch.Tensor]:
    """Extract compact pre-NMS CenterHead proposals for Stage 3B.

    CenterHead has dense heatmap cells rather than decoder queries. We use the
    pre-NMS top-K decoded cells as proposals and gather the local shared head
    feature at each cell. Post scores remain mapping/filter metadata only.
    """
    task_preds = [item[0] if isinstance(item, (list, tuple)) else item for item in preds_dicts]
    if not task_preds:
        raise ValueError("empty CenterHead prediction container")
    for pred in task_preds:
        for key in ("heatmap", "reg", "height", "dim", "rot", "stage3_shared_features"):
            if key not in pred:
                raise KeyError(f"CenterHead prediction is missing {key}")

    coder = head.bbox_coder
    batch_size = task_preds[0]["heatmap"].shape[0]
    device = task_preds[0]["heatmap"].device
    task_offsets, offset = [], 0
    for count in head.num_classes:
        task_offsets.append(offset)
        offset += int(count)
    if offset != num_classes:
        raise ValueError("CenterHead task classes do not match num_classes")

    logits_list, features_list, boxes_list, log_size_list = [], [], [], []
    score_list, keep_list, source_index_list, label_list = [], [], [], []
    for task_id, pred in enumerate(task_preds):
        heat_logits = pred["heatmap"].float()
        heat = heat_logits.sigmoid()
        per_task_k = min(int(coder.max_num), heat.shape[-1] * heat.shape[-2])
        scores, inds, clses, ys, xs = coder._topk(heat, K=per_task_k)

        task_logits = coder._transpose_and_gather_feat(heat_logits, inds)
        global_logits = heat_logits.new_full((batch_size, per_task_k, num_classes), -20.0)
        start = task_offsets[task_id]
        global_logits[..., start : start + task_logits.shape[-1]] = task_logits
        logits_list.append(global_logits)

        shared_features = pred["stage3_shared_features"].float()
        features_list.append(coder._transpose_and_gather_feat(shared_features, inds))

        reg = coder._transpose_and_gather_feat(pred["reg"].float(), inds).view(batch_size, per_task_k, 2)
        height = coder._transpose_and_gather_feat(pred["height"].float(), inds).view(batch_size, per_task_k, 1)
        dim_raw = coder._transpose_and_gather_feat(pred["dim"].float(), inds).view(batch_size, per_task_k, 3)
        dim = torch.exp(dim_raw) if getattr(head, "norm_bbox", False) else dim_raw
        rot_raw = pred["rot"].float()
        rot_s = coder._transpose_and_gather_feat(rot_raw[:, 0:1], inds).view(batch_size, per_task_k, 1)
        rot_c = coder._transpose_and_gather_feat(rot_raw[:, 1:2], inds).view(batch_size, per_task_k, 1)
        yaw = torch.atan2(rot_s, rot_c)
        xs = xs.view(batch_size, per_task_k, 1) + reg[..., 0:1]
        ys = ys.view(batch_size, per_task_k, 1) + reg[..., 1:2]
        xs = xs * coder.out_size_factor * coder.voxel_size[0] + coder.pc_range[0]
        ys = ys * coder.out_size_factor * coder.voxel_size[1] + coder.pc_range[1]
        z = height - dim[..., 2:3] * 0.5
        if "vel" in pred:
            vel = coder._transpose_and_gather_feat(pred["vel"].float(), inds).view(batch_size, per_task_k, 2)
        else:
            vel = heat_logits.new_zeros(batch_size, per_task_k, 2)
        boxes = torch.cat((xs, ys, z, dim, yaw, vel), dim=-1)
        boxes_list.append(boxes)
        log_size_list.append(dim.clamp_min(1e-4).log())

        keep = torch.ones((batch_size, per_task_k), dtype=torch.bool, device=device)
        post_range = getattr(coder, "post_center_range", None)
        if post_range is not None:
            post_range = torch.as_tensor(post_range, device=device, dtype=boxes.dtype)
            keep &= (boxes[..., :3] >= post_range[:3]).all(dim=-1)
            keep &= (boxes[..., :3] <= post_range[3:]).all(dim=-1)
        if getattr(coder, "score_threshold", None):
            keep &= scores > float(coder.score_threshold)
        keep_list.append(keep)
        score_list.append(scores)
        source_index_list.append(inds + task_id * heat.shape[-1] * heat.shape[-2])
        label_list.append(clses.long() + task_offsets[task_id])

    class_logits_all = torch.cat(logits_list, dim=1)
    features_all = torch.cat(features_list, dim=1)
    boxes_all = torch.cat(boxes_list, dim=1)
    log_size_all = torch.cat(log_size_list, dim=1)
    scores_all = torch.cat(score_list, dim=1)
    keep_all = torch.cat(keep_list, dim=1)
    source_indices_all = torch.cat(source_index_list, dim=1)
    labels_all = torch.cat(label_list, dim=1)

    proposals = min(int(num_proposals), scores_all.shape[1])
    top_scores, top_index = scores_all.topk(proposals, dim=1)
    class_logits = _gather_proposals(class_logits_all, top_index)
    features = _gather_proposals(features_all, top_index)
    decoded_boxes = _gather_proposals(boxes_all, top_index)
    log_size = _gather_proposals(log_size_all, top_index)
    keep = _gather_proposals(keep_all.long(), top_index).bool()
    query_indices = _gather_proposals(source_indices_all, top_index).long()
    query_labels = _gather_proposals(labels_all, top_index).long()

    proposal_ids = torch.arange(proposals, device=device).view(1, -1).expand(batch_size, -1)
    post_nms_ids = torch.full((batch_size, proposals), -1, dtype=torch.long, device=device)
    for batch_index in range(batch_size):
        count = int(keep[batch_index].sum().item())
        post_nms_ids[batch_index, keep[batch_index]] = torch.arange(count, device=device)

    velocity = decoded_boxes.new_zeros(batch_size, proposals, 2)
    if decoded_boxes.shape[-1] >= 9:
        velocity = decoded_boxes[..., 7:9]
    return {
        "class_probability_source": "pre_nms_logits",
        "class_logits": _finite("stage3_class_logits", class_logits),
        "features": _finite("stage3_query_features", features),
        "proposal_ids": proposal_ids,
        "query_indices": query_indices,
        "query_labels": query_labels,
        "post_nms_ids": post_nms_ids,
        "post_score_for_mapping": _finite("post_score_for_mapping", top_scores),
        "decoded_boxes": _finite("decoded_boxes", decoded_boxes),
        "means": {
            "center_xy": decoded_boxes[..., 0:2],
            "z": decoded_boxes[..., 2:3],
            "log_size": log_size,
            "yaw": decoded_boxes[..., 6],
            "velocity": velocity,
        },
        "kept_post_nms_mask": keep,
    }


def match_stage3_proposals_to_gt(
    decoded_boxes: torch.Tensor,
    gt_bboxes_3d: Iterable[Any],
    gt_labels_3d: Iterable[torch.Tensor],
    num_classes: int = 10,
    positive_distance_threshold: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Build K+1 classification and positive-only regression targets."""
    batch_size, proposals = decoded_boxes.shape[:2]
    device = decoded_boxes.device
    class_target = torch.full((batch_size, proposals), num_classes, dtype=torch.long, device=device)
    positive = torch.zeros((batch_size, proposals), dtype=torch.bool, device=device)
    matched_gt = torch.full((batch_size, proposals), -1, dtype=torch.long, device=device)
    target = {
        "center_xy": torch.zeros(batch_size, proposals, 2, device=device, dtype=decoded_boxes.dtype),
        "z": torch.zeros(batch_size, proposals, 1, device=device, dtype=decoded_boxes.dtype),
        "log_size": torch.zeros(batch_size, proposals, 3, device=device, dtype=decoded_boxes.dtype),
        "yaw": torch.zeros(batch_size, proposals, device=device, dtype=decoded_boxes.dtype),
        "velocity": torch.zeros(batch_size, proposals, 2, device=device, dtype=decoded_boxes.dtype),
    }
    velocity_mask = torch.zeros((batch_size, proposals), dtype=torch.bool, device=device)
    distances = torch.full((batch_size, proposals), -1.0, device=device, dtype=decoded_boxes.dtype)

    gt_boxes_list = list(gt_bboxes_3d)
    gt_labels_list = list(gt_labels_3d)
    for batch_index in range(batch_size):
        if batch_index >= len(gt_boxes_list) or gt_boxes_list[batch_index] is None:
            continue
        gt_tensor = gt_boxes_list[batch_index].tensor.to(device=device, dtype=decoded_boxes.dtype)
        labels = gt_labels_list[batch_index].to(device=device).long()
        if gt_tensor.numel() == 0:
            continue
        dists = torch.cdist(decoded_boxes[batch_index, :, :2], gt_tensor[:, :2])
        min_dist, min_index = dists.min(dim=1)
        pos = min_dist <= float(positive_distance_threshold)
        if not bool(pos.any()):
            continue
        positive[batch_index, pos] = True
        matched_gt[batch_index, pos] = min_index[pos]
        distances[batch_index] = min_dist
        selected = gt_tensor[min_index[pos]]
        class_target[batch_index, pos] = labels[min_index[pos]].clamp(0, num_classes - 1)
        target["center_xy"][batch_index, pos] = selected[:, 0:2]
        target["z"][batch_index, pos] = selected[:, 2:3]
        target["log_size"][batch_index, pos] = selected[:, 3:6].clamp_min(1e-4).log()
        target["yaw"][batch_index, pos] = selected[:, 6]
        if gt_tensor.shape[1] >= 9:
            vel = selected[:, 7:9]
            valid = torch.isfinite(vel).all(dim=-1) & (vel.abs() < 100.0).all(dim=-1)
            target["velocity"][batch_index, pos] = torch.where(valid[:, None], vel, torch.zeros_like(vel))
            pos_indices = pos.nonzero(as_tuple=False).squeeze(-1)
            velocity_mask[batch_index, pos_indices[valid]] = True

    output = {
        "class_target": class_target,
        "positive_mask": positive,
        "velocity_mask": velocity_mask,
        "matched_gt_index": matched_gt,
        "match_distance_xy": distances,
        **target,
    }
    for key, value in output.items():
        if torch.is_tensor(value) and value.is_floating_point():
            _finite(key, value)
    return output
