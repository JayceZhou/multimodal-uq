"""Minimal F1 probability head with no detector write path."""

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


QUERY_FEATURE_DIM = 128
CLASS_COUNT = 10
STATE_DIM = 8
BOX_DIM = 9
HIDDEN_DIM = 64
GEOMETRY_INPUT_DIM = QUERY_FEATURE_DIM + CLASS_COUNT + STATE_DIM + 1
DENSITY_GEOMETRY_INPUT_DIM = GEOMETRY_INPUT_DIM + 1
MIN_SCALE = 1e-3
MIN_DF = 2.1
MIN_KAPPA = 1e-4
MAX_KAPPA = 1e4
COMPETITION_TEMPERATURE = 0.25
DUPLICATE_CENTER_METERS = 2.0
COMPETITION_WEIGHT = 0.1


def _require_finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains a non-finite value")


def _as_query_scalar(name: str, value: Tensor, batch: int, queries: int) -> Tensor:
    if value.shape == (batch, queries):
        value = value.unsqueeze(-1)
    if value.shape != (batch, queries, 1):
        raise ValueError(
            f"{name} must have shape {(batch, queries)} or "
            f"{(batch, queries, 1)}, got {tuple(value.shape)}"
        )
    return value


def wrap_angle(value: Tensor) -> Tensor:
    """Map angles to [-pi, pi)."""
    return torch.remainder(value + math.pi, 2.0 * math.pi) - math.pi


def log_i0(value: Tensor) -> Tensor:
    """Stable log(I0(x)) from the exponentially scaled Bessel function."""
    return torch.log(torch.special.i0e(value)) + value.abs()


def state_from_boxes(boxes: Tensor) -> Tensor:
    """Convert decoded [x,y,z,l,w,h,yaw,vx,vy] boxes to the F1 state."""
    if boxes.shape[-1] != BOX_DIM:
        raise ValueError(f"decoded boxes must end in {BOX_DIM}, got {boxes.shape[-1]}")
    if not torch.all(boxes[..., 3:6] > 0):
        raise ValueError("decoded length, width, and height must be positive")
    state = torch.cat(
        (
            boxes[..., 0:3],
            boxes[..., 3:6].log(),
            boxes[..., 7:9],
        ),
        dim=-1,
    )
    _require_finite("linear state", state)
    return state


def calibrated_class_log_probs(logits: Tensor, temperature: Tensor) -> Tensor:
    """Calibrate a detached copy of logits without changing detector scores."""
    logits = logits.detach()
    temperature = torch.as_tensor(
        temperature, dtype=logits.dtype, device=logits.device
    )
    if temperature.ndim == 0:
        pass
    elif temperature.shape != (logits.shape[-1],):
        raise ValueError(
            f"temperature must be scalar or shape {(logits.shape[-1],)}, "
            f"got {tuple(temperature.shape)}"
        )
    if not torch.all(temperature > 0):
        raise ValueError("categorical temperature must be positive")
    result = F.log_softmax(logits / temperature, dim=-1)
    _require_finite("class log probabilities", result)
    return result


class ProbabilisticQueryHead(nn.Module):
    """F1 side-channel head; every detector-derived input is detached."""

    def __init__(self, include_query_density: bool = False) -> None:
        super().__init__()
        self.include_query_density = bool(include_query_density)
        self.existence = nn.Sequential(
            nn.Linear(QUERY_FEATURE_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        geometry_input_dim = (
            DENSITY_GEOMETRY_INPUT_DIM
            if self.include_query_density
            else GEOMETRY_INPUT_DIM
        )
        self.geometry = nn.Sequential(
            nn.Linear(geometry_input_dim, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, STATE_DIM + 2),
        )

    def forward(
        self,
        query_feature: Tensor,
        class_logits: Tensor,
        decoded_boxes: Tensor,
        seed_score: Tensor,
        query_density: Tensor,
    ) -> Dict[str, Tensor]:
        if query_feature.ndim != 3:
            raise ValueError("query_feature must have [batch,query,feature] layout")
        batch, queries, feature_dim = query_feature.shape
        if feature_dim != QUERY_FEATURE_DIM:
            raise ValueError(
                f"query feature dimension must be {QUERY_FEATURE_DIM}, got {feature_dim}"
            )
        if class_logits.shape != (batch, queries, CLASS_COUNT):
            raise ValueError("class_logits shape does not match the frozen schema")
        if decoded_boxes.shape != (batch, queries, BOX_DIM):
            raise ValueError("decoded_boxes shape does not match the frozen schema")

        query_feature = query_feature.detach()
        class_logits = class_logits.detach()
        decoded_boxes = decoded_boxes.detach()
        seed_score = _as_query_scalar(
            "seed_score", seed_score.detach(), batch, queries
        )
        query_density = _as_query_scalar(
            "query_density", query_density.detach(), batch, queries
        )
        mean_state = state_from_boxes(decoded_boxes)

        existence_logit = self.existence(query_feature).squeeze(-1)
        conditioned_values = [
            query_feature,
            class_logits,
            mean_state,
            seed_score,
        ]
        if self.include_query_density:
            conditioned_values.append(query_density)
        conditioned = torch.cat(
            conditioned_values,
            dim=-1,
        )
        raw = self.geometry(conditioned)
        scale = F.softplus(raw[..., :STATE_DIM]) + MIN_SCALE
        degrees_of_freedom = F.softplus(raw[..., STATE_DIM]) + MIN_DF
        yaw_raw = raw[..., STATE_DIM + 1]
        yaw_scale = F.softplus(yaw_raw) + MIN_SCALE
        yaw_kappa = (F.softplus(yaw_raw) + MIN_KAPPA).clamp(max=MAX_KAPPA)

        result = {
            "existence_logit": existence_logit,
            "scale": scale,
            "degrees_of_freedom": degrees_of_freedom,
            "yaw_scale": yaw_scale,
            "yaw_kappa": yaw_kappa,
            "class_logits": class_logits,
            "mean_boxes": decoded_boxes,
            "mean_state": mean_state,
            "yaw_mean": decoded_boxes[..., 6],
        }
        for name, value in result.items():
            _require_finite(name, value)
        return result


def bernoulli_nll(
    logits: Tensor, target: Tensor, reduction: str = "mean"
) -> Tensor:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"unsupported reduction: {reduction}")
    return F.binary_cross_entropy_with_logits(
        logits, target.to(dtype=logits.dtype), reduction=reduction
    )


def gaussian_log_prob(
    target: Tensor, mean: Tensor, scale: Tensor
) -> Tensor:
    residual = (target - mean) / scale
    dimension = target.shape[-1]
    result = -0.5 * (
        dimension * math.log(2.0 * math.pi)
        + 2.0 * scale.log().sum(dim=-1)
        + residual.square().sum(dim=-1)
    )
    _require_finite("Gaussian log probability", result)
    return result


def student_t_log_prob(
    target: Tensor, mean: Tensor, scale: Tensor, degrees_of_freedom: Tensor
) -> Tensor:
    """Multivariate Student-t log density with a diagonal scale matrix."""
    dimension = target.shape[-1]
    degrees_of_freedom = degrees_of_freedom.to(dtype=target.dtype)
    squared_mahalanobis = ((target - mean) / scale).square().sum(dim=-1)
    result = (
        torch.lgamma((degrees_of_freedom + dimension) / 2.0)
        - torch.lgamma(degrees_of_freedom / 2.0)
        - 0.5 * dimension * torch.log(degrees_of_freedom * math.pi)
        - scale.log().sum(dim=-1)
        - 0.5
        * (degrees_of_freedom + dimension)
        * torch.log1p(squared_mahalanobis / degrees_of_freedom)
    )
    _require_finite("Student-t log probability", result)
    return result


def von_mises_log_prob(
    target: Tensor, mean: Tensor, kappa: Tensor
) -> Tensor:
    result = (
        kappa * torch.cos(wrap_angle(target - mean))
        - math.log(2.0 * math.pi)
        - log_i0(kappa)
    )
    _require_finite("von Mises log probability", result)
    return result


def wrapped_gaussian_log_prob(
    target: Tensor, mean: Tensor, scale: Tensor, wrap_count: int = 3
) -> Tensor:
    if wrap_count < 1:
        raise ValueError("wrap_count must be positive")
    residual = target - mean
    offsets = torch.arange(
        -wrap_count,
        wrap_count + 1,
        device=residual.device,
        dtype=residual.dtype,
    )
    shifted = residual.unsqueeze(-1) + offsets * (2.0 * math.pi)
    component = (
        -0.5 * (shifted / scale.unsqueeze(-1)).square()
        - scale.log().unsqueeze(-1)
        - 0.5 * math.log(2.0 * math.pi)
    )
    result = torch.logsumexp(component, dim=-1)
    _require_finite("wrapped Gaussian log probability", result)
    return result


def duplicate_neighborhood(
    class_logits: Tensor,
    decoded_boxes: Tensor,
    gt_labels: Tensor,
    gt_boxes: Tensor,
) -> Tensor:
    """Return [batch,gt,query] frozen same-class/2 m candidate masks."""
    if class_logits.ndim != 3 or decoded_boxes.ndim != 3:
        raise ValueError("query tensors must have [batch,query,feature] layout")
    if gt_labels.ndim != 2 or gt_boxes.ndim != 3:
        raise ValueError("GT tensors must have [batch,gt,feature] layout")
    predicted = class_logits.detach().argmax(dim=-1)
    same_class = predicted.unsqueeze(1) == gt_labels.detach().unsqueeze(-1)
    distance = torch.linalg.vector_norm(
        decoded_boxes.detach()[:, None, :, :2]
        - gt_boxes.detach()[:, :, None, :2],
        dim=-1,
    )
    return same_class & (distance <= DUPLICATE_CENTER_METERS)


def query_target_log_mass(
    output: Dict[str, Tensor],
    gt_state: Tensor,
    gt_yaw: Tensor,
    gt_labels: Tensor,
    temperature: Tensor,
    family: str = "student_t",
) -> Tensor:
    """Return log rho with layout [batch,gt,query]."""
    batch, queries = output["existence_logit"].shape
    if gt_state.ndim != 3 or gt_state.shape[0] != batch:
        raise ValueError("gt_state must have [batch,gt,state] layout")
    gt_count = gt_state.shape[1]
    if gt_state.shape[-1] != STATE_DIM:
        raise ValueError(f"gt_state must end in {STATE_DIM}")
    if gt_yaw.shape != (batch, gt_count) or gt_labels.shape != (batch, gt_count):
        raise ValueError("GT yaw/label shape mismatch")

    class_log_probs = calibrated_class_log_probs(
        output["class_logits"], temperature
    )
    class_mass = class_log_probs.unsqueeze(1).expand(
        batch, gt_count, queries, CLASS_COUNT
    )
    class_index = gt_labels[:, :, None, None].expand(
        batch, gt_count, queries, 1
    )
    class_mass = class_mass.gather(-1, class_index).squeeze(-1)

    target = gt_state[:, :, None, :]
    mean = output["mean_state"][:, None, :, :]
    scale = output["scale"][:, None, :, :]
    if family == "student_t":
        geometry_mass = student_t_log_prob(
            target,
            mean,
            scale,
            output["degrees_of_freedom"][:, None, :],
        )
        yaw_mass = von_mises_log_prob(
            gt_yaw[:, :, None],
            output["yaw_mean"][:, None, :],
            output["yaw_kappa"][:, None, :],
        )
    elif family == "gaussian":
        geometry_mass = gaussian_log_prob(target, mean, scale)
        yaw_mass = wrapped_gaussian_log_prob(
            gt_yaw[:, :, None],
            output["yaw_mean"][:, None, :],
            output["yaw_scale"][:, None, :],
        )
    else:
        raise ValueError(f"unknown probability family: {family}")

    existence_mass = -F.softplus(
        -output["existence_logit"][:, None, :]
    )
    result = existence_mass + class_mass + geometry_mass + yaw_mass
    _require_finite("query-target log mass", result)
    return result


def competition_loss(
    log_mass: Tensor,
    neighborhood: Tensor,
    temperature: float = COMPETITION_TEMPERATURE,
) -> Tensor:
    """Smooth winner-concentration loss over non-empty GT neighborhoods."""
    if log_mass.shape != neighborhood.shape:
        raise ValueError("log_mass and neighborhood must have identical shape")
    if temperature <= 0:
        raise ValueError("competition temperature must be positive")
    valid_gt = neighborhood.any(dim=-1)
    if not valid_gt.any():
        return log_mass.sum() * 0.0
    masked = log_mass.masked_fill(~neighborhood, -torch.inf)
    log_weight = F.log_softmax(masked / temperature, dim=-1)
    log_numerator = torch.logsumexp(masked + log_weight, dim=-1)
    log_denominator = torch.logsumexp(masked, dim=-1)
    loss = (log_denominator - log_numerator)[valid_gt].mean()
    _require_finite("competition loss", loss)
    return loss


def f1_loss(
    output: Dict[str, Tensor],
    matched_gt_index: Tensor,
    gt_state: Tensor,
    gt_yaw: Tensor,
    gt_labels: Tensor,
    temperature: Tensor,
    family: str = "student_t",
    neighborhood: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Compute the frozen F1 objective for a fixed-size synthetic/real batch."""
    if matched_gt_index.shape != output["existence_logit"].shape:
        raise ValueError("matched_gt_index shape mismatch")
    positive = matched_gt_index >= 0
    if not positive.any():
        raise ValueError("a training batch must contain a positive query")
    existence_target = positive.to(dtype=output["existence_logit"].dtype)
    existence = bernoulli_nll(output["existence_logit"], existence_target)

    log_class = calibrated_class_log_probs(output["class_logits"], temperature)
    safe_index = matched_gt_index.clamp(min=0)
    batch_index = torch.arange(
        matched_gt_index.shape[0], device=matched_gt_index.device
    )[:, None]
    matched_label = gt_labels[batch_index, safe_index]
    matched_state = gt_state[batch_index, safe_index]
    matched_yaw = gt_yaw[batch_index, safe_index]
    class_value = log_class.gather(
        -1, matched_label.unsqueeze(-1)
    ).squeeze(-1)

    if family == "student_t":
        geometry_value = student_t_log_prob(
            matched_state,
            output["mean_state"],
            output["scale"],
            output["degrees_of_freedom"],
        )
        yaw_value = von_mises_log_prob(
            matched_yaw, output["yaw_mean"], output["yaw_kappa"]
        )
    elif family == "gaussian":
        geometry_value = gaussian_log_prob(
            matched_state, output["mean_state"], output["scale"]
        )
        yaw_value = wrapped_gaussian_log_prob(
            matched_yaw, output["yaw_mean"], output["yaw_scale"]
        )
    else:
        raise ValueError(f"unknown probability family: {family}")

    classification = -class_value[positive].mean()
    geometry = -geometry_value[positive].mean()
    yaw = -yaw_value[positive].mean()
    competition = output["existence_logit"].sum() * 0.0
    if neighborhood is not None:
        log_mass = query_target_log_mass(
            output, gt_state, gt_yaw, gt_labels, temperature, family
        )
        competition = competition_loss(log_mass, neighborhood)
    total = (
        existence
        + classification
        + geometry
        + yaw
        + COMPETITION_WEIGHT * competition
    )
    values = {
        "total": total,
        "existence": existence,
        "classification": classification,
        "geometry": geometry,
        "yaw": yaw,
        "competition": competition,
    }
    for name, value in values.items():
        _require_finite(f"F1 {name} loss", value)
    return values
