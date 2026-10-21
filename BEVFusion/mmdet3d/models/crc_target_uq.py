"""CRC-TargetUQ heads and post-hoc calibration primitives.

The detector/query features consumed here are frozen.  Probability, ordinal
risk and predictive-set scale are intentionally separate outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .target_uq import StructuredTargetUQAdapter, wrap_angle


COMPONENTS: Tuple[str, ...] = (
    "existence",
    "conditional_class",
    "xy",
    "z",
    "size",
    "yaw",
    "velocity",
)


def _require_finite(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value


class ClassConditionalDiagonalDensity(nn.Module):
    """One-component diagonal GMM per class with frozen quantile scaling."""

    def __init__(self, class_count: int, feature_dim: int, variance_floor: float = 1e-4) -> None:
        super().__init__()
        self.class_count = int(class_count)
        self.feature_dim = int(feature_dim)
        self.variance_floor = float(variance_floor)
        self.register_buffer("means", torch.zeros(class_count, feature_dim))
        self.register_buffer("variances", torch.ones(class_count, feature_dim))
        self.register_buffer("lower", torch.zeros(class_count))
        self.register_buffer("upper", torch.ones(class_count))
        self.register_buffer("counts", torch.zeros(class_count, dtype=torch.long))

    @torch.no_grad()
    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> "ClassConditionalDiagonalDensity":
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("density features have the wrong shape")
        labels = labels.long()
        if labels.shape != features.shape[:1]:
            raise ValueError("density labels must align with features")
        if not bool(((labels >= 0) & (labels < self.class_count)).all()):
            raise ValueError("density labels must be foreground class indices")
        global_mean = features.double().mean(0)
        global_var = features.double().var(0, unbiased=False).clamp_min(self.variance_floor)
        for class_index in range(self.class_count):
            selected = features[labels == class_index].double()
            self.counts[class_index] = len(selected)
            if len(selected) < 2:
                mean, variance = global_mean, global_var
            else:
                mean = selected.mean(0)
                variance = selected.var(0, unbiased=False).clamp_min(self.variance_floor)
            self.means[class_index].copy_(mean.float())
            self.variances[class_index].copy_(variance.float())
            nll = self._raw(selected.float() if len(selected) else features.float(), class_index)
            self.lower[class_index] = torch.quantile(nll, 0.001)
            self.upper[class_index] = torch.quantile(nll, 0.999).clamp_min(self.lower[class_index] + 1e-6)
        return self

    def _raw(self, features: torch.Tensor, class_index: int) -> torch.Tensor:
        variance = self.variances[class_index].to(features)
        mean = self.means[class_index].to(features)
        return 0.5 * (
            ((features - mean).square() / variance).sum(-1)
            + variance.log().sum()
            + self.feature_dim * math.log(2.0 * math.pi)
        )

    def forward(self, features: torch.Tensor, class_indices: torch.Tensor) -> torch.Tensor:
        if features.shape[:-1] != class_indices.shape or features.shape[-1] != self.feature_dim:
            raise ValueError("density inputs do not align")
        flat_features = features.reshape(-1, self.feature_dim)
        flat_classes = class_indices.reshape(-1).long()
        if not bool(((flat_classes >= 0) & (flat_classes < self.class_count)).all()):
            raise ValueError("predicted density class is out of range")
        means = self.means.index_select(0, flat_classes).to(flat_features)
        variances = self.variances.index_select(0, flat_classes).to(flat_features)
        raw = 0.5 * (
            ((flat_features - means).square() / variances).sum(-1)
            + variances.log().sum(-1)
            + self.feature_dim * math.log(2.0 * math.pi)
        )
        lower = self.lower.index_select(0, flat_classes).to(raw)
        upper = self.upper.index_select(0, flat_classes).to(raw)
        normalized = ((raw - lower) / (upper - lower).clamp_min(1e-6)).clamp(0.0, 1.0)
        return _require_finite("query_density", normalized.reshape(class_indices.shape))


class _AffineCoupling(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, parity: int) -> None:
        super().__init__()
        mask = (torch.arange(feature_dim) % 2 == parity).float()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * feature_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self.mask.to(value)
        scale, shift = self.net(value * mask).chunk(2, dim=-1)
        scale = torch.tanh(scale) * (1.0 - mask)
        shift = shift * (1.0 - mask)
        transformed = value * mask + (1.0 - mask) * (value * torch.exp(scale) + shift)
        return transformed, scale.sum(-1)


class _RealNVP(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32, blocks: int = 32) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.blocks = nn.ModuleList(
            [_AffineCoupling(feature_dim, hidden_dim, index % 2) for index in range(blocks)]
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        transformed = value
        log_determinant = value.new_zeros(value.shape[:-1])
        for block in self.blocks:
            transformed, current = block(transformed)
            log_determinant = log_determinant + current
        base = -0.5 * (transformed.square().sum(-1) + self.feature_dim * math.log(2.0 * math.pi))
        return _require_finite("realnvp_log_prob", base + log_determinant)


class ClassConditionalRealNVPDensity(nn.Module):
    """Query2Uncertainty-style class-conditioned RealNVP density.

    This is a faithful architectural reimplementation, not official code.
    """

    def __init__(
        self,
        class_count: int,
        feature_dim: int,
        hidden_dim: int = 32,
        coupling_blocks: int = 32,
    ) -> None:
        super().__init__()
        self.class_count = int(class_count)
        self.feature_dim = int(feature_dim)
        self.flows = nn.ModuleList(
            [_RealNVP(feature_dim, hidden_dim, coupling_blocks) for _ in range(class_count)]
        )
        self.register_buffer("lower", torch.zeros(class_count))
        self.register_buffer("upper", torch.ones(class_count))

    def negative_log_likelihood(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = features.reshape(-1, self.feature_dim)
        labels = labels.reshape(-1).long()
        values = []
        for class_index, flow in enumerate(self.flows):
            selected = features[labels == class_index]
            if len(selected):
                values.append(-flow.log_prob(selected))
        if not values:
            raise ValueError("RealNVP density batch has no valid class")
        return torch.cat(values).mean()

    def raw_negative_log_density(self, features: torch.Tensor, class_indices: torch.Tensor) -> torch.Tensor:
        shape = class_indices.shape
        flat_features = features.reshape(-1, self.feature_dim)
        flat_classes = class_indices.reshape(-1).long()
        result = flat_features.new_empty(len(flat_features))
        for class_index, flow in enumerate(self.flows):
            selected = flat_classes == class_index
            if bool(selected.any()):
                result[selected] = -flow.log_prob(flat_features[selected])
        return _require_finite("realnvp_negative_log_density", result.reshape(shape))

    @torch.no_grad()
    def fit_normalization(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        raw = self.raw_negative_log_density(features, labels)
        for class_index in range(self.class_count):
            selected = raw[labels == class_index]
            if not len(selected):
                raise ValueError(f"class {class_index} has no normalization samples")
            self.lower[class_index] = torch.quantile(selected, 0.001)
            self.upper[class_index] = torch.quantile(selected, 0.999).clamp_min(self.lower[class_index] + 1e-6)

    def forward(self, features: torch.Tensor, class_indices: torch.Tensor) -> torch.Tensor:
        raw = self.raw_negative_log_density(features, class_indices)
        flat_classes = class_indices.reshape(-1).long()
        lower = self.lower.index_select(0, flat_classes).reshape(class_indices.shape).to(raw)
        upper = self.upper.index_select(0, flat_classes).reshape(class_indices.shape).to(raw)
        return ((raw - lower) / (upper - lower).clamp_min(1e-6)).clamp(0.0, 1.0)


def apply_density_aware_temperature(
    logits_k1: torch.Tensor,
    density_score: torch.Tensor,
    existence_base: float,
    existence_gain: float,
    class_base: float,
    class_gain: float,
) -> Dict[str, torch.Tensor]:
    exist_temperature = F.softplus(
        logits_k1.new_tensor(existence_base) + float(existence_gain) * density_score
    ) + 0.05
    class_temperature = F.softplus(
        logits_k1.new_tensor(class_base) + float(class_gain) * density_score
    ) + 0.05
    foreground = logits_k1[..., :-1]
    existence_logit = torch.logsumexp(foreground, dim=-1) - logits_k1[..., -1]
    p_exist = torch.sigmoid(existence_logit / exist_temperature)
    p_class = (foreground / class_temperature[..., None]).softmax(-1)
    p_all = torch.cat((p_exist[..., None] * p_class, 1.0 - p_exist[..., None]), dim=-1)
    return {
        "p_exist": _require_finite("density_calibrated_p_exist", p_exist),
        "p_class_given_exist": _require_finite("density_calibrated_p_class", p_class),
        "p_all": _require_finite("density_calibrated_p_all", p_all),
        "logits_k1": p_all.clamp_min(1e-30).log(),
    }


def fit_density_aware_temperature(
    logits_k1: torch.Tensor,
    labels: torch.Tensor,
    density_score: torch.Tensor,
) -> Dict[str, float]:
    logits_k1 = logits_k1.detach().float()
    labels = labels.detach().long()
    density_score = density_score.detach().float()
    if logits_k1.shape[:-1] != labels.shape or labels.shape != density_score.shape:
        raise ValueError("density-aware calibration inputs do not align")
    positive = labels != logits_k1.shape[-1] - 1
    existence_target = positive.float()
    existence_logit = (
        torch.logsumexp(logits_k1[:, :-1], dim=-1) - logits_k1[:, -1]
    )
    parameters = nn.Parameter(torch.zeros(4, device=logits_k1.device))
    optimizer = torch.optim.LBFGS(
        [parameters], lr=0.25, max_iter=75, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        exist_temperature = (
            F.softplus(parameters[0] + parameters[1] * density_score) + 0.05
        )
        class_temperature = (
            F.softplus(parameters[2] + parameters[3] * density_score) + 0.05
        )
        loss = F.binary_cross_entropy_with_logits(
            existence_logit / exist_temperature, existence_target
        )
        if bool(positive.any()):
            loss = loss + F.cross_entropy(
                logits_k1[positive, :-1]
                / class_temperature[positive, None],
                labels[positive],
            )
        loss.backward()
        return loss

    optimizer.step(closure)
    values = parameters.detach().cpu().tolist()
    return {
        "method": "density_aware_temperature",
        "existence_base": values[0],
        "existence_gain": values[1],
        "class_base": values[2],
        "class_gain": values[3],
    }


def _proposal_density(center_xy: torch.Tensor, radius: float = 5.0) -> torch.Tensor:
    distance = torch.cdist(center_xy.float(), center_xy.float())
    return (distance <= radius).float().mean(-1)


def structured_ranking_summaries(
    distribution: Mapping[str, torch.Tensor],
    means: Mapping[str, torch.Tensor],
    density_score: Optional[torch.Tensor],
) -> torch.Tensor:
    probability = distribution["p_class_given_exist"].detach().clamp_min(1e-8)
    entropy = -(probability * probability.log()).sum(-1)
    top = probability.topk(2, dim=-1).values
    margin = top[..., 0] - top[..., 1]
    predicted_class = probability.argmax(-1)
    one_hot = F.one_hot(predicted_class, probability.shape[-1]).to(probability)

    def covariance_summary(value: torch.Tensor) -> Sequence[torch.Tensor]:
        covariance = value.detach().float()
        eigenvalues = torch.linalg.eigvalsh(covariance)
        return (
            covariance.diagonal(dim1=-2, dim2=-1).sum(-1),
            torch.linalg.slogdet(covariance).logabsdet,
            eigenvalues[..., -1],
        )

    xy = covariance_summary(distribution["center_xy_cov"])
    velocity = covariance_summary(distribution["velocity_cov"])
    center_xy = means["center_xy"].detach().float()
    values = [
        distribution["p_exist"].detach(),
        entropy,
        margin,
        *xy,
        *velocity,
        distribution["yaw_kappa"].detach(),
    ]
    values.extend(one_hot.unbind(-1))
    values.append(center_xy.norm(dim=-1))
    values.extend(means["log_size"].detach().float().unbind(-1))
    values.append(_proposal_density(center_xy))
    if density_score is not None:
        values.append(density_score.detach().float())
    return _require_finite("structured_ranking_summaries", torch.stack(values, dim=-1))


class _RiskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, output_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class CRCTargetUQ(nn.Module):
    """Structured Target-UQ plus independent component-wise ordinal risks."""

    def __init__(
        self,
        in_channels: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        use_density: bool = True,
        use_query_features: bool = True,
        use_self_quality: bool = True,
        shared_head: bool = False,
        structured_variant: str = "structured",
        classifier_mode: str = "feature_k1",
    ) -> None:
        super().__init__()
        if hidden_dim not in (64, 128):
            raise ValueError("hidden_dim is outside the preregistered grid")
        if dropout not in (0.0, 0.1):
            raise ValueError("dropout is outside the preregistered grid")
        if not use_query_features and not use_self_quality and not use_density:
            raise ValueError("at least one self-only ranking feature is required")
        self.use_density = bool(use_density)
        self.use_query_features = bool(use_query_features)
        self.use_self_quality = bool(use_self_quality)
        self.shared_head = bool(shared_head)
        self.structured = StructuredTargetUQAdapter(
            num_classes=10,
            in_channels=in_channels,
            hidden_channels=32,
            variant=structured_variant,
            fixed_log_var=0.0,
            classifier_mode=classifier_mode,
        )
        summary_dim = 25 + int(use_density)
        input_dim = int(use_query_features) * in_channels + int(use_self_quality) * summary_dim
        if use_density and not use_self_quality:
            input_dim += 1
        if shared_head:
            self.risk_head = _RiskMLP(input_dim, hidden_dim, dropout, len(COMPONENTS))
        else:
            self.risk_heads = nn.ModuleDict(
                {name: _RiskMLP(input_dim, hidden_dim, dropout) for name in COMPONENTS}
            )

    def forward(
        self,
        class_logits: torch.Tensor,
        means: Mapping[str, torch.Tensor],
        features: torch.Tensor,
        density_score: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        if self.use_density and density_score is None:
            raise ValueError("density_score is required for this CRC variant")
        distribution = self.structured(class_logits, dict(means), features)
        pieces = []
        if self.use_query_features:
            pieces.append(features.detach().float())
        if self.use_self_quality:
            pieces.append(structured_ranking_summaries(distribution, means, density_score if self.use_density else None))
        elif self.use_density:
            pieces.append(density_score.detach().float().unsqueeze(-1))
        ranking_input = torch.cat(pieces, dim=-1)
        if self.shared_head:
            all_scores = self.risk_head(ranking_input)
            risks = {name: all_scores[..., index] for index, name in enumerate(COMPONENTS)}
        else:
            risks = {name: head(ranking_input).squeeze(-1) for name, head in self.risk_heads.items()}
        for name, score in risks.items():
            _require_finite(f"risk/{name}", score)
        return {"distribution": distribution, "risk": risks}


def component_error_targets(
    distribution: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    positive = targets["positive_mask"].bool()
    valid = targets["classification_mask"].bool()
    velocity = positive & targets["velocity_mask"].bool()
    p_exist = distribution["p_exist"].detach()
    class_prediction = distribution["p_class_given_exist"].detach().argmax(-1)
    class_target = targets["class_target"].long()
    errors = {
        "existence": ((p_exist >= 0.5) != positive).float(),
        "conditional_class": (class_prediction != class_target).float(),
        "xy": (distribution["center_xy_mean"].detach() - targets["center_xy"]).norm(dim=-1),
        "z": (distribution["z_mean"].detach() - targets["z"]).abs().squeeze(-1),
        "size": (distribution["log_size_mean"].detach() - targets["log_size"]).norm(dim=-1),
        "yaw": wrap_angle(distribution["yaw_mean"].detach() - targets["yaw"]).abs(),
        "velocity": (distribution["velocity_mean"].detach() - targets["velocity"]).norm(dim=-1),
    }
    masks = {
        "existence": valid,
        "conditional_class": positive,
        "xy": positive,
        "z": positive,
        "size": positive,
        "yaw": positive,
        "velocity": velocity,
    }
    return errors, masks


def pairwise_logistic_ranking_loss(
    scores: torch.Tensor,
    errors: torch.Tensor,
    mask: torch.Tensor,
    max_pairs: int = 4096,
    seed: int = 0,
) -> torch.Tensor:
    selected_scores = scores[mask].reshape(-1)
    selected_errors = errors[mask].reshape(-1)
    if len(selected_scores) < 2:
        return scores.new_zeros(())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    pair_count = min(max_pairs, len(selected_scores))
    first = torch.randperm(len(selected_scores), generator=generator)[:pair_count].to(scores.device)
    second = torch.randperm(len(selected_scores), generator=generator)[:pair_count].to(scores.device)
    unequal = selected_errors[first] != selected_errors[second]
    if not bool(unequal.any()):
        return scores.new_zeros(())
    first, second = first[unequal], second[unequal]
    direction = torch.sign(selected_errors[first] - selected_errors[second])
    margin = direction * (selected_scores[first] - selected_scores[second])
    return F.softplus(-margin).mean()


def crc_ranking_loss(
    risks: Mapping[str, torch.Tensor],
    distribution: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    errors, masks = component_error_targets(distribution, targets)
    pieces = {
        name: pairwise_logistic_ranking_loss(risks[name], errors[name], masks[name], seed=seed + index)
        for index, name in enumerate(COMPONENTS)
    }
    return sum(pieces.values()), pieces


def apply_factorized_temperature(
    logits_k1: torch.Tensor, existence_temperature: float, class_temperature: float
) -> Dict[str, torch.Tensor]:
    if existence_temperature <= 0 or class_temperature <= 0:
        raise ValueError("temperatures must be positive")
    foreground = logits_k1[..., :-1]
    background = logits_k1[..., -1]
    existence_logit = torch.logsumexp(foreground, dim=-1) - background
    p_exist = torch.sigmoid(existence_logit / float(existence_temperature))
    p_class = (foreground / float(class_temperature)).softmax(-1)
    p_all = torch.cat((p_exist[..., None] * p_class, 1.0 - p_exist[..., None]), dim=-1)
    return {
        "p_exist": _require_finite("calibrated_p_exist", p_exist),
        "p_class_given_exist": _require_finite("calibrated_p_class", p_class),
        "p_all": _require_finite("calibrated_p_all", p_all),
        "logits_k1": p_all.clamp_min(1e-30).log(),
    }


def fit_factorized_temperature(logits_k1: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    logits_k1 = logits_k1.detach().float()
    labels = labels.detach().long()
    positive = labels != logits_k1.shape[-1] - 1
    existence_target = positive.float()
    existence_logit = torch.logsumexp(logits_k1[:, :-1], dim=-1) - logits_k1[:, -1]
    log_exist_temperature = nn.Parameter(logits_k1.new_zeros(()))
    log_class_temperature = nn.Parameter(logits_k1.new_zeros(()))
    parameters = [log_exist_temperature]
    if bool(positive.any()):
        parameters.append(log_class_temperature)
    optimizer = torch.optim.LBFGS(parameters, lr=0.25, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        exist_temperature = log_exist_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(existence_logit / exist_temperature, existence_target)
        if bool(positive.any()):
            class_temperature = log_class_temperature.exp().clamp(0.05, 20.0)
            loss = loss + F.cross_entropy(logits_k1[positive, :-1] / class_temperature, labels[positive])
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "method": "factorized_temperature",
        "existence_temperature": float(log_exist_temperature.detach().exp().clamp(0.05, 20.0).cpu()),
        "class_temperature": float(log_class_temperature.detach().exp().clamp(0.05, 20.0).cpu()),
    }


def conformal_quantile(values: torch.Tensor, level: float) -> float:
    if not 0.0 < level < 1.0 or values.numel() == 0:
        raise ValueError("conformal quantile needs non-empty values and level in (0,1)")
    sorted_values = torch.sort(values.detach().double().reshape(-1)).values
    index = min(len(sorted_values), math.ceil((len(sorted_values) + 1) * level)) - 1
    return float(sorted_values[index])


@dataclass(frozen=True)
class MondrianScaleTable:
    levels: Tuple[float, ...]
    minimum_count: int
    groups: Dict[str, Dict[str, object]]

    @staticmethod
    def _key(class_index: Optional[int], range_index: Optional[int]) -> str:
        if class_index is None and range_index is None:
            return "global"
        if class_index is None:
            return f"range:{range_index}"
        if range_index is None:
            return f"class:{class_index}"
        return f"class:{class_index}|range:{range_index}"

    @classmethod
    def fit(
        cls,
        scores: torch.Tensor,
        class_indices: torch.Tensor,
        range_indices: torch.Tensor,
        levels: Iterable[float] = (0.5, 0.8, 0.9, 0.95),
        minimum_count: int = 100,
    ) -> "MondrianScaleTable":
        scores = scores.detach().double().reshape(-1)
        class_indices = class_indices.detach().long().reshape(-1)
        range_indices = range_indices.detach().long().reshape(-1)
        if not (len(scores) == len(class_indices) == len(range_indices)) or not len(scores):
            raise ValueError("Mondrian inputs must be aligned and non-empty")
        frozen_levels = tuple(float(level) for level in levels)
        groups: Dict[str, Dict[str, object]] = {}
        class_values = sorted(set(class_indices.tolist()))
        range_values = sorted(set(range_indices.tolist()))
        keys = [(None, None)]
        keys.extend((value, None) for value in class_values)
        keys.extend((None, value) for value in range_values)
        keys.extend((class_value, range_value) for class_value in class_values for range_value in range_values)
        for class_value, range_value in keys:
            selected = torch.ones(len(scores), dtype=torch.bool, device=scores.device)
            if class_value is not None:
                selected &= class_indices == class_value
            if range_value is not None:
                selected &= range_indices == range_value
            current = scores[selected]
            groups[cls._key(class_value, range_value)] = {
                "count": len(current),
                "quantiles": {
                    str(level): conformal_quantile(current, level) for level in frozen_levels
                } if len(current) else {},
            }
        return cls(frozen_levels, int(minimum_count), groups)

    def lookup(self, class_index: int, range_index: int, level: float) -> Tuple[float, str]:
        for key in (
            self._key(class_index, range_index),
            self._key(class_index, None),
            self._key(None, range_index),
            self._key(None, None),
        ):
            group = self.groups[key]
            if int(group["count"]) >= self.minimum_count or key == "global":
                return float(group["quantiles"][str(float(level))]), key
        raise RuntimeError("global Mondrian group is unexpectedly unavailable")

    def as_dict(self) -> Dict[str, object]:
        return {
            "levels": list(self.levels),
            "minimum_count": self.minimum_count,
            "fallback": ["class_range", "class", "range", "global"],
            "groups": self.groups,
        }
