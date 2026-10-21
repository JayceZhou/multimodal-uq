"""Stage 4/5 preliminary reliability/conflict feature-gating baseline.

This is retained for later comparison and is not the final Stage 3 method.
"""

import torch
from torch import nn
from torch.nn import functional as F


class Stage3ReliabilityConflict(nn.Module):
    MODES = {"reliability", "conflict", "cosine", "l1", "full", "oracle"}

    def __init__(
        self,
        in_channels=(80, 256),
        hidden_channels=32,
        mode="full",
        gate_scale=0.5,
        point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        reliability_pos_weight=1.0,
        conflict_pos_weight=4.0,
        lambda_rel=1.0,
        lambda_conf=1.0,
        lambda_clean=0.1,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unsupported Stage 3 mode: {mode}")
        self.mode = mode
        self.gate_scale = float(gate_scale)
        self.point_cloud_range = tuple(float(value) for value in point_cloud_range)
        self.reliability_pos_weight = float(reliability_pos_weight)
        self.conflict_pos_weight = float(conflict_pos_weight)
        self.lambda_rel = float(lambda_rel)
        self.lambda_conf = float(lambda_conf)
        self.lambda_clean = float(lambda_clean)

        self.camera_projection = nn.Conv2d(in_channels[0], hidden_channels, 1)
        self.lidar_projection = nn.Conv2d(in_channels[1], hidden_channels, 1)
        self.camera_reliability = nn.Conv2d(hidden_channels, 1, 1)
        self.lidar_reliability = nn.Conv2d(hidden_channels, 1, 1)
        self.conflict = nn.Sequential(
            nn.Conv2d(hidden_channels * 4 + 2, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(3, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 2, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    @staticmethod
    def _presence(metas, device, dtype):
        values = []
        for meta in metas:
            condition = meta.get("stage3_condition", {})
            present = condition.get("modality_present_mask", {"camera": True, "lidar": True})
            values.append([float(present["camera"]), float(present["lidar"])])
        return torch.tensor(values, device=device, dtype=dtype)[:, :, None, None]

    def _feature_difference(self, camera, lidar):
        if self.mode == "cosine":
            return ((1.0 - F.cosine_similarity(camera, lidar, dim=1, eps=1e-6)) / 2.0).clamp(0, 1)[:, None]
        numerator = (camera - lidar).abs().mean(dim=1, keepdim=True)
        denominator = camera.abs().mean(dim=1, keepdim=True) + lidar.abs().mean(dim=1, keepdim=True) + 1e-6
        return (numerator / denominator).clamp(0, 1)

    def forward(self, features, metas):
        if len(features) != 2:
            raise ValueError("Stage 3 requires Camera and LiDAR BEV features")
        camera_feature, lidar_feature = features
        camera = self.camera_projection(camera_feature)
        lidar = self.lidar_projection(lidar_feature)
        camera_logit = self.camera_reliability(camera)
        lidar_logit = self.lidar_reliability(lidar)
        presence = self._presence(metas, camera.device, camera.dtype)
        reliability = torch.cat((camera_logit.sigmoid(), lidar_logit.sigmoid()), dim=1) * presence

        learned_conflict = self.conflict(torch.cat((
            camera, lidar, (camera - lidar).abs(), camera * lidar,
            camera_logit, lidar_logit,
        ), dim=1)).sigmoid()
        if self.mode in ("cosine", "l1"):
            conflict = self._feature_difference(camera, lidar)
        else:
            conflict = learned_conflict

        gate_reliability = reliability if self.mode in ("reliability", "full", "oracle") else torch.ones_like(reliability)
        gate_conflict = conflict if self.mode in ("conflict", "cosine", "l1", "full", "oracle") else torch.zeros_like(conflict)
        gate = (1.0 + self.gate_scale * torch.tanh(self.gate(torch.cat((gate_reliability, gate_conflict), dim=1)))) * presence
        maps = {
            "camera_reliability_logit": camera_logit,
            "lidar_reliability_logit": lidar_logit,
            "camera_reliability": reliability[:, 0:1],
            "lidar_reliability": reliability[:, 1:2],
            "conflict": conflict,
            "learned_conflict": learned_conflict,
            "camera_gate": gate[:, 0:1],
            "lidar_gate": gate[:, 1:2],
            "presence": presence,
        }
        return [camera_feature * maps["camera_gate"], lidar_feature * maps["lidar_gate"]], maps

    def _box_mask(self, box, height, width, device):
        x0, y0, _, x1, y1, _ = self.point_cloud_range
        ys = y0 + (torch.arange(height, device=device) + 0.5) * ((y1 - y0) / height)
        xs = x0 + (torch.arange(width, device=device) + 0.5) * ((x1 - x0) / width)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dx, dy = xx - box[0], yy - box[1]
        cosine, sine = torch.cos(box[6]), torch.sin(box[6])
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        return (local_x.abs() <= box[3] / 2) & (local_y.abs() <= box[4] / 2)

    @staticmethod
    def _balanced_bce(logits, labels, pos_weight):
        weight = logits.new_tensor(pos_weight)
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)

    def loss(self, maps, gt_bboxes_3d, metas):
        rel_logits, rel_labels, conf_logits, conf_labels = [], [], [], []
        object_stats = []
        height, width = maps["conflict"].shape[-2:]
        for batch_index, (boxes, meta) in enumerate(zip(gt_bboxes_3d, metas)):
            targets = meta.get("stage3_targets")
            if targets is None:
                continue
            box_tensor = boxes.tensor.to(maps["conflict"].device)
            if len(targets) != len(box_tensor):
                raise ValueError(f"Stage 3 target/GT mismatch: {len(targets)} != {len(box_tensor)}")
            for object_index, (box, target) in enumerate(zip(box_tensor, targets)):
                if not target.get("defined", True):
                    continue
                mask = self._box_mask(box, height, width, box.device)
                if not mask.any():
                    continue
                camera_logit = maps["camera_reliability_logit"][batch_index, 0][mask].mean()
                lidar_logit = maps["lidar_reliability_logit"][batch_index, 0][mask].mean()
                conflict_probability = maps["learned_conflict"][batch_index, 0][mask].mean()
                rel_logits.extend((camera_logit, lidar_logit))
                rel_labels.extend((float(target["camera"]["reliable_detection"]), float(target["lidar"]["reliable_detection"])))
                conf_logits.append(torch.logit(conflict_probability.clamp(1e-6, 1 - 1e-6)))
                conf_labels.append(float(target["conflict"]))
                object_stats.append({
                    "sample_token": meta.get("token"), "object_index": object_index,
                    "camera_reliability": float(camera_logit.detach().sigmoid()),
                    "lidar_reliability": float(lidar_logit.detach().sigmoid()),
                    "conflict": float(conflict_probability.detach()),
                    "camera_gate": float(maps["camera_gate"][batch_index, 0][mask].detach().mean()),
                    "lidar_gate": float(maps["lidar_gate"][batch_index, 0][mask].detach().mean()),
                })
        if not rel_logits or not conf_logits:
            raise ValueError("Stage 3 batch has no defined GT target regions")
        rel_logits = torch.stack(rel_logits)
        rel_labels = rel_logits.new_tensor(rel_labels)
        conf_logits = torch.stack(conf_logits)
        conf_labels = conf_logits.new_tensor(conf_labels)
        rel_loss = self._balanced_bce(rel_logits, rel_labels, self.reliability_pos_weight)
        conf_loss = self._balanced_bce(conf_logits, conf_labels, self.conflict_pos_weight)
        clean = maps["conflict"].new_tensor([
            float(meta.get("stage3_condition", {}).get("family") == "clean") for meta in metas
        ])[:, None, None, None]
        clean_denominator = clean.sum().clamp_min(1.0)
        clean_prior = (((maps["camera_gate"] - 1).abs() + (maps["lidar_gate"] - 1).abs()) * clean).mean((1, 2, 3)).sum() / clean_denominator
        return {
            "loss/stage3/reliability": rel_loss * self.lambda_rel,
            "loss/stage3/conflict": conf_loss * self.lambda_conf,
            "loss/stage3/clean_gate_prior": clean_prior * self.lambda_clean,
            "stats/stage3/reliability_brier": ((rel_logits.sigmoid() - rel_labels) ** 2).mean().detach(),
            "stats/stage3/conflict_brier": ((conf_logits.sigmoid() - conf_labels) ** 2).mean().detach(),
            "stats/stage3/reliability_positive_rate": rel_labels.mean().detach(),
            "stats/stage3/conflict_positive_rate": conf_labels.mean().detach(),
            "object_stats": object_stats,
        }
