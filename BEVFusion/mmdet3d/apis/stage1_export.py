import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class Stage1Exporter:
    LEVELS = ("camera_bev", "lidar_bev", "prefusion", "fused_bev")

    def __init__(
        self,
        model,
        dataset,
        output_dir,
        export_level,
        config,
        checkpoint,
        seed=0,
        sample_tokens=None,
        every_n=1,
        max_samples=None,
        dtype="float16",
        run_id=None,
        bev_ranges=None,
    ):
        self.model = model
        self.dataset = dataset
        self.root = Path(output_dir)
        self.export_level = export_level
        self.config = str(Path(config).resolve())
        self.checkpoint = str(Path(checkpoint).resolve())
        self.seed = seed
        self.sample_tokens = set(sample_tokens or [])
        self.every_n = every_n
        self.max_samples = max_samples
        self.dtype = getattr(torch, dtype)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.bev_ranges = bev_ranges or {}
        self.cursor = 0
        self.exported = 0
        self.captured = {}
        self.handles = []
        self.files = []

        sensors = set(model.encoders.keys())
        self.modality = "fusion" if len(sensors) > 1 else next(iter(sensors))
        self.run_dir = self.root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.config, self.run_dir / Path(self.config).name)
        self.manifest_path = self.run_dir / "manifest.json"
        self._write_manifest("running")
        if export_level in ("features", "all"):
            self._register_hooks()

    def _register_hooks(self):
        if "camera" in self.model.encoders:
            module = self.model.encoders["camera"]["vtransform"]
            self.handles.append(module.register_forward_hook(self._capture("camera_bev")))
        if "lidar" in self.model.encoders:
            module = self.model.encoders["lidar"]["backbone"]
            self.handles.append(module.register_forward_hook(self._capture("lidar_bev")))
        if self.model.fuser is not None:
            self.handles.append(self.model.fuser.register_forward_pre_hook(self._capture_prefusion))
            self.handles.append(self.model.fuser.register_forward_hook(self._capture("fused_bev")))

    def _capture(self, name):
        def hook(module, inputs, output):
            tensor = _first_tensor(output)
            if tensor is not None:
                self.captured[name] = tensor

        return hook

    def _capture_prefusion(self, module, inputs):
        tensors = [_first_tensor(value) for value in inputs[0]]
        self.captured["prefusion"] = [value for value in tensors if value is not None]

    def _selected(self, index, token):
        return (
            index % self.every_n == 0
            and (not self.sample_tokens or token in self.sample_tokens)
            and (self.max_samples is None or self.exported < self.max_samples)
        )

    def write_batch(self, results):
        for batch_index, result in enumerate(results):
            index = self.cursor + batch_index
            info = self.dataset.data_infos[index]
            token = info["token"]
            if self._selected(index, token):
                if self.export_level in ("predictions", "all"):
                    self._write_prediction(result, info)
                if self.export_level in ("features", "all"):
                    self._write_features(batch_index, info)
                self.exported += 1
        self.cursor += len(results)
        self.captured.clear()

    def _write_prediction(self, result, info):
        scores = result["scores_3d"].detach().float().cpu()
        labels = result["labels_3d"].detach().long().cpu()
        boxes = result["boxes_3d"].tensor.detach().float().cpu()
        classes = len(self.dataset.CLASSES)
        eps = torch.finfo(torch.float32).eps
        safe_scores = scores.clamp(eps, 1 - eps)
        probabilities = torch.empty((len(scores), classes), dtype=torch.float32)
        if classes == 1:
            probabilities[:, 0] = 1
        else:
            probabilities[:] = ((1 - safe_scores) / (classes - 1)).unsqueeze(1)
            probabilities.scatter_(1, labels[:, None], safe_scores[:, None])
        record = {
            "schema_version": 1,
            "sample_token": info["token"],
            "timestamp": info["timestamp"],
            "modality": self.modality,
            "class_logits": probabilities.log(),
            "class_logits_source": "post_nms_detection_score",
            "class_probabilities": probabilities,
            "class_indices": labels,
            "box_mean": boxes,
            "detection_score": scores,
            "object_index": torch.arange(len(scores)),
            "coordinate_system": "lidar",
            "model_checkpoint": self.checkpoint,
            "config": self.config,
            "run_id": self.run_id,
            "box_covariance": None,
            "velocity_covariance": None,
            "orientation_concentration": None,
            "modality_reliability": None,
            "conflict_score": None,
        }
        self._save(record, "predictions", self.modality, info["token"])

    def _write_features(self, batch_index, info):
        captured = dict(self.captured)
        if "prefusion" not in captured:
            sensor = captured.get(f"{self.modality}_bev")
            if sensor is not None:
                captured["prefusion"] = [sensor]
                captured["fused_bev"] = sensor
        for level in self.LEVELS:
            value = captured.get(level)
            if value is None:
                continue
            if isinstance(value, list):
                parts = [part[batch_index].detach().cpu() for part in value]
                feature = torch.cat(parts, dim=0)
            else:
                feature = value[batch_index].detach().cpu()
            feature = feature.to(self.dtype)
            bev_range = self.bev_ranges.get(level)
            resolution = None
            if bev_range and feature.ndim >= 2:
                resolution = [
                    (bev_range[2] - bev_range[0]) / feature.shape[-1],
                    (bev_range[3] - bev_range[1]) / feature.shape[-2],
                ]
            modality = {
                "camera_bev": "camera",
                "lidar_bev": "lidar",
            }.get(level, "fusion")
            record = {
                "schema_version": 1,
                "sample_token": info["token"],
                "timestamp": info["timestamp"],
                "level": level,
                "modality": modality,
                "feature": feature,
                "shape": list(feature.shape),
                "dtype": str(feature.dtype).split(".")[-1],
                "bev_range": bev_range,
                "resolution": resolution,
                "coordinate_system": "lidar",
                "run_id": self.run_id,
            }
            self._save(record, "features", level, info["token"])

    def _save(self, record, kind, name, token):
        path = self.run_dir / kind / name / f"{token}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, path)
        self.files.append(str(path.relative_to(self.run_dir)))

    def _write_manifest(self, status, metrics=None):
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            git_dirty = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except (OSError, subprocess.CalledProcessError):
            commit = None
            git_dirty = None
        digest = hashlib.sha256()
        with open(self.checkpoint, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest = {
            "run_id": self.run_id,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "git_commit": commit,
            "git_dirty": git_dirty,
            "config": self.config,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": digest.hexdigest(),
            "data_version": getattr(self.dataset, "version", None),
            "seed": self.seed,
            "modality": self.modality,
            "export_level": self.export_level,
            "outputs": self.files,
            "exported_samples": self.exported,
            "metrics": metrics,
            "status": status,
        }
        self.manifest_path.write_text(
            json.dumps(
                _json_safe(manifest),
                indent=2,
                default=str,
                allow_nan=False,
            )
            + "\n"
        )

    def close(self, status="complete", metrics=None):
        for handle in self.handles:
            handle.remove()
        self._write_manifest(status, metrics)
