import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mmdet.datasets.builder import PIPELINES


CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

LOWLIGHT_CANDIDATES = {
    "legacy_candidate": {
        1: {"exposure_scale": 0.70, "gamma": 1.20},
        2: {"exposure_scale": 0.40, "gamma": 1.60},
        3: {"exposure_scale": 0.20, "gamma": 2.20},
    },
    "calibrated_candidate": {
        1: {"exposure_scale": 0.80, "gamma": 1.10},
        2: {"exposure_scale": 0.60, "gamma": 1.30},
        3: {"exposure_scale": 0.40, "gamma": 1.60},
    },
    "formal_candidate": {
        1: {"exposure_scale": 0.80, "gamma": 1.10},
        2: {"exposure_scale": 0.60, "gamma": 1.30},
        3: {"exposure_scale": 0.50, "gamma": 1.45},
    },
}

SEVERITIES = {
    "lidar_dropout": {
        0: {"drop_ratio": 0.0},
        1: {"drop_ratio": 0.25},
        2: {"drop_ratio": 0.50},
        3: {"drop_ratio": 0.75},
    },
    "lidar_beam_reduction": {
        0: {"retained_beams": 32},
        1: {"retained_beams": 24},
        2: {"retained_beams": 16},
        3: {"retained_beams": 8},
    },
    "camera_extrinsic_translation": {
        0: {"translation_m": 0.0, "direction": [1.0, 0.0, 0.0]},
        1: {"translation_m": 0.10, "direction": [1.0, 0.0, 0.0]},
        2: {"translation_m": 0.25, "direction": [1.0, 0.0, 0.0]},
        3: {"translation_m": 0.50, "direction": [1.0, 0.0, 0.0]},
    },
    "camera_extrinsic_rotation": {
        0: {"rotation_deg": 0.0, "axis": [0.0, 0.0, 1.0]},
        1: {"rotation_deg": 1.0, "axis": [0.0, 0.0, 1.0]},
        2: {"rotation_deg": 3.0, "axis": [0.0, 0.0, 1.0]},
        3: {"rotation_deg": 5.0, "axis": [0.0, 0.0, 1.0]},
    },
    "camera_temporal_offset": {
        0: {"previous_frames": 0},
        1: {"previous_frames": 1},
        2: {"previous_frames": 2},
        3: {"previous_frames": 3},
    },
    "camera_missing": {0: {}, 1: {}},
    "lidar_missing": {0: {}, 1: {}},
}

FAMILIES = {
    "camera_lowlight": "degradation",
    "lidar_dropout": "degradation",
    "lidar_beam_reduction": "degradation",
    "camera_extrinsic_translation": "conflict",
    "camera_extrinsic_rotation": "conflict",
    "camera_temporal_offset": "conflict",
    "camera_missing": "missing",
    "lidar_missing": "missing",
}


def derive_seed(global_seed, sample_token, corruption_type, severity, unit_token=""):
    payload = json.dumps(
        [int(global_seed), sample_token, corruption_type, int(severity), unit_token],
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _mean_brightness(image):
    return float(np.asarray(image, dtype=np.float32).mean() / 255.0)


@PIPELINES.register_module()
class Stage2Corruption:
    """Apply one deterministic Stage 2 sensor degradation before augmentation."""

    def __init__(
        self,
        corruption_type,
        severity=0,
        seed=20260707,
        affected_sensors="all",
        parameters=None,
        family=None,
        lowlight_candidate="calibrated_candidate",
        temporal_history_file=None,
        temporal_history=None,
    ):
        if corruption_type == "camera_lowlight":
            if lowlight_candidate not in LOWLIGHT_CANDIDATES:
                raise ValueError(f"unknown lowlight candidate: {lowlight_candidate}")
            severities = {0: {"exposure_scale": 1.0, "gamma": 1.0}}
            severities.update(LOWLIGHT_CANDIDATES[lowlight_candidate])
        else:
            severities = SEVERITIES.get(corruption_type)
        if severities is None:
            raise ValueError(f"unsupported Stage 2 corruption: {corruption_type}")
        if severity not in severities:
            raise ValueError(f"invalid severity {severity} for {corruption_type}")
        expected_family = FAMILIES[corruption_type]
        if family is not None and family != expected_family:
            raise ValueError(f"{corruption_type} belongs to the {expected_family} family")
        self.corruption_type = corruption_type
        self.severity = int(severity)
        self.seed = int(seed)
        self.affected_sensors = affected_sensors
        self.parameters = dict(severities[severity])
        self.parameters.update(parameters or {})
        self.family = expected_family
        self.lowlight_candidate = lowlight_candidate
        self.temporal_history = temporal_history
        if temporal_history_file:
            self.temporal_history = json.loads(Path(temporal_history_file).read_text())

    def __call__(self, results):
        token = results["token"]
        record = self._record(results, token)
        if self.severity == 0:
            results["stage2_manifest"] = record
            return results
        operation = {
            "camera_lowlight": self._camera_lowlight,
            "lidar_dropout": self._lidar_dropout,
            "lidar_beam_reduction": self._lidar_beam_reduction,
            "camera_extrinsic_translation": self._camera_extrinsic,
            "camera_extrinsic_rotation": self._camera_extrinsic,
            "camera_temporal_offset": self._camera_temporal_offset,
            "camera_missing": self._modality_missing,
            "lidar_missing": self._modality_missing,
        }[self.corruption_type]
        operation(results, record)
        results["stage2_manifest"] = record
        return results

    def _record(self, results, token):
        return {
            "run_id": None,
            "sample_token": token,
            "corruption_family": self.family,
            "corruption_type": self.corruption_type,
            "severity": self.severity,
            "global_seed": self.seed,
            "derived_seed": derive_seed(
                self.seed, token, self.corruption_type, self.severity
            ),
            "affected_modality": self.corruption_type.split("_", 1)[0],
            "affected_sensor_channels": [],
            "modality_present_mask": {"camera": True, "lidar": True},
            "corruption_parameters": dict(self.parameters),
            "original_timestamp": results.get("timestamp"),
            "substituted_timestamp": None,
            "temporal_offset_ms": None,
            "extrinsic_delta_4x4": None,
            "original_point_count": None,
            "retained_point_count": None,
            "original_brightness": None,
            "corrupted_brightness": None,
            "valid": True,
            "invalid_reason": None,
            "config": {
                "family": self.family,
                "type": self.corruption_type,
                "severity": self.severity,
                "seed": self.seed,
                "affected_sensors": self.affected_sensors,
                "parameters": dict(self.parameters),
                "lowlight_candidate": (
                    self.lowlight_candidate
                    if self.corruption_type == "camera_lowlight"
                    else None
                ),
            },
            "checkpoint_sha256": None,
            "git_commit": None,
            "dataset_version": None,
        }

    def _selected_camera_indices(self, results):
        channels = tuple(results.get("camera_channels", CAMERA_CHANNELS))
        if len(channels) != len(results["img"]):
            raise ValueError("camera_channels and img must have the same length")
        if self.affected_sensors == "all" or self.affected_sensors == ["all"]:
            return channels, range(len(channels))
        requested = (
            {self.affected_sensors}
            if isinstance(self.affected_sensors, str)
            else set(self.affected_sensors)
        )
        if not requested:
            raise ValueError("affected_sensors cannot be empty")
        unknown = requested.difference(channels)
        if unknown:
            raise ValueError(f"unknown camera channels: {sorted(unknown)}")
        return channels, [index for index, name in enumerate(channels) if name in requested]

    def _camera_lowlight(self, results, record):
        channels, indices = self._selected_camera_indices(results)
        images = list(results["img"])
        exposure = float(self.parameters["exposure_scale"])
        gamma = float(self.parameters["gamma"])
        if exposure < 0 or gamma <= 0:
            raise ValueError("exposure_scale must be non-negative and gamma positive")
        stats = []
        for index in indices:
            image = images[index]
            pixels = np.asarray(image, dtype=np.float32) / 255.0
            if pixels.ndim != 3 or pixels.shape[2] != 3:
                raise ValueError("camera_lowlight expects RGB images")
            corrupted = np.power(np.clip(pixels * exposure, 0.0, 1.0), gamma)
            array = np.rint(corrupted * 255.0).astype(np.uint8)
            output = Image.fromarray(array, "RGB")
            images[index] = output
            gray = array.mean(2)
            stats.append(
                {
                    "channel": channels[index],
                    "original_brightness": _mean_brightness(image),
                    "corrupted_brightness": _mean_brightness(output),
                    "black_pixel_ratio": float((gray <= 5).mean()),
                    "saturated_pixel_ratio": float((gray >= 250).mean()),
                }
            )
        results["img"] = images
        record["affected_sensor_channels"] = [item["channel"] for item in stats]
        record["original_brightness"] = float(
            np.mean([item["original_brightness"] for item in stats])
        )
        record["corrupted_brightness"] = float(
            np.mean([item["corrupted_brightness"] for item in stats])
        )
        record["corruption_parameters"]["camera_brightness"] = stats
        record["corruption_parameters"]["black_pixel_ratio"] = float(
            np.mean([item["black_pixel_ratio"] for item in stats])
        )
        record["corruption_parameters"]["saturated_pixel_ratio"] = float(
            np.mean([item["saturated_pixel_ratio"] for item in stats])
        )
        record["corruption_parameters"]["candidate"] = self.lowlight_candidate

    def _lidar_dropout(self, results, record):
        if self.affected_sensors not in ("all", ["all"], ["LIDAR_TOP"], "LIDAR_TOP"):
            raise ValueError("lidar_dropout only supports LIDAR_TOP")
        points = results["points"]
        if points.tensor.shape[1] < 5:
            raise ValueError("lidar_dropout needs the nuScenes time-lag point field")
        ratio = float(self.parameters["drop_ratio"])
        if not 0.0 <= ratio < 1.0:
            raise ValueError("drop_ratio must be in [0, 1)")

        time_lags = points.tensor[:, 4].detach().cpu().numpy()
        sweep_lags = np.asarray(
            [
                (results["timestamp"] - sweep["timestamp"]) / 1e6
                for sweep in results.get("sweeps", [])
            ],
            dtype=np.float64,
        )
        keep = np.zeros(len(points), dtype=bool)
        sweep_records = []
        for lag in np.unique(time_lags):
            indices = np.flatnonzero(time_lags == lag)
            if abs(float(lag)) < 1e-7:
                sweep_token = results["token"]
            else:
                if not len(sweep_lags):
                    raise ValueError("historical points are present without sweep metadata")
                sweep_index = int(np.argmin(np.abs(sweep_lags - float(lag))))
                if abs(sweep_lags[sweep_index] - float(lag)) > 1e-3:
                    raise ValueError("could not match point time lag to sweep metadata")
                sweep_token = results["sweeps"][sweep_index]["sample_data_token"]
            seed = derive_seed(
                self.seed,
                results["token"],
                self.corruption_type,
                self.severity,
                sweep_token,
            )
            retained = int(round(len(indices) * (1.0 - ratio)))
            chosen = np.random.default_rng(seed).choice(
                indices, size=retained, replace=False
            )
            keep[chosen] = True
            sweep_records.append(
                {
                    "sweep_token": sweep_token,
                    "time_lag_s": float(lag),
                    "derived_seed": seed,
                    "original_point_count": int(len(indices)),
                    "retained_point_count": retained,
                }
            )

        record["affected_sensor_channels"] = ["LIDAR_TOP"]
        record["original_point_count"] = int(len(points))
        record["retained_point_count"] = int(keep.sum())
        record["corruption_parameters"]["sweeps"] = sweep_records
        results["points"] = points[torch.from_numpy(keep).to(points.tensor.device)]

    def _lidar_beam_reduction(self, results, record):
        if self.affected_sensors not in ("all", ["all"], ["LIDAR_TOP"], "LIDAR_TOP"):
            raise ValueError("lidar_beam_reduction only supports LIDAR_TOP")
        points = results["points"]
        rings = np.asarray(results.get("stage2_ring_indices"))
        if rings.shape != (len(points),):
            raise ValueError(
                "lidar_beam_reduction requires preserve_ring=True on the multi-sweep loader"
            )
        if not np.all((rings >= 0) & (rings < 32) & (rings == rings.astype(int))):
            raise ValueError("invalid nuScenes ring indices")
        retained_beams = int(self.parameters["retained_beams"])
        keep_rings = np.rint(np.linspace(0, 31, retained_beams)).astype(int)
        keep = np.isin(rings.astype(int), keep_rings)
        time_lags = points.tensor[:, 4].detach().cpu().numpy()
        sweep_records = []
        for lag in np.unique(time_lags):
            group = time_lags == lag
            sweep_records.append(
                {
                    "sweep_token": self._sweep_token(results, float(lag)),
                    "time_lag_s": float(lag),
                    "original_ring_count": int(np.unique(rings[group]).size),
                    "retained_ring_count": int(np.unique(rings[group & keep]).size),
                    "original_point_count": int(group.sum()),
                    "retained_point_count": int((group & keep).sum()),
                }
            )
        record["affected_sensor_channels"] = ["LIDAR_TOP"]
        record["original_point_count"] = int(len(points))
        record["retained_point_count"] = int(keep.sum())
        record["corruption_parameters"].update(
            approximate_ring=False,
            original_ring_count=int(np.unique(rings).size),
            retained_ring_count=int(np.unique(rings[keep]).size),
            retained_ring_indices=keep_rings.tolist(),
            sweeps=sweep_records,
        )
        results["points"] = points[torch.from_numpy(keep).to(points.tensor.device)]
        results["stage2_ring_indices"] = rings[keep]

    def _sweep_token(self, results, lag):
        if abs(lag) < 1e-7:
            return results["token"]
        sweep_lags = np.asarray(
            [
                (results["timestamp"] - sweep["timestamp"]) / 1e6
                for sweep in results.get("sweeps", [])
            ]
        )
        if not len(sweep_lags):
            raise ValueError("historical points are present without sweep metadata")
        index = int(np.argmin(np.abs(sweep_lags - lag)))
        if abs(sweep_lags[index] - lag) > 1e-3:
            raise ValueError("could not match point time lag to sweep metadata")
        return results["sweeps"][index]["sample_data_token"]

    def _camera_extrinsic(self, results, record):
        channels, indices = self._selected_camera_indices(results)
        delta = np.eye(4, dtype=np.float32)
        if self.corruption_type.endswith("translation"):
            direction = np.asarray(self.parameters["direction"], dtype=np.float64)
            norm = np.linalg.norm(direction)
            if norm == 0:
                raise ValueError("translation direction must be nonzero")
            delta[:3, 3] = direction / norm * float(self.parameters["translation_m"])
        else:
            axis = np.asarray(self.parameters["axis"], dtype=np.float64)
            norm = np.linalg.norm(axis)
            if norm == 0:
                raise ValueError("rotation axis must be nonzero")
            x, y, z = axis / norm
            skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            theta = np.deg2rad(float(self.parameters["rotation_deg"]))
            delta[:3, :3] = (
                np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)
            )

        for index in indices:
            camera2lidar = delta @ np.asarray(results["camera2lidar"][index])
            lidar2camera = np.linalg.inv(camera2lidar).astype(np.float32)
            results["camera2lidar"][index] = camera2lidar.astype(np.float32)
            results["lidar2camera"][index] = lidar2camera
            results["lidar2image"][index] = (
                np.asarray(results["camera_intrinsics"][index]) @ lidar2camera
            ).astype(np.float32)
            results["camera2ego"][index] = (
                np.asarray(results["lidar2ego"]) @ camera2lidar
            ).astype(np.float32)
        record["affected_sensor_channels"] = [channels[index] for index in indices]
        record["extrinsic_delta_4x4"] = delta.tolist()
        record["corruption_parameters"].update(
            multiplication_order="T_camera_to_lidar_prime = delta_lidar @ T_camera_to_lidar",
            perturbation_coordinate_system="keyframe_lidar",
        )

    def _camera_temporal_offset(self, results, record):
        if self.temporal_history is None:
            raise ValueError("camera_temporal_offset requires temporal history metadata")
        channels, indices = self._selected_camera_indices(results)
        depth = int(self.parameters["previous_frames"])
        replacements = []
        for index in indices:
            channel = channels[index]
            current_token = results["camera_tokens"][index]
            history = self.temporal_history.get(current_token, [])
            if len(history) < depth:
                record["valid"] = False
                record["invalid_reason"] = f"{channel} has only {len(history)} previous frames"
                results["stage2_invalid"] = True
                return
            replacement = history[depth - 1]
            replacements.append((index, channel, replacement))
        images = list(results["img"])
        offsets = []
        for index, channel, replacement in replacements:
            with Image.open(replacement["data_path"]) as image:
                images[index] = image.copy()
            offset_ms = (
                replacement["timestamp"] - results["camera_timestamps"][index]
            ) / 1000.0
            offsets.append(
                {
                    "channel": channel,
                    "sample_data_token": replacement["sample_data_token"],
                    "substituted_timestamp": replacement["timestamp"],
                    "temporal_offset_ms": offset_ms,
                }
            )
        results["img"] = images
        record["affected_sensor_channels"] = [item["channel"] for item in offsets]
        record["original_timestamp"] = {
            channels[index]: results["camera_timestamps"][index] for index in indices
        }
        record["substituted_timestamp"] = {
            item["channel"]: item["substituted_timestamp"] for item in offsets
        }
        record["temporal_offset_ms"] = {
            item["channel"]: item["temporal_offset_ms"] for item in offsets
        }
        record["corruption_parameters"]["camera_offsets"] = offsets
        values = [item["temporal_offset_ms"] for item in offsets]
        record["corruption_parameters"]["offset_spread_ms"] = max(values) - min(values)

    def _modality_missing(self, results, record):
        missing = self.corruption_type.split("_", 1)[0]
        record["affected_sensor_channels"] = CAMERA_CHANNELS if missing == "camera" else ["LIDAR_TOP"]
        record["modality_present_mask"][missing] = False
        record["corruption_parameters"].update(
            missing_mode="feature_branch_missing",
            raw_sensor_input_unchanged=True,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(type={self.corruption_type!r}, "
            f"severity={self.severity}, seed={self.seed})"
        )
