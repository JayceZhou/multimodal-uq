"""AV2 Sensor detector dataset adapter for the frozen BRIDGE train split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mmcv
import numpy as np

from mmdet.datasets import DATASETS

from ..core.bbox import LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset


RING_CAMERA_ORDER = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
)

BRIDGE_DYNAMIC_CLASSES = (
    "ARTICULATED_BUS",
    "BICYCLE",
    "BICYCLIST",
    "BOX_TRUCK",
    "BUS",
    "DOG",
    "LARGE_VEHICLE",
    "MOTORCYCLE",
    "MOTORCYCLIST",
    "PEDESTRIAN",
    "RAILED_VEHICLE",
    "REGULAR_VEHICLE",
    "SCHOOL_BUS",
    "STROLLER",
    "TRUCK",
    "TRUCK_CAB",
    "VEHICULAR_TRAILER",
    "WHEELCHAIR",
    "WHEELED_DEVICE",
    "WHEELED_RIDER",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_id_file(path: Path):
    identifiers = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate log identity in smoke ID file: {path}")
    return identifiers


@DATASETS.register_module()
class AV2Dataset(Custom3DDataset):
    """Read converter infos and enforce a pre-frozen log-identity partition.

    No configuration option samples or creates a partition.  The dataset
    first validates the SHA-256 of the frozen split JSON, then filters exactly
    the IDs named by ``split_key``.  The optional D0 list is allowed only as a
    subset of that same detector-train identity set.
    """

    CLASSES = BRIDGE_DYNAMIC_CLASSES

    def __init__(
        self,
        ann_file,
        pipeline=None,
        dataset_root=None,
        object_classes=None,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        split_file=None,
        split_key=None,
        expected_split_sha256=None,
        expected_manifest_sha256=None,
        expected_partition_size=None,
        smoke_log_ids_file=None,
        with_velocity=True,
    ):
        if split_file is None or split_key is None:
            raise ValueError("AV2Dataset requires a frozen split_file and split_key")
        self.split_file = Path(split_file)
        self.split_key = str(split_key)
        self.expected_split_sha256 = expected_split_sha256
        self.expected_manifest_sha256 = expected_manifest_sha256
        self.expected_partition_size = expected_partition_size
        self.smoke_log_ids_file = Path(smoke_log_ids_file) if smoke_log_ids_file else None
        self.with_velocity = bool(with_velocity)
        super().__init__(
            dataset_root=dataset_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=object_classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )
        if tuple(self.CLASSES) != BRIDGE_DYNAMIC_CLASSES:
            raise ValueError("AV2 detector classes must equal the frozen dynamic taxonomy")
        if self.modality is None:
            self.modality = dict(use_camera=True, use_lidar=True, use_radar=False, use_map=False, use_external=False)
        if not self.modality.get("use_camera") or not self.modality.get("use_lidar"):
            raise ValueError("AV2 detector authority requires both seven cameras and LiDAR")
        if self.modality.get("use_radar") or self.modality.get("use_map"):
            raise ValueError("AV2 detector migration excludes radar and map inputs")

    def _frozen_ids(self):
        if not self.split_file.is_file():
            raise FileNotFoundError(self.split_file)
        split_sha = _sha256(self.split_file)
        if self.expected_split_sha256 and split_sha != self.expected_split_sha256:
            raise RuntimeError("frozen AV2 split SHA-256 mismatch")
        split = json.loads(self.split_file.read_text(encoding="utf-8"))
        if self.expected_manifest_sha256 and split.get("manifest_sha256") != self.expected_manifest_sha256:
            raise RuntimeError("AV2 split does not point to the frozen train manifest")
        identifiers = split.get(self.split_key)
        if not isinstance(identifiers, list) or len(identifiers) != len(set(identifiers)):
            raise RuntimeError(f"invalid frozen AV2 split key: {self.split_key}")
        if self.expected_partition_size is not None and len(identifiers) != int(self.expected_partition_size):
            raise RuntimeError("AV2 split partition has the wrong frozen size")
        identifiers = list(identifiers)
        if self.smoke_log_ids_file is not None:
            smoke_ids = _read_id_file(self.smoke_log_ids_file)
            if not set(smoke_ids) <= set(identifiers):
                raise RuntimeError("D0 smoke IDs are not a subset of the frozen partition")
            identifiers = smoke_ids
        return identifiers

    def load_annotations(self, ann_file):
        payload = mmcv.load(ann_file)
        if not isinstance(payload, dict) or "infos" not in payload or "metadata" not in payload:
            raise ValueError("AV2 info file has an invalid schema")
        metadata = payload["metadata"]
        if metadata.get("dataset") != "argoverse2_sensor" or tuple(metadata.get("ring_camera_order", ())) != RING_CAMERA_ORDER:
            raise ValueError("AV2 info file does not satisfy the frozen sensor contract")
        if self.expected_manifest_sha256 and metadata.get("source_manifest_sha256") != self.expected_manifest_sha256:
            raise RuntimeError("AV2 info file comes from a different train manifest")
        requested_ids = self._frozen_ids()
        info_ids = {str(info.get("log_id")) for info in payload["infos"]}
        if not set(requested_ids) <= info_ids:
            missing = sorted(set(requested_ids) - info_ids)
            raise RuntimeError(f"frozen split identities missing from AV2 infos: {missing[:3]}")
        selected = [info for info in payload["infos"] if str(info["log_id"]) in set(requested_ids)]
        present = {str(info["log_id"]) for info in selected}
        if present != set(requested_ids):
            raise RuntimeError("AV2 info filtering changed a frozen log identity")
        for info in selected:
            if tuple(info.get("camera_channels", ())) != RING_CAMERA_ORDER:
                raise RuntimeError("AV2 info does not have the fixed ring-camera order")
            if tuple(info.get("cams", {}).keys()) != RING_CAMERA_ORDER:
                raise RuntimeError("AV2 info contains a missing, extra, or reordered camera")
        self.metadata = metadata
        return sorted(selected, key=lambda item: (str(item["log_id"]), int(item["timestamp"])))

    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        mask = np.asarray(info["gt_detector_mask"], dtype=bool)
        return sorted({self.cat2id[name] for name in np.asarray(info["gt_names"])[mask] if name in self.cat2id})

    def get_data_info(self, index):
        info = self.data_infos[index]
        data = dict(
            token=info["token"],
            sample_idx=info["token"],
            log_id=info["log_id"],
            lidar_path=info["lidar_path"],
            sweeps=[],
            timestamp=int(info["timestamp"]),
            ego2global=np.asarray(info["ego2city"], dtype=np.float32),
            lidar2ego=np.asarray(info["lidar2ego"], dtype=np.float32),
        )
        if self.modality["use_camera"]:
            data.update(
                camera_channels=list(RING_CAMERA_ORDER),
                camera_tokens=[],
                camera_timestamps=[],
                image_paths=[],
                lidar2camera=[],
                lidar2image=[],
                camera2ego=[],
                camera2lidar=[],
                camera_intrinsics=[],
                camera_distortions=[],
            )
            for channel in RING_CAMERA_ORDER:
                camera = info["cams"][channel]
                data["camera_tokens"].append(camera["token"])
                data["camera_timestamps"].append(int(camera["timestamp"]))
                data["image_paths"].append(camera["data_path"])
                data["lidar2camera"].append(np.asarray(camera["lidar2camera"], dtype=np.float32))
                data["lidar2image"].append(np.asarray(camera["lidar2image"], dtype=np.float32))
                data["camera2ego"].append(np.asarray(camera["camera2ego"], dtype=np.float32))
                data["camera2lidar"].append(np.asarray(camera["camera2lidar"], dtype=np.float32))
                data["camera_intrinsics"].append(np.asarray(camera["cam_intrinsic"], dtype=np.float32))
                data["camera_distortions"].append(np.asarray(camera["distortion"], dtype=np.float32))
        data["ann_info"] = self.get_ann_info(index)
        return data

    def get_ann_info(self, index):
        info = self.data_infos[index]
        mask = np.asarray(info["gt_detector_mask"], dtype=bool)
        boxes = np.asarray(info["gt_boxes"], dtype=np.float32)[mask]
        names = np.asarray(info["gt_names"])[mask]
        velocity = np.asarray(info["gt_velocity"], dtype=np.float32)[mask]
        if self.with_velocity:
            if not np.all(np.isfinite(velocity)):
                raise RuntimeError("invalid GT velocity reached AV2 detector supervision")
            boxes = np.concatenate((boxes, velocity), axis=1)
        labels = np.asarray([self.cat2id[name] for name in names], dtype=np.int64)
        return dict(
            gt_bboxes_3d=LiDARInstance3DBoxes(boxes, box_dim=boxes.shape[-1]),
            gt_labels_3d=labels,
            gt_names=names,
            gt_track_ids=np.asarray(info["gt_track_ids"])[mask],
        )

    def evaluate(self, *args, **kwargs):
        raise RuntimeError(
            "AV2 detector acceptance/checkpoint metric is intentionally not frozen yet; "
            "Phase D training is not authorized by this migration implementation."
        )
