"""Deterministically convert the frozen AV2 train source into BEVFusion infos.

This converter deliberately knows nothing about BRIDGE relation labels, future
flips, hard cases, AV2 validation, or learned UQ.  It consumes the full frozen
700-log train source and records only detector-relevant current-frame data.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from bisect import bisect_left, bisect_right
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

from mmdet3d.datasets.av2_dataset import BRIDGE_DYNAMIC_CLASSES, RING_CAMERA_ORDER
from mmdet3d.datasets.av2_geometry import (
    av2_cuboid_to_lidar_box,
    invert_se3,
    se3_from_av2_record,
    transform_points,
    transform_planar_vectors,
)


ANNOTATION_LIDAR_MAX_GAP_NS = 10_000_000
CAMERA_MAX_LAG_NS = 75_000_000
VELOCITY_MAX_GAP_NS = 110_000_000
INFO_SCHEMA_VERSION = "av2_bevfusion_detector_infos_v1"


class AlignmentUnavailable(ValueError):
    """A timestamp has no legal fixed-contract sensor alignment."""

    def __init__(self, sensor: str, message: str):
        super().__init__(message)
        self.sensor = sensor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_once(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _pyarrow_feather():
    try:
        import pyarrow
        import pyarrow.feather as feather
    except ModuleNotFoundError as error:  # pragma: no cover - execution environment gate
        raise RuntimeError("AV2 info conversion requires pyarrow==7.0.0") from error
    if pyarrow.__version__ != "7.0.0":
        raise RuntimeError(f"AV2 info conversion requires pyarrow==7.0.0, found {pyarrow.__version__}")
    return feather


def _parse_timestamp_paths(directory: Path, suffix: str):
    result = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix != suffix:
            raise ValueError(f"unexpected sensor entry: {path}")
        try:
            result.append((int(path.stem), path))
        except ValueError as error:
            raise ValueError(f"unparseable timestamp path: {path}") from error
    if not result:
        raise ValueError(f"no {suffix} sensor files in {directory}")
    return result


def _nearest_index(sorted_times, timestamp: int) -> int:
    index = bisect_left(sorted_times, timestamp)
    if index == 0:
        return 0
    if index == len(sorted_times):
        return len(sorted_times) - 1
    before, after = sorted_times[index - 1], sorted_times[index]
    return index - 1 if timestamp - before <= after - timestamp else index


def _nearest_item(items, timestamp: int, maximum_gap_ns: int, name: str):
    times = [item[0] for item in items]
    index = _nearest_index(times, timestamp)
    candidate = items[index]
    if abs(candidate[0] - timestamp) > maximum_gap_ns:
        raise ValueError(f"{name} has no observation within {maximum_gap_ns} ns of {timestamp}")
    return candidate


def _nearest_pose(poses, timestamp: int):
    return _nearest_item(poses, timestamp, ANNOTATION_LIDAR_MAX_GAP_NS, "ego pose")[1]


def _previous_camera(camera_items, timestamp: int, channel: str):
    times = [item[0] for item in camera_items]
    index = bisect_right(times, timestamp) - 1
    if index < 0:
        raise AlignmentUnavailable("camera", f"{channel} has no nonfuture image for {timestamp}")
    candidate = camera_items[index]
    lag = timestamp - candidate[0]
    if lag < 0 or lag > CAMERA_MAX_LAG_NS:
        raise AlignmentUnavailable("camera", f"{channel} nonfuture image lag {lag} ns exceeds frozen contract")
    return candidate


def _camera_intrinsic(record):
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[0, 0] = float(record["fx_px"])
    intrinsic[1, 1] = float(record["fy_px"])
    intrinsic[0, 2] = float(record["cx_px"])
    intrinsic[1, 2] = float(record["cy_px"])
    if not np.all(np.isfinite(intrinsic)) or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("camera intrinsic is invalid")
    distortion = np.asarray((record["k1"], record["k2"], record["k3"]), dtype=np.float64)
    if not np.all(np.isfinite(distortion)):
        raise ValueError("camera distortion is invalid")
    return intrinsic, distortion


def _read_records(path: Path, columns):
    table = _pyarrow_feather().read_table(path, columns=list(columns)).to_pydict()
    count = len(next(iter(table.values()), ()))
    return [{name: table[name][index] for name in columns} for index in range(count)]


def _read_log(log: Path):
    annotation_columns = (
        "timestamp_ns", "track_uuid", "category", "length_m", "width_m", "height_m",
        "qw", "qx", "qy", "qz", "tx_m", "ty_m", "tz_m",
    )
    se3_columns = ("timestamp_ns", "qw", "qx", "qy", "qz", "tx_m", "ty_m", "tz_m")
    calibration_columns = ("sensor_name", "qw", "qx", "qy", "qz", "tx_m", "ty_m", "tz_m")
    intrinsic_columns = ("sensor_name", "fx_px", "fy_px", "cx_px", "cy_px", "k1", "k2", "k3", "height_px", "width_px")
    annotations = _read_records(log / "annotations.feather", annotation_columns)
    poses = [(int(row["timestamp_ns"]), se3_from_av2_record(row)) for row in _read_records(log / "city_SE3_egovehicle.feather", se3_columns)]
    poses.sort(key=lambda value: value[0])
    if not annotations or not poses:
        raise ValueError(f"empty annotations or pose source: {log.name}")
    extrinsics = {row["sensor_name"]: se3_from_av2_record(row) for row in _read_records(log / "calibration" / "egovehicle_SE3_sensor.feather", calibration_columns)}
    intrinsics = {row["sensor_name"]: row for row in _read_records(log / "calibration" / "intrinsics.feather", intrinsic_columns)}
    if not set(RING_CAMERA_ORDER) <= set(extrinsics) or not set(RING_CAMERA_ORDER) <= set(intrinsics):
        raise ValueError(f"missing ring calibration: {log.name}")
    lidar = _parse_timestamp_paths(log / "sensors" / "lidar", ".feather")
    cameras = {channel: _parse_timestamp_paths(log / "sensors" / "cameras" / channel, ".jpg") for channel in RING_CAMERA_ORDER}
    return annotations, poses, extrinsics, intrinsics, lidar, cameras


def _annotation_frames(annotations, poses):
    frames = defaultdict(list)
    previous = {}
    for record in sorted(annotations, key=lambda item: (int(item["timestamp_ns"]), str(item["track_uuid"]))):
        timestamp = int(record["timestamp_ns"])
        track_id = str(record["track_uuid"])
        category = str(record["category"])
        ego_from_box = av2_cuboid_to_lidar_box(record)
        city_from_ego = _nearest_pose(poses, timestamp)
        city_center = transform_points(city_from_ego, ego_from_box[:3])
        velocity_city = np.full(2, np.nan, dtype=np.float64)
        velocity_valid = False
        if track_id in previous:
            previous_time, previous_category, previous_center = previous[track_id]
            gap = timestamp - previous_time
            if previous_category == category and 0 < gap <= VELOCITY_MAX_GAP_NS:
                velocity_city = (city_center[:2] - previous_center[:2]) / (gap / 1e9)
                velocity_valid = bool(np.all(np.isfinite(velocity_city)))
        previous[track_id] = (timestamp, category, city_center)
        frames[timestamp].append(
            dict(
                category=category,
                track_id=track_id,
                ego_box=ego_from_box,
                city_velocity_xy=velocity_city,
                velocity_valid=velocity_valid,
            )
        )
    return frames


def _make_info(log_id, timestamp, rows, poses, extrinsics, intrinsics, lidar, cameras):
    try:
        lidar_timestamp, lidar_path = _nearest_item(lidar, timestamp, ANNOTATION_LIDAR_MAX_GAP_NS, "lidar")
    except ValueError as error:
        raise AlignmentUnavailable("lidar", str(error)) from error
    city_from_ego = _nearest_pose(poses, timestamp)
    ego_from_city = invert_se3(city_from_ego)
    cam_infos = OrderedDict()
    for channel in RING_CAMERA_ORDER:
        camera_timestamp, image_path = _previous_camera(cameras[channel], lidar_timestamp, channel)
        ego_from_camera = extrinsics[channel]
        camera_from_ego = invert_se3(ego_from_camera)
        intrinsic, distortion = _camera_intrinsic(intrinsics[channel])
        lidar_from_camera = ego_from_camera  # AV2 LiDAR is already the egovehicle reference frame.
        lidar_from_camera = np.asarray(lidar_from_camera, dtype=np.float64)
        camera_from_lidar = camera_from_ego
        cam_infos[channel] = dict(
            token=f"{log_id}:{channel}:{camera_timestamp}",
            timestamp=int(camera_timestamp),
            data_path=str(image_path),
            camera2ego=ego_from_camera.astype(np.float32),
            camera2lidar=lidar_from_camera.astype(np.float32),
            lidar2camera=camera_from_lidar.astype(np.float32),
            lidar2image=(intrinsic @ camera_from_lidar).astype(np.float32),
            cam_intrinsic=intrinsic.astype(np.float32),
            distortion=distortion.astype(np.float32),
        )
    rows = sorted(rows, key=lambda item: (item["track_id"], item["category"]))
    boxes = np.asarray([row["ego_box"] for row in rows], dtype=np.float32).reshape(-1, 7)
    names = np.asarray([row["category"] for row in rows])
    tracks = np.asarray([row["track_id"] for row in rows])
    velocity_valid = np.asarray([row["velocity_valid"] for row in rows], dtype=bool)
    city_velocities = np.asarray([row["city_velocity_xy"] for row in rows], dtype=np.float64).reshape(-1, 2)
    ego_velocity = np.full((len(rows), 2), np.nan, dtype=np.float32)
    if velocity_valid.any():
        ego_velocity[velocity_valid] = transform_planar_vectors(ego_from_city, city_velocities[velocity_valid]).astype(np.float32)
    detector_mask = np.asarray([(name in BRIDGE_DYNAMIC_CLASSES) and valid for name, valid in zip(names, velocity_valid)], dtype=bool)
    if not np.all(np.isfinite(boxes)) or not np.all(np.isfinite(city_from_ego)) or not np.all(np.isfinite(ego_velocity[velocity_valid])):
        raise ValueError(f"non-finite detector info at {log_id}:{timestamp}")
    return dict(
        token=f"{log_id}:{lidar_timestamp}",
        log_id=log_id,
        timestamp=int(lidar_timestamp),
        annotation_timestamp=int(timestamp),
        lidar_path=str(lidar_path),
        lidar2ego=np.eye(4, dtype=np.float32),
        ego2city=city_from_ego.astype(np.float32),
        camera_channels=RING_CAMERA_ORDER,
        cams=cam_infos,
        gt_boxes=boxes,
        gt_names=names,
        gt_track_ids=tracks,
        gt_velocity=ego_velocity,
        gt_velocity_valid=velocity_valid,
        gt_detector_mask=detector_mask,
    )


def _validate_authority_inputs(train_root: Path, audit_manifest: Path, split_file: Path):
    manifest = json.loads(audit_manifest.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "argoverse2_sensor" or manifest.get("split") != "train" or not all(manifest.get("checks", {}).values()):
        raise RuntimeError("AV2 info conversion requires the frozen PASS train audit")
    log_ids = list(manifest.get("log_ids", ()))
    if len(log_ids) != 700 or len(log_ids) != len(set(log_ids)):
        raise RuntimeError("frozen train manifest does not contain exactly 700 identities")
    if any(not (train_root / log_id).is_dir() for log_id in log_ids):
        raise RuntimeError("one or more frozen train identities are absent locally")
    split = json.loads(split_file.read_text(encoding="utf-8"))
    split_ids = set(split.get("detector_train_ids", ())) | set(split.get("uq_fit_ids", ())) | set(split.get("perception_dev_ids", ()))
    if split.get("manifest_sha256") != sha256(audit_manifest) or split_ids != set(log_ids) or len(split_ids) != 700:
        raise RuntimeError("frozen 500/100/100 split does not match the audited train manifest")
    return manifest, log_ids


def _convert_log(task):
    train_root, log_id = task
    log = Path(train_root) / log_id
    annotations, poses, extrinsics, intrinsics, lidar, cameras = _read_log(log)
    frames = _annotation_frames(annotations, poses)
    infos, emitted, camera_alignment_excluded, lidar_alignment_excluded = [], 0, 0, 0
    for timestamp, rows in sorted(frames.items()):
        try:
            infos.append(_make_info(log_id, timestamp, rows, poses, extrinsics, intrinsics, lidar, cameras))
            emitted += 1
        except AlignmentUnavailable as error:
            camera_alignment_excluded += int(error.sensor == "camera")
            lidar_alignment_excluded += int(error.sensor == "lidar")
    if emitted == 0:
        raise RuntimeError(f"all annotation anchors lack legal 7-camera alignment: {log_id}")
    return log_id, infos, dict(
        info_count=emitted,
        annotation_timestamp_count=len(frames),
        annotation_count=len(annotations),
        excluded_no_legal_nonfuture_camera_alignment=camera_alignment_excluded,
        excluded_no_lidar_reference_alignment=lidar_alignment_excluded,
    )


def create_av2_infos(train_root: Path, audit_manifest: Path, split_file: Path, output: Path, workers: int = 8):
    """Create one all-700 detector-info file and its immutable manifest."""
    if output.exists():
        raise FileExistsError(output)
    manifest, log_ids = _validate_authority_inputs(train_root, audit_manifest, split_file)
    infos = []
    counts = {}
    if not 1 <= workers <= 32:
        raise ValueError("AV2 info converter workers must be in [1, 32]")
    # executor.map yields in the audited manifest order, so concurrency cannot
    # perturb token ordering or the byte-level pickle payload.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, (log_id, log_infos, log_counts) in enumerate(executor.map(_convert_log, ((str(train_root), log_id) for log_id in log_ids)), start=1):
            infos.extend(log_infos)
            counts[log_id] = log_counts
            if index % 25 == 0 or index == len(log_ids):
                print(json.dumps({"stage": "convert", "completed_logs": index, "total_logs": len(log_ids)}), flush=True)
    per_log = {info["log_id"] for info in infos}
    if per_log != set(log_ids):
        raise RuntimeError("converter lost an audited train log")
    metadata = dict(
        dataset="argoverse2_sensor",
        split="train_700_all_infos",
        info_schema_version=INFO_SCHEMA_VERSION,
        ring_camera_order=RING_CAMERA_ORDER,
        source_manifest_sha256=sha256(audit_manifest),
        source_path_size_manifest_sha256=manifest["path_size_manifest_sha256"],
        inner_split_sha256=sha256(split_file),
        coordinate_contract="raw AV2 cuboids are current ego-frame; ego2city is city_SE3_egovehicle; detector boxes use current ego/LiDAR frame",
        image_alignment="nearest nonfuture ring image to LiDAR reference, maximum lag 75 ms",
        lidar_reference="egomotion-compensated AV2 LiDAR frame in egovehicle coordinates",
        velocity_supervision="class-consistent previous same-track city difference <=110 ms rotated into current ego by the frozen planar SO(2) polar factor; invalid history is NaN and excluded by gt_detector_mask",
        prohibited_fields=("bridge", "future_flip", "hard_case", "relation", "validation"),
    )
    payload = dict(infos=infos, metadata=metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output)
    return dict(
        analysis_id="bridge_rg0_av2_detector_migration_v1",
        status="FROZEN_ALL_700_AV2_TRAIN_DETECTOR_INFOS",
        info_file=str(output),
        info_file_sha256=sha256(output),
        source_manifest_sha256=metadata["source_manifest_sha256"],
        inner_split_sha256=metadata["inner_split_sha256"],
        log_count=len(per_log),
        info_count=len(infos),
        counts=counts,
        camera_alignment_excluded_info_count=sum(value["excluded_no_legal_nonfuture_camera_alignment"] for value in counts.values()),
        lidar_alignment_excluded_info_count=sum(value["excluded_no_lidar_reference_alignment"] for value in counts.values()),
        coordinate_contract=metadata["coordinate_contract"],
        workers=workers,
        validation_read=False,
    )
