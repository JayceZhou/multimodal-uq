"""Frozen AV2 Sensor coordinate helpers for the detector migration.

All transforms use column-vector homogeneous coordinates.  The source AV2
files name transforms by destination/source, e.g. ``city_SE3_egovehicle`` is
``T_city_ego`` and ``egovehicle_SE3_sensor`` is ``T_ego_sensor``.
"""

from __future__ import annotations

import math

import numpy as np


def quaternion_wxyz_to_matrix(quaternion) -> np.ndarray:
    """Return the finite, proper rotation corresponding to AV2 ``wxyz``."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must be a finite four-vector in wxyz order")
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion has zero norm")
    w, x, y, z = quaternion / norm
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=1e-10):
        raise ValueError("quaternion did not produce an orthonormal rotation")
    return rotation


def se3_from_av2_record(record) -> np.ndarray:
    """Build ``T_destination_source`` from one AV2 pose/calibration record."""
    rotation = quaternion_wxyz_to_matrix(
        (record["qw"], record["qx"], record["qy"], record["qz"])
    )
    translation = np.asarray(
        (record["tx_m"], record["ty_m"], record["tz_m"]), dtype=np.float64
    )
    if not np.all(np.isfinite(translation)):
        raise ValueError("SE(3) translation is non-finite")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def invert_se3(transform) -> np.ndarray:
    """Invert a finite rigid ``4 x 4`` transform without a general inverse."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("transform must be a finite 4 x 4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-10, rtol=0):
        raise ValueError("transform is not homogeneous SE(3)")
    rotation = transform[:3, :3]
    # Dataset infos retain model-facing calibrations as float32.  Their
    # orthogonality residual is normally around 1e-7 after deserialization,
    # whereas the transform application below still promotes to float64.
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-6, rtol=2e-6):
        raise ValueError("transform rotation is not orthonormal")
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def transform_points(transform, points) -> np.ndarray:
    """Apply an SE(3) transform to a ``(..., 3)`` point array."""
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1:] != (3,) or not np.all(np.isfinite(points)):
        raise ValueError("points must have a finite final dimension of three")
    return points @ transform[:3, :3].T + transform[:3, 3]


def transform_vectors(transform, vectors) -> np.ndarray:
    """Apply only the rotational part of an SE(3) transform to vectors."""
    transform = np.asarray(transform, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape[-1:] != (3,) or not np.all(np.isfinite(vectors)):
        raise ValueError("vectors must have a finite final dimension of three")
    return vectors @ transform[:3, :3].T


def planar_rotation(transform) -> np.ndarray:
    """Return the SO(2) polar factor of an ego/city transform's XY block.

    BRIDGE velocity and heading are two-dimensional ground-plane quantities.
    Applying a full 3-D vehicle pose and then discarding its z component is
    not reversible on grades or banked roads.  The polar factor is the unique
    closest planar rotation and its inverse is exactly the corresponding
    factor of the inverse SE(3) transform.
    """
    transform = np.asarray(transform, dtype=np.float64)
    block = transform[:2, :2]
    if block.shape != (2, 2) or not np.all(np.isfinite(block)):
        raise ValueError("transform has an invalid XY block")
    left, _, right_t = np.linalg.svd(block)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    return rotation


def transform_planar_vectors(transform, vectors) -> np.ndarray:
    """Apply the frozen SO(2) ground-plane rotation to N x 2 row vectors."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape[-1:] != (2,) or not np.all(np.isfinite(vectors)):
        raise ValueError("planar vectors must have a finite final dimension of two")
    return vectors @ planar_rotation(transform).T


def av2_quaternion_yaw(record) -> float:
    """Physical AV2 object heading in its stated ego/city XY frame."""
    rotation = quaternion_wxyz_to_matrix(
        (record["qw"], record["qx"], record["qy"], record["qz"])
    )
    return math.atan2(rotation[1, 0], rotation[0, 0])


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def av2_cuboid_to_lidar_box(record) -> np.ndarray:
    """Convert one AV2 ego-frame cuboid to this fork's LiDAR box convention.

    AV2 supplies a gravity centre and a quaternion whose local x-axis is its
    length direction.  This fork stores a bottom-centre LiDAR box and rotates
    row vectors, hence the required yaw sign inversion.  AV2's length/width
    field order already matches the fork's x/y size order.
    """
    length = float(record["length_m"])
    width = float(record["width_m"])
    height = float(record["height_m"])
    center = np.asarray((record["tx_m"], record["ty_m"], record["tz_m"]), dtype=np.float64)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite((length, width, height))):
        raise ValueError("AV2 cuboid has non-finite fields")
    if min(length, width, height) <= 0:
        raise ValueError("AV2 cuboid dimensions must be strictly positive")
    return np.asarray(
        (
            center[0],
            center[1],
            center[2] - height / 2.0,
            length,
            width,
            height,
            wrap_angle(-av2_quaternion_yaw(record)),
        ),
        dtype=np.float64,
    )


def detector_boxes_ego_to_city(boxes, city_from_ego) -> np.ndarray:
    """The sole detector-output export from current ego to city coordinates.

    ``boxes`` is N x 9 in this BEVFusion fork's LiDAR convention
    ``(x, y, z_bottom, length, width, height, yaw, vx, vy)``.  The returned
    array keeps that convention in city axes, including correctly rotated
    velocity and physical box heading.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] not in (7, 9) or not np.all(np.isfinite(boxes)):
        raise ValueError("detector boxes must be finite N x 7 or N x 9 arrays")
    city_from_ego = np.asarray(city_from_ego, dtype=np.float64)
    output = boxes.copy()
    output[:, :3] = transform_points(city_from_ego, boxes[:, :3])
    physical_ego = np.stack((np.cos(-boxes[:, 6]), np.sin(-boxes[:, 6])), axis=1)
    physical_city = transform_planar_vectors(city_from_ego, physical_ego)
    output[:, 6] = np.asarray(
        [wrap_angle(-math.atan2(vector[1], vector[0])) for vector in physical_city], dtype=np.float64
    )
    if boxes.shape[1] == 9:
        output[:, 7:9] = transform_planar_vectors(city_from_ego, boxes[:, 7:9])
    return output


def detector_boxes_city_to_ego(boxes, city_from_ego) -> np.ndarray:
    """Inverse of :func:`detector_boxes_ego_to_city`, used only for tests."""
    return detector_boxes_ego_to_city(boxes, invert_se3(city_from_ego))
