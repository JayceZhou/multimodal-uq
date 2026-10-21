"""AV2-only raw sensor loaders used by the frozen detector migration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import get_points_type

from .transforms_3d import ImageAug3D


@PIPELINES.register_module()
class LoadAV2PointsFromFeather:
    """Load one egomotion-compensated AV2 LiDAR frame without a file copy.

    The frozen feature layout is ``x, y, z, intensity/255, 0``.  AV2's raw
    per-point firing offset and laser number are intentionally not silently
    substituted for the BEVFusion sweep-time field; this migration uses one
    LiDAR reference frame only, so the fifth feature is exactly zero.
    """

    def __init__(self, coord_type="LIDAR"):
        if coord_type != "LIDAR":
            raise ValueError("AV2 detector migration supports LiDAR coordinates only")
        self.coord_type = coord_type

    def __call__(self, results):
        try:
            import pyarrow
            import pyarrow.feather as feather
        except ModuleNotFoundError as error:  # pragma: no cover - environment gate
            raise RuntimeError("AV2 loading requires pyarrow==7.0.0") from error
        if pyarrow.__version__ != "7.0.0":
            raise RuntimeError(f"AV2 loading requires pyarrow==7.0.0, found {pyarrow.__version__}")
        path = Path(results["lidar_path"])
        table = feather.read_table(path, columns=("x", "y", "z", "intensity"))
        columns = table.to_pydict()
        points = np.column_stack(
            (
                np.asarray(columns["x"], dtype=np.float32),
                np.asarray(columns["y"], dtype=np.float32),
                np.asarray(columns["z"], dtype=np.float32),
                np.asarray(columns["intensity"], dtype=np.float32) / 255.0,
                np.zeros(table.num_rows, dtype=np.float32),
            )
        )
        if not len(points) or not np.all(np.isfinite(points)):
            raise ValueError(f"non-finite or empty AV2 LiDAR frame: {path}")
        points_class = get_points_type(self.coord_type)
        results["points"] = points_class(points, points_dim=5, attribute_dims=None)
        return results


@PIPELINES.register_module()
class LoadAV2MultiViewImageFromFiles:
    """Load exactly seven AV2 ring views and apply deterministic undistortion."""

    def __init__(self, undistort=True):
        self.undistort = bool(undistort)
        self._map_cache = {}

    @staticmethod
    def _cache_key(image, intrinsic, distortion):
        return (
            image.size,
            tuple(np.asarray(intrinsic, dtype=np.float64).reshape(-1).round(12)),
            tuple(np.asarray(distortion, dtype=np.float64).reshape(-1).round(12)),
        )

    def _undistort(self, image, intrinsic, distortion):
        intrinsic = np.asarray(intrinsic, dtype=np.float64)[:3, :3]
        distortion = np.asarray(distortion, dtype=np.float64)
        if intrinsic.shape != (3, 3) or distortion.shape != (3,):
            raise ValueError("AV2 ring camera calibration has an invalid shape")
        if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(distortion)):
            raise ValueError("AV2 ring camera calibration is non-finite")
        key = self._cache_key(image, intrinsic, distortion)
        if key not in self._map_cache:
            width, height = image.size
            radial = np.asarray((distortion[0], distortion[1], 0.0, 0.0, distortion[2]))
            self._map_cache[key] = cv2.initUndistortRectifyMap(
                intrinsic, radial, np.eye(3), intrinsic, (width, height), cv2.CV_32FC1
            )
        map_x, map_y = self._map_cache[key]
        pixels = cv2.remap(np.asarray(image), map_x, map_y, interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(pixels)

    def __call__(self, results):
        paths = results["image_paths"]
        distortions = results["camera_distortions"]
        intrinsics = results["camera_intrinsics"]
        if not (len(paths) == len(distortions) == len(intrinsics) == 7):
            raise ValueError("AV2 detector input must contain exactly seven ring cameras")
        images = []
        for path, intrinsic, distortion in zip(paths, intrinsics, distortions):
            with Image.open(path) as source:
                image = source.convert("RGB")
            images.append(self._undistort(image, intrinsic, distortion) if self.undistort else image)
        results["filename"] = paths
        results["img"] = images
        results["img_shape"] = images[0].size
        results["ori_shape"] = images[0].size
        results["pad_shape"] = images[0].size
        results["scale_factor"] = 1.0
        return results


@PIPELINES.register_module()
class AV2ImageAug3D(ImageAug3D):
    """ImageAug3D with an independently correct native shape per ring view."""

    def sample_augmentation_for_image(self, image):
        width, height = image.size
        final_height, final_width = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resized_width, resized_height = int(width * resize), int(height * resize)
            crop_height = int((1 - np.random.uniform(*self.bot_pct_lim)) * resized_height) - final_height
            crop_width = int(np.random.uniform(0, max(0, resized_width - final_width)))
            flip = bool(self.rand_flip and np.random.choice([0, 1]))
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resized_width, resized_height = int(width * resize), int(height * resize)
            crop_height = int((1 - np.mean(self.bot_pct_lim)) * resized_height) - final_height
            crop_width = int(max(0, resized_width - final_width) / 2)
            flip = False
            rotate = 0.0
        return resize, (resized_width, resized_height), (
            crop_width, crop_height, crop_width + final_width, crop_height + final_height
        ), flip, rotate

    def __call__(self, data):
        new_images, transforms = [], []
        for image in data["img"]:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation_for_image(image)
            post_rotation = torch.eye(2)
            post_translation = torch.zeros(2)
            new_image, rotation, translation = self.img_transform(
                image, post_rotation, post_translation, resize, resize_dims, crop, flip, rotate
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_images.append(new_image)
            transforms.append(transform.numpy())
        data["img"] = new_images
        data["img_aug_matrix"] = transforms
        return data
