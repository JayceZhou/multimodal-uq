import json
import tempfile
import unittest
from pathlib import Path

import mmcv
import numpy as np
import torch
from torch import nn

from mmdet3d.apis.stage1_export import Stage1Exporter


class _Boxes:
    def __init__(self):
        self.tensor = torch.tensor([[1, 2, 3, 4, 5, 6, 0.1, 0.2, 0.3]])


class _Dataset:
    CLASSES = tuple(str(index) for index in range(10))
    version = "v1.0-mini"
    data_infos = [{"token": "sample-token", "timestamp": 123456}]


class _Model:
    def __init__(self, sensors):
        self.encoders = {}
        if "camera" in sensors:
            self.encoders["camera"] = {"vtransform": nn.Identity()}
        if "lidar" in sensors:
            self.encoders["lidar"] = {"backbone": nn.Identity()}
        self.fuser = nn.Identity() if len(sensors) > 1 else None


class Stage1ExporterTest(unittest.TestCase):
    def _export(self, root, sensors, level="all"):
        config = root / "config.yaml"
        checkpoint = root / "checkpoint.pth"
        config.write_text("model: {}\n")
        checkpoint.write_bytes(b"checkpoint")
        exporter = Stage1Exporter(
            _Model(sensors),
            _Dataset(),
            root,
            level,
            config,
            checkpoint,
            run_id="test-run-" + "-".join(sensors),
            bev_ranges={name: [-10, -10, 10, 10] for name in Stage1Exporter.LEVELS},
        )
        exporter.captured = {
            name: torch.ones(1, 2, 4, 5)
            for name in ("camera_bev", "lidar_bev")
            if name.split("_")[0] in sensors
        }
        exporter.captured["prefusion"] = list(exporter.captured.values())
        exporter.captured["fused_bev"] = torch.ones(1, 3, 4, 5)
        exporter.write_batch(
            [{"boxes_3d": _Boxes(), "scores_3d": torch.tensor([0.8]), "labels_3d": torch.tensor([2])}]
        )
        exporter.close(metrics={"undefined": float("nan")})
        return exporter

    def test_three_modalities_prediction_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sensors, modality in [
                (["camera"], "camera"),
                (["lidar"], "lidar"),
                (["camera", "lidar"], "fusion"),
            ]:
                exporter = self._export(root, sensors)
                path = exporter.run_dir / "predictions" / modality / "sample-token.pt"
                record = torch.load(path, weights_only=False)
                self.assertEqual(record["sample_token"], "sample-token")
                self.assertEqual(record["modality"], modality)
                self.assertEqual(tuple(record["class_logits"].shape), (1, 10))
                self.assertEqual(tuple(record["box_mean"].shape), (1, 9))
                self.assertTrue(torch.isfinite(record["class_logits"]).all())
                self.assertTrue(torch.isfinite(record["box_mean"]).all())

    def test_feature_files_are_reloadable_and_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            exporter = self._export(Path(directory), ["camera", "lidar"])
            manifest = json.loads(
                exporter.manifest_path.read_text(),
                parse_constant=lambda value: self.fail(f"invalid JSON: {value}"),
            )
            self.assertIsNone(manifest["metrics"]["undefined"])
            for level in Stage1Exporter.LEVELS:
                path = exporter.run_dir / "features" / level / "sample-token.pt"
                record = torch.load(path, weights_only=False)
                self.assertEqual(record["sample_token"], "sample-token")
                self.assertEqual(record["shape"], list(record["feature"].shape))
                self.assertEqual(record["dtype"], "float16")
                self.assertTrue(torch.isfinite(record["feature"]).all())
                expected_modality = (
                    level.split("_")[0]
                    if level in ("camera_bev", "lidar_bev")
                    else "fusion"
                )
                self.assertEqual(record["modality"], expected_modality)

    def test_mini_tokens_timestamps_and_calibration_are_aligned(self):
        root = Path("data/nuscenes")
        info_path = root / "nuscenes_mini_infos_val.pkl"
        sample_path = root / "v1.0-mini/sample.json"
        if not info_path.exists() or not sample_path.exists():
            self.skipTest("nuScenes mini data is not available")
        exists = lambda path: Path(path).exists() or (root / path).exists()
        raw = {item["token"]: item for item in json.loads(sample_path.read_text())}
        infos = mmcv.load(info_path)["infos"][:5]
        for info in infos:
            self.assertEqual(info["timestamp"], raw[info["token"]]["timestamp"])
            self.assertTrue(exists(info["lidar_path"]))
            self.assertEqual(len(info["cams"]), 6)
            for camera in info["cams"].values():
                self.assertTrue(exists(camera["data_path"]))
                self.assertEqual(np.asarray(camera["sensor2lidar_rotation"]).shape, (3, 3))
                self.assertEqual(np.asarray(camera["sensor2lidar_translation"]).shape, (3,))
                self.assertEqual(np.asarray(camera["cam_intrinsic"]).shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
