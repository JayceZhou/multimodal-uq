import tempfile
import unittest
from pathlib import Path

import mmcv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from mmcv import Config

from mmdet3d.apis.stage2_manifest import (
    Stage2ManifestWriter,
    inject_stage2_corruption,
    load_sample_records,
)
from mmdet3d.core.points import LiDARPoints
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmdet3d.datasets.pipelines.stage2_corruptions import Stage2Corruption
from mmdet3d.models.fusion_models.stage2_missing_hook import Stage2MissingHook
from tools.stage2_build_pre_pilot import build_pilot
from tools.stage2_run_pilot import build_plan


def _sample(points_per_sweep=100):
    keyframe = np.column_stack(
        [
            np.arange(points_per_sweep),
            np.zeros(points_per_sweep),
            np.ones(points_per_sweep),
            np.ones(points_per_sweep),
            np.zeros(points_per_sweep),
        ]
    )
    sweep = keyframe.copy()
    sweep[:, 0] += 1000
    sweep[:, 4] = 0.1
    intrinsics = []
    for _ in range(6):
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0] = matrix[1, 1] = 100
        matrix[0, 2] = matrix[1, 2] = 50
        intrinsics.append(matrix)
    identity = [np.eye(4, dtype=np.float32) for _ in range(6)]
    return {
        "token": "sample-token",
        "timestamp": 1_000_000,
        "img": [Image.fromarray(np.full((12, 16, 3), 128, np.uint8)) for _ in range(6)],
        "camera_channels": [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ],
        "camera_tokens": [f"camera-{index}" for index in range(6)],
        "camera_timestamps": [1_000_000] * 6,
        "camera_intrinsics": intrinsics,
        "camera2lidar": [matrix.copy() for matrix in identity],
        "lidar2camera": [matrix.copy() for matrix in identity],
        "lidar2image": [intrinsics[index].copy() for index in range(6)],
        "camera2ego": [matrix.copy() for matrix in identity],
        "lidar2ego": np.eye(4, dtype=np.float32),
        "points": LiDARPoints(np.concatenate([keyframe, sweep]), points_dim=5),
        "stage2_ring_indices": np.tile(
            np.arange(32), int(np.ceil(points_per_sweep * 2 / 32))
        )[: points_per_sweep * 2],
        "sweeps": [{"timestamp": 900_000, "sample_data_token": "sweep-token"}],
        "gt_bboxes_3d": object(),
    }


class _DropoutDataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        sample = _sample(20)
        sample["token"] = f"sample-{index}"
        output = Stage2Corruption("lidar_dropout", 2, 20260707)(sample)
        return output["points"].tensor


class Stage2CorruptionTest(unittest.TestCase):
    def test_config_injection_is_after_all_sensor_loaders(self):
        cfg = Config(
            {
                "data": {
                    "test": {
                        "pipeline": [
                            {"type": "LoadMultiViewImageFromFiles"},
                            {"type": "LoadPointsFromFile"},
                            {"type": "LoadPointsFromMultiSweeps"},
                            {"type": "ImageAug3D"},
                        ]
                    }
                }
            }
        )
        inject_stage2_corruption(cfg, "lidar_dropout", 2, 7)
        self.assertEqual(cfg.data.test.pipeline[3].type, "Stage2Corruption")
        self.assertEqual(cfg.data.test.pipeline[3].severity, 2)

        beam_cfg = Config(cfg.to_dict())
        beam_cfg.data.test.pipeline.pop(3)
        inject_stage2_corruption(beam_cfg, "lidar_beam_reduction", 2, 7)
        self.assertTrue(beam_cfg.data.test.pipeline[2].preserve_ring)

    def test_severity_zero_preserves_sensor_inputs_and_gt(self):
        for corruption_type in (
            "camera_lowlight",
            "lidar_dropout",
            "lidar_beam_reduction",
            "camera_extrinsic_translation",
            "camera_extrinsic_rotation",
            "camera_temporal_offset",
            "camera_missing",
            "lidar_missing",
        ):
            sample = _sample()
            images = sample["img"]
            points = sample["points"]
            gt = sample["gt_bboxes_3d"]
            output = Stage2Corruption(corruption_type, 0)(sample)
            self.assertIs(output, sample)
            self.assertIs(output["img"], images)
            self.assertIs(output["points"], points)
            self.assertIs(output["gt_bboxes_3d"], gt)
            self.assertEqual(output["token"], "sample-token")

    def test_lowlight_is_deterministic_monotonic_and_sensor_selective(self):
        means = []
        for severity in (0, 1, 2, 3):
            sample = _sample()
            output = Stage2Corruption("camera_lowlight", severity)(sample)
            means.append(np.asarray(output["img"][0]).mean())
        self.assertTrue(all(left > right for left, right in zip(means, means[1:])))

        first = Stage2Corruption(
            "camera_lowlight", 2, affected_sensors="CAM_FRONT"
        )(_sample())
        second = Stage2Corruption(
            "camera_lowlight", 2, affected_sensors="CAM_FRONT"
        )(_sample())
        np.testing.assert_array_equal(first["img"][0], second["img"][0])
        self.assertEqual(np.asarray(first["img"][1]).mean(), 128)
        self.assertEqual(
            first["stage2_manifest"]["affected_sensor_channels"], ["CAM_FRONT"]
        )
        legacy = Stage2Corruption(
            "camera_lowlight", 2, lowlight_candidate="legacy_candidate"
        )(_sample())
        calibrated = Stage2Corruption(
            "camera_lowlight", 2, lowlight_candidate="calibrated_candidate"
        )(_sample())
        self.assertLess(
            np.asarray(legacy["img"][0]).mean(),
            np.asarray(calibrated["img"][0]).mean(),
        )
        self.assertEqual(
            legacy["stage2_manifest"]["corruption_parameters"]["candidate"],
            "legacy_candidate",
        )
        formal = Stage2Corruption(
            "camera_lowlight", 3, lowlight_candidate="formal_candidate"
        )(_sample())
        old_s3 = Stage2Corruption(
            "camera_lowlight", 3, lowlight_candidate="calibrated_candidate"
        )(_sample())
        self.assertGreater(
            np.asarray(formal["img"][0]).mean(),
            np.asarray(old_s3["img"][0]).mean(),
        )

    def test_dropout_is_per_sweep_deterministic_and_seeded(self):
        first = Stage2Corruption("lidar_dropout", 2, 7)(_sample())
        repeat = Stage2Corruption("lidar_dropout", 2, 7)(_sample())
        other_seed = Stage2Corruption("lidar_dropout", 2, 8)(_sample())
        self.assertTrue(torch.equal(first["points"].tensor, repeat["points"].tensor))
        self.assertFalse(torch.equal(first["points"].tensor, other_seed["points"].tensor))
        record = first["stage2_manifest"]
        self.assertEqual(record["original_point_count"], 200)
        self.assertEqual(record["retained_point_count"], 100)
        sweeps = record["corruption_parameters"]["sweeps"]
        self.assertEqual({item["sweep_token"] for item in sweeps}, {"sample-token", "sweep-token"})
        self.assertEqual({item["retained_point_count"] for item in sweeps}, {50})
        self.assertEqual(first["points"].tensor.dtype, torch.float32)

    def test_dropout_count_decreases_with_severity(self):
        counts = [
            len(Stage2Corruption("lidar_dropout", severity)(_sample())["points"])
            for severity in (0, 1, 2, 3)
        ]
        self.assertEqual(counts, [200, 150, 100, 50])

    def test_beam_reduction_uses_real_fixed_ring_indices(self):
        counts = []
        for severity, expected_rings in ((1, 24), (2, 16), (3, 8)):
            output = Stage2Corruption("lidar_beam_reduction", severity)(_sample(96))
            record = output["stage2_manifest"]
            self.assertFalse(record["corruption_parameters"]["approximate_ring"])
            self.assertEqual(
                record["corruption_parameters"]["retained_ring_count"], expected_rings
            )
            self.assertEqual(
                len(record["corruption_parameters"]["sweeps"]), 2
            )
            counts.append(len(output["points"]))
        self.assertEqual(counts, [144, 96, 48])

    def test_raw_nuscenes_fifth_field_is_ring_index(self):
        info_path = Path("outputs/stage1/smoke/nuscenes_mini_infos_val_5.pkl")
        if not info_path.exists():
            self.skipTest("nuScenes mini infos are unavailable")
        info = mmcv.load(info_path)["infos"][0]
        raw = np.fromfile(info["lidar_path"], dtype=np.float32).reshape(-1, 5)
        np.testing.assert_array_equal(np.unique(raw[:, 4]), np.arange(32))

    def test_extrinsic_left_multiplication_and_projection_direction(self):
        clean = _sample()
        gt = clean["gt_bboxes_3d"]
        point = np.asarray([0.0, 0.0, 10.0, 1.0])
        clean_u = (clean["lidar2image"][0] @ point)[0] / 10.0
        output = Stage2Corruption(
            "camera_extrinsic_translation", 1, affected_sensors="CAM_FRONT"
        )(clean)
        projected = output["lidar2image"][0] @ point
        self.assertLess(projected[0] / projected[2], clean_u)
        self.assertAlmostEqual(output["camera2lidar"][0][0, 3], 0.1)
        self.assertIs(output["gt_bboxes_3d"], gt)
        self.assertIn("delta_lidar @", output["stage2_manifest"]["corruption_parameters"]["multiplication_order"])

        zero = _sample()
        before = [matrix.copy() for matrix in zero["lidar2image"]]
        Stage2Corruption("camera_extrinsic_rotation", 0)(zero)
        for left, right in zip(before, zero["lidar2image"]):
            np.testing.assert_array_equal(left, right)

    def test_temporal_offset_uses_prev_and_marks_short_chains_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            histories = {}
            for camera_index in range(6):
                entries = []
                for depth in range(1, 4):
                    path = Path(directory) / f"{camera_index}-{depth}.png"
                    Image.fromarray(np.full((12, 16, 3), depth, np.uint8)).save(path)
                    entries.append(
                        {
                            "sample_data_token": f"prev-{camera_index}-{depth}",
                            "timestamp": 1_000_000 - depth * 100_000,
                            "data_path": str(path),
                        }
                    )
                histories[f"camera-{camera_index}"] = entries
            output = Stage2Corruption(
                "camera_temporal_offset", 3, temporal_history=histories
            )(_sample())
            self.assertTrue(output["stage2_manifest"]["valid"])
            self.assertEqual(np.asarray(output["img"][0]).mean(), 3)
            self.assertEqual(
                output["stage2_manifest"]["temporal_offset_ms"]["CAM_FRONT"], -300
            )

            histories["camera-0"] = histories["camera-0"][:1]
            invalid = Stage2Corruption(
                "camera_temporal_offset", 3, temporal_history=histories
            )(_sample())
            self.assertFalse(invalid["stage2_manifest"]["valid"])
            self.assertTrue(invalid["stage2_invalid"])

    def test_feature_branch_missing_record_and_zero_hook(self):
        sample = _sample()
        output = Stage2Corruption("camera_missing", 1)(sample)
        self.assertFalse(output["stage2_manifest"]["modality_present_mask"]["camera"])
        self.assertIs(output["points"], sample["points"])
        hook = Stage2MissingHook(["camera", "lidar"], "camera")
        camera = torch.ones(1, 80, 2, 2)
        lidar = torch.ones(1, 256, 2, 2)
        features = hook(None, ([camera, lidar],))[0]
        self.assertEqual(float(features[0].abs().max()), 0.0)
        self.assertIs(features[1], lidar)

    def test_local_rng_does_not_change_global_rng_state(self):
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        Stage2Corruption("lidar_dropout", 2)(_sample())
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        np.testing.assert_array_equal(numpy_state[1], after[1])
        self.assertEqual(numpy_state[2:], after[2:])
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))

    def test_multiworker_dataloader_is_reproducible(self):
        def collect():
            loader = DataLoader(
                _DropoutDataset(), num_workers=2, multiprocessing_context="spawn"
            )
            return [batch.clone() for batch in loader]

        first, second = collect(), collect()
        self.assertEqual(len(first), len(second))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))

    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = Stage2ManifestWriter(
                directory,
                "test-run",
                config="configs/stage2/corruptions.yaml",
                dataset_version="v1.0-mini",
            )
            record = Stage2Corruption("camera_lowlight", 1)(_sample())[
                "stage2_manifest"
            ]
            writer.add(record)
            manifest = writer.close(metrics={"bad": float("nan")})
            loaded = load_sample_records(Path(directory) / "samples.jsonl")
            self.assertEqual(manifest["sample_count"], 1)
            self.assertEqual(loaded[0]["sample_token"], "sample-token")
            self.assertEqual(loaded[0]["run_id"], "test-run")
            self.assertIsNone(manifest["metrics"]["bad"])

    def test_pilot_selection_is_seeded_and_covers_requested_features(self):
        scenes = []
        conditions = (
            "camera_weak_lidar_strong",
            "lidar_weak_camera_strong",
            "both_strong",
            "both_weak",
        )
        for index in range(40):
            scenes.append(
                {
                    "sample_token": f"token-{index}",
                    "city": "boston" if index % 2 else "singapore",
                    "time_of_day": "night" if index % 3 == 0 else "day",
                    "weather": "rain" if index % 4 == 0 else "not_rain",
                    "dynamic_object_count": 1,
                    "valid_severity_3": index % 5 != 0,
                    "condition_counts": {conditions[index % 4]: 1},
                    "category_counts": {NuScenesDataset.CLASSES[index % 10]: 1},
                }
            )
        first, _ = build_pilot(scenes, 20, 7)
        second, _ = build_pilot(scenes, 20, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 20)

    def test_pilot_run_plan_has_unique_fixed_subset_tasks(self):
        tasks = build_plan()["tasks"]
        self.assertEqual(len(tasks), 39)
        self.assertEqual(len({task["run_id"] for task in tasks}), 39)
        self.assertTrue(
            all(
                task["token_set"] == "197"
                for task in tasks
                if task["corruption_type"] == "camera_temporal_offset"
            )
        )
        self.assertTrue(
            all(
                task["token_set"] == "200"
                for task in tasks
                if task["corruption_type"] != "camera_temporal_offset"
            )
        )


if __name__ == "__main__":
    unittest.main()
