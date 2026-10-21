import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.stage3d_b_native_assignment import assignment_from_cost, detection_gt
from tools.stage3d_b_collect_labels import SCHEMES
from tools.stage3d_b_train_branch import adapter, loss_with_ignore, state_sha256
from tools.stage3d_b_aggregate import native_smoke_passed, third_branch_noncollapse


class Stage3DBAssignmentTest(unittest.TestCase):
    def test_one_to_one_and_two_meter_gate(self):
        cost = np.array([[0.0, 2.0], [1.0, 0.0], [0.5, 0.5]])
        proposal = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        gt = np.array([[0.5, 0.0], [10.5, 0.0]])
        matched, distance, ignore, _ = assignment_from_cost(cost, proposal, gt)
        self.assertEqual(matched.tolist(), [0, 1, -1])
        self.assertEqual(len(set(matched[matched >= 0].tolist())), 2)
        self.assertTrue(np.all(distance[matched >= 0] <= 2.0))
        self.assertFalse(ignore.any())

    def test_ignore_boundaries(self):
        proposal = np.array([[2.0, 0.0], [2.0001, 0.0], [4.0, 0.0], [4.0001, 0.0]])
        gt = np.array([[0.0, 0.0]])
        # Force the first proposal to take the only assignment.
        cost = np.array([[0.0], [1.0], [2.0], [3.0]])
        matched, _, ignore, nearest = assignment_from_cost(cost, proposal, gt)
        self.assertEqual(matched.tolist(), [0, -1, -1, -1])
        self.assertEqual(ignore.tolist(), [False, True, True, False])
        self.assertTrue(np.allclose(nearest, [2.0, 2.0001, 4.0, 4.0001]))

    def test_empty_gt(self):
        matched, distance, ignore, nearest = assignment_from_cost(
            np.empty((2, 0)), np.zeros((2, 2)), np.empty((0, 2))
        )
        self.assertEqual(matched.tolist(), [-1, -1])
        self.assertFalse(ignore.any())
        self.assertTrue(np.isnan(distance).all() and np.isnan(nearest).all())

    def test_detection_gt_empty_frame(self):
        indices, boxes, labels = detection_gt({
            "gt_names": np.asarray([]),
            "num_lidar_pts": np.asarray([]),
            "gt_boxes": np.empty((0, 7), dtype=np.float32),
            "gt_velocity": np.empty((0, 2), dtype=np.float32),
        })
        self.assertEqual(indices.dtype, np.int64)
        self.assertEqual(boxes.shape, (0, 9))
        self.assertEqual(labels.shape, (0,))

    def test_deterministic_and_strict_json(self):
        cost = np.array([[0.0, 1.0], [1.0, 0.0]])
        xy = np.array([[0.0, 0.0], [1.0, 0.0]])
        first = assignment_from_cost(cost, xy, xy)
        second = assignment_from_cost(cost, xy, xy)
        for left, right in zip(first, second):
            self.assertTrue(np.array_equal(left, right, equal_nan=True))
        with self.assertRaises(ValueError):
            json.dumps({"bad": float("nan")}, allow_nan=False)

    def test_label_parent_layout(self):
        with tempfile.TemporaryDirectory() as root:
            for scheme in SCHEMES:
                (Path(root) / "labels" / "camera" / scheme).mkdir(parents=True, exist_ok=True)
            self.assertTrue(all((Path(root) / "labels" / "camera" / scheme).is_dir() for scheme in SCHEMES))

    def test_initialization_hash_and_ignore_loss(self):
        import torch
        torch.manual_seed(20260708)
        model = adapter(4)
        self.assertEqual(state_sha256(model.state_dict()), state_sha256(model.state_dict()))
        features = torch.zeros(1, 2, 4)
        logits = torch.zeros(1, 2, 10)
        means = {
            "center_xy": torch.zeros(1, 2, 2), "z": torch.zeros(1, 2, 1),
            "log_size": torch.zeros(1, 2, 3), "yaw": torch.zeros(1, 2),
            "velocity": torch.zeros(1, 2, 2),
        }
        targets = {
            "class_target": torch.tensor([[10, 0]]),
            "classification_mask": torch.tensor([[True, False]]),
            "positive_mask": torch.zeros(1, 2, dtype=torch.bool),
            "velocity_mask": torch.zeros(1, 2, dtype=torch.bool),
            "center_xy": torch.zeros(1, 2, 2), "z": torch.zeros(1, 2, 1),
            "log_size": torch.zeros(1, 2, 3), "yaw": torch.zeros(1, 2),
            "velocity": torch.zeros(1, 2, 2),
        }
        distribution = model(logits, means, features)
        first, _ = loss_with_ignore(distribution, targets)
        targets["class_target"][0, 1] = 9
        second, _ = loss_with_ignore(distribution, targets)
        self.assertEqual(float(first), float(second))

    def test_native_smoke_forbidden_source_flags_must_be_false(self):
        smoke = {"status": "pass", "checks": {
            "B_C_positive_identical": True, "deterministic_repeat": True,
            "distance_gates": True, "one_to_one": True, "strict_json": True,
            "post_nms_score_used_as_probability": False,
            "target_uq_probability_used": False,
        }}
        self.assertTrue(native_smoke_passed(smoke))
        smoke["checks"]["target_uq_probability_used"] = True
        self.assertFalse(native_smoke_passed(smoke))

    def test_third_branch_noncollapse_uses_frozen_point_tolerances(self):
        import tools.stage3d_b_aggregate as aggregate
        stability = {
            reference: {
                metric: {"estimate": aggregate.INCREASE_LIMITS[metric], "ci95": [-1.0, 1.0]}
                for metric in aggregate.PROBABILITY_METRICS
            }
            for reference in aggregate.REFERENCES
        }
        gate = {"hard_validity": {"passed": True}, "stability": stability}
        self.assertTrue(third_branch_noncollapse(gate))
        gate["stability"]["hungarian"]["conditional_class_nll"]["estimate"] = 0.020001
        self.assertFalse(third_branch_noncollapse(gate))


if __name__ == "__main__":
    unittest.main()
