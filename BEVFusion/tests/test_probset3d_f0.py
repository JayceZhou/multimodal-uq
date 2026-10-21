import itertools
import unittest

import numpy as np

from tools import probset3d_f0_score as f0


class ProbSet3DF0Test(unittest.TestCase):
    def setUp(self):
        self.parameters = f0.fixture_parameters()
        self.scene = f0.fixture_scene()
        self.grid = {
            "voxel_size": [1.0, 1.0],
            "out_size_factor": 1,
            "point_cloud_range": [-4.0, -4.0],
        }

    def test_murty_matches_exhaustive_assignments(self):
        cost = np.asarray([[1.0, 2.0, 4.0], [3.0, 1.0, 2.0]])
        expected = sorted(
            cost[0, left] + cost[1, right]
            for left, right in itertools.permutations(range(3), 2)
        )
        actual = [value for value, _ in f0.murty_k_best(cost, len(expected))]
        np.testing.assert_array_equal(actual, expected)

    def test_mb_and_pmb_are_permutation_invariant(self):
        permuted = f0.permute_scene(
            self.scene,
            query_permutation=np.asarray([2, 0, 1]),
            gt_permutation=np.asarray([1, 0]),
        )
        for model in ("E2", "E3"):
            original = f0.score_set(
                self.scene, self.parameters, model, self.grid, f0.K_BEST
            )
            changed = f0.score_set(
                permuted, self.parameters, model, self.grid, f0.K_BEST
            )
            self.assertAlmostEqual(original["nll"], changed["nll"], places=12)
            self.assertLessEqual(original["reconstruction_error"], 1e-10)
            if model == "E3":
                self.assertLessEqual(
                    original["ppp_spatial_normalization_error"], 1e-12
                )

    def test_error_injections_have_required_directions(self):
        baseline = f0.score_set(
            self.scene, self.parameters, "E2", self.grid, f0.K_BEST
        )["nll"]
        deleted = f0.score_set(
            f0.remove_query(self.scene, 0),
            self.parameters,
            "E2",
            self.grid,
            f0.K_BEST,
        )["nll"]
        duplicated = f0.score_set(
            f0.duplicate_query(self.scene, 0),
            self.parameters,
            "E2",
            self.grid,
            f0.K_BEST,
        )["nll"]
        self.assertGreater(deleted, baseline)
        self.assertGreater(duplicated, baseline)

    def test_empty_target_set_is_finite(self):
        scene = f0.copy_scene(self.scene)
        scene["gt_boxes"] = np.empty((0, 9), dtype=np.float64)
        scene["gt_labels"] = np.empty(0, dtype=np.int64)
        value = f0.score_set(
            scene, self.parameters, "E3", self.grid, f0.K_BEST
        )
        self.assertTrue(np.isfinite(value["nll"]))
        self.assertLessEqual(value["ppp_spatial_normalization_error"], 1e-12)


if __name__ == "__main__":
    unittest.main()
