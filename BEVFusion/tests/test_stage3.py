import copy
import gzip
import inspect
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path("tools").resolve()))

from mmdet3d.datasets.pipelines.stage3_condition import Stage3Condition
from mmdet3d.models.fusion_models.bevfusion import BEVFusion
from mmdet3d.models.fusion_models.stage3 import Stage3ReliabilityConflict
from mmdet3d.models.target_uq import (
    StructuredTargetUQAdapter,
    assert_post_nms_mapping_complete,
    cholesky_from_params,
    covariance_from_cholesky,
    distribution_records,
    gaussian_nll_diag,
    gaussian_nll_full,
    kplus1_probabilities,
    match_stage3_proposals_to_gt,
    require_pre_nms_logits,
    target_uq_loss,
    centerhead_stage3_proposals,
    transfusion_stage3_proposals,
    von_mises_nll,
)
from mmdet3d.core.bbox.coders.centerpoint_bbox_coders import CenterPointBBoxCoder
from mmdet3d.core.bbox.coders.transfusion_bbox_coder import TransFusionBBoxCoder
from tools.stage2_analyze_pilot import disagreement, threshold_for_precision
from tools.stage3_run_real_target_uq import calibrated_probabilities, export_distribution_jsonl, stage3_degradation_conditions
from tools.stage3_export_target_uq_from_checkpoint import validate_checkpoint_metadata
from tools.stage3_calibrate_camera import (
    apply_calibration_numpy,
    build_scene_folds,
    dev_gate,
    select_candidate,
)
from tools.stage3_build_labels import paired_induced
from tools.stage3_build_splits import stratified_take
from tests.test_stage2_corruptions import _sample


class _Boxes:
    def __init__(self, tensor):
        self.tensor = tensor


class Stage3Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_scene_split_is_exact_disjoint_and_deterministic(self):
        items = [
            {"token": f"scene-{index}", "stratum": ("a" if index % 2 else "b",)}
            for index in range(100)
        ]
        first, rest = stratified_take(items, 10, 20260708)
        second, _ = stratified_take(items, 10, 20260708)
        self.assertEqual([item["token"] for item in first], [item["token"] for item in second])
        self.assertFalse({item["token"] for item in first} & {item["token"] for item in rest})

    def test_actual_split_manifest_has_no_val_or_internal_overlap(self):
        path = Path("outputs/stage3/splits/manifest.json")
        if not path.exists():
            self.skipTest("split builder has not run")
        value = json.loads(path.read_text(), parse_constant=lambda token: self.fail(token))
        self.assertTrue(value["checks"]["all_stage3_intersections_zero"])
        self.assertTrue(value["checks"]["official_val_scene_and_sample_overlap_zero"])
        self.assertTrue(all(item["scene_count"] == item["sample_count"] == 0 for item in value["official_val_overlap"].values()))

    def test_precision_threshold_is_class_independent_input(self):
        low = threshold_for_precision([(0.9, True), (0.8, False), (0.7, True)], 0.8)
        high = threshold_for_precision([(0.6, True), (0.5, True), (0.4, False)], 0.8)
        self.assertNotEqual(low["threshold"], high["threshold"])

    def test_paired_conflict_rule_and_negative_controls(self):
        clean = {"conflict": False, "dual_detected": True, "class_mismatch": False, "center_gap": 0.2, "yaw_gap": 0.1}
        corrupt = {"conflict": True, "dual_detected": True, "class_mismatch": False, "center_gap": 3.0, "yaw_gap": 0.1}
        self.assertTrue(paired_induced("camera_extrinsic_rotation", clean, corrupt))
        self.assertTrue(paired_induced("camera_temporal_offset", clean, corrupt))
        self.assertFalse(paired_induced("camera_extrinsic_translation", clean, corrupt))
        self.assertFalse(paired_induced("clean", clean, corrupt))
        self.assertFalse(paired_induced("camera_extrinsic_rotation", dict(clean, conflict=True), corrupt))

    def test_complementarity_is_not_feature_conflict(self):
        camera = {"box": np.zeros(9), "label": 0}
        lidar = {"box": np.zeros(9), "label": 0}
        state = disagreement(camera, lidar)
        self.assertFalse(state["conflict"])
        self.assertFalse(paired_induced("clean", state, state))

    def test_temporal_missing_history_raises_instead_of_clean_fallback(self):
        sample = _sample()
        sample["stage3_condition"] = {"type": "camera_temporal_offset", "severity": 2, "family": "conflict"}
        with self.assertRaises(ValueError):
            Stage3Condition(temporal_history_file=None)(sample)

    def test_missing_gate_zero_and_clean_initial_gate_identity(self):
        module = Stage3ReliabilityConflict(in_channels=(4, 4), hidden_channels=4)
        features = [torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)]
        metas = [
            {"stage3_condition": {"modality_present_mask": {"camera": True, "lidar": True}}},
            {"stage3_condition": {"modality_present_mask": {"camera": False, "lidar": True}}},
        ]
        gated, maps = module(features, metas)
        self.assertTrue(torch.equal(gated[0][0], features[0][0]))
        self.assertTrue(torch.equal(gated[1][0], features[1][0]))
        self.assertEqual(float(maps["camera_gate"][1].abs().max()), 0.0)
        self.assertEqual(float(gated[0][1].abs().max()), 0.0)

    def test_loss_uses_only_defined_gt_regions_and_is_finite(self):
        module = Stage3ReliabilityConflict(in_channels=(4, 4), hidden_channels=4)
        features = [torch.randn(1, 4, 12, 12), torch.randn(1, 4, 12, 12)]
        metas = [{
            "token": "sample", "stage3_condition": {"family": "clean", "modality_present_mask": {"camera": True, "lidar": True}},
            "stage3_targets": [{"defined": True, "camera": {"reliable_detection": True}, "lidar": {"reliable_detection": False}, "conflict": False}],
        }]
        _, maps = module(features, metas)
        boxes = [_Boxes(torch.tensor([[0.0, 0.0, 0.0, 20.0, 20.0, 1.0, 0.0]]))]
        losses = module.loss(maps, boxes, metas)
        self.assertTrue(all(torch.isfinite(value) for key, value in losses.items() if key != "object_stats"))
        self.assertEqual(len(losses["object_stats"]), 1)

    def test_five_sample_forward_backward_smoke(self):
        module = Stage3ReliabilityConflict(in_channels=(4, 4), hidden_channels=4)
        features = [torch.randn(5, 4, 20, 20), torch.randn(5, 4, 20, 20)]
        targets = [{"defined": True, "camera": {"reliable_detection": True}, "lidar": {"reliable_detection": False}, "conflict": False}]
        metas = [{"token": str(index), "stage3_condition": {"family": "clean", "modality_present_mask": {"camera": True, "lidar": True}}, "stage3_targets": copy.deepcopy(targets)} for index in range(5)]
        _, maps = module(features, metas)
        boxes = [_Boxes(torch.tensor([[0.0, 0.0, 0.0, 20.0, 20.0, 1.0, 0.0]])) for _ in range(5)]
        losses = module.loss(maps, boxes, metas)
        sum(value for key, value in losses.items() if key.startswith("loss/")).backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in module.parameters()))

    def test_checkpoint_round_trip(self):
        module = Stage3ReliabilityConflict(in_channels=(4, 4), hidden_channels=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage3.pt"
            torch.save(module.state_dict(), path)
            restored = Stage3ReliabilityConflict(in_channels=(4, 4), hidden_channels=4)
            restored.load_state_dict(torch.load(path, weights_only=True))
            self.assertTrue(all(torch.equal(left, right) for left, right in zip(module.parameters(), restored.parameters())))

    def test_only_stage3_or_requested_fuser_parameters_are_trainable(self):
        model = torch.nn.Module()
        model.encoders = torch.nn.Linear(2, 2)
        model.fuser = torch.nn.Linear(2, 2)
        model.stage3 = torch.nn.Linear(2, 2)
        model.stage3_fuser_trainable = False
        BEVFusion._freeze_stage3_base(model)
        names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertEqual(names, ["stage3.weight", "stage3.bias"])

    def test_stage3_default_is_disabled(self):
        self.assertIsNone(inspect.signature(BEVFusion.__init__).parameters["stage3"].default)


class Stage3TargetUQTest(unittest.TestCase):
    def _toy(self, variant="structured"):
        torch.manual_seed(20260708)
        batch, proposals, classes, channels = 2, 4, 10, 3
        features = torch.randn(batch, proposals, channels)
        logits = torch.randn(batch, proposals, classes)
        means = {
            "center_xy": torch.randn(batch, proposals, 2),
            "z": torch.randn(batch, proposals, 1),
            "log_size": torch.zeros(batch, proposals, 3),
            "yaw": torch.zeros(batch, proposals),
            "velocity": torch.randn(batch, proposals, 2),
        }
        targets = {
            "class_target": torch.full((batch, proposals), classes),
            "positive_mask": torch.zeros(batch, proposals, dtype=torch.bool),
            "center_xy": means["center_xy"].clone(),
            "z": means["z"].clone(),
            "log_size": means["log_size"].clone(),
            "yaw": means["yaw"].clone(),
            "velocity": means["velocity"].clone(),
            "velocity_mask": torch.zeros(batch, proposals, dtype=torch.bool),
        }
        targets["positive_mask"][:, 0] = True
        targets["velocity_mask"][0, 0] = True
        targets["class_target"][targets["positive_mask"]] = 0
        adapter = StructuredTargetUQAdapter(classes, channels, variant=variant)
        return adapter, features, logits, means, targets

    def test_kplus1_categorical_probabilities_normalize(self):
        logits = torch.tensor([[[2.0, 0.0]]])
        probs = kplus1_probabilities(logits, torch.zeros(1, 1, 1))
        self.assertTrue(torch.allclose(probs["p_all"].sum(-1), torch.ones(1, 1)))
        self.assertTrue(torch.allclose(probs["p_class_given_exist"].sum(-1), torch.ones(1, 1)))

    def test_feature_classifier_modes_are_pre_nms_and_normalized(self):
        for mode, source in (
            ("feature_k1", "pre_nms_query_feature_k1_classifier"),
            ("separated", "pre_nms_query_feature_separated_classifier"),
        ):
            adapter, features, logits, means, _ = self._toy()
            adapter = StructuredTargetUQAdapter(10, 3, variant="structured", classifier_mode=mode)
            dist = adapter(logits, means, features)
            self.assertEqual(dist["class_probability_source"], source)
            self.assertTrue(torch.allclose(dist["p_all"].sum(-1), torch.ones_like(dist["p_exist"]), atol=1e-6))

    def test_calibrator_probability_outputs_normalize(self):
        logits = torch.randn(2, 4, 11)
        methods = [
            {"method": "identity"},
            {"method": "global_k1_temperature", "temperature": 2.0},
            {"method": "separated", "classification_temperature": 1.5, "existence_temperature": 2.0, "existence_bias": 0.5},
            {"method": "vector_scaling", "temperature": 1.2, "bias": [0.1] * 11},
        ]
        for method in methods:
            probs = calibrated_probabilities(logits, method)
            self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(2, 4), atol=1e-6))

    def test_stage3c_degradation_conditions_are_branch_specific(self):
        self.assertIn(("camera_lowlight", 1), stage3_degradation_conditions("camera", "stage3c"))
        self.assertIn(("lidar_beam_reduction", 3), stage3_degradation_conditions("lidar", "stage3c"))
        self.assertIn(("camera_missing", 1), stage3_degradation_conditions("fusion", "stage3c"))
        self.assertIn(("lidar_missing", 1), stage3_degradation_conditions("fusion", "stage3c"))

    def test_distribution_full_gzip_schema_keeps_probability_source(self):
        adapter, features, logits, means, targets = self._toy()
        sample = {
            "sample_token": "tok",
            "features": features[0],
            "class_logits": logits[0],
            "means": {key: value[0] for key, value in means.items()},
            "decoded_boxes": torch.zeros(4, 9),
            "proposal_ids": torch.arange(4),
            "post_nms_ids": torch.tensor([0, -1, -1, -1]),
            "post_score_for_mapping": torch.zeros(4),
            "kept_post_nms_mask": torch.tensor([True, False, False, False]),
            "targets": {
                "matched_gt_index": torch.tensor([3, -1, -1, -1]),
                "positive_mask": targets["positive_mask"][0],
                "velocity_mask": targets["velocity_mask"][0],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dist.jsonl.gz"
            result = export_distribution_jsonl(
                adapter,
                [sample],
                "camera",
                {"method": "identity"},
                torch.device("cpu"),
                1,
                path,
                split="stage3_dev",
                calibration_id="cal.json",
                checkpoint_id="ckpt.pt",
            )
            self.assertEqual(result["record_count"], 4)
            with gzip.open(path, "rt") as stream:
                row = json.loads(stream.readline(), parse_constant=lambda token: self.fail(token))
            self.assertEqual(row["split"], "stage3_dev")
            self.assertEqual(row["calibration_id"], "cal.json")
            self.assertEqual(row["checkpoint_id"], "ckpt.pt")
            self.assertEqual(row["class_probability_source"], "pre_nms_heatmap_logits")
            self.assertFalse(row["score_is_probability"])
            self.assertAlmostEqual(sum(row["calibrated_kplus1_probability"]), 1.0, places=5)

    def test_pre_nms_logits_source_is_required(self):
        require_pre_nms_logits("pre_nms_logits")
        with self.assertRaises(ValueError):
            require_pre_nms_logits("post_nms_score")

    def _transfusion_preds(self):
        batch, proposals, classes, channels = 1, 3, 10, 4
        heatmap = torch.arange(batch * classes * proposals, dtype=torch.float32).view(batch, classes, proposals) / 10.0
        return {
            "heatmap": heatmap.clone(),
            "query_heatmap_score": torch.ones(batch, 1, proposals),
            "center": torch.tensor([[[0.0, 10.0, 50.0], [0.0, 0.0, 0.0]]]),
            "height": torch.full((batch, 1, proposals), 0.5),
            "dim": torch.zeros(batch, 3, proposals),
            "rot": torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]),
            "vel": torch.zeros(batch, 2, proposals),
            "stage3_query_features": torch.randn(batch, proposals, channels),
            "stage3_query_indices": torch.tensor([[2, 1, 0]]),
            "stage3_query_labels": torch.tensor([[0, 1, 2]]),
        }

    def _transfusion_coder(self):
        return TransFusionBBoxCoder(
            pc_range=[0.0, 0.0],
            out_size_factor=1,
            voxel_size=[1.0, 1.0],
            post_center_range=[-1.0, -1.0, -5.0, 20.0, 20.0, 5.0],
            score_threshold=0.0,
            code_size=10,
        )

    def _centerhead(self):
        class Head:
            pass

        head = Head()
        head.num_classes = [10]
        head.norm_bbox = True
        head.bbox_coder = CenterPointBBoxCoder(
            pc_range=[-1.0, -1.0],
            out_size_factor=1,
            voxel_size=[1.0, 1.0],
            post_center_range=[-10.0, -10.0, -10.0, 10.0, 10.0, 10.0],
            max_num=6,
            score_threshold=0.0,
            code_size=9,
        )
        return head

    def _centerhead_preds(self):
        batch, classes, height, width = 1, 10, 4, 4
        heatmap = torch.zeros(batch, classes, height, width)
        heatmap[0, 3, 1, 2] = 5.0
        heatmap[0, 1, 0, 0] = 4.0
        return [({
            "heatmap": heatmap,
            "reg": torch.zeros(batch, 2, height, width),
            "height": torch.ones(batch, 1, height, width),
            "dim": torch.zeros(batch, 3, height, width),
            "rot": torch.cat((torch.zeros(batch, 1, height, width), torch.ones(batch, 1, height, width)), dim=1),
            "vel": torch.zeros(batch, 2, height, width),
            "stage3_shared_features": torch.randn(batch, 8, height, width),
        },)]

    def test_real_pre_nms_logits_are_extracted_before_score_mapping(self):
        pred = self._transfusion_preds()
        proposals = transfusion_stage3_proposals([[pred]], self._transfusion_coder(), num_classes=10, num_proposals=3)
        expected = pred["heatmap"].permute(0, 2, 1)
        self.assertTrue(torch.equal(proposals["class_logits"], expected))
        self.assertEqual(proposals["class_probability_source"], "pre_nms_logits")
        self.assertNotIn("p_exist", proposals)

    def test_post_nms_mapping_keeps_pre_nms_ids_without_probability_leakage(self):
        proposals = transfusion_stage3_proposals([[self._transfusion_preds()]], self._transfusion_coder())
        self.assertEqual(proposals["post_nms_ids"].tolist(), [[0, 1, -1]])
        self.assertTrue(bool((proposals["post_score_for_mapping"] >= 0).all()))
        require_pre_nms_logits(proposals["class_probability_source"])

    def test_centerhead_pre_nms_features_are_exported_without_score_probability(self):
        proposals = centerhead_stage3_proposals(self._centerhead_preds(), self._centerhead(), num_proposals=3)
        self.assertEqual(tuple(proposals["class_logits"].shape), (1, 3, 10))
        self.assertEqual(tuple(proposals["features"].shape), (1, 3, 8))
        self.assertEqual(proposals["class_probability_source"], "pre_nms_logits")
        self.assertNotIn("p_exist", proposals)
        self.assertTrue(bool((proposals["post_score_for_mapping"] >= 0).all()))

    def test_proposal_gt_matching_sets_background_and_positive_regression(self):
        proposals = transfusion_stage3_proposals([[self._transfusion_preds()]], self._transfusion_coder())
        gt = _Boxes(torch.tensor([[0.2, 0.1, 0.0, 2.0, 4.0, 1.5, 0.3, 1.0, -1.0]]))
        labels = torch.tensor([3])
        targets = match_stage3_proposals_to_gt(
            proposals["decoded_boxes"], [gt], [labels], num_classes=10, positive_distance_threshold=1.0
        )
        self.assertTrue(bool(targets["positive_mask"][0, 0]))
        self.assertFalse(bool(targets["positive_mask"][0, 1]))
        self.assertEqual(int(targets["class_target"][0, 0]), 3)
        self.assertEqual(int(targets["class_target"][0, 1]), 10)
        self.assertTrue(bool(targets["velocity_mask"][0, 0]))

    def test_post_nms_mapping_fields_are_complete(self):
        adapter, features, logits, means, _ = self._toy()
        dist = adapter(logits, means, features)
        records = distribution_records(dist, ["a", "b"], "camera")
        assert_post_nms_mapping_complete(records)
        bad = dict(records[0])
        del bad["post_nms_detection_id"]
        with self.assertRaises(ValueError):
            assert_post_nms_mapping_complete([bad])

    def test_cholesky_covariance_is_strictly_positive_definite(self):
        chol = cholesky_from_params(torch.tensor([[[0.0, 0.2, -1.0]]]))
        cov = covariance_from_cholesky(chol)
        eig = torch.linalg.eigvalsh(cov)
        self.assertTrue(bool((eig > 0).all()))

    def test_diagonal_and_full_covariance_shapes(self):
        for variant in ("diagonal", "structured"):
            adapter, features, logits, means, _ = self._toy(variant)
            dist = adapter(logits, means, features)
            self.assertEqual(tuple(dist["center_xy_cov"].shape[-2:]), (2, 2))
            self.assertEqual(tuple(dist["velocity_cov"].shape[-2:]), (2, 2))

    def test_gaussian_nll_matches_unit_normal_reference(self):
        value = torch.zeros(1, 2)
        mean = torch.zeros(1, 2)
        log_var = torch.zeros(1, 2)
        diag = gaussian_nll_diag(value, mean, log_var)
        full = gaussian_nll_full(value, mean, torch.eye(2).view(1, 2, 2))
        expected = torch.tensor([math.log(2.0 * math.pi)])
        self.assertTrue(torch.allclose(diag, expected, atol=1e-6))
        self.assertTrue(torch.allclose(full, expected, atol=1e-6))

    def test_von_mises_nll_handles_periodic_boundary(self):
        left = von_mises_nll(torch.tensor([math.pi - 0.01]), torch.tensor([-math.pi + 0.01]), torch.tensor([3.0]))
        right = von_mises_nll(torch.tensor([0.01]), torch.tensor([-0.01]), torch.tensor([3.0]))
        self.assertTrue(torch.allclose(left, right, atol=1e-5))

    def test_velocity_mask_excludes_missing_velocity_targets(self):
        adapter, features, logits, means, targets = self._toy()
        losses = target_uq_loss(adapter(logits, means, features), targets)
        self.assertEqual(float(losses["stats/stage3_target_uq/velocity_count"]), 1.0)
        self.assertTrue(torch.isfinite(losses["loss/stage3_target_uq/vel"]))

    def test_no_nan_inf_under_extreme_variance_parameters(self):
        params = torch.tensor([[[-80.0, 50.0, 80.0]]])
        cov = covariance_from_cholesky(cholesky_from_params(params))
        self.assertTrue(torch.isfinite(cov).all())

    def test_stage3_target_uq_checkpoint_round_trip(self):
        adapter, features, logits, means, _ = self._toy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target_uq.pt"
            torch.save(adapter.state_dict(), path)
            restored = StructuredTargetUQAdapter(10, 3, variant="structured")
            restored.load_state_dict(torch.load(path, weights_only=True))
            left = adapter(logits, means, features)["p_exist"]
            right = restored(logits, means, features)["p_exist"]
            self.assertTrue(torch.equal(left, right))

    def test_full_train_checkpoint_rejects_branch_or_variant_mismatch(self):
        checkpoint = {
            "branch": "camera",
            "variant": "structured",
            "classifier_mode": "feature_k1",
        }
        validate_checkpoint_metadata(checkpoint, "camera", "structured", "feature_k1")
        with self.assertRaisesRegex(ValueError, "branch"):
            validate_checkpoint_metadata(checkpoint, "lidar", "structured", "feature_k1")
        with self.assertRaisesRegex(ValueError, "variant"):
            validate_checkpoint_metadata(checkpoint, "camera", "diagonal", "feature_k1")

    def test_calibration_config_names_label_cal_only(self):
        text = Path("configs/stage3/target_uq/default.yaml").read_text()
        self.assertIn("split: stage3_label_cal", text)
        self.assertIn("use_official_val_for_selection: false", text)

    def test_strict_json_rejects_nan(self):
        with self.assertRaises(ValueError):
            json.dumps({"bad": float("nan")}, allow_nan=False)

    def test_target_uq_seed_is_deterministic(self):
        torch.manual_seed(123)
        first = StructuredTargetUQAdapter(10, 3, variant="structured").state_dict()
        torch.manual_seed(123)
        second = StructuredTargetUQAdapter(10, 3, variant="structured").state_dict()
        self.assertTrue(all(torch.equal(first[key], second[key]) for key in first))

    def test_stage3c_cal_scene_folds_are_deterministic_and_disjoint(self):
        mapping = {f"sample-{scene}-{index}": f"scene-{scene}" for scene in range(10) for index in range(3)}
        first = build_scene_folds(mapping)
        second = build_scene_folds(mapping)
        self.assertEqual(first, second)
        scene_sets = [set(fold["scenes"]) for fold in first]
        sample_sets = [set(fold["samples"]) for fold in first]
        self.assertEqual(len(set.union(*scene_sets)), 10)
        self.assertEqual(len(set.union(*sample_sets)), 30)
        self.assertFalse(any(scene_sets[i] & scene_sets[j] for i in range(5) for j in range(i)))
        self.assertFalse(any(sample_sets[i] & sample_sets[j] for i in range(5) for j in range(i)))

    def test_stage3c_cal_separated_probability_reconstruction(self):
        logits = np.asarray([[2.0, 1.0, -1.0], [0.1, 0.2, 0.3]])
        calibration = {
            "method": "separated_vector",
            "existence_temperature": 1.2,
            "class_temperature": 0.8,
            "class_bias": [0.1, -0.1],
        }
        probability = apply_calibration_numpy(logits, calibration)
        self.assertTrue(np.isfinite(probability).all())
        self.assertTrue(np.allclose(probability.sum(1), 1.0))
        conditional = probability[:, :-1] / probability[:, :-1].sum(1, keepdims=True)
        self.assertTrue(np.allclose(conditional.sum(1), 1.0))

    def test_stage3c_cal_identity_selection_and_dev_fallback(self):
        identity = {
            "ece": 0.01, "adaptive_ece": 0.01, "existence_ece": 0.01,
            "conditional_class_ece": 0.01, "categorical_nll": 0.5,
            "brier": 0.3, "existence_nll": 0.4, "finite": True,
            "normalized": True, "complexity": 0,
        }
        candidate = dict(identity, categorical_nll=0.499, brier=0.299, complexity=1)
        selected, _ = select_candidate({"identity": identity, "scalar": candidate})
        self.assertEqual(selected, "identity")
        failed = dict(identity, ece=0.02)
        passed, violations, action = dev_gate(identity, failed)
        self.assertFalse(passed)
        self.assertIn("ece", violations)
        self.assertEqual(action, "identity_fallback")

    def test_stage3c_cal_actual_fold_and_canonical_artifacts(self):
        fold_path = Path("outputs/stage3/target_uq/manifests/stage3c_cal_camera_folds.json")
        canonical_path = Path("outputs/stage3/target_uq/manifests/stage4_canonical_inputs.json")
        validation_path = Path("outputs/stage3/target_uq/metrics/stage3c_cal_camera_export_validation.json")
        if not all(path.exists() for path in (fold_path, canonical_path, validation_path)):
            self.skipTest("Stage 3C-Cal artifacts have not been generated")
        folds = json.loads(fold_path.read_text(), parse_constant=lambda value: self.fail(value))
        self.assertEqual(folds["checks"]["label_cal_dev_sample_overlap"], 0)
        self.assertEqual(sum(item["sample_count"] for item in folds["folds"]), 2814)
        canonical = json.loads(canonical_path.read_text(), parse_constant=lambda value: self.fail(value))
        self.assertFalse(canonical["checks"]["official_val_used"])
        self.assertFalse(canonical["checks"]["stage4_training_started"])
        self.assertFalse(canonical["checks"]["stage5_training_started"])
        self.assertIn("recovered", canonical["fusion"]["exports"]["train"]["path"])
        validation = json.loads(validation_path.read_text(), parse_constant=lambda value: self.fail(value))
        self.assertEqual(sum(item["record_count"] for item in validation["splits"].values()), 5626000)


if __name__ == "__main__":
    unittest.main()
