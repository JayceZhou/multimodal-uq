import json
from pathlib import Path

from mmdet.datasets.builder import PIPELINES

from .stage2_corruptions import Stage2Corruption


@PIPELINES.register_module()
class Stage3Condition:
    """Apply the per-info Stage 3 condition through the frozen Stage 2 code."""

    def __init__(self, seed=20260708, temporal_history_file=None):
        self.seed = int(seed)
        self.temporal_history = (
            json.loads(Path(temporal_history_file).read_text())
            if temporal_history_file else None
        )
        self.transforms = {}

    def __call__(self, results):
        condition = dict(results.get("stage3_condition", {"type": "clean", "severity": 0, "family": "clean"}))
        corruption_type = condition.get("type", "clean")
        if corruption_type == "clean":
            transform_type, severity = "camera_lowlight", 0
        else:
            transform_type, severity = corruption_type, int(condition["severity"])
        key = transform_type, severity
        if key not in self.transforms:
            self.transforms[key] = Stage2Corruption(
                transform_type,
                severity,
                seed=self.seed,
                lowlight_candidate="formal_candidate",
                temporal_history=self.temporal_history,
            )
        results = self.transforms[key](results)
        if results.get("stage2_invalid"):
            raise ValueError(
                f"Stage 3 temporal sample {results['token']} lacks requested history; clean fallback is forbidden"
            )
        presence = results["stage2_manifest"]["modality_present_mask"]
        condition.update(
            type=corruption_type,
            severity=int(condition.get("severity", 0)),
            family=condition.get("family", "clean" if corruption_type == "clean" else results["stage2_manifest"]["corruption_family"]),
            modality_present_mask=presence,
        )
        results["stage3_condition"] = condition
        return results
