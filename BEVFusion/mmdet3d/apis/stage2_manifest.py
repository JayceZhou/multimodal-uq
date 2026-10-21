import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from mmcv import ConfigDict


STAGE2_LOADERS = {
    "LoadMultiViewImageFromFiles",
    "LoadPointsFromFile",
    "LoadPointsFromMultiSweeps",
}


def inject_stage2_corruption(
    cfg,
    corruption_type,
    severity=0,
    seed=20260707,
    affected_sensors=None,
    lowlight_candidate="calibrated_candidate",
    temporal_history_file=None,
):
    """Insert one corruption after sensor loading in each test pipeline."""
    datasets = cfg.data.test if isinstance(cfg.data.test, list) else [cfg.data.test]
    for dataset in datasets:
        pipeline = dataset.pipeline
        if any(step.type == "Stage2Corruption" for step in pipeline):
            raise ValueError("test pipeline already contains Stage2Corruption")
        loader_indices = [
            index for index, step in enumerate(pipeline) if step.type in STAGE2_LOADERS
        ]
        if not loader_indices:
            raise ValueError("test pipeline has no supported sensor loader")
        if corruption_type == "lidar_beam_reduction":
            for step in pipeline:
                if step.type == "LoadPointsFromMultiSweeps":
                    step.preserve_ring = True
        family = (
            "missing"
            if corruption_type.endswith("_missing")
            else "conflict"
            if corruption_type.startswith("camera_extrinsic")
            or corruption_type == "camera_temporal_offset"
            else "degradation"
        )
        pipeline.insert(
            max(loader_indices) + 1,
            ConfigDict(
                {
                    "type": "Stage2Corruption",
                    "corruption_type": corruption_type,
                    "severity": severity,
                    "seed": seed,
                    "affected_sensors": affected_sensors or "all",
                    "family": family,
                    "lowlight_candidate": lowlight_candidate,
                    "temporal_history_file": temporal_history_file,
                }
            ),
        )


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class Stage2ManifestWriter:
    """Collect records in the parent process and write strict JSON manifests."""

    def __init__(
        self,
        output_dir,
        run_id,
        config=None,
        checkpoint=None,
        dataset_version=None,
        token_list_sha256=None,
        command=None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.config = str(config) if config is not None else None
        self.checkpoint_sha256 = _sha256(checkpoint)
        self.dataset_version = dataset_version
        self.git_commit = _git_commit()
        self.token_list_sha256 = token_list_sha256
        self.command = command or " ".join(sys.argv)
        self.records = []

    def add(self, record):
        record = dict(record)
        record.update(
            run_id=self.run_id,
            config=record.get("config", self.config),
            checkpoint_sha256=self.checkpoint_sha256,
            git_commit=self.git_commit,
            dataset_version=self.dataset_version,
        )
        self.records.append(_json_safe(record))

    def close(self, status="complete", metrics=None):
        samples_path = self.output_dir / "samples.jsonl"
        samples_path.write_text(
            "".join(
                json.dumps(record, allow_nan=False, sort_keys=True) + "\n"
                for record in self.records
            )
        )
        manifest = {
            "run_id": self.run_id,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "config": self.config,
            "checkpoint_sha256": self.checkpoint_sha256,
            "git_commit": self.git_commit,
            "dataset_version": self.dataset_version,
            "token_list_sha256": self.token_list_sha256,
            "command": self.command,
            "sample_records": samples_path.name,
            "sample_count": len(self.records),
            "metrics": _json_safe(metrics),
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        return manifest


def load_sample_records(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]
