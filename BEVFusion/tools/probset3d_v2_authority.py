#!/usr/bin/env python3
"""Fresh deterministic ProbSet3D-V2 clean query authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import tempfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/probset3d-v2-matplotlib")

import numpy as np
import torch
from mmdet3d.datasets import build_dataloader, build_dataset

from tools import probset3d_f1_authority as base
from tools import stage3_run_real_target_uq as stage3
from tools.probset3d_f0_export import exact_local_max
from tools.probset3d_v2_sweeps import (
    CONTRACT_ID,
    bind_pipeline,
    canonicalize_infos,
    selection_digest,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = ROOT / "outputs/probset3d_v2/authority"
RUN_ROOTS = {name: V2_ROOT / name for name in ("run_a", "run_b", "run_c")}
DECISION_ROOT = V2_ROOT / "decision"
PROBE_ROOT = V2_ROOT / "host_probe"
COMPARATOR_FREEZE = V2_ROOT / "comparator_fix_freeze.json"
SWEEP_ACCEPTANCE = ROOT / "outputs/probset3d_v2/sweep_authority/sweep_authority_acceptance.json"
PROTOCOL = ROOT / "probset3d_v2/P0_V2_PROTOCOL_AMENDMENT.md"
CONTRACT = ROOT / "probset3d_v2/DETERMINISTIC_SWEEP_AUTHORITY.md"
SCHEDULES = {
    "run_a": ("canonical", 0),
    "run_b": ("reverse", 2),
    "run_c": ("rotate", 4),
}
COMPARATOR_NEW_EXCLUSIONS = (
    "query_authority_manifest.json.validation",
    "query_authority_manifest.json.validation_sha256",
    "probset3d_v2_binding.json.authority_manifest_sha256",
    "probset3d_v2_binding.json.authority_validation_sha256",
)


def strict_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen V2 artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def configure_determinism(device: int) -> None:
    random.seed(base.SEED)
    np.random.seed(base.SEED)
    torch.manual_seed(base.SEED)
    torch.cuda.manual_seed_all(base.SEED)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def select_dataset_v2(
    cfg,
    info_path,
    shard_index,
    shard_count,
    traversal,
    workers,
    sample_scene,
):
    dataset_cfg, samples_per_gpu = stage3.dataset_cfg_for(cfg, str(info_path))
    bind_pipeline(dataset_cfg)
    dataset = build_dataset(dataset_cfg)
    dataset.data_infos = canonicalize_infos(dataset.data_infos)
    full_count = len(dataset.data_infos)
    canonical = sorted(
        range(full_count),
        key=lambda index: (
            sample_scene[dataset.data_infos[index]["token"]],
            dataset.data_infos[index]["token"],
        ),
    )
    selected = np.asarray(
        np.array_split(np.asarray(canonical, dtype=np.int64), shard_count)[
            shard_index
        ],
        dtype=np.int64,
    )
    if traversal == "reverse":
        selected = selected[::-1]
    elif traversal == "rotate" and len(selected):
        selected = np.roll(selected, len(selected) // 3)
    dataset.data_infos = [dataset.data_infos[int(index)] for index in selected]
    if hasattr(dataset, "flag"):
        dataset.flag = np.asarray(dataset.flag)[selected]
    loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers,
        dist=False,
        shuffle=False,
        seed=base.SEED,
    )
    return dataset, loader, full_count


def bind_base() -> None:
    if not json.loads(SWEEP_ACCEPTANCE.read_text(encoding="utf-8"))["accepted"]:
        raise ValueError("V2 sweep authority is not accepted")
    base.RUN_ROOTS = RUN_ROOTS
    base.A0_ROOT = V2_ROOT
    base.select_dataset_a0 = select_dataset_v2


def export(args) -> None:
    expected = SCHEDULES[args.run]
    if (args.traversal, args.workers) != expected:
        raise ValueError(f"V2 schedule for {args.run} is frozen to {expected}")
    bind_base()
    base.export(args)


def smoke(args) -> None:
    root = Path(tempfile.mkdtemp(prefix="probset3d-v2-authority-", dir="/tmp"))
    base.RUN_ROOTS = {"run_a": root / "run_a"}
    base.A0_ROOT = root
    base.select_dataset_a0 = select_dataset_v2
    probe = argparse.Namespace(
        run="run_a",
        split="dev",
        device=args.device,
        shard_index=0,
        shard_count=1,
        part_size=2,
        max_samples=2,
        traversal="canonical",
        workers=0,
    )
    base.export(probe)
    print(json.dumps({"accepted": True, "smoke_root": str(root)}, sort_keys=True))


def validate(args) -> None:
    bind_base()
    base.validate(args)
    run_root = RUN_ROOTS[args.run]
    p0 = json.loads(base.P0.read_text(encoding="utf-8"))
    sweep_digests = {}
    for split in base.SPLITS:
        with (ROOT / p0["splits"][split]["info_path"]).open("rb") as stream:
            payload = pickle.load(stream)
        infos = payload["infos"] if isinstance(payload, dict) else payload
        sweep_digests[split] = selection_digest(infos)
    binding = {
        "accepted": True,
        "authority_manifest": "query_authority_manifest.json",
        "authority_manifest_sha256": sha256(
            run_root / "query_authority_manifest.json"
        ),
        "authority_validation": "query_authority_validation.json",
        "authority_validation_sha256": sha256(
            run_root / "query_authority_validation.json"
        ),
        "base_exporter": str(base.Path(base.__file__).resolve().relative_to(ROOT)),
        "base_exporter_sha256": sha256(Path(base.__file__).resolve()),
        "contract": str(CONTRACT.relative_to(ROOT)),
        "contract_id": CONTRACT_ID,
        "contract_sha256": sha256(CONTRACT),
        "p0_v2_amendment": str(PROTOCOL.relative_to(ROOT)),
        "p0_v2_amendment_sha256": sha256(PROTOCOL),
        "run": args.run,
        "schedule": {
            "traversal": SCHEDULES[args.run][0],
            "workers": SCHEDULES[args.run][1],
        },
        "schema_version": "probset3d_v2_authority_binding_v1",
        "scientific_metrics_generated_or_read": False,
        "sweep_contract_helper": str(Path(__file__).resolve().relative_to(ROOT)),
        "sweep_contract_helper_sha256": sha256(Path(__file__).resolve()),
        "sweep_selection_digests": sweep_digests,
        "sweep_selection_rule": "nearest nine past sweeps after canonical lag/timestamp/path sort",
    }
    strict_write(run_root / "probset3d_v2_binding.json", binding)
    print(json.dumps(binding, indent=2, sort_keys=True))


def normalized_manifest(run_root: Path):
    validation = json.loads(
        (run_root / "query_authority_validation.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (run_root / "query_authority_manifest.json").read_text(encoding="utf-8")
    )
    binding = json.loads(
        (run_root / "probset3d_v2_binding.json").read_text(encoding="utf-8")
    )
    for value in (validation, manifest, binding):
        value.pop("run", None)
    manifest.pop("worker_counts", None)
    manifest.pop("input_traversals", None)
    manifest.pop("validation", None)
    manifest.pop("validation_sha256", None)
    binding.pop("schedule", None)
    binding.pop("authority_manifest_sha256", None)
    binding.pop("authority_validation_sha256", None)
    for split in manifest["splits"].values():
        split["content_files"] = [
            {
                key: row[key]
                for key in (
                    "compressed_sha256",
                    "uncompressed_sha256",
                    "size_bytes",
                )
            }
            for row in split["content_files"]
        ]
    return validation, manifest, binding


def freeze_comparator() -> None:
    original_decision_path = DECISION_ROOT / "authority_decision.json"
    original = json.loads(original_decision_path.read_text(encoding="utf-8"))
    normalized = [normalized_manifest(root) for root in RUN_ROOTS.values()]
    scientific_content_passed = (
        all(value["accepted"] for value in original["splits"].values())
        and original["host_byte_identity"]["accepted"]
    )
    accepted = normalized[0] == normalized[1] == normalized[2] and scientific_content_passed
    value = {
        "accepted": accepted,
        "authority_content_modified": False,
        "comparator_source": str(Path(__file__).resolve().relative_to(ROOT)),
        "comparator_source_sha256": sha256(Path(__file__).resolve()),
        "f0_v2_only_authorized": accepted,
        "f1_authorized": False,
        "newly_excluded_run_dependent_fields": list(COMPARATOR_NEW_EXCLUSIONS),
        "original_authority_decision": str(original_decision_path.relative_to(ROOT)),
        "original_authority_decision_sha256": sha256(original_decision_path),
        "preexisting_protocol_exclusions": [
            "*.run",
            "query_authority_manifest.json.worker_counts",
            "query_authority_manifest.json.input_traversals",
            "probset3d_v2_binding.json.schedule",
            "content_files[].path",
        ],
        "retained_strict_comparisons": [
            "all normalized compressed and uncompressed content hashes and sizes",
            "all sample, scene, and query counts",
            "all per-key query scientific content",
            "all assignment and duplicate-graph content",
            "all split summaries",
            "independent repeated-host byte identity",
        ],
        "run_artifact_sha256": {
            run: {
                name: sha256(root / name)
                for name in (
                    "query_authority_manifest.json",
                    "query_authority_validation.json",
                    "probset3d_v2_binding.json",
                )
            }
            for run, root in RUN_ROOTS.items()
        },
        "schema_version": "probset3d_v2_comparator_fix_freeze_v1",
        "scientific_content_comparison_passed": scientific_content_passed,
    }
    strict_write(COMPARATOR_FREEZE, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    if not accepted:
        raise SystemExit(1)


def compare() -> None:
    split_results = {}
    passed = True
    for run_root in RUN_ROOTS.values():
        for name in (
            "query_authority_manifest.json",
            "query_authority_validation.json",
            "probset3d_v2_binding.json",
        ):
            if not (run_root / name).is_file():
                raise FileNotFoundError(run_root / name)
    for split in base.SPLITS:
        per_run = {}
        per_sample = {}
        summaries = {}
        for run, root in RUN_ROOTS.items():
            digest, samples = base.canonical_sample_digests(
                root / "clean_query", split, include_scene=True
            )
            per_run[run] = digest
            per_sample[run] = samples
            summaries[run] = json.loads(
                (root / "query_authority_validation.json").read_text(
                    encoding="utf-8"
                )
            )["splits"][split]
        content_equal = per_sample["run_a"] == per_sample["run_b"] == per_sample["run_c"]
        summary_equal = summaries["run_a"] == summaries["run_b"] == summaries["run_c"]
        split_passed = content_equal and summary_equal
        passed &= split_passed
        split_results[split] = {
            "accepted": split_passed,
            "normalized_uncompressed_sha256": per_run,
            "per_key_content_identical": content_equal,
            "summary_identical": summary_equal,
            "summary": summaries["run_a"],
        }
    normalized = {
        run: normalized_manifest(root) for run, root in RUN_ROOTS.items()
    }
    json_equal = normalized["run_a"] == normalized["run_b"] == normalized["run_c"]
    passed &= json_equal
    host_probe = compare_host_probes()
    passed &= host_probe["accepted"]
    decision = {
        "accepted": passed,
        "endpoint": (
            "PASS — PROBSET3D-V2 AUTHORITY ESTABLISHED"
            if passed
            else "STOP PROBSET3D-V2 AUTHORITY REBASELINE"
        ),
        "f1_training_authorized": False,
        "host_byte_identity": host_probe,
        "normalized_scientific_json_identical": json_equal,
        "old_a0_data_promoted": False,
        "old_authority_used_as_identity_baseline": False,
        "schema_version": "probset3d_v2_authority_decision_v1",
        "scientific_metrics_generated_or_read": False,
        "splits": split_results,
    }
    strict_write(DECISION_ROOT / "authority_decision.json", decision)
    rows = [
        "# ProbSet3D-V2 Authority Decision",
        "",
        f"Decision: **{decision['endpoint']}**",
        "",
        "| Gate | Result |",
        "| --- | --- |",
        f"| A/B/C normalized query content | {'PASS' if all(v['accepted'] for v in split_results.values()) else 'FAIL'} |",
        f"| A/B/C normalized manifests | {'PASS' if json_equal else 'FAIL'} |",
        f"| Independent repeated-host byte identity | {'PASS' if host_probe['accepted'] else 'FAIL'} |",
        "| Old A0/INVALID data promoted | PASS — no |",
        "| F1 training performed | PASS — no |",
        "",
        "This decision authorizes only the complete F0-V2 rerun.",
    ]
    report = DECISION_ROOT / "authority_decision.md"
    if report.exists():
        raise FileExistsError(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


def tensor_sha(value) -> str:
    if hasattr(value, "tensor"):
        value = value.tensor
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def host_probe(args) -> None:
    configure_determinism(args.device)
    p0 = json.loads(base.P0.read_text(encoding="utf-8"))
    info_path = ROOT / p0["splits"]["dev"]["info_path"]
    with info_path.open("rb") as stream:
        payload = pickle.load(stream)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    canonical = sorted(infos, key=lambda item: item["token"])
    zero = [item for item in canonical if not item.get("sweeps")]
    full = [item for item in canonical if len(item.get("sweeps", ())) > 9]
    chosen = (zero[:2] + full[:6])[:8]
    tokens = [item["token"] for item in chosen]
    cfg = stage3.load_cfg(str(base.CONFIG))
    dataset_cfg, _ = stage3.dataset_cfg_for(cfg, str(info_path))
    bind_pipeline(dataset_cfg)
    dataset = build_dataset(dataset_cfg)
    dataset.data_infos = canonicalize_infos(chosen)
    if hasattr(dataset, "flag"):
        dataset.flag = np.asarray(dataset.flag[: len(chosen)])
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        seed=base.SEED,
    )
    model = stage3.build_frozen_model(cfg, str(base.CHECKPOINT), args.device)
    module = model.module if hasattr(model, "module") else model
    state_before = base.state_sha256(module)
    head = module.heads["object"]
    captured = []

    def capture_dense(_module, _inputs, output):
        captured.append(output.detach())

    handle = head.heatmap_head.register_forward_hook(capture_dense)
    records = []
    try:
        for expected, data in zip(tokens, loader):
            point_input = stage3.dc_first(data, "points")
            with torch.inference_mode():
                output = model(
                    return_loss=False,
                    rescale=True,
                    return_stage3_target_uq=True,
                    stage3_branch="fusion",
                    stage3_num_proposals=200,
                    **data,
                )[0]
            dense = captured.pop(0)
            item = output["stage3_target_uq"]
            if item["sample_token"] != expected:
                raise ValueError("host probe sample order changed")
            _, local_mask, seed_index, seed_class, seed_score = exact_local_max(
                head, dense
            )
            means = {
                name: tensor_sha(value)
                for name, value in sorted(item["means"].items())
            }
            records.append(
                {
                    "sample_token": expected,
                    "raw_point_input": tensor_sha(point_input),
                    "dense_heatmap": tensor_sha(dense),
                    "local_max_mask": tensor_sha(local_mask),
                    "top200_indices": tensor_sha(seed_index),
                    "seed_labels": tensor_sha(seed_class),
                    "seed_scores": tensor_sha(seed_score),
                    "query_features": tensor_sha(item["features"]),
                    "classification_logits": tensor_sha(item["class_logits"]),
                    "regression_means": means,
                    "decoded_box_means": tensor_sha(item["decoded_boxes"]),
                    "final_detection_boxes": tensor_sha(output["boxes_3d"]),
                    "final_detection_scores": tensor_sha(output["scores_3d"]),
                    "final_detection_labels_and_order": tensor_sha(output["labels_3d"]),
                }
            )
    finally:
        handle.remove()
    result = {
        "accepted": base.state_sha256(module) == state_before,
        "authoritative_same_sample_formal_test_cache_found": False,
        "checkpoint_sha256": sha256(base.CHECKPOINT),
        "detector_state_sha256_before": state_before,
        "detector_state_sha256_after": base.state_sha256(module),
        "name": args.name,
        "records": records,
        "sample_count": len(records),
        "schema_version": "probset3d_v2_host_probe_v1",
        "scientific_metrics_generated_or_read": False,
        "sweep_contract_id": CONTRACT_ID,
        "workers": args.workers,
    }
    strict_write(PROBE_ROOT / f"{args.name}.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def compare_host_probes():
    paths = [PROBE_ROOT / "repeat_a.json", PROBE_ROOT / "repeat_b.json"]
    if not all(path.is_file() for path in paths):
        return {"accepted": False, "reason": "missing repeated-host probe"}
    values = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    records_equal = values[0]["records"] == values[1]["records"]
    return {
        "accepted": all(value["accepted"] for value in values) and records_equal,
        "field_hashes_byte_identical": records_equal,
        "probe_sha256": [sha256(path) for path in paths],
        "sample_count": values[0]["sample_count"],
        "workers": [value["workers"] for value in values],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    exporter = sub.add_parser("export")
    exporter.add_argument("--run", choices=tuple(RUN_ROOTS), required=True)
    exporter.add_argument("--split", choices=base.SPLITS, required=True)
    exporter.add_argument("--device", type=int, required=True)
    exporter.add_argument("--shard-index", type=int, required=True)
    exporter.add_argument("--shard-count", type=int, required=True)
    exporter.add_argument("--part-size", type=int, default=32)
    exporter.add_argument("--max-samples", type=int)
    exporter.add_argument("--traversal", choices=("canonical", "reverse", "rotate"), required=True)
    exporter.add_argument("--workers", type=int, required=True)
    validator = sub.add_parser("validate")
    validator.add_argument("--run", choices=tuple(RUN_ROOTS), required=True)
    probe = sub.add_parser("host-probe")
    probe.add_argument("--name", choices=("repeat_a", "repeat_b"), required=True)
    probe.add_argument("--device", type=int, choices=(0, 1, 2, 3), required=True)
    probe.add_argument("--workers", type=int, required=True)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--device", type=int, choices=(0, 1, 2, 3), required=True)
    sub.add_parser("compare")
    sub.add_parser("freeze-comparator")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "export":
        export(args)
    elif args.command == "validate":
        validate(args)
    elif args.command == "host-probe":
        host_probe(args)
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "freeze-comparator":
        freeze_comparator()
    else:
        compare()


if __name__ == "__main__":
    main()
