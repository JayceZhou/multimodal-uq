#!/usr/bin/env python3
import argparse
import gzip
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tools.stage3_run_real_target_uq import (
    ROOT,
    build_frozen_model,
    build_loader,
    collect_samples,
    export_distribution_jsonl,
    load_cfg,
    new_adapter,
    strict_json,
)


ANN = {
    "train": "outputs/stage3/splits/nuscenes_infos_stage3_train.pkl",
    "label_cal": "outputs/stage3/splits/nuscenes_infos_stage3_label_cal.pkl",
    "dev": "outputs/stage3/splits/nuscenes_infos_stage3_dev.pkl",
}
SPLIT_NAME = {"train": "stage3_train", "label_cal": "stage3_label_cal", "dev": "stage3_dev"}
MAX_SAMPLES = {"train": 22508, "label_cal": 2814, "dev": 2808}


def validate_checkpoint_metadata(checkpoint, branch, variant, classifier_mode):
    expected = {"branch": branch, "variant": variant, "classifier_mode": classifier_mode}
    mismatches = {
        key: {"expected": value, "actual": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"adapter checkpoint metadata mismatch: {mismatches}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--variant", default="structured")
    parser.add_argument("--classifier-mode", default="feature_k1")
    parser.add_argument("--distribution-key", default="structured_feature_k1")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--splits", nargs="+", default=["label_cal", "dev"], choices=sorted(ANN))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--fixed-log-var", type=float, default=0.0)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--positive-distance-threshold", type=float, default=2.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--copy-prefix-from")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.cuda.set_device(args.device)
    cfg = load_cfg(args.config)
    model = build_frozen_model(cfg, args.checkpoint, args.device)
    device = torch.device(f"cuda:{args.device}")
    ckpt = torch.load(args.adapter_checkpoint, map_location=device, weights_only=True)
    validate_checkpoint_metadata(ckpt, args.branch, args.variant, args.classifier_mode)
    adapter_args = SimpleNamespace(
        hidden_channels=args.hidden_channels,
        fixed_log_var=args.fixed_log_var,
        classifier_modes=[args.classifier_mode],
    )
    in_channels = int(ckpt["model"]["net.0.weight"].shape[1])
    adapter = new_adapter(args.variant, in_channels, adapter_args, args.classifier_mode).to(device)
    adapter.load_state_dict(ckpt["model"])
    calibration = json.loads(Path(args.calibration).read_text())["parameters"]
    exports = {}
    for split in args.splits:
        start_index = args.start_index if split == "train" else 0
        _, loader = build_loader(cfg, ANN[split], start_index=start_index)
        pack = collect_samples(
            model,
            loader,
            MAX_SAMPLES[split] - start_index,
            args.branch,
            args.positive_distance_threshold,
        )
        path = ROOT / "distributions_full" / args.branch / f"{args.output_prefix}{SPLIT_NAME[split]}_{args.distribution_key}.jsonl.gz"
        export_path = path.with_name(path.name.replace(".jsonl.gz", ".tail.jsonl.gz")) if start_index else path
        item = export_distribution_jsonl(
            adapter,
            pack["samples"],
            args.branch,
            calibration,
            device,
            args.eval_batch_size,
            export_path,
            split=SPLIT_NAME[split],
            calibration_id=args.calibration,
            checkpoint_id=args.adapter_checkpoint,
        )
        if start_index:
            if not args.copy_prefix_from:
                raise ValueError("--copy-prefix-from is required with --start-index")
            copied_records = start_index * 200
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(args.copy_prefix_from, "rt") as source, gzip.open(path, "wt") as target:
                for _ in range(copied_records):
                    target.write(next(source))
            with path.open("ab") as target, export_path.open("rb") as tail:
                shutil.copyfileobj(tail, target)
            item = {
                "path": str(path),
                "record_count": copied_records + item["record_count"],
                "schema": "stage3_target_uq_distribution_jsonl_gzip",
                "recovery": {
                    "source_truncated_file": args.copy_prefix_from,
                    "complete_prefix_records": copied_records,
                    "tail_path": str(export_path),
                    "tail_records": item["record_count"],
                    "start_sample_index": start_index,
                },
            }
        exports[split] = item
    strict_json(ROOT / "manifests" / f"{args.output_prefix}recovery_export_manifest.json", {
        "stage": "stage3c_recovery_export_from_checkpoint",
        "branch": args.branch,
        "adapter_checkpoint": args.adapter_checkpoint,
        "calibration": args.calibration,
        "official_val_used": False,
        "post_score_usage": "post_score_for_mapping only; never probability or calibration",
        "exports": exports,
    })
    print(json.dumps(exports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
