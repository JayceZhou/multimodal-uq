"""Recover perception backgrounds without changing the running clean recovery."""
import argparse
import json
import pickle
from pathlib import Path

import recover as recovery
from mmdet3d.apis.stage2_manifest import inject_stage2_corruption

HERE, ROOT = recovery.HERE, recovery.ROOT
CLEAN = recovery.OUT
OUT = HERE / "context_recovery_v1"
CONDITIONS = {"dropout_s3": ("lidar_dropout", 3, "dev", 2),
              "rotation_s3": ("camera_extrinsic_rotation", 3, "dev", 2),
              "official_clean": ("clean", 0, "official", 4)}
OFFICIAL = ROOT / "outputs/stage2/full/nuscenes_infos_val_6019.pkl"


def freeze():
    sources = json.loads((CLEAN / "pre_recovery_freeze.json").read_text())["source_sha256"]
    sources.update({str(Path(__file__).resolve().relative_to(ROOT)): recovery.sha(__file__),
                    str(OFFICIAL.relative_to(ROOT)): recovery.sha(OFFICIAL)})
    recovery.write(OUT / "freeze.json", {"sources": sources, "conditions": CONDITIONS,
        "targets": "retained R0 target inputs, labels and row identities remain authoritative",
        "background": "new matched perception observations; current plus nearest nine past sweeps",
        "legacy_byte_identity_claimed": False, "corruption_seed": 20260708,
        "selection": "all currently matched agents; <=30m, nearest eight, instance-token tie break; no future filter"})


def check():
    value = json.loads((OUT / "freeze.json").read_text())
    for name, digest in value["sources"].items():
        if recovery.sha(ROOT / name) != digest:
            raise ValueError("Context source changed: " + name)
    return value


def configure(condition):
    check()
    corruption, severity, split, shards = CONDITIONS[condition]
    load_clean = recovery.load_split
    load_cfg = recovery.stage3.load_cfg
    setup_gpu = recovery.setup_gpu

    def load_split(_condition):
        if split != "official":
            return load_clean(split)
        with OFFICIAL.open("rb") as stream:
            payload = pickle.load(stream)
        infos = payload["infos"] if isinstance(payload, dict) else payload
        assert len(infos) == 6019 and len({i["token"] for i in infos}) == 6019
        return {"info_path": str(OFFICIAL)}, {i["token"]: i for i in infos}

    def config(path):
        cfg = load_cfg(path)
        if corruption != "clean":
            inject_stage2_corruption(cfg, corruption, severity, 20260708, None,
                                     "calibrated_candidate", None)
        return cfg

    recovery.OUT = OUT
    recovery.check_freeze = check
    recovery.load_split = load_split
    recovery.stage3.load_cfg = config
    recovery.setup_gpu = lambda gpu, seed=20260708: setup_gpu(gpu, seed)
    recovery.SHARDS = {condition: shards}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "export", "run"))
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--gpu", type=int, choices=recovery.GPUS)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "export":
        configure(args.condition)
        recovery.export(args.condition, args.shard, args.gpu)
    else:
        raise ValueError("Schedule explicit export workers after checking available GPUs")


if __name__ == "__main__":
    main()
