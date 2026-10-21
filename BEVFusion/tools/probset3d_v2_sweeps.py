#!/usr/bin/env python3
"""ProbSet3D-V2 deterministic nuScenes sweep contract helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "probset3d/P0_DATA_SPLIT_AUTHORITY.json"
OUT = ROOT / "outputs/probset3d_v2/sweep_authority"
SWEEPS_NUM = 9
CONTRACT_ID = "probset3d_v2_sweep_authority_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sweeps(info, sweeps_num=SWEEPS_NUM):
    key_timestamp = int(info["timestamp"])
    seen = set()
    rows = []
    for sweep in info.get("sweeps", ()): 
        timestamp = int(sweep["timestamp"])
        path = str(sweep["data_path"])
        lag = key_timestamp - timestamp
        if lag <= 0:
            raise ValueError(f"non-past sweep for {info['token']}: {path}")
        identity = (timestamp, path)
        if identity in seen:
            raise ValueError(f"duplicate sweep for {info['token']}: {identity}")
        seen.add(identity)
        rotation = sweep["sensor2lidar_rotation"]
        translation = sweep["sensor2lidar_translation"]
        if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
            raise ValueError(f"invalid sweep rotation for {info['token']}")
        if len(translation) != 3:
            raise ValueError(f"invalid sweep translation for {info['token']}")
        rows.append((lag, -timestamp, path, sweep))
    rows.sort(key=lambda row: row[:3])
    return [row[3] for row in rows[:sweeps_num]]


def canonicalize_infos(infos):
    result = []
    for info in infos:
        copied = dict(info)
        copied["sweeps"] = canonical_sweeps(info)
        result.append(copied)
    return result


def bind_pipeline(dataset_cfg):
    found = 0
    for transform in dataset_cfg.pipeline:
        if transform["type"] != "LoadPointsFromMultiSweeps":
            continue
        found += 1
        if int(transform["sweeps_num"]) != SWEEPS_NUM:
            raise ValueError("V2 requires exactly nine historical sweeps")
        if not bool(transform["pad_empty_sweeps"]):
            raise ValueError("V2 requires pad_empty_sweeps=true")
        if not bool(transform["remove_close"]):
            raise ValueError("V2 requires remove_close=true")
        transform["test_mode"] = True
    if found != 1:
        raise ValueError(f"expected one multi-sweep transform, found {found}")
    return dataset_cfg


def selection_digest(infos):
    digest = hashlib.sha256()
    for info in sorted(infos, key=lambda value: value["token"]):
        digest.update(info["token"].encode("ascii"))
        for sweep in canonical_sweeps(info):
            digest.update(str(sweep["timestamp"]).encode("ascii"))
            digest.update(str(sweep["data_path"]).encode("utf-8"))
    return digest.hexdigest()


def run_fixtures() -> None:
    p0 = json.loads(P0.read_text(encoding="utf-8"))
    splits = {}
    for split in ("train", "label_cal", "dev"):
        authority = p0["splits"][split]
        path = ROOT / authority["info_path"]
        if sha256(path) != authority["info_sha256"]:
            raise ValueError(f"P0 info hash mismatch for {split}")
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        infos = payload["infos"] if isinstance(payload, dict) else payload
        selected = canonicalize_infos(infos)
        counts = [len(item["sweeps"]) for item in selected]
        order_invariant = selection_digest(infos) == selection_digest(
            [
                {**item, "sweeps": list(reversed(item.get("sweeps", ())))}
                for item in reversed(infos)
            ]
        )
        if not order_invariant or len(infos) != authority["sample_count"]:
            raise ValueError(f"V2 sweep fixture failure for {split}")
        splits[split] = {
            "info_path": authority["info_path"],
            "info_sha256": authority["info_sha256"],
            "max_selected_history": max(counts),
            "min_selected_history": min(counts),
            "sample_count": len(infos),
            "selection_digest": selection_digest(infos),
            "source_order_invariant": order_invariant,
            "zero_history_sample_count": sum(count == 0 for count in counts),
        }
    result = {
        "accepted": True,
        "contract_id": CONTRACT_ID,
        "current_frame_included_separately": True,
        "historical_sweeps_num": SWEEPS_NUM,
        "normal_frame_budget": 1 + SWEEPS_NUM,
        "selection_key": [
            "keyframe_timestamp_us - sweep_timestamp_us",
            "-sweep_timestamp_us",
            "data_path",
        ],
        "splits": splits,
        "source": str(Path(__file__).resolve().relative_to(ROOT)),
        "source_sha256": sha256(Path(__file__).resolve()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "sweep_authority_acceptance.json"
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(json.dumps({"accepted": True, "splits": splits}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fixtures",))
    args = parser.parse_args()
    if args.command == "fixtures":
        run_fixtures()
