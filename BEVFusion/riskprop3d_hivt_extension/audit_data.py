"""Audit time, labels, coordinate transforms and feature separation before training."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "tools"))
import numpy as np
from nuscenes.nuscenes import NuScenes
from build_context import TRACKS, load_infos, parts
from tools import riskprop3d_r0 as r0
from tools.stage3d_b_native_assignment import detection_gt
from recover import sha, write


def query_alignment(condition, geometry, infos, nusc):
    by_token = {}
    for row, token in enumerate(geometry["sample_token"]):
        by_token.setdefault(str(token), []).append(row)
    source_keys = {"box": "decoded_box_mean", "feature": "query_features", "logits": "class_logits", "seed": "seed_score"}
    maximum = {key: 0. for key in source_keys}
    checked, missing = 0, 0
    for _, part in parts(condition):
        for index, token in enumerate(part["sample_tokens"]):
            token = str(token)
            if token not in by_token:
                continue
            sample = nusc.get("sample", token)
            source_indices, _, _ = detection_gt(infos[token])
            assignment = part["native_matched_gt"][index]
            query_by_instance = {}
            for query in np.flatnonzero(assignment >= 0):
                ann_token = sample["anns"][int(source_indices[int(assignment[query])])]
                instance = nusc.get("sample_annotation", ann_token)["instance_token"]
                query_by_instance[instance] = int(query)
            for row in by_token[token]:
                instance = str(geometry["instance_token"][row])
                if instance not in query_by_instance:
                    missing += 1
                    continue
                query = query_by_instance[instance]
                for key, name in source_keys.items():
                    difference = np.abs(geometry[key][row].astype(np.float64) - part[name][index, query])
                    maximum[key] = max(maximum[key], float(np.max(difference)))
                checked += 1
    assert checked + missing == len(geometry["sample_token"])
    return {"matched_targets_checked": checked, "missing": missing, "max_absolute_difference": maximum,
            "all_target_head_inputs_exact": missing == 0 and all(value == 0 for value in maximum.values())}


def audit():
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(HERE.parent / "data/nuscenes"), verbose=False)
    records = {}
    split_scenes = {}
    for condition in TRACKS:
        directory = HERE / "data_v1" / condition
        manifest = json.loads((directory / "manifest.json").read_text())
        assert sha(directory / "geometry.npz") == manifest["geometry_sha256"]
        assert sha(directory / "labels.npz") == manifest["labels_sha256"]
        with np.load(directory / "geometry.npz", allow_pickle=False) as raw_geometry, np.load(directory / "labels.npz", allow_pickle=False) as raw_labels:
            geometry = {key: raw_geometry[key] for key in raw_geometry.files}
            labels = {"future": raw_labels["future"]}
            import hashlib
            keys = [str(s) + "/" + str(i) for s, i in zip(geometry["sample_token"], geometry["instance_token"])]
            indices = sorted(range(len(keys)), key=lambda i: hashlib.sha256(keys[i].encode()).digest())[:64]
            infos = load_infos(condition)
            frames = r0.frame_authority(infos)
            max_roundtrip, max_label_error, max_time_error = 0., 0., 0.
            elapsed = []
            for row in indices:
                token = str(geometry["sample_token"][row])
                sample = nusc.get("sample", token)
                instance = str(geometry["instance_token"][row])
                anns = [nusc.get("sample_annotation", a) for a in sample["anns"]]
                matches = [a for a in anns if a["instance_token"] == instance]
                assert len(matches) == 1
                ann = matches[0]
                origin, rotation = frames[token]
                point = np.asarray(ann["translation"])
                roundtrip = rotation @ r0.global_to_local(point, frames[token]) + origin
                max_roundtrip = max(max_roundtrip, float(np.abs(roundtrip - point).max()))
                previous_sample = sample
                for step in range(6):
                    ann = nusc.get("sample_annotation", ann["next"])
                    future_sample = nusc.get("sample", ann["sample_token"])
                    assert future_sample["token"] == previous_sample["next"]
                    assert future_sample["timestamp"] > previous_sample["timestamp"]
                    max_time_error = max(max_time_error, abs((future_sample["timestamp"] - previous_sample["timestamp"]) / 1e6 - .5))
                    expected = r0.global_to_local(ann["translation"], frames[token])[:2]
                    max_label_error = max(max_label_error, float(np.abs(labels["future"][row, step] - expected).max()))
                    previous_sample = future_sample
                elapsed.append((previous_sample["timestamp"] - sample["timestamp"]) / 1e6)
                neighbors = geometry["neighbor_index"][row]
                neighbors = neighbors[neighbors >= 0]
                assert instance not in geometry["agent_instance"][neighbors].tolist()
                assert len(set(geometry["agent_instance"][neighbors].tolist())) == len(neighbors)
                distance = np.linalg.norm(geometry["agent_xy"][neighbors, -1] - geometry["target_xy"][row, -1], axis=-1)
                assert np.all(distance <= 30.00001) and len(neighbors) <= 8
            assert max_roundtrip < 1e-8 and max_label_error < 1e-4
            assert "future" not in geometry and "gt_input" not in geometry and "ego_future" not in geometry
            if condition in ("train", "label_cal", "clean_s0", "official_clean"):
                split_scenes[condition] = set(geometry["scene_token"].tolist())
            records[condition] = {"rows": len(keys), "checked_rows": indices, "max_coordinate_roundtrip_error": max_roundtrip,
                "max_future_label_error": max_label_error, "max_future_interval_error_seconds": max_time_error,
                "checked_six_keyframe_horizon_seconds_range": [min(elapsed), max(elapsed)],
                "full_history_max_nominal_interval_deviation_seconds": manifest["max_history_time_deviation_seconds"],
                "manifest_sha256": sha(directory / "manifest.json"), "feature_label_separation": True}
            if condition in ("train", "label_cal", "clean_s0"):
                records[condition]["full_query_alignment"] = query_alignment(condition, geometry, infos, nusc)
    pairs = [(a, b) for i, a in enumerate(split_scenes) for b in list(split_scenes)[i + 1:]]
    assert all(not split_scenes[a] & split_scenes[b] for a, b in pairs)
    write(HERE / "data_v1/audit.json", {"status": "PASS", "records": records,
        "split_scene_overlap": {a + "/" + b: 0 for a, b in pairs}, "label_checks_entered_model": False,
        "HD_map_sha256": {p.name: sha(p) for p in sorted((HERE.parent / "data/nuscenes/maps/expansion").glob("*.json"))},
        "HD_map_projection": "2D global lane points embedded at global z=0, transformed with frozen full LiDAR/global pose, then XY retained",
        "source_sha256": sha(__file__)})


if __name__ == "__main__":
    audit()
