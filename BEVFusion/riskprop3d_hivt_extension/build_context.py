"""Construct fixed perception histories and HD geometry for retained targets."""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from tools import riskprop3d_r0 as r0
from tools.stage3d_b_native_assignment import detection_gt
from recover import sha, write

OUTPUT = HERE / "data_v1"
TRACKS = {
    "train": ROOT / "outputs/riskprop3d_r0/tracks/train.npz",
    "label_cal": ROOT / "outputs/riskprop3d_r0/tracks/label_cal.npz",
    "clean_s0": ROOT / "outputs/riskprop3d_r0/tracks/clean_s0.npz",
    "dropout_s3": ROOT / "outputs/riskprop3d_r0/tracks/dropout_s3.npz",
    "rotation_s3": ROOT / "outputs/riskprop3d_final_confirmation/tracks/external/rotation_s3.npz",
    "official_clean": ROOT / "outputs/riskprop3d_final_confirmation/tracks/official/clean.npz"}


def load_infos(condition):
    if condition != "official_clean":
        return r0.load_infos(condition if condition in ("train", "label_cal") else "dev")
    with (ROOT / "outputs/stage2/full/nuscenes_infos_val_6019.pkl").open("rb") as stream:
        value = pickle.load(stream)
    infos = value["infos"] if isinstance(value, dict) else value
    return {i["token"]: i for i in infos}


def parts(condition):
    split = "dev" if condition == "clean_s0" else condition
    folder = HERE / ("recovery_v1" if split in ("train", "label_cal", "dev") else "context_recovery_v1")
    paths = sorted((folder / "authority" / split).glob("part-*.npz"))
    if not paths:
        roots = sorted((folder / "raw").glob(split + "_*"))
        for root in roots:
            accepted = json.loads((root / "accepted.json").read_text())
            for record in accepted["parts"]:
                path = root / "parts" / record["file"]
                assert sha(path) == record["sha256"]
                paths.append(path)
    if not paths:
        raise FileNotFoundError("Complete perception exports required: " + condition)
    for path in paths:
        with np.load(path, allow_pickle=False) as value:
            yield path, {name: value[name] for name in value.files}


def observations(condition, infos, frames, nusc):
    result, sources = {}, []
    for path, part in parts(condition):
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path)})
        for row, token in enumerate(part["sample_tokens"]):
            token = str(token)
            assert token not in result and token in infos
            source_indices, gt_box, labels = detection_gt(infos[token])
            offsets = part["gt_boxes_offsets"]
            np.testing.assert_allclose(part["gt_boxes"][offsets[row]:offsets[row + 1]], gt_box, atol=1e-4, rtol=0)
            sample = nusc.get("sample", token)
            assert len(sample["anns"]) == len(infos[token]["gt_boxes"])
            matches = part["native_matched_gt"][row]
            positive = np.flatnonzero(matches >= 0)
            assert len(set(matches[positive].tolist())) == len(positive)
            current = {}
            for query in positive:
                gt = int(matches[query])
                ann = nusc.get("sample_annotation", sample["anns"][int(source_indices[gt])])
                np.testing.assert_allclose(r0.global_to_local(ann["translation"], frames[token]), gt_box[gt, :3], atol=1e-4, rtol=0)
                instance = ann["instance_token"]
                assert instance not in current
                current[instance] = {"box": part["decoded_box_mean"][row, query].copy(), "class": int(labels[gt])}
            result[token] = current
    assert set(result) == set(infos), (len(result), len(infos))
    return result, sources


def history(nusc, token, instance, observations, frames):
    values = [(token, observations[token][instance])]
    previous = nusc.get("sample", token)["prev"]
    while previous and len(values) < 5:
        value = observations.get(previous, {}).get(instance)
        if value is None:
            break
        values.append((previous, value))
        previous = nusc.get("sample", previous)["prev"]
    values.reverse()
    xy = np.asarray([r0.transform_box(value["box"], frames[past], frames[token])[:2]
                     for past, value in values], dtype=np.float32)
    pad = 5 - len(xy)
    mask = np.asarray([False] * pad + [True] * len(xy))
    return np.concatenate((np.repeat(xy[:1], pad, axis=0), xy), axis=0), mask


def map_points(api, frame, centers):
    origin, rotation = frame
    global_centers = np.c_[centers, np.zeros(len(centers))] @ rotation.T + origin
    bounds = (global_centers[:, :2].min(axis=0) - 52, global_centers[:, :2].max(axis=0) + 52)
    records = api.get_records_in_patch((*bounds[0], *bounds[1]), ["lane", "lane_connector"], mode="intersect")
    tokens = sorted(records["lane"] + records["lane_connector"])
    if not tokens:
        return np.empty((0, 2), dtype=np.float32)
    lanes = api.discretize_lanes(tokens, resolution_meters=2.0)
    points = np.concatenate([np.asarray(lanes[token])[:, :2] for token in tokens if len(lanes[token])], axis=0)
    points = np.c_[points, np.zeros(len(points))]
    return ((points - origin) @ rotation)[:, :2].astype(np.float32)


def build(condition):
    target = OUTPUT / condition
    if target.exists():
        raise FileExistsError(target)
    started = time.time()
    infos = load_infos(condition)
    frames = r0.frame_authority(infos)
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(ROOT / "data/nuscenes"), verbose=False)
    obs, source_hashes = observations(condition, infos, frames, nusc)
    with np.load(TRACKS[condition], allow_pickle=False) as data:
        selected = np.flatnonzero(data["class"] <= 4)
        tracks = {k: data[k][selected].copy() for k in ("input", "future", "scene_token", "sample_token", "instance_token", "class")}
        if "feature" in data:
            heads = {k: data[k][selected].copy() for k in ("feature", "logits", "box", "seed")}
        else:
            scene_path = ROOT / ("outputs/riskprop3d_r1/scenes/" + condition + ".npz")
            with np.load(scene_path, allow_pickle=False) as scenes:
                assert np.array_equal(scenes["sample_token"], data["sample_token"])
                assert np.array_equal(scenes["instance_token"], data["instance_token"])
                indices = scenes["target_agent_index"][selected]
                heads = {k: scenes["agent_" + k][indices].copy() for k in ("feature", "logits", "box", "seed")}
            source_hashes.append({"path": str(scene_path.relative_to(ROOT)), "sha256": sha(scene_path)})
    target_xy = tracks["input"][:, :50].reshape(-1, 5, 10)[..., :2]
    target_mask = np.c_[tracks["input"][:, 50:54] > 0, np.ones(len(selected), dtype=bool)]
    np.testing.assert_array_equal(heads["box"][:, :2], target_xy[:, -1])
    count = len(selected)
    neighbors = np.full((count, 8), -1, dtype=np.int32)
    agents_xy, agents_mask, agent_ids = [], [], []
    lane_xy, lane_offsets, lane_selected, lane_row_offsets = [], [0], [], [0]
    frame_index = np.empty(count, dtype=np.int32)
    row_lanes = [None] * count
    maps = {}
    rows_by_sample = {}
    for row, token in enumerate(tracks["sample_token"]):
        rows_by_sample.setdefault(str(token), []).append(row)
    current_difference, missing_current, empty_maps, timestamp_deviations = [], 0, 0, []
    for frame_number, (token, rows) in enumerate(rows_by_sample.items()):
        sample = nusc.get("sample", token)
        location = nusc.get("log", nusc.get("scene", sample["scene_token"])["log_token"])["location"]
        if location not in maps:
            maps[location] = NuScenesMap(dataroot=str(ROOT / "data/nuscenes"), map_name=location)
        local_agents = {}
        for instance, value in sorted(obs[token].items()):
            index = len(agents_xy)
            xy, mask = history(nusc, token, instance, obs, frames)
            agents_xy.append(xy)
            agents_mask.append(mask)
            agent_ids.append(instance)
            local_agents[instance] = index
        centers = np.asarray([v["box"][:2] for v in obs[token].values()] + [target_xy[r, -1] for r in rows])
        points = map_points(maps[location], frames[token], centers)
        offset = lane_offsets[-1]
        lane_xy.append(points)
        lane_offsets.append(offset + len(points))
        for row in rows:
            frame_index[row] = frame_number
            instance = str(tracks["instance_token"][row])
            center = target_xy[row, -1]
            candidates = [(float(np.linalg.norm(v["box"][:2] - center)), other, local_agents[other])
                          for other, v in obs[token].items() if other != instance]
            chosen = sorted(c for c in candidates if c[0] <= 30)[:8]
            neighbors[row, :len(chosen)] = [c[2] for c in chosen]
            nodes = np.asarray([center] + [agents_xy[c[2]][-1] for c in chosen])
            keep = np.zeros(len(points), dtype=bool)
            for node in nodes:
                keep |= np.linalg.norm(points - node, axis=-1) < 50
            indices = np.flatnonzero(keep) + offset
            row_lanes[row] = indices.astype(np.int32)
            empty_maps += not len(indices)
            if instance in obs[token]:
                current_difference.append(float(np.max(np.abs(obs[token][instance]["box"] - heads["box"][row]))))
            else:
                missing_current += 1
            past = token
            for slot in range(3, -1, -1):
                previous = nusc.get("sample", past)["prev"]
                if not target_mask[row, slot]:
                    continue
                assert previous and previous in infos
                timestamp_deviations.append(abs((infos[past]["timestamp"] - infos[previous]["timestamp"]) / 1e6 - .5))
                past = previous
            for slot in np.flatnonzero(target_mask[row, :4]):
                if slot < 3 and target_mask[row, slot + 1]:
                    assert np.isfinite(target_xy[row, slot]).all()
        if frame_number % 100 == 0:
            print(json.dumps({"condition": condition, "frames": frame_number + 1, "total_frames": len(rows_by_sample), "seconds": time.time() - started}), flush=True)
    for indices in row_lanes:
        lane_selected.append(indices)
        lane_row_offsets.append(lane_row_offsets[-1] + len(indices))
    arrays = dict(target_xy=target_xy, target_mask=target_mask, neighbor_index=neighbors,
                  agent_xy=np.asarray(agents_xy, dtype=np.float32), agent_mask=np.asarray(agents_mask),
                  agent_instance=np.asarray(agent_ids), lane_xy=np.concatenate(lane_xy),
                  lane_index=np.concatenate(lane_selected), lane_row_offsets=np.asarray(lane_row_offsets),
                  frame_index=frame_index, original_row=selected,
                  sample_token=tracks["sample_token"], scene_token=tracks["scene_token"],
                  instance_token=tracks["instance_token"], **heads)
    assert all(np.isfinite(v).all() for v in arrays.values() if np.issubdtype(v.dtype, np.floating))
    target.mkdir(parents=True)
    with (target / "geometry.npz").open("xb") as stream:
        np.savez(stream, **arrays)
    with (target / "labels.npz").open("xb") as stream:
        np.savez(stream, future=tracks["future"])
    write(target / "manifest.json", {"condition": condition, "targets": count,
        "scenes": len(set(tracks["scene_token"].tolist())), "source_tracks": str(TRACKS[condition].relative_to(ROOT)),
        "source_tracks_sha256": sha(TRACKS[condition]), "sources": source_hashes,
        "geometry_sha256": sha(target / "geometry.npz"), "labels_sha256": sha(target / "labels.npz"),
        "builder_sha256": sha(__file__), "contract_sha256": sha(HERE / "DATA_CONTRACT.md"),
        "background_current_missing_retained_target": missing_current,
        "current_box_max_absolute_difference": max(current_difference),
        "current_box_exact_retained_targets": sum(v == 0 for v in current_difference),
        "empty_map_targets": empty_maps, "max_history_time_deviation_seconds": max(timestamp_deviations),
        "target_history_source": "original retained perception; unchanged",
        "neighbor_history_source": "same-condition regenerated perception; no GT positions",
        "future_labels_in_geometry": False, "seconds": time.time() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=TRACKS)
    args = parser.parse_args()
    build(args.condition)
