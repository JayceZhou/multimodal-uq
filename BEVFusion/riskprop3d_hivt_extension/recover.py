"""Regenerate missing clean query inputs and the original three F1 head fits."""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hivt_recovery_mpl")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "recovery_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch
from mmdet3d.datasets import build_dataset, build_dataloader
from tools import probset3d_v3_formal as formal
from tools import probset3d_v3_pilot as pilot
from tools import probset3d_f1_authority as original
from tools import stage3_run_real_target_uq as stage3
from tools.probset3d_f0_export import exact_local_max
from tools.probset3d_v2_sweeps import canonicalize_infos, bind_pipeline
from tools.stage3d_b_native_assignment import cost_components, detection_gt, deterministic_assignment_from_cost
from probset3d.f1.implementation.probabilistic_query import ProbabilisticQueryHead, f1_loss

GPUS = (0, 2, 3)
SEEDS = formal.SEEDS
SPLITS = ("train", "label_cal", "dev")
COUNTS = {"train": 22508, "label_cal": 2814, "dev": 2808}
SHARDS = {"train": 6, "label_cal": 1, "dev": 1}
FIXED = ("sample_tokens", "scene_tokens", "query_id", "seed_flat_index", "seed_class",
         "seed_score", "query_features", "class_logits", "decoded_box_mean", "native_matched_gt")
VARIABLE = ("gt_boxes", "gt_labels", "duplicate_edge_query_id", "duplicate_edge_gt_index")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def setup_gpu(physical, seed=20260730):
    if physical not in GPUS or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("Explicitly pin a qualified physical GPU with CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda:0")


def load_split(split):
    p0 = json.loads(original.P0.read_text())
    record = p0["splits"][split]
    for key, hash_key in (("info_path", "info_sha256"), ("sample_tokens_path", "sample_tokens_sha256"),
                          ("scene_tokens_path", "scene_tokens_sha256")):
        if sha(ROOT / record[key]) != record[hash_key]:
            raise ValueError("Original P0 identity changed: " + split + "/" + key)
    with (ROOT / record["info_path"]).open("rb") as stream:
        payload = pickle.load(stream)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    tokens = json.loads((ROOT / record["sample_tokens_path"]).read_text())["sample_tokens"]
    if len(infos) != COUNTS[split] or set(tokens) != {item["token"] for item in infos}:
        raise ValueError("Original split membership/count changed")
    return record, {item["token"]: item for item in infos}


def check_freeze():
    frozen = json.loads((OUT / "pre_recovery_freeze.json").read_text())
    for name, digest in frozen["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Recovery source changed: " + name)
    return frozen


def freeze():
    if sha(original.CONFIG) != original.EXPECTED_CONFIG_SHA256 or sha(original.CHECKPOINT) != original.EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Original detector/config identity mismatch")
    files = [Path(__file__).resolve(), HERE / "RECOVERY_PROTOCOL.md", original.CONFIG, original.CHECKPOINT,
             original.P0, original.NUSCENES_SAMPLES,
             ROOT / "tools/probset3d_v3_formal.py", ROOT / "tools/probset3d_v3_pilot.py",
             ROOT / "tools/probset3d_f1_authority.py", ROOT / "tools/probset3d_v2_sweeps.py",
             ROOT / "tools/stage3d_b_native_assignment.py", ROOT / "tools/stage3_run_real_target_uq.py",
             ROOT / "probset3d/f1/implementation/probabilistic_query.py",
             ROOT / "mmdet3d/models/fusion_models/bevfusion.py", ROOT / "mmdet3d/models/heads/bbox/transfusion.py",
             ROOT / "mmdet3d/core/bbox/coders/transfusion_bbox_coder.py"]
    splits = {}
    for split in SPLITS:
        record, infos = load_split(split)
        files.extend(ROOT / record[key] for key in ("info_path", "sample_tokens_path", "scene_tokens_path"))
        splits[split] = {"samples": len(infos), "info_path": record["info_path"]}
    write(OUT / "pre_recovery_freeze.json", {
        "authorization": "author requested recovery or rerunning the missing earlier inputs",
        "status": "FROZEN_BEFORE_RECOVERY", "recovery_is_byte_restoration": False,
        "original_results_rewritten": False, "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "splits": splits, "head_seeds": list(SEEDS), "physical_gpus": list(GPUS),
        "epochs": formal.EPOCHS, "batch_size": formal.BATCH_SIZE,
        "optimizer": "AdamW lr=1e-3 wd=1e-4 clip=5 FP32", "checkpoint_rule": "last epoch 20",
        "legacy_density_column_restored": False, "sweeps": "current + nearest nine past; deterministic",
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "official_metrics_read": False, "sealed_test_accessed": False,
    })


class Writer:
    def __init__(self, root):
        self.root = Path(root)
        if self.root.exists():
            raise FileExistsError(self.root)
        self.root.mkdir(parents=True)
        self.rows, self.parts = [], []

    def append(self, row):
        self.rows.append(row)
        if len(self.rows) == 32:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        arrays = {key: np.stack([row[key] for row in self.rows]) for key in FIXED}
        for key in VARIABLE:
            width = 9 if key == "gt_boxes" else None
            dtype = np.float32 if key == "gt_boxes" else np.int16
            arrays[key], arrays[key + "_offsets"] = original.variable_arrays(self.rows, key, width, dtype)
        path = self.root / ("part-{:05d}.npz".format(len(self.parts)))
        with path.open("xb") as stream:
            np.savez(stream, **arrays)
        self.parts.append({"file": path.name, "sample_count": len(self.rows), "sha256": sha(path),
                           "size_bytes": path.stat().st_size})
        self.rows = []

    def close(self):
        self.flush()
        return self.parts


def unpack_rows(path):
    with np.load(path, allow_pickle=False) as arrays:
        for index in range(len(arrays["sample_tokens"])):
            row = {key: arrays[key][index].copy() for key in FIXED}
            for key in VARIABLE:
                offset = arrays[key + "_offsets"]
                row[key] = arrays[key][offset[index]:offset[index + 1]].copy()
            yield row


def export(split, shard, gpu, smoke=False):
    check_freeze()
    device = setup_gpu(gpu)
    record, infos = load_split(split)
    sample_scene = original.load_sample_scene_map()
    tokens = sorted(infos)
    selected = list(np.array_split(np.asarray(tokens), SHARDS[split])[shard])
    if smoke:
        selected = selected[:2]
    cfg = stage3.load_cfg(str(original.CONFIG))
    dataset_cfg, batch_size = stage3.dataset_cfg_for(cfg, str(ROOT / record["info_path"]))
    bind_pipeline(dataset_cfg)
    dataset = build_dataset(dataset_cfg)
    canonical = {item["token"]: item for item in canonicalize_infos(dataset.data_infos)}
    index_by_token = {item["token"]: i for i, item in enumerate(dataset.data_infos)}
    if hasattr(dataset, "flag"):
        dataset.flag = dataset.flag[[index_by_token[token] for token in selected]]
    dataset.data_infos = [canonical[token] for token in selected]
    loader = build_dataloader(dataset, samples_per_gpu=batch_size, workers_per_gpu=0,
                              dist=False, shuffle=False, seed=20260730)
    model = stage3.build_frozen_model(cfg, str(original.CHECKPOINT), 0)
    module = model.module
    assert not any(p.requires_grad for p in module.parameters())
    state_before = original.state_sha256(module)
    head = module.heads["object"]
    captured = []
    handle = head.heatmap_head.register_forward_hook(lambda _m, _i, value: captured.append(value.detach()))
    name = "{}_{:02d}".format(split, shard)
    target = OUT / ("smoke/raw" if smoke else "raw") / name
    writer = Writer(target / "parts")
    seen = []
    started = time.time()
    try:
        for index, data in enumerate(loader):
            with torch.inference_mode():
                output = model(return_loss=False, rescale=True, return_stage3_target_uq=True,
                               stage3_branch="fusion", stage3_num_proposals=200, **data)[0]
            if len(captured) != 1:
                raise ValueError("Dense capture count differs")
            dense = captured.pop()
            item = output["stage3_target_uq"]
            token = item["sample_token"]
            assert token == selected[index] and token not in seen
            if smoke:
                with torch.inference_mode():
                    repeated = model(return_loss=False, rescale=True, return_stage3_target_uq=True,
                                     stage3_branch="fusion", stage3_num_proposals=200, **data)[0]["stage3_target_uq"]
                assert len(captured) == 1 and torch.equal(captured.pop(), dense)
                for key in ("class_logits", "features", "decoded_boxes", "query_indices", "query_labels"):
                    assert torch.equal(item[key], repeated[key]), (token, key)
            _, boxes, labels = detection_gt(infos[token])
            query_box = item["decoded_boxes"].float().numpy()
            features = item["features"].float().numpy()
            logits = item["class_logits"].float().numpy()
            query_ids = item["proposal_ids"].numpy().astype(np.int16)
            assert np.array_equal(query_ids, np.arange(200))
            _, _, seed_index, seed_class, seed_score = exact_local_max(head, dense)
            seed_index = seed_index[0].cpu().numpy().astype(np.int32)
            seed_class = seed_class[0].cpu().numpy().astype(np.int16)
            seed_score = seed_score[0].cpu().float().numpy()
            assert np.array_equal(seed_index, item["query_indices"].numpy())
            assert np.array_equal(seed_class, item["query_labels"].numpy())
            costs = cost_components("fusion", model, item["class_logits"].float(),
                                    item["decoded_boxes"].float(), boxes, labels, device)
            matched = deterministic_assignment_from_cost(costs["total"], query_box[:, :2], boxes[:, :2],
                                                         query_ids=query_ids)[0].astype(np.int16)
            positive = matched[matched >= 0]
            assert len(positive) == len(set(positive.tolist())) and np.all(matched < len(boxes))
            assert query_box.shape == (200, 9) and features.shape == (200, 128) and logits.shape == (200, 10)
            assert all(np.isfinite(a).all() for a in (query_box, features, logits, seed_score, boxes))
            assert np.all(query_box[:, 3:6] > 0) and torch.isfinite(dense).all()
            edge_query, edge_gt = original.duplicate_edges(seed_class, query_box, labels, boxes)
            writer.append({"sample_tokens": np.asarray(token, dtype="U32"),
                           "scene_tokens": np.asarray(sample_scene[token], dtype="U32"),
                           "query_id": query_ids, "seed_flat_index": seed_index, "seed_class": seed_class,
                           "seed_score": seed_score, "query_features": features, "class_logits": logits,
                           "decoded_box_mean": query_box, "native_matched_gt": matched,
                           "gt_boxes": boxes, "gt_labels": labels.astype(np.int16),
                           "duplicate_edge_query_id": edge_query, "duplicate_edge_gt_index": edge_gt})
            seen.append(token)
            if index == 0 or (index + 1) % 100 == 0:
                print(json.dumps({"event": "export_progress", "split": split, "shard": shard,
                                  "frames": index + 1, "expected": len(selected),
                                  "seconds": time.time() - started}), flush=True)
    finally:
        handle.remove()
    assert seen == selected and state_before == original.state_sha256(module)
    assert not any(p.grad is not None for p in module.parameters())
    parts = writer.close()
    result = {"accepted": True, "split": split, "shard": shard, "samples": len(seen), "parts": parts,
              "physical_gpu": gpu, "wall_time_seconds": time.time() - started,
              "peak_cuda_allocation_bytes": torch.cuda.max_memory_allocated(), "detector_state_sha256": state_before,
              "detector_updates": 0, "query_density_column_present": False, "is_engineering_smoke": smoke}
    write(target / "accepted.json", result)
    if smoke:
        rows = list(itertools.chain.from_iterable(unpack_rows(target / "parts" / p["file"]) for p in parts))
        batch = pilot.make_batch(rows, device)
        for family, neighbors in (("gaussian", None), ("student_t", None),
                                  ("student_t", batch["neighborhood"])):
            probability_head = ProbabilisticQueryHead().to(device)
            value = probability_head(batch["query_feature"], batch["class_logits"], batch["decoded_boxes"],
                                     batch["seed_score"], batch["query_density"])
            loss = f1_loss(value, batch["matched_gt_index"], batch["gt_state"], batch["gt_yaw"],
                           batch["gt_labels"], torch.ones(10, device=device), family=family,
                           neighborhood=neighbors)["total"]
            loss.backward()
            gradients = [p.grad for p in probability_head.parameters() if p.grad is not None]
            assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in gradients)
            assert any(torch.count_nonzero(g) for g in gradients)
            assert torch.equal(value["mean_state"], pilot.state_from_boxes(batch["decoded_boxes"]))
        write(OUT / "smoke" / "engineering_check.json", {**result, "head_gradient_check": True,
                                                         "head_optimizer_updates": 0})


def canonicalize():
    check_freeze()
    records = {}
    for split in SPLITS:
        writer = Writer(OUT / "authority" / split)
        seen = []
        for shard in range(SHARDS[split]):
            root = OUT / "raw" / ("{}_{:02d}".format(split, shard))
            accepted = json.loads((root / "accepted.json").read_text())
            assert accepted["accepted"] and not accepted["is_engineering_smoke"]
            for part in accepted["parts"]:
                path = root / "parts" / part["file"]
                assert sha(path) == part["sha256"]
                for row in unpack_rows(path):
                    token = str(row["sample_tokens"])
                    if seen and token <= seen[-1]:
                        raise ValueError("Non-canonical or repeated sample token")
                    seen.append(token)
                    writer.append(row)
        _, infos = load_split(split)
        assert len(seen) == COUNTS[split] and set(seen) == set(infos)
        records[split] = {"files": writer.close(), "sample_count": len(seen)}
    scene_map = original.load_sample_scene_map()
    sets = {split: {scene_map[t] for t in load_split(split)[1]} for split in SPLITS}
    assert all(not sets[a] & sets[b] for a, b in itertools.combinations(SPLITS, 2))
    write(OUT / "authority/manifest.json", {"accepted": True, "schema": "hivt_recovery_minimal_query_v1",
                                           "splits": records, "legacy_density_column_present": False,
                                           "original_manifest_hash_claimed": False})


def rows(split):
    manifest = json.loads((OUT / "authority/manifest.json").read_text())
    for item in manifest["splits"][split]["files"]:
        path = OUT / "authority" / split / item["file"]
        assert sha(path) == item["sha256"]
        yield from unpack_rows(path)


def baseline():
    counts = np.zeros(10, dtype=np.int64)
    sums = np.zeros((10, 128), dtype=np.float64)
    squares = np.zeros_like(sums)
    for row in rows("train"):
        match = row["native_matched_gt"]
        positive = match >= 0
        labels = row["gt_labels"][match[positive]]
        feature = row["query_features"][positive].astype(np.float64)
        for category in range(10):
            selected = feature[labels == category]
            counts[category] += len(selected)
            sums[category] += selected.sum(0)
            squares[category] += (selected * selected).sum(0)
    assert np.all(counts >= 2)
    means = sums / counts[:, None]
    variances = np.maximum(squares / counts[:, None] - means * means, 1e-4)
    density = {"means": means.tolist(), "variances": variances.tolist(), "counts": counts.tolist()}
    by_class = [[] for _ in range(10)]
    for row in rows("train"):
        match = row["native_matched_gt"]
        positive = match >= 0
        labels = row["gt_labels"][match[positive]]
        raw = formal.density_raw(row["query_features"][positive], labels, density)
        for category in range(10):
            selected = raw[labels == category]
            if len(selected):
                by_class[category].append(selected)
    density["lower"] = [float(np.quantile(np.concatenate(v), .001)) for v in by_class]
    density["upper"] = [float(np.quantile(np.concatenate(v), .999)) for v in by_class]
    scores, targets, densities, logits, labels = [], [], [], [], []
    for row in rows("label_cal"):
        match = row["native_matched_gt"]
        positive = match >= 0
        value = row["class_logits"]
        scores.append(value.max(-1))
        targets.append(positive.astype(np.float64))
        densities.append(formal.density_score(row["query_features"], value.argmax(-1), density))
        logits.append(value[positive])
        labels.append(row["gt_labels"][match[positive]])
    score, target, density_value = np.concatenate(scores), np.concatenate(targets), np.concatenate(densities)
    temperature = formal.fit_class_temperature(np.concatenate(logits), np.concatenate(labels))
    write(OUT / "baseline.json", {"class_temperature": temperature.tolist(), "density": density,
                                  "m0_existence": formal.logistic_fit(np.column_stack((score, np.ones_like(score))), target).tolist(),
                                  "m5_existence": formal.logistic_fit(np.column_stack((score, density_value, np.ones_like(score))), target).tolist(),
                                  "authority_sha256": sha(OUT / "authority/manifest.json"),
                                  "recomputed_baseline": True, "original_baseline_hash_claimed": False})


def train(seed, gpu):
    check_freeze()
    device = setup_gpu(gpu, seed)
    pilot.MANIFEST = OUT / "authority/manifest.json"
    config = json.loads((OUT / "baseline.json").read_text())
    temperature = torch.tensor(config["class_temperature"], device=device, dtype=torch.float32)
    heads = {name: ProbabilisticQueryHead().to(device) for name in formal.LEARNED}
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4) for name, head in heads.items()}
    checkpoint = OUT / "checkpoints" / ("seed_{}.pt".format(seed))
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    started, visits, updates = time.time(), 0, 0
    log_path = OUT / "training" / ("seed_{}_epochs.jsonl".format(seed))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as stream:
        for epoch in range(formal.EPOCHS):
            epoch_visits = 0
            total = {name: 0. for name in heads}
            epoch_updates = 0
            for batch in formal.training_batches(device):
                inputs = (batch["query_feature"], batch["class_logits"], batch["decoded_boxes"],
                          batch["seed_score"], batch["query_density"])
                for name, family, neighbors in (("M1", "gaussian", None), ("M2", "student_t", None),
                                                 ("M3", "student_t", batch["neighborhood"])):
                    optimizers[name].zero_grad(set_to_none=True)
                    value = heads[name](*inputs)
                    losses = f1_loss(value, batch["matched_gt_index"], batch["gt_state"], batch["gt_yaw"],
                                     batch["gt_labels"], temperature, family=family, neighborhood=neighbors)
                    assert all(torch.isfinite(v) for v in losses.values())
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(heads[name].parameters(), 5., error_if_nonfinite=True)
                    optimizers[name].step()
                    total[name] += float(losses["total"].detach()) * len(batch["query_feature"])
                epoch_visits += len(batch["query_feature"])
                epoch_updates += 1
            assert epoch_visits == COUNTS["train"]
            assert all(torch.isfinite(p).all() for head in heads.values() for p in head.parameters())
            visits += epoch_visits
            updates += epoch_updates
            record = {"epoch": epoch + 1, "seed": seed, "physical_gpu": gpu, "scene_visits": epoch_visits,
                      "updates_per_model": epoch_updates, "mean_loss": {k: v / epoch_visits for k, v in total.items()},
                      "seconds": time.time() - started}
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps({"event": "head_epoch", **record}), flush=True)
    formal.write_torch(checkpoint, {"epochs": formal.EPOCHS, "models": {k: v.state_dict() for k, v in heads.items()},
                                   "seed": seed, "train_scene_visits": visits, "updates_per_model": updates,
                                   "probability_head_sha256": sha(ROOT / "probset3d/f1/implementation/probabilistic_query.py"),
                                   "wall_time_seconds": time.time() - started, "recovery_manifest_sha256": sha(OUT / "authority/manifest.json"),
                                   "recovery_baseline_sha256": sha(OUT / "baseline.json"), "original_checkpoint_hash_claimed": False})
    write(OUT / "training" / ("seed_{}_complete.json".format(seed)), {"checkpoint_sha256": sha(checkpoint),
                                                                   "seed": seed, "finite": True, "updates_per_model": updates})


def parallel(tasks, stage):
    pending, active, completed, failed = list(tasks), {}, [], []
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    while pending or active:
        if not failed:
            for gpu in GPUS:
                if gpu in active or not pending:
                    continue
                name, arguments = pending.pop(0)
                stream = (OUT / "logs" / (name + ".log")).open("x", encoding="utf-8")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
                command = [sys.executable, str(Path(__file__)), *arguments, "--gpu", str(gpu)]
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active[gpu] = (process, stream, name)
                print(json.dumps({"event": "start", "stage": stage, "task": name, "gpu": gpu, "pid": process.pid}), flush=True)
        for gpu, (process, stream, name) in list(active.items()):
            code = process.poll()
            if code is not None:
                stream.close()
                (completed if code == 0 else failed).append({"task": name, "gpu": gpu, "exit_code": code})
                del active[gpu]
        write(OUT / "status.json", {"stage": stage, "active": {str(g): {"task": v[2], "pid": v[0].pid} for g, v in active.items()},
                                    "completed": completed, "failed": failed, "pending": len(pending)}, replace=True)
        if failed and not active:
            raise RuntimeError("Recovery task failed: " + json.dumps(failed))
        time.sleep(5)


def run():
    check_freeze()
    assert json.loads((OUT / "smoke/engineering_check.json").read_text())["head_gradient_check"]
    tasks = [("export_{}_{:02d}".format(s, i), ["export", "--split", s, "--shard", str(i)])
             for s in SPLITS for i in range(SHARDS[s])]
    parallel(tasks, "clean_query_export")
    write(OUT / "status.json", {"stage": "canonicalize_and_fit_baseline"}, replace=True)
    canonicalize()
    baseline()
    parallel([("train_seed_{}".format(seed), ["train", "--seed", str(seed)]) for seed in SEEDS], "head_training")
    manifest = {"status": "RECOVERY_COMPLETE", "authority": "authority/manifest.json",
                "authority_sha256": sha(OUT / "authority/manifest.json"), "baseline": "baseline.json",
                "baseline_sha256": sha(OUT / "baseline.json"), "old_results_rewritten": False,
                "original_head_hashes_claimed": False,
                "heads": {str(seed): {"path": "checkpoints/seed_{}.pt".format(seed),
                                      "sha256": sha(OUT / "checkpoints" / ("seed_{}.pt".format(seed)))} for seed in SEEDS}}
    write(OUT / "recovery_manifest.json", manifest)
    write(OUT / "status.json", {"stage": "RECOVERY_COMPLETE", "manifest": "recovery_manifest.json"}, replace=True)


def self_check():
    import tempfile
    with tempfile.TemporaryDirectory(prefix="hivt-recovery-check-") as directory:
        root = Path(directory)
        writer = Writer(root / "parts")
        row = {key: np.asarray("token", dtype="U32") if key in ("sample_tokens", "scene_tokens")
               else np.zeros((200,), dtype=np.float32) for key in FIXED}
        row.update(gt_boxes=np.ones((2, 9), dtype=np.float32), gt_labels=np.array([0, 1], dtype=np.int16),
                   duplicate_edge_query_id=np.array([2, 3], dtype=np.int16), duplicate_edge_gt_index=np.array([0, 1], dtype=np.int16))
        for _ in range(33):
            writer.append(row)
        parts = writer.close()
        restored = list(itertools.chain.from_iterable(unpack_rows(root / "parts" / p["file"]) for p in parts))
        assert [p["sample_count"] for p in parts] == [32, 1] and len(restored) == 33
        assert all(np.array_equal(row[k], restored[-1][k]) for k in row)
    print("PASS: part boundaries and variable-length fields round trip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "smoke", "export", "train", "run", "self-check"))
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--gpu", type=int, choices=GPUS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "smoke":
        export("dev", 0, args.gpu, True)
    elif args.command == "export":
        export(args.split, args.shard, args.gpu)
    elif args.command == "train":
        train(args.seed, args.gpu)
    elif args.command == "run":
        try:
            run()
        except Exception as error:
            write(OUT / "failure.json", {"type": type(error).__name__, "message": str(error), "time_unix": time.time()})
            raise
    else:
        self_check()


if __name__ == "__main__":
    main()
