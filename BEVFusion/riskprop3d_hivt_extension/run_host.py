"""Engineering qualification, three seeded native host fits, and clean calibration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from hivt_runtime import ContextDataset, HiVT, MODEL, collate, forward, loss, perturb
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
OUT = HERE / "host_v1"
SEEDS = (202610130, 202610131, 202610132)


def sha(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def gpu(seed, physical):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical) or physical not in (0, 2, 3):
        raise RuntimeError("Pin a declared GPU 0, 2 or 3")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda:0")


def hashed_rows(dataset, count):
    a = dataset.arrays
    hashes = [hashlib.sha256((str(s) + "/" + str(i)).encode()).digest()
              for s, i in zip(a["sample_token"], a["instance_token"])]
    return np.asarray(sorted(range(len(dataset)), key=lambda i: hashes[i])[:count])


def engineering(physical):
    if not (HERE / "data_v1/audit.json").exists():
        subprocess.check_call([sys.executable, str(HERE / "audit_data.py")])
    device = gpu(SEEDS[0], physical)
    dataset = ContextDataset("train", labels=True)
    rows = hashed_rows(dataset, 8)
    batch = collate([dataset[int(i)] for i in rows]).to(device)
    model = HiVT(**MODEL).to(device)
    optimizers, _ = model.configure_optimizers()
    optimizer = optimizers[0]
    model.eval()
    initial = float(loss(model, batch).detach())
    positions = batch.positions.clone()
    labels = batch.label.clone()
    with torch.no_grad():
        first, second = forward(model, batch), forward(model, batch)
        repeat_difference = max(float((a - b).abs().max()) for a, b in zip(first, second))
    assert torch.equal(batch.positions, positions) and torch.equal(batch.label, labels) and batch.y is None
    model.train()
    curve = []
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        value = loss(model, batch)
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        curve.append(float(value.detach()))
    model.eval()
    final = float(loss(model, batch).detach())
    assert final < initial, (initial, final)
    dev = ContextDataset("clean_s0")
    checks = hashed_rows(dev, 8)
    context = collate([dev[int(i)] for i in checks]).to(device)
    with torch.no_grad():
        normal = forward(model, context)[0]
        no_map = context.clone()
        no_map.lane_actor_index = no_map.lane_actor_index[:, :0]
        no_map.lane_actor_vectors = no_map.lane_actor_vectors[:0]
        no_neighbors = context.clone()
        no_neighbors.edge_index = no_neighbors.edge_index[:, :0]
        map_difference = float((forward(model, no_map)[0] - normal).abs().max())
        neighbor_difference = float((forward(model, no_neighbors)[0] - normal).abs().max())
    assert map_difference > 0 and neighbor_difference > 0
    timing = []
    with torch.no_grad():
        for size in (32, 64, 128):
            data = collate([dev[int(i)] for i in hashed_rows(dev, size)]).to(device)
            forward(model, data)
            torch.cuda.synchronize()
            started = time.time()
            for _ in range(5):
                forward(model, data)
            torch.cuda.synchronize()
            timing.append({"graphs": size, "seconds_per_forward": (time.time() - started) / 5,
                           "peak_allocation_bytes": torch.cuda.max_memory_allocated()})
    write(OUT / "engineering.json", {"status": "PASS", "gpu": physical,
        "overfit_rows": rows.tolist(), "initial_loss": initial, "final_loss": final,
        "overfit_loss": curve, "updates": 100, "dev_check_rows": checks.tolist(),
        "map_ablation_max_mean_difference": map_difference, "neighbor_ablation_max_mean_difference": neighbor_difference,
        "repeat_max_difference": repeat_difference, "timing": timing,
        "labels_in_forward": False, "original_input_mutated": False,
        "native_scatter_reduction": "GPU atomic reductions; bitwise repeatability not assumed"})


def freeze():
    if json.loads((OUT / "engineering.json").read_text())["status"] != "PASS":
        raise RuntimeError("Engineering checks required")
    files = [Path(__file__), HERE / "hivt_runtime.py", HERE / "DATA_CONTRACT.md",
             HERE / "host_v1/engineering.json", HERE / "data_v1/audit.json", HERE / "TIME_AUDIT_NOTE.md"]
    files += sorted((HERE / "third_party/MapUncertaintyPrediction/HiVT_modified").glob("models/*.py"))
    files += sorted((HERE / "third_party/MapUncertaintyPrediction/HiVT_modified").glob("losses/*.py"))
    files += sorted((HERE / "third_party/MapUncertaintyPrediction/HiVT_modified").glob("metrics/*.py"))
    files.append(HERE / "third_party/MapUncertaintyPrediction/HiVT_modified/utils.py")
    files.append(HERE / "runtime_compatibility.patch")
    write(OUT / "freeze.json", {"model": MODEL, "host_seeds": SEEDS,
        "head_seeds": [20260731, 20260801, 20260802], "state_seeds": [202610140, 202610141, 202610142],
        "future_seeds": [202610150, 202610151, 202610152], "epochs": 64, "batch_size": 32,
        "optimizer": "native AdamW parameter groups", "clip_norm": 5., "AMP": False,
        "scheduler": "native CosineAnnealingLR T_max=64", "checkpoint": "lowest clean-dev target minFDE; earliest tie",
        "calibration": "one positive scale factor minimizing clean label-cal six-step mixture Laplace NLL",
        "supervision": "retained target only; future labels excluded from model forward",
        "sources": {str(p.relative_to(HERE)): sha(p) for p in files},
        "torch": torch.__version__, "python": sys.version, "level": "I only", "law_results_read": False,
        "reference_commit": "0a11ee4b925e4967f2f889079d4086b663d396dc",
        "private_dependencies": {"torch_geometric": "2.5.3", "torch_scatter": "2.1.2+pt24cu124", "torch_sparse": "0.6.18+pt24cu124",
                                 "pytorch_lightning": "2.3.3", "torchmetrics": "1.4.3", "lightning_utilities": "0.11.9"}})


def check_freeze():
    value = json.loads((OUT / "freeze.json").read_text())
    for name, digest in value["sources"].items():
        assert sha(HERE / name) == digest, name
    return value


def dev_metrics(model, loader, device):
    values = []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            means, scales, probabilities, _, _ = forward(model, data)
            errors = torch.linalg.vector_norm(means - data.label[:, None], dim=-1)
            prediction_mean = (probabilities[:, :, None, None] * means).sum(1)
            mean_error = torch.linalg.vector_norm(prediction_mean - data.label, dim=-1)
            values.append(torch.stack((errors[:, :, -1].min(1).values, errors.mean(-1).min(1).values,
                                       mean_error.mean(-1), mean_error[:, -1]), axis=1).cpu().numpy())
    array = np.concatenate(values)
    return dict(zip(("min_fde", "min_ade", "mean_ade", "mean_fde"), array.mean(0).astype(float).tolist()))


def train(replicate, physical):
    config = check_freeze()
    seed = SEEDS[replicate]
    device = gpu(seed, physical)
    directory = OUT / ("run_" + str(replicate))
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir()
    training = ContextDataset("train", labels=True)
    development = ContextDataset("clean_s0", labels=True)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(training, batch_size=config["batch_size"], shuffle=True, generator=generator,
                        num_workers=2, collate_fn=collate, persistent_workers=True)
    dev_loader = DataLoader(development, batch_size=64, shuffle=False, num_workers=2,
                            collate_fn=collate, persistent_workers=True)
    model = HiVT(**config["model"]).to(device)
    optimizers, schedulers = model.configure_optimizers()
    optimizer, scheduler = optimizers[0], schedulers[0]
    initial = dev_metrics(model, dev_loader, device)
    write(directory / "epoch0.json", initial)
    best, updates = float("inf"), 0
    started = time.time()
    curve = []
    with (directory / "epochs.jsonl").open("x") as stream:
        for epoch in range(config["epochs"]):
            model.train()
            total, seen = 0., 0
            for data in loader:
                data = data.to(device)
                optimizer.zero_grad(set_to_none=True)
                value = loss(model, data)
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_norm"], error_if_nonfinite=True)
                optimizer.step()
                total += float(value.detach()) * len(data.target_index)
                seen += len(data.target_index)
                updates += 1
            assert seen == len(training)
            metrics = dev_metrics(model, dev_loader, device)
            record = {"epoch": epoch + 1, "loss": total / seen, "development": metrics,
                      "seen": seen, "updates": updates, "lr": scheduler.get_last_lr()[0], "seconds": time.time() - started}
            curve.append(record)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(record), flush=True)
            if metrics["min_fde"] < best:
                best = metrics["min_fde"]
                payload = {"state_dict": model.state_dict(), "model": config["model"], "host_seed": seed,
                           "replicate": replicate, "epoch": epoch + 1, "development": metrics,
                           "freeze_sha256": sha(OUT / "freeze.json")}
                torch.save(payload, directory / "best.pt.partial")
                (directory / "best.pt.partial").replace(directory / "best.pt")
            scheduler.step()
    assert min(r["loss"] for r in curve[1:]) < curve[0]["loss"]
    assert best < initial["min_fde"], (best, initial)
    write(directory / "training_complete.json", {"status": "PASS", "host_seed": seed, "gpu": physical,
        "epochs": len(curve), "updates": updates, "checkpoint_sha256": sha(directory / "best.pt"),
        "initial_development": initial, "selected_min_fde": best, "seconds": time.time() - started,
        "peak_allocation_bytes": torch.cuda.max_memory_allocated(), "training_manifest_sha256": sha(HERE / "data_v1/train/manifest.json"),
        "dev_manifest_sha256": sha(HERE / "data_v1/clean_s0/manifest.json")})


def load_model(replicate, device):
    checkpoint = torch.load(OUT / ("run_" + str(replicate)) / "best.pt", map_location="cpu", weights_only=True)
    model = HiVT(**checkpoint["model"]).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    return model


def calibrate(replicate, physical):
    check_freeze()
    device = gpu(SEEDS[replicate], physical)
    model = load_model(replicate, device)
    dataset = ContextDataset("label_cal", labels=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, collate_fn=collate)
    distances, scales, probabilities = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, b, p, inverse, local_means = forward(model, data)
            local_label = torch.bmm(data.label - (data.positions[data.target_index, -1] + data.origin)[:, None], inverse.transpose(1, 2))
            distances.append((local_label[:, None] - local_means).abs().cpu().numpy())
            scales.append(b.cpu().numpy())
            probabilities.append(p.cpu().numpy())
    distance, scale, probability = map(np.concatenate, (distances, scales, probabilities))

    def objective(log_factor):
        b = scale.astype(np.float64) * np.exp(log_factor)
        log_density = (-np.log(2 * b) - distance / b).sum(axis=(-1, -2))
        return float(-logsumexp(np.log(probability) + log_density, axis=1).mean())

    result = minimize_scalar(objective, bounds=(-8, 8), method="bounded", options={"xatol": 1e-7})
    assert result.success and abs(float(result.x)) < 7.99
    write(OUT / ("run_" + str(replicate)) / "calibration.json", {"scale_factor": float(np.exp(result.x)),
        "initial_mixture_nll": objective(0), "calibrated_mixture_nll": float(result.fun),
        "rows": len(dataset), "source_manifest_sha256": sha(HERE / "data_v1/label_cal/manifest.json"),
        "checkpoint_sha256": sha(OUT / ("run_" + str(replicate)) / "best.pt"), "all_A_groups_share_factor": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("engineering", "freeze", "train", "calibrate"))
    parser.add_argument("--replicate", type=int, choices=range(3))
    parser.add_argument("--gpu", type=int, choices=(0, 2, 3))
    args = parser.parse_args()
    if args.command == "engineering":
        engineering(args.gpu)
    elif args.command == "freeze":
        freeze()
    elif args.command == "train":
        train(args.replicate, args.gpu)
    else:
        calibrate(args.replicate, args.gpu)
