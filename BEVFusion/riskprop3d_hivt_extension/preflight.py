"""Check the frozen inputs and the native HiVT runtime before host training."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / "third_party/MapUncertaintyPrediction"
COMMIT = "0a11ee4b925e4967f2f889079d4086b663d396dc"
sys.path.insert(0, str(HERE / "python"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file(path, expected=None):
    path = Path(path)
    result = {"path": str(path), "exists": path.is_file(), "expected_sha256": expected}
    result["sha256"] = sha256(path) if result["exists"] else None
    result["valid"] = result["exists"] and (expected is None or result["sha256"] == expected)
    return result


def inspect_inputs():
    import numpy as np

    authority = ROOT / "outputs/riskprop3d_level1_matched_law/frozen_config.json"
    frozen = json.loads(authority.read_text())
    required = [check_file(authority)]
    for seed in frozen["head_seeds"]:
        name = "outputs/probset3d_v3/f1_formal/checkpoints/seed_{}.pt".format(seed)
        required.append(check_file(ROOT / name, frozen["source_sha256"][name]))
    r1_path = ROOT / "outputs/riskprop3d_r1/prepare_manifest.json"
    r1 = json.loads(r1_path.read_text())
    required.append(check_file(r1_path))
    required.append(check_file(
        ROOT / "outputs/probset3d_v3/f1_authority/run_a/manifest.json",
        r1["f1_v3_authority_manifest_sha256"],
    ))
    required.append(check_file(ROOT / "outputs/probset3d_v3/f1_formal/baseline.json"))
    records = {}
    for name in ("train", "label_cal", "clean_s0"):
        path = ROOT / "outputs/riskprop3d_r0/tracks" / (name + ".npz")
        required.append(check_file(path, frozen["source_sha256"].get(str(path.relative_to(ROOT)))))
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as arrays:
            keep = np.isin(arrays["class"], (0, 1, 2, 3, 4))
            assert arrays["input"].shape == (len(keep), 64)
            assert arrays["future"].shape == (len(keep), 6, 2)
            records[name] = {"rows": len(keep), "vehicle_rows": int(keep.sum()),
                             "vehicle_scenes": len(np.unique(arrays["scene_token"][keep]))}
        scene_path = ROOT / "outputs/riskprop3d_r1/scenes" / (name + ".npz")
        required.append(check_file(scene_path, r1["records"][name]["sha256"]))
    return required, records


def gpu_smoke():
    import torch

    assignment = os.environ.get("CUDA_VISIBLE_DEVICES")
    if assignment not in ("0", "2", "3"):
        raise RuntimeError("Pin one qualified physical GPU with CUDA_VISIBLE_DEVICES=0, 2 or 3")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    sys.path.insert(0, str(SOURCE / "HiVT_modified"))
    from models.hivt import HiVT
    from utils import TemporalData

    torch.manual_seed(202610130)
    device = torch.device("cuda:0")
    model = HiVT(historical_steps=5, future_steps=6, num_modes=6, rotate=True,
                 node_dim=2, edge_dim=2, embed_dim=128, num_heads=8, dropout=0.1,
                 num_temporal_layers=4, num_global_layers=3, local_radius=50,
                 parallel=False, lr=5e-4, weight_decay=1e-4, T_max=64, method="base").to(device)
    positions = torch.tensor([[[float(t), 0.] for t in range(5)],
                              [[float(t), 3.] for t in range(5)]], device=device)
    x = torch.zeros_like(positions)
    x[:, 1:] = positions[:, 1:] - positions[:, :-1]
    lane_index = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], device=device)
    lane_positions = torch.tensor([[0., 1.], [6., 1.]], device=device)
    bos = torch.zeros((2, 5), dtype=torch.bool, device=device)
    bos[:, 0] = True
    data = TemporalData(x=x, positions=positions, num_nodes=2,
                        edge_index=torch.tensor([[0, 1], [1, 0]], device=device),
                        y=None, padding_mask=torch.zeros((2, 11), dtype=torch.bool, device=device),
                        bos_mask=bos, rotate_angles=torch.zeros(2, device=device),
                        lane_vectors=torch.tensor([[6., 0.], [6., 0.]], device=device),
                        lane_actor_index=lane_index,
                        lane_actor_vectors=lane_positions[lane_index[0]] - positions[lane_index[1], -1])
    model.eval()
    with torch.no_grad():
        first = model(data.clone())
        second = model(data.clone())
    assert first[0].shape == (6, 2, 6, 4) and first[1].shape == (2, 6)
    assert all(torch.isfinite(value).all() for value in first)
    assert torch.all(first[0][..., 2:] > 0)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert data.y is None and torch.equal(data.positions, positions)
    model.train()
    prediction, logits = model(data.clone())
    # Labels exist only in this synthetic gradient check, never in the forward input.
    labels = torch.randn((2, 6, 2), device=device)
    distance = torch.linalg.vector_norm(prediction[..., :2] - labels, dim=-1).sum(-1)
    best = distance.argmin(0)
    regression = model.reg_loss(prediction[best, torch.arange(2, device=device)], labels)
    classification = model.cls_loss(logits, torch.softmax(-distance / 6, dim=0).t().detach())
    loss = regression + classification
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert torch.isfinite(loss) and gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) > 0 for g in gradients)
    return {"status": "PASS", "physical_gpu": assignment,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "synthetic_loss": float(loss.detach()),
            "optimizer_updates": 0, "real_data_training": False,
            "peak_tensor_memory_bytes": torch.cuda.max_memory_allocated(0)}


def self_check():
    with tempfile.TemporaryDirectory(prefix="hivt-preflight-") as directory:
        path = Path(directory) / "input.bin"
        assert not check_file(path)["valid"]
        path.write_bytes(b"frozen")
        expected = hashlib.sha256(b"frozen").hexdigest()
        assert check_file(path, expected)["valid"]
        path.write_bytes(b"changed")
        assert not check_file(path, expected)["valid"]
    print("PASS: missing, matching and changed input checks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "H0_attempt1_preflight.json")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    output = args.output.resolve()
    if HERE not in output.parents or output.exists():
        raise ValueError("Output must be a new file inside the HiVT experiment directory")
    commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    assert commit == COMMIT
    required, records = inspect_inputs()
    blockers = [item["path"] for item in required if not item["valid"]]
    smoke = None
    if args.gpu_smoke:
        try:
            smoke = gpu_smoke()
        except Exception as error:
            smoke = {"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}
            blockers.append("native HiVT GPU runtime smoke")
    packages = {}
    for name in ("torch", "torch-geometric", "torch-scatter", "torch-sparse",
                 "pytorch-lightning", "torchmetrics", "lightning-utilities"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"time_beijing": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "status": "INPUTS_MISSING" if blockers else "PREFLIGHT_PASS",
              "source_commit": commit, "plan_sha256": sha256(HERE / "SUPPLEMENTARY_PLAN.md"),
              "runner_sha256": sha256(Path(__file__)), "required_inputs": required,
              "row_inventory": records, "blockers": blockers, "packages": packages,
              "gpu_smoke": smoke, "formal_training_started": False,
              "stochastic_law_results_read": False, "sealed_test_accessed": False,
              "source_hashes": {str(p.relative_to(SOURCE)): sha256(p)
                                for p in sorted((SOURCE / "HiVT_modified").rglob("*.py"))}}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "blockers": blockers, "output": str(output)}))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
