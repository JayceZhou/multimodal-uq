#!/usr/bin/env python3
"""Run the minimal F1-V3 legality pilot without scientific readout."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from probset3d.f1.implementation.probabilistic_query import (
    CLASS_COUNT,
    MAX_KAPPA,
    MIN_DF,
    MIN_KAPPA,
    MIN_SCALE,
    ProbabilisticQueryHead,
    f1_loss,
    state_from_boxes,
)


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "outputs/probset3d_v3/f1_authority/authority.json"
MANIFEST = ROOT / "outputs/probset3d_v3/f1_authority/run_a/manifest.json"
HOST_BEFORE = ROOT / "outputs/probset3d_v2/authority/host_probe/repeat_a.json"
CHECKPOINT = ROOT / "pretrained/bevfusion-det.pth"
OUT = ROOT / "outputs/probset3d_v3/f1_pilot/pilot.json"
SEED = 20260730
BATCH_SIZE = 8
QUERY_COUNT = 200
HEAD_STEPS = 100
TEMPERATURE_STEPS = 25
TEMPERATURE_SCENES = BATCH_SIZE * TEMPERATURE_STEPS
EXPECTED_COUNTS = {
    "train": (22508, 4501600),
    "label_cal": (2814, 562800),
    "dev": (2808, 561600),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def strict_write(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen pilot: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def state_sha256(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


def output_sha256(output) -> str:
    return state_sha256(output)


def array_slice(part, name: str, row: int):
    offsets = part[name + "_offsets"]
    return np.asarray(part[name][offsets[row] : offsets[row + 1]])


def authority_parts(split: str):
    manifest = strict_load(MANIFEST)
    root = MANIFEST.parent
    for item in manifest["splits"][split]["files"]:
        path = root / split / item["file"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"authority content hash mismatch: {path}")
        yield path


def make_batch(rows, device: torch.device):
    max_gt = max(len(row["gt_labels"]) for row in rows)
    if max_gt == 0:
        raise ValueError("pilot batch contains no GT")
    gt_boxes = np.zeros((len(rows), max_gt, 9), dtype=np.float32)
    gt_boxes[..., 3:6] = 1.0
    gt_labels = np.zeros((len(rows), max_gt), dtype=np.int64)
    neighborhood = np.zeros((len(rows), max_gt, QUERY_COUNT), dtype=bool)
    for index, row in enumerate(rows):
        count = len(row["gt_labels"])
        gt_boxes[index, :count] = row["gt_boxes"]
        gt_labels[index, :count] = row["gt_labels"]
        neighborhood[
            index,
            row["duplicate_edge_gt_index"],
            row["duplicate_edge_query_id"],
        ] = True
    decoded = torch.from_numpy(
        np.stack([row["decoded_box_mean"] for row in rows])
    ).to(device)
    return {
        "query_feature": torch.from_numpy(
            np.stack([row["query_features"] for row in rows])
        ).to(device),
        "class_logits": torch.from_numpy(
            np.stack([row["class_logits"] for row in rows])
        ).to(device),
        "decoded_boxes": decoded,
        "seed_score": torch.from_numpy(
            np.stack([row["seed_score"] for row in rows])
        ).to(device),
        "query_density": torch.zeros(
            len(rows), QUERY_COUNT, dtype=torch.float32, device=device
        ),
        "matched_gt_index": torch.from_numpy(
            np.stack([row["native_matched_gt"] for row in rows]).astype(np.int64)
        ).to(device),
        "gt_state": state_from_boxes(torch.from_numpy(gt_boxes).to(device)),
        "gt_yaw": torch.from_numpy(gt_boxes[..., 6]).to(device),
        "gt_labels": torch.from_numpy(gt_labels).to(device),
        "neighborhood": torch.from_numpy(neighborhood).to(device),
    }


def head_batches(device: torch.device):
    pending = []
    yielded = 0
    for path in authority_parts("train"):
        with np.load(path, allow_pickle=False) as part:
            for row in range(len(part["sample_tokens"])):
                pending.append(
                    {
                        "query_features": np.asarray(part["query_features"][row]),
                        "class_logits": np.asarray(part["class_logits"][row]),
                        "decoded_box_mean": np.asarray(part["decoded_box_mean"][row]),
                        "seed_score": np.asarray(part["seed_score"][row]),
                        "native_matched_gt": np.asarray(part["native_matched_gt"][row]),
                        "gt_boxes": array_slice(part, "gt_boxes", row),
                        "gt_labels": array_slice(part, "gt_labels", row),
                        "duplicate_edge_query_id": array_slice(
                            part, "duplicate_edge_query_id", row
                        ).astype(np.int64),
                        "duplicate_edge_gt_index": array_slice(
                            part, "duplicate_edge_gt_index", row
                        ).astype(np.int64),
                    }
                )
                if len(pending) == BATCH_SIZE:
                    yield make_batch(pending, device)
                    pending = []
                    yielded += 1
                    if yielded == HEAD_STEPS:
                        return
    raise ValueError("train authority ended before fixed pilot schedule")


def temperature_batches(device: torch.device):
    pending = []
    consumed = 0
    for path in authority_parts("label_cal"):
        with np.load(path, allow_pickle=False) as part:
            for row in range(len(part["sample_tokens"])):
                pending.append(
                    {
                        "class_logits": np.asarray(part["class_logits"][row]),
                        "native_matched_gt": np.asarray(part["native_matched_gt"][row]),
                        "gt_labels": array_slice(part, "gt_labels", row),
                    }
                )
                consumed += 1
                if len(pending) == BATCH_SIZE:
                    selected_logits = []
                    selected_labels = []
                    for item in pending:
                        positive = np.flatnonzero(item["native_matched_gt"] >= 0)
                        if len(positive):
                            selected_logits.append(item["class_logits"][positive])
                            selected_labels.append(
                                item["gt_labels"][item["native_matched_gt"][positive]]
                            )
                    if not selected_logits:
                        raise ValueError("temperature batch contains no matched query")
                    yield (
                        torch.from_numpy(np.concatenate(selected_logits)).to(device),
                        torch.from_numpy(
                            np.concatenate(selected_labels).astype(np.int64)
                        ).to(device),
                    )
                    pending = []
                    if consumed == TEMPERATURE_SCENES:
                        return
    raise ValueError("label_cal authority ended before fixed pilot schedule")


def initial_temperature(device: torch.device):
    value = math.log(math.expm1(0.95))
    return torch.nn.Parameter(torch.full((CLASS_COUNT,), value, device=device))


def temperature_value(raw: torch.Tensor):
    return F.softplus(raw) + 0.05


def run_once(device: torch.device):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    head = ProbabilisticQueryHead().to(device)
    initial_state = state_sha256(head.state_dict())
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    probe = None
    finite_steps = 0
    head_gradient_failures = 0
    input_gradient_failures = 0
    positive_count = 0
    negative_count = 0
    duplicate_count = 0
    competition_steps = 0
    for batch in head_batches(device):
        if probe is None:
            probe = {
                name: value.detach().clone()
                for name, value in batch.items()
                if torch.is_tensor(value)
            }
        inputs = tuple(
            batch[name]
            for name in (
                "query_feature",
                "class_logits",
                "decoded_boxes",
                "seed_score",
                "query_density",
            )
        )
        input_hashes = [tensor_sha256(value) for value in inputs]
        for value in inputs:
            value.requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        output = head(*inputs)
        losses = f1_loss(
            output,
            batch["matched_gt_index"],
            batch["gt_state"],
            batch["gt_yaw"],
            batch["gt_labels"],
            torch.ones(CLASS_COUNT, device=device),
            family="student_t",
            neighborhood=batch["neighborhood"],
        )
        if not all(torch.isfinite(value) for value in losses.values()):
            raise ValueError("non-finite pilot loss")
        losses["total"].backward()
        input_gradient_failures += int(any(value.grad is not None for value in inputs))
        gradients = [parameter.grad for parameter in head.parameters()]
        head_gradient_failures += int(
            any(value is None or not torch.isfinite(value).all() for value in gradients)
        )
        if input_hashes != [tensor_sha256(value) for value in inputs]:
            raise ValueError("frozen detector-derived tensor changed")
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in head.parameters()):
            raise ValueError("non-finite probability-head parameter")
        finite_steps += 1
        positive = int((batch["matched_gt_index"] >= 0).sum())
        positive_count += positive
        negative_count += BATCH_SIZE * QUERY_COUNT - positive
        duplicate_count += int((batch["neighborhood"].sum(-1) > 1).sum())
        competition_steps += int(float(losses["competition"].detach()) > 0)
        history.append([float(value.detach()) for value in losses.values()])

    raw_temperature = initial_temperature(device)
    temperature_optimizer = torch.optim.AdamW(
        [raw_temperature], lr=1e-2, weight_decay=0.0
    )
    temperature_history = []
    for logits, labels in temperature_batches(device):
        temperature_optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits.detach() / temperature_value(raw_temperature), labels)
        if not torch.isfinite(loss):
            raise ValueError("non-finite temperature loss")
        loss.backward()
        if raw_temperature.grad is None or not torch.isfinite(raw_temperature.grad).all():
            raise ValueError("non-finite temperature gradient")
        temperature_optimizer.step()
        temperature_history.append(float(loss.detach()))
    if len(temperature_history) != TEMPERATURE_STEPS or probe is None:
        raise ValueError("fixed pilot schedule was not completed")

    with torch.inference_mode():
        final_output = head(
            probe["query_feature"],
            probe["class_logits"],
            probe["decoded_boxes"],
            probe["seed_score"],
            probe["query_density"],
        )
    probability = final_output["existence_logit"].sigmoid()
    scale = final_output["scale"]
    degrees = final_output["degrees_of_freedom"]
    kappa = final_output["yaw_kappa"]
    temperature = temperature_value(raw_temperature)
    legality = {
        "existence": bool(torch.isfinite(probability).all() and ((probability > 0) & (probability < 1)).all()),
        "scale": bool(torch.isfinite(scale).all() and (scale >= MIN_SCALE).all()),
        "degrees_of_freedom": bool(torch.isfinite(degrees).all() and (degrees > MIN_DF).all()),
        "kappa": bool(torch.isfinite(kappa).all() and (kappa >= MIN_KAPPA).all() and (kappa <= MAX_KAPPA).all()),
        "temperature": bool(torch.isfinite(temperature).all() and (temperature > 0).all()),
    }
    collapse = {
        "existence_not_all_zero_or_one": not bool((probability == 0).all() or (probability == 1).all()),
        "scale_not_all_at_bound": not bool((scale <= MIN_SCALE + 1e-7).all() or (scale >= 1e6).all()),
        "df_not_all_at_minimum": not bool((degrees <= MIN_DF + 1e-7).all()),
        "kappa_not_all_saturated": not bool((kappa >= MAX_KAPPA).all()),
        "competition_nonzero": competition_steps > 0,
        "duplicates_consumed": duplicate_count > 0,
        "positive_and_negative_consumed": positive_count > 0 and negative_count > 0,
    }
    final_state = state_sha256(head.state_dict())
    if initial_state == final_state:
        raise ValueError("probability head did not update")

    with tempfile.TemporaryDirectory(prefix="probset3d-v3-pilot-", dir="/tmp") as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        torch.save(
            {"head": head.state_dict(), "temperature": raw_temperature.detach().cpu()},
            checkpoint,
        )
        restored = ProbabilisticQueryHead().to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        restored.load_state_dict(payload["head"])
        with torch.inference_mode():
            restored_output = restored(
                probe["query_feature"],
                probe["class_logits"],
                probe["decoded_boxes"],
                probe["seed_score"],
                probe["query_density"],
            )
        roundtrip = all(
            torch.equal(value, restored_output[name])
            for name, value in final_output.items()
        ) and torch.equal(raw_temperature.detach().cpu(), payload["temperature"])

    history_digest = hashlib.sha256()
    history_digest.update(np.asarray(history, dtype=np.float64).tobytes())
    history_digest.update(np.asarray(temperature_history, dtype=np.float64).tobytes())
    history_hash = history_digest.hexdigest()
    result = {
        "collapse": collapse,
        "finite_head_steps": finite_steps,
        "final_state_sha256": final_state,
        "head_gradient_failures": head_gradient_failures,
        "history_sha256": history_hash,
        "initial_state_sha256": initial_state,
        "input_gradient_failures": input_gradient_failures,
        "legality": legality,
        "output_sha256": output_sha256(final_output),
        "query_density_values_read": 0,
        "roundtrip_byte_identical": roundtrip,
        "temperature_sha256": tensor_sha256(raw_temperature),
    }
    return result


def host_after_pilot(device: int):
    from tools import probset3d_v2_authority as host

    before = strict_load(HOST_BEFORE)
    with tempfile.TemporaryDirectory(prefix="probset3d-v3-host-", dir="/tmp") as directory:
        host.PROBE_ROOT = Path(directory)
        args = argparse.Namespace(name="after_pilot", device=device, workers=0)
        with contextlib.redirect_stdout(io.StringIO()):
            host.host_probe(args)
        after = strict_load(Path(directory) / "after_pilot.json")
    return {
        "byte_identical": before["records"] == after["records"],
        "checkpoint_identical": before["checkpoint_sha256"] == after["checkpoint_sha256"],
        "detector_state_unchanged": (
            after["accepted"]
            and after["detector_state_sha256_before"]
            == after["detector_state_sha256_after"]
        ),
        "sample_count": after["sample_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, choices=(0, 1, 2, 3), default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    authority = strict_load(AUTHORITY)
    manifest = strict_load(MANIFEST)
    counts_valid = all(
        (manifest["splits"][split]["sample_count"], manifest["splits"][split]["query_record_count"])
        == expected
        for split, expected in EXPECTED_COUNTS.items()
    )
    authority_valid = bool(
        authority["accepted"]
        and authority["run_a_run_b_scientific_content_identical"]
        and authority["host_detection_output_byte_identical"]
        and manifest["accepted"]
        and not manifest["legacy_query_density_column_read_allowed"]
        and counts_valid
    )
    if not authority_valid:
        raise ValueError("F1-V3 authority is not accepted")

    checkpoint_before = sha256(CHECKPOINT)
    first = run_once(torch.device("cpu"))
    second = run_once(torch.device("cpu"))
    deterministic_keys = (
        "collapse",
        "finite_head_steps",
        "final_state_sha256",
        "head_gradient_failures",
        "history_sha256",
        "initial_state_sha256",
        "input_gradient_failures",
        "legality",
        "output_sha256",
        "query_density_values_read",
        "roundtrip_byte_identical",
        "temperature_sha256",
    )
    same_seed = all(first[key] == second[key] for key in deterministic_keys)
    host = host_after_pilot(args.device)
    checkpoint_after = sha256(CHECKPOINT)
    gates = {
        "P01_authority_complete": authority_valid,
        "P02_host_byte_identity": host["byte_identical"],
        "P03_gradient_isolation": (
            first["input_gradient_failures"] == second["input_gradient_failures"] == 0
            and first["head_gradient_failures"] == second["head_gradient_failures"] == 0
        ),
        "P04_frozen_parameter_identity": (
            checkpoint_before == checkpoint_after
            and host["checkpoint_identical"]
            and host["detector_state_unchanged"]
        ),
        "P05_finite_optimization": (
            first["finite_head_steps"] == second["finite_head_steps"] == HEAD_STEPS
        ),
        "P06_parameter_legality": all(first["legality"].values()) and all(second["legality"].values()),
        "P07_no_trivial_collapse": all(first["collapse"].values()) and all(second["collapse"].values()),
        "P08_checkpoint_roundtrip": first["roundtrip_byte_identical"] and second["roundtrip_byte_identical"],
        "P09_same_seed_determinism": same_seed,
        "P10_forbidden_read_isolation": (
            first["query_density_values_read"] == second["query_density_values_read"] == 0
        ),
    }
    passed = all(gates.values())
    result = {
        "authority_sha256": sha256(AUTHORITY),
        "detector_checkpoint_sha256": checkpoint_after,
        "endpoint": (
            "PASS — GO TO F1 FORMAL SCIENTIFIC TEST"
            if passed
            else "INVALID — F1-V3 LEGALITY PILOT FAILED"
        ),
        "formal_f1_scientific_test_authorized": passed,
        "gates": gates,
        "head_steps_per_run": HEAD_STEPS,
        "host_probe": host,
        "passed": passed,
        "probability_head_sha256": sha256(
            ROOT / "probset3d/f1/implementation/probabilistic_query.py"
        ),
        "same_seed_runs": 2,
        "schema_version": "probset3d_f1_v3_legality_pilot_v1",
        "scientific_metrics_generated_or_read": False,
        "seed": SEED,
        "temperature_steps_per_run": TEMPERATURE_STEPS,
    }
    strict_write(OUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
