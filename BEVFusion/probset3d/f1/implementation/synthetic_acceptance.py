#!/usr/bin/env python3
"""Run F1 implementation acceptance without reading any real query result."""

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch

from probset3d.f1.implementation.probabilistic_query import (
    BOX_DIM,
    CLASS_COUNT,
    COMPETITION_TEMPERATURE,
    DENSITY_GEOMETRY_INPUT_DIM,
    DUPLICATE_CENTER_METERS,
    GEOMETRY_INPUT_DIM,
    HIDDEN_DIM,
    MAX_KAPPA,
    MIN_DF,
    MIN_KAPPA,
    MIN_SCALE,
    QUERY_FEATURE_DIM,
    STATE_DIM,
    ProbabilisticQueryHead,
    calibrated_class_log_probs,
    competition_loss,
    duplicate_neighborhood,
    f1_loss,
    gaussian_log_prob,
    state_from_boxes,
    student_t_log_prob,
    von_mises_log_prob,
    wrap_angle,
)


ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE = ROOT / "probset3d/f1/acceptance"
SEED = 20260730

HASH_AUTHORITIES = {
    "probset3d/P0_DATA_SPLIT_AUTHORITY.json":
        "381b7265d9aa5a1a52618b918ac4a5ea21ca47b48c23995ba58de5640b87f4a9",
    "outputs/probset3d/f0/decision/decision.json":
        "c174aea4b6b2c7f7dabb47f704f93078247bad84387d87094f410bd84d5e338f",
    "outputs/probset3d/f0/decision/manifest.json":
        "d37a1809da2928a3c06307eb3682448e5fb37d1bd40d5ddf1d05daa7af7a2cab",
    "configs/nuscenes/det/transfusion/default.yaml":
        "772e0abef1a86a42f28511ab208dda3b0e3860eb2bbb4e42d70e04fe53b3c2d9",
    "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml":
        "cb7bbbc25e8fdf67c2e914b11661828b684312300deb2afc410793192713e04b",
    "pretrained/bevfusion-det.pth":
        "ee0a389213922343508db40dff0ec04767b36d8f512da34b093939c37aaf2122",
    "mmdet3d/models/heads/bbox/transfusion.py":
        "66883ac65d9cbe172f15e8c1aa8981587c5c2345740970eb154b1ee6d348f3c4",
    "tools/stage3d_b_native_assignment.py":
        "ebdd704d21b41ccbfbf3b1449b40787433f2f1a00984afc7672c070370f31da1",
    "mmdet3d/models/target_uq.py":
        "d0f41a87158ba26018c4e96c971b344764952a62328debe222c315f6f0a1edb0",
    "mmdet3d/models/crc_target_uq.py":
        "41f073d59fade01b622b0f1d38302440e572f2fce0ddb067ca810011036b81ee",
    "tools/transactions_query2u_density.py":
        "b6c57a105fc68fe5c1a4fd497cc7700886a22702d571c0c4e8cd763025b96537",
    "outputs/stage3/target_uq/checkpoints/stage3c/stage3c_structured_fusion_real_fusion_pilot_structured_feature_k1.pt":
        "68119e8507212e60b8f7b88d98267e82b893fac8d3142a7619f8ccf1684db34f",
    "outputs/stage3/target_uq/calibrators/stage3c/stage3c_structured_fusion_structured_feature_k1.json":
        "b9d6a0d3cc35fc7420343c95b1655ebcf2440c1ba0bd849fcc184d15f160b3ef",
    "outputs/transactions_extension/completion_run/query2u_density/fusion/checkpoint.pt":
        "8fa810e35d2b21b4c8b0ace724024a79e4cb146829ec1282815bc582546a10d7",
    "outputs/transactions_extension/completion_run/query2u_density/fusion/training.json":
        "209c05ac7cd568cb4a863778cfa72b93d8bb6c5c0a91a4df60f346c416767a4e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


def strict_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def synthetic_batch() -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    batch, queries, gt_count = 2, 12, 3
    feature = torch.randn(
        batch, queries, QUERY_FEATURE_DIM, generator=generator
    )
    logits = torch.randn(batch, queries, CLASS_COUNT, generator=generator)
    boxes = torch.randn(batch, queries, BOX_DIM, generator=generator)
    boxes[..., 3:6] = torch.rand(
        batch, queries, 3, generator=generator
    ) * 4.0 + 0.5
    boxes[..., 6] = torch.rand(
        batch, queries, generator=generator
    ) * (2.0 * torch.pi) - torch.pi
    seed_score = torch.rand(batch, queries, generator=generator)
    query_density = torch.randn(batch, queries, generator=generator)
    gt_boxes = boxes[:, :gt_count].clone()
    gt_boxes[..., :3] += 0.1
    gt_state = state_from_boxes(gt_boxes)
    gt_yaw = gt_boxes[..., 6] + 0.02
    gt_labels = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)
    matched = torch.full((batch, queries), -1, dtype=torch.long)
    matched[:, :gt_count] = torch.arange(gt_count)
    neighborhood = torch.zeros(batch, gt_count, queries, dtype=torch.bool)
    for index in range(gt_count):
        neighborhood[:, index, index] = True
        neighborhood[:, index, index + gt_count] = True
    return {
        "feature": feature,
        "logits": logits,
        "boxes": boxes,
        "seed_score": seed_score,
        "query_density": query_density,
        "gt_boxes": gt_boxes,
        "gt_state": gt_state,
        "gt_yaw": gt_yaw,
        "gt_labels": gt_labels,
        "matched": matched,
        "neighborhood": neighborhood,
        "proposal_ids": torch.arange(queries).expand(batch, -1).clone(),
        "final_detections": boxes[:, :5].clone(),
    }


def run_train_step() -> Dict[str, Any]:
    torch.manual_seed(SEED)
    fixture = synthetic_batch()
    inputs = [
        fixture["feature"],
        fixture["logits"],
        fixture["boxes"],
        fixture["seed_score"],
        fixture["query_density"],
    ]
    for value in inputs:
        value.requires_grad_(True)
    before = {name: tensor_sha256(value) for name, value in fixture.items()}
    model = ProbabilisticQueryHead()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    output = model(*inputs)
    losses = f1_loss(
        output,
        fixture["matched"],
        fixture["gt_state"],
        fixture["gt_yaw"],
        fixture["gt_labels"],
        torch.ones(CLASS_COUNT),
        family="student_t",
        neighborhood=fixture["neighborhood"],
    )
    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    gradients_finite = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    input_gradients_absent = all(value.grad is None for value in inputs)
    optimizer.step()
    after = {name: tensor_sha256(value) for name, value in fixture.items()}
    values_finite = all(
        torch.isfinite(value).all()
        for value in list(output.values()) + list(losses.values())
    )
    return {
        "detector_inputs_unchanged": before == after,
        "gradients_finite": gradients_finite,
        "input_gradients_absent": input_gradients_absent,
        "loss": float(losses["total"].detach()),
        "losses": {
            name: float(value.detach()) for name, value in losses.items()
        },
        "model_digest": state_dict_sha256(model),
        "output_digest": hashlib.sha256(
            "".join(
                tensor_sha256(output[name]) for name in sorted(output)
            ).encode()
        ).hexdigest(),
        "returned_logits_identical": torch.equal(
            output["class_logits"], fixture["logits"]
        ),
        "returned_means_identical": torch.equal(
            output["mean_boxes"], fixture["boxes"]
        ),
        "values_finite": values_finite,
    }


def check_authority_hashes() -> str:
    failures = []
    for relative, expected in HASH_AUTHORITIES.items():
        path = ROOT / relative
        actual = sha256(path)
        if actual != expected:
            failures.append(f"{relative}: {actual}")
    if failures:
        raise AssertionError("authority hash mismatch: " + "; ".join(failures))
    return f"{len(HASH_AUTHORITIES)} immutable identities matched"


def check_schema_and_parameter_budget() -> str:
    model = ProbabilisticQueryHead()
    count = sum(parameter.numel() for parameter in model.parameters())
    density_count = sum(
        parameter.numel()
        for parameter in ProbabilisticQueryHead(
            include_query_density=True
        ).parameters()
    )
    assert GEOMETRY_INPUT_DIM == 147
    assert DENSITY_GEOMETRY_INPUT_DIM == 148
    assert HIDDEN_DIM == 64
    assert count == 18443
    assert density_count == 18507
    assert count < 100000
    return (
        f"primary_input=147/parameters={count}; "
        f"density_ablation_input=148/parameters={density_count}"
    )


def check_input_isolation() -> str:
    result = run_train_step()
    assert result["detector_inputs_unchanged"]
    assert result["input_gradients_absent"]
    assert result["returned_logits_identical"]
    assert result["returned_means_identical"]
    return "all detector-derived tensors byte-identical and gradient-free"


def check_finite_backward() -> str:
    result = run_train_step()
    assert result["values_finite"]
    assert result["gradients_finite"]
    assert torch.isfinite(torch.tensor(result["loss"]))
    return f"finite forward/loss/backward; synthetic total={result['loss']:.6f}"


def check_domains_and_large_kappa() -> str:
    torch.manual_seed(SEED)
    fixture = synthetic_batch()
    output = ProbabilisticQueryHead()(
        fixture["feature"],
        fixture["logits"],
        fixture["boxes"],
        fixture["seed_score"],
        fixture["query_density"],
    )
    assert torch.all(output["scale"] >= MIN_SCALE)
    assert torch.all(output["degrees_of_freedom"] > MIN_DF)
    assert torch.all(output["yaw_kappa"] >= MIN_KAPPA)
    assert torch.all(output["yaw_kappa"] <= MAX_KAPPA)
    extreme = von_mises_log_prob(
        torch.tensor([0.0, torch.pi]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([MAX_KAPPA, MAX_KAPPA]),
    )
    assert torch.isfinite(extreme).all()
    return "scale/df/kappa domains valid; kappa=10000 log density finite"


def check_yaw_periodicity() -> str:
    epsilon = 1e-4
    left = torch.tensor([-torch.pi + epsilon])
    right = torch.tensor([torch.pi - epsilon])
    kappa = torch.tensor([20.0])
    boundary = von_mises_log_prob(left, right, kappa)
    opposite = von_mises_log_prob(torch.tensor([0.0]), right, kappa)
    shifted = von_mises_log_prob(left + 2.0 * torch.pi, right, kappa)
    assert torch.allclose(boundary, shifted, atol=1e-6, rtol=0)
    assert boundary.item() > opposite.item()
    assert abs(wrap_angle(left - right).item()) < 3e-4
    return "2pi invariance exact within 1e-6; boundary residual stays local"


def check_student_tail_fixture() -> str:
    target = torch.full((1, STATE_DIM), 8.0)
    mean = torch.zeros_like(target)
    scale = torch.ones_like(target)
    student_nll = -student_t_log_prob(
        target, mean, scale, torch.tensor([3.0])
    )
    gaussian_nll = -gaussian_log_prob(target, mean, scale)
    assert student_nll.item() < gaussian_nll.item()
    return (
        f"outlier NLL Student-t={student_nll.item():.6f} "
        f"< Gaussian={gaussian_nll.item():.6f}"
    )


def check_competition_fixture() -> str:
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    diffuse = competition_loss(torch.zeros(1, 1, 3), mask)
    concentrated = competition_loss(
        torch.tensor([[[0.0, -10.0, -10.0]]]), mask
    )
    shifted = competition_loss(
        torch.tensor([[[-50.0, -60.0, -60.0]]]), mask
    )
    assert concentrated.item() < diffuse.item()
    assert torch.allclose(concentrated, shifted, atol=1e-6, rtol=0)

    learnable = torch.tensor(
        [[[0.0, -0.1, -0.2]]], requires_grad=True
    )
    before = competition_loss(learnable, mask)
    gradient = torch.autograd.grad(before, learnable)[0]
    after = competition_loss(learnable - gradient, mask)
    assert after.item() < before.item()
    return (
        f"diffuse={diffuse.item():.6f}; concentrated={concentrated.item():.6f}; "
        "common-shift invariant; descent concentrates mass"
    )


def check_duplicate_neighborhood_fixture() -> str:
    logits = torch.full((1, 4, CLASS_COUNT), -10.0)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 1] = 10.0
    logits[0, 2, 2] = 10.0
    logits[0, 3, 1] = 10.0
    boxes = torch.zeros(1, 4, BOX_DIM)
    boxes[..., 3:6] = 1.0
    boxes[0, :, 0] = torch.tensor([0.1, 1.5, 0.2, 2.1])
    gt_boxes = torch.zeros(1, 1, BOX_DIM)
    gt_boxes[..., 3:6] = 1.0
    mask = duplicate_neighborhood(
        logits, boxes, torch.tensor([[1]]), gt_boxes
    )
    expected = torch.tensor([[[True, True, False, False]]])
    assert torch.equal(mask, expected)
    return "same-class queries at 0.1/1.5 m included; wrong class/2.1 m excluded"


def check_temperature_isolation() -> str:
    logits = torch.tensor([[[2.0, 1.0] + [0.0] * 8]])
    before = tensor_sha256(logits)
    global_value = calibrated_class_log_probs(logits, torch.tensor(2.0))
    per_class = calibrated_class_log_probs(
        logits, torch.linspace(0.5, 1.4, CLASS_COUNT)
    )
    assert tensor_sha256(logits) == before
    assert torch.isfinite(global_value).all() and torch.isfinite(per_class).all()
    return "global/per-class calibration finite; source logits unchanged"


def check_same_seed_determinism() -> str:
    first = run_train_step()
    second = run_train_step()
    assert first == second
    return (
        f"model={first['model_digest'][:16]}..., "
        f"output={first['output_digest'][:16]}..., losses identical"
    )


def check_no_detector_integration() -> str:
    source = (ROOT / "probset3d/f1/implementation/probabilistic_query.py").read_text()
    forbidden = (
        "mmdet3d.models",
        "return_loss=False",
        "load_checkpoint",
        "build_model",
        "nms",
    )
    found = [token for token in forbidden if token in source]
    assert not found
    return "standalone torch module; no detector build/forward/NMS import path"


def run_check(
    gate: str, description: str, function: Callable[[], str]
) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        evidence = function()
        return {
            "description": description,
            "evidence": evidence,
            "gate": gate,
            "passed": True,
            "seconds": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "description": description,
            "evidence": f"{type(error).__name__}: {error}",
            "gate": gate,
            "passed": False,
            "seconds": time.perf_counter() - started,
        }


def markdown_report(decision: Dict[str, Any]) -> str:
    rows = [
        "| Gate | Result | Evidence |",
        "| --- | --- | --- |",
    ]
    for item in decision["gates"]:
        evidence = item["evidence"].replace("|", "\\|")
        rows.append(
            f"| {item['gate']} | {'PASS' if item['passed'] else 'FAIL'} | "
            f"{evidence} |"
        )
    return "\n".join(
        [
            "# ProbSet3D F1 Implementation Acceptance",
            "",
            f"Decision: **{decision['endpoint']}**",
            "",
            "This is a synthetic implementation-validity decision, not a",
            "scientific F1 result. No real query tensor, nuScenes sample, pilot",
            "metric, degradation result, official val, internal test, or sealed",
            "partial artifact was read.",
            "",
            *rows,
            "",
            "## Research state after acceptance",
            "",
            "- `F0 SUPPORTED`",
            "- `F1 AUTHORIZED`",
            "- `PROBABILISTIC QUERY EFFECTIVENESS UNKNOWN`",
            "- `FULL PROBSET3D METHOD NOT YET SUPPORTED`",
            "",
            "A pass authorizes only the preregistered legality pilot. It does not",
            "authorize F2, K-best training, low-rank covariance, a residual",
            "Poisson branch, decoder unfreezing, a second host, or official val.",
            "",
        ]
    )


def main() -> int:
    checks: List[Dict[str, Any]] = []
    definitions = [
        ("A01_AUTHORITY_HASHES", "Immutable P0/F0/host/baseline identities",
         check_authority_hashes),
        ("A02_SCHEMA_AND_SIZE", "Frozen schema and minimal head size",
         check_schema_and_parameter_budget),
        ("A03_INPUT_ISOLATION", "No detector input mutation or gradient",
         check_input_isolation),
        ("A04_FINITE_BACKWARD", "Finite forward, objective, gradients, update",
         check_finite_backward),
        ("A05_PARAMETER_DOMAINS", "Positive scale, df>2.1, stable concentration",
         check_domains_and_large_kappa),
        ("A06_YAW_PERIODICITY", "Circular yaw boundary and 2pi invariance",
         check_yaw_periodicity),
        ("A07_HEAVY_TAIL_FIXTURE", "Artificial outlier has finite heavy-tail score",
         check_student_tail_fixture),
        ("A08_COMPETITION_FIXTURE", "Competition favors one duplicate winner",
         check_competition_fixture),
        ("A09_DUPLICATE_NEIGHBORHOOD", "Frozen artificial duplicate cluster",
         check_duplicate_neighborhood_fixture),
        ("A10_TEMPERATURE_ISOLATION", "Categorical calibration is side-channel-only",
         check_temperature_isolation),
        ("A11_SAME_SEED_DETERMINISM", "Repeated synthetic update is identical",
         check_same_seed_determinism),
        ("A12_NO_DETECTOR_INTEGRATION", "Implementation has no host write path",
         check_no_detector_integration),
    ]
    for gate, description, function in definitions:
        result = run_check(gate, description, function)
        checks.append(result)
        print(f"{gate}: {'PASS' if result['passed'] else 'FAIL'}")

    passed = all(item["passed"] for item in checks)
    endpoint = (
        "F1 IMPLEMENTATION ACCEPTED — GO TO LEGALITY-ONLY PILOT"
        if passed
        else "INVALID F1 IMPLEMENTATION — STOP BEFORE PILOT"
    )
    implementation_source = (
        ROOT / "probset3d/f1/implementation/probabilistic_query.py"
    )
    acceptance_source = Path(__file__).resolve()
    freeze = {
        "constants": {
            "box_dimension": BOX_DIM,
            "class_count": CLASS_COUNT,
            "competition_temperature": COMPETITION_TEMPERATURE,
            "competition_weight": 0.1,
            "duplicate_center_meters": DUPLICATE_CENTER_METERS,
            "density_ablation_geometry_input_dimension":
                DENSITY_GEOMETRY_INPUT_DIM,
            "hidden_dimension": HIDDEN_DIM,
            "max_kappa": MAX_KAPPA,
            "min_df": MIN_DF,
            "min_kappa": MIN_KAPPA,
            "min_scale": MIN_SCALE,
            "query_feature_dimension": QUERY_FEATURE_DIM,
            "primary_geometry_input_dimension": GEOMETRY_INPUT_DIM,
            "state_dimension": STATE_DIM,
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "forbidden_real_data_reads": 0,
        "head_parameter_count": sum(
            parameter.numel() for parameter in ProbabilisticQueryHead().parameters()
        ),
        "implementation": {
            "path": str(implementation_source.relative_to(ROOT)),
            "sha256": sha256(implementation_source),
        },
        "preregistration_and_authority": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in sorted((ROOT / "probset3d/f1").glob("preregistration/*"))
            + sorted((ROOT / "probset3d/f1").glob("authority/*"))
        },
        "schema_version": "probset3d_f1_implementation_freeze_v1",
        "synthetic_acceptance": {
            "path": str(acceptance_source.relative_to(ROOT)),
            "seed": SEED,
            "sha256": sha256(acceptance_source),
        },
    }
    freeze_path = ACCEPTANCE / "implementation_freeze.json"
    strict_json(freeze_path, freeze)
    decision = {
        "endpoint": endpoint,
        "f1_scientific_result_read": False,
        "f1_scientific_status": "PROBABILISTIC QUERY EFFECTIVENESS UNKNOWN",
        "full_probset3d_status": "FULL PROBSET3D METHOD NOT YET SUPPORTED",
        "gates": checks,
        "implementation_freeze": str(freeze_path.relative_to(ROOT)),
        "implementation_freeze_sha256": sha256(freeze_path),
        "passed": passed,
        "passed_gate_count": sum(item["passed"] for item in checks),
        "required_gate_count": len(checks),
        "schema_version": "probset3d_f1_acceptance_decision_v1",
        "scope": {
            "bevfusion_modified": False,
            "checkpoint_deserialized": False,
            "degradation_data_read": False,
            "detector_forward_run": False,
            "official_val_read": False,
            "real_f1_results_read": False,
            "synthetic_only": True,
        },
    }
    report_path = ACCEPTANCE / "acceptance_report.md"
    report_path.write_text(markdown_report(decision))
    decision["acceptance_report"] = str(report_path.relative_to(ROOT))
    decision["acceptance_report_sha256"] = sha256(report_path)
    strict_json(ACCEPTANCE / "acceptance_decision.json", decision)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
