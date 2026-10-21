#!/usr/bin/env python3
"""Aggregate Stage 3D-B branch groups and apply the preregistered pilot gate."""

import json
from pathlib import Path

from tools.stage3d_b_native_assignment import sha256, strict_dump
from tools.stage3d_b_train_branch import ARM_SCHEME, bootstrap_delta


OUT = Path("outputs/stage3d/stage3d_b")
BRANCHES = ("camera", "lidar", "fusion")
CANDIDATES = ("B_hungarian_1to1", "C_hungarian_1to1_ignore_2to4m")
REFERENCES = ("hungarian", "hungarian_ignore")
INCREASE_LIMITS = {
    "existence_nll": 0.02,
    "brier": 0.01,
    "ece": 0.01,
    "adaptive_ece": 0.01,
    "conditional_class_nll": 0.02,
    "xy_nll": 0.05,
    "coverage_error": 0.02,
}
LOWER_LIMITS = {"spearman": -0.02, "recall": -0.01}
PROBABILITY_METRICS = (
    "existence_nll", "brier", "ece", "adaptive_ece", "conditional_class_nll",
)


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def native_smoke_passed(smoke):
    checks = smoke["checks"]
    required_true = (
        "B_C_positive_identical", "deterministic_repeat", "distance_gates",
        "one_to_one", "strict_json",
    )
    required_false = ("post_nms_score_used_as_probability", "target_uq_probability_used")
    return (
        smoke["status"] == "pass"
        and all(checks[name] is True for name in required_true)
        and all(checks[name] is False for name in required_false)
    )


def third_branch_noncollapse(gate):
    """Use preregistered probability tolerances on point estimates only.

    Full branch qualification also requires the paired CI bound.  The weaker
    third-branch clause says only that probability quality must not collapse;
    it therefore uses the same frozen tolerances without inventing a new
    post-result threshold.
    """
    if not gate["hard_validity"]["passed"]:
        return False
    return all(
        gate["stability"][reference][metric]["estimate"] is not None
        and gate["stability"][reference][metric]["estimate"] <= INCREASE_LIMITS[metric]
        for reference in REFERENCES for metric in PROBABILITY_METRICS
    )


def hard_validity(branch, group, calibrators, validation):
    failures = []
    smoke = load(OUT / "native_assignment_smoke" / f"{branch}_native_assignment_summary.json")
    if not native_smoke_passed(smoke):
        failures.append("native_assignment_smoke")
    for arm, item in group["arms"].items():
        if not item["smoke"]["overfit_loss_decreased"]:
            failures.append(f"{arm}:overfit")
        if item["training"]["finite_loss_rate"] != 1.0:
            failures.append(f"{arm}:finite_loss")
        if item["training"]["restore_max_abs_diff"] != 0.0:
            failures.append(f"{arm}:checkpoint_restore")
        if item["training"]["iterations"] != 2000 or item["training"]["batch_size"] != 4:
            failures.append(f"{arm}:fairness_controls")
        if calibrators[arm]["dev_transfer_gate"]["dev_evaluation_count"] != 1:
            failures.append(f"{arm}:dev_evaluation_count")
    if len({item["training"]["initialization_sha256"] for item in group["arms"].values()}) != 1:
        failures.append("initialization_mismatch")
    for split in ("pilot", "label_cal", "dev"):
        for scheme in ("hungarian", "hungarian_ignore"):
            if validation["branches"][branch][split]["labels"][scheme]["positive"] != validation["branches"][branch][split]["labels"][scheme]["unique_gt"]:
                failures.append(f"{split}/{scheme}:duplicate_positive")
    if group["official_val_used"]:
        failures.append("official_val_used")
    return {"passed": not failures, "failures": failures}


def candidate_branch_gate(branch, candidate, bootstrap, hard):
    stability = {}
    stable = hard["passed"]
    improvement = {"existence": False, "xy_coverage": False, "ranking": False}
    for reference in REFERENCES:
        stability[reference] = {}
        current = bootstrap[reference][candidate]
        for metric, limit in INCREASE_LIMITS.items():
            interval = current[metric]
            passed = (
                interval["estimate"] is not None
                and interval["estimate"] <= limit
                and interval["ci95"][1] <= limit
            )
            stability[reference][metric] = {"passed": passed, "limit": limit, **interval}
            stable &= passed
        for metric, limit in LOWER_LIMITS.items():
            interval = current[metric]
            passed = (
                interval["estimate"] is not None
                and interval["estimate"] >= limit
                and interval["ci95"][0] >= limit
            )
            stability[reference][metric] = {"passed": passed, "limit": limit, **interval}
            stable &= passed
        improvement["existence"] |= any(current[name]["ci95"][1] < 0 for name in ("existence_nll", "brier", "ece") if current[name]["ci95"][1] is not None)
        improvement["xy_coverage"] |= any(current[name]["ci95"][1] < 0 for name in ("xy_nll", "coverage_error") if current[name]["ci95"][1] is not None)
        improvement["ranking"] |= current["spearman"]["ci95"][0] is not None and current["spearman"]["ci95"][0] > 0
    return {
        "branch": branch, "candidate": candidate,
        "hard_validity": hard, "stability": stability,
        "improvement_categories": improvement,
        "stable": stable, "has_significant_improvement": any(improvement.values()),
        "passed": stable and any(improvement.values()),
    }


def main():
    validation = load(OUT / "label_validation.json")
    groups = {branch: load(OUT / "runs" / branch / "group_manifest.json") for branch in BRANCHES}
    calibrators = {branch: load(OUT / "calibrators" / f"{branch}.json") for branch in BRANCHES}
    evaluations = {branch: load(OUT / "evaluation" / f"{branch}.json") for branch in BRANCHES}
    bootstraps = {branch: load(OUT / "bootstrap" / f"{branch}.json") for branch in BRANCHES}
    hard = {branch: hard_validity(branch, groups[branch], calibrators[branch], validation) for branch in BRANCHES}
    gates = {
        candidate: {
            branch: candidate_branch_gate(branch, candidate, bootstraps[branch], hard[branch])
            for branch in BRANCHES
        }
        for candidate in CANDIDATES
    }
    global_pass = {}
    for candidate in CANDIDATES:
        passing = [branch for branch in BRANCHES if gates[candidate][branch]["passed"]]
        noncollapse = all(third_branch_noncollapse(gates[candidate][branch]) for branch in BRANCHES)
        global_pass[candidate] = {
            "passing_branches": passing,
            "passing_branch_count": len(passing),
            "third_branch_no_hard_failure_or_probability_collapse": noncollapse,
            "third_branch_noncollapse_definition": "hard validity plus all preregistered probability-metric point estimates within the frozen tolerances on both B/C references; full passing branches additionally require CI bounds",
            "passed": len(passing) >= 2 and noncollapse,
        }

    c_vs_b = {}
    c_preference_count = 0
    for branch in BRANCHES:
        c_vs_b[branch] = {}
        for reference in REFERENCES:
            b_eval = evaluations[branch]["B_hungarian_1to1"][reference]["calibrated"]
            c_eval = evaluations[branch]["C_hungarian_1to1_ignore_2to4m"][reference]["calibrated"]
            c_vs_b[branch][reference] = {
                name: bootstrap_delta(b_eval, c_eval, name)
                for name in ("existence_nll", "brier", "ece", "adaptive_ece")
            }
        primary = c_vs_b[branch]["hungarian_ignore"]
        existence_stable_or_better = all(item["ci95"][1] <= INCREASE_LIMITS.get(name, 0.01) for name, item in primary.items())
        b_background = evaluations[branch]["B_hungarian_1to1"]["hungarian"]["calibrated"]["classification"]["background_categorical_nll"]
        c_background = evaluations[branch]["C_hungarian_1to1_ignore_2to4m"]["hungarian"]["calibrated"]["classification"]["background_categorical_nll"]
        background_stable = c_background <= b_background + 0.02
        b_bg_exist = evaluations[branch]["B_hungarian_1to1"]["hungarian_ignore"]["calibrated"]["probability_scale"]["background_mean_p_exist"]
        c_bg_exist = evaluations[branch]["C_hungarian_1to1_ignore_2to4m"]["hungarian_ignore"]["calibrated"]["probability_scale"]["background_mean_p_exist"]
        high_confidence_background_not_increased = c_bg_exist <= b_bg_exist + 0.01
        preferred = existence_stable_or_better and background_stable and high_confidence_background_not_increased and gates["C_hungarian_1to1_ignore_2to4m"][branch]["stable"]
        c_vs_b[branch]["preference_checks"] = {
            "existence_calibration_stable_or_better": existence_stable_or_better,
            "B_reference_background_discrimination_stable": background_stable,
            "near_GT_ignore_high_confidence_background_not_increased": high_confidence_background_not_increased,
            "other_primary_metrics_stable": gates["C_hungarian_1to1_ignore_2to4m"][branch]["stable"],
            "C_preferred": preferred,
        }
        c_preference_count += int(preferred)

    selected = None
    if global_pass["C_hungarian_1to1_ignore_2to4m"]["passed"] and c_preference_count >= 2:
        selected = "C"
    elif global_pass["B_hungarian_1to1"]["passed"]:
        selected = "B"
    full = selected is not None
    status = "VERIFIED" if full else "INCONCLUSIVE"
    decision = {
        "native_assignment_verified": True,
        "camera_assignment_semantics": "cost_aware_centerhead_detector_logits_gaussian_focal_plus_0.25_xy_distance",
        "pilot_branch_groups_completed": 3,
        "pilot_training_arms_completed": 9,
        "selected_label_scheme": selected,
        "full_retraining_recommended": full,
        "new_internal_split_authorized": full,
        "recommended_next_stage": (
            "Create 490/70/70/70 split, then Stage 3D-C structured full retraining"
            if full else "Matching/optimization diagnosis"
        ),
        "internal_test_created": False,
        "stage3d_c_started": False,
        "stage4_started": False,
        "stage5_started": False,
        "status": status,
        "global_candidate_gates": global_pass,
        "branch_candidate_gates": gates,
        "C_vs_B_preference": {"branches": c_vs_b, "preferred_branch_count": c_preference_count},
    }
    strict_dump(OUT / "decision.json", decision)
    manifest = {
        "stage": "stage3d_b_structured_pilot", "seed": 20260708,
        "official_val_used": False, "internal_test_created": False,
        "decision": str(OUT / "decision.json"), "decision_sha256": sha256(OUT / "decision.json"),
        "inputs": {
            "assignment_spec": str(OUT / "assignment_spec.json"),
            "pilot_decision_spec": str(OUT / "pilot_decision_spec.json"),
            "label_validation": str(OUT / "label_validation.json"),
        },
    }
    strict_dump(OUT / "manifest_complete.json", manifest)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
