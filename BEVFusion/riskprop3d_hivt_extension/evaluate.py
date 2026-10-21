"""Paired XY law propagation through the frozen native Laplace-mixture host."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from hivt_runtime import ContextDataset, collate, forward, perturb
from run_host import HERE, OUT as HOST, SEEDS, gpu, load_model, sha, write
sys.path.append(str(HERE.parent))
from probset3d.f1.implementation.probabilistic_query import ProbabilisticQueryHead
import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import chi2

OUT = HERE / "evaluation_v1"
CONDITIONS = ("clean_s0", "dropout_s3", "rotation_s3", "official_clean")
LAWS = ("A0", "AG", "AH", "AM")
HEAD_SEEDS = (20260731, 20260801, 20260802)


def freeze():
    engineering = json.loads((HOST / "engineering.json").read_text())
    timing = engineering["timing"][-1]
    inputs = sum(json.loads((HERE / "data_v1" / c / "manifest.json").read_text())["targets"] for c in CONDITIONS)
    estimate = inputs * 3 * 3 * 128 * timing["seconds_per_forward"] / timing["graphs"]
    # The protocol permits a prospective hash subset only when measured cost demands it.
    population = "full retained vehicle targets"
    rows = {}
    for condition in CONDITIONS:
        dataset = ContextDataset(condition)
        rows[condition] = list(range(len(dataset)))
    sources = {"evaluate.py": sha(__file__), "hivt_runtime.py": sha(HERE / "hivt_runtime.py"),
               "host_v1/freeze.json": sha(HOST / "freeze.json"), "evaluate_controls.py": sha(HERE / "evaluate_controls.py"),
               "summarize.py": sha(HERE / "summarize.py"), "data_v1/audit.json": sha(HERE / "data_v1/audit.json"),
               "../tools/riskprop3d_r0.py": sha(HERE.parent / "tools/riskprop3d_r0.py"),
               "../probset3d/f1/implementation/probabilistic_query.py": sha(HERE.parent / "probset3d/f1/implementation/probabilistic_query.py"),
               "../outputs/riskprop3d_r0/predictor.pt": sha(HERE.parent / "outputs/riskprop3d_r0/predictor.pt")}
    for condition in CONDITIONS:
        path = HERE / "data_v1" / condition / "manifest.json"
        sources[str(path.relative_to(HERE))] = sha(path)
    for i, seed in enumerate(HEAD_SEEDS):
        for path in (HOST / ("run_" + str(i)) / "best.pt", HOST / ("run_" + str(i)) / "calibration.json",
                     HOST / ("run_" + str(i)) / "support_probe.json",
                     HERE / "recovery_v1/checkpoints" / ("seed_{}.pt".format(seed))):
            sources[str(path.relative_to(HERE))] = sha(path)
    write(OUT / "freeze.json", {"status": "FROZEN_BEFORE_LAW_RESULTS", "population": population,
        "rows": rows, "conditions": CONDITIONS, "laws": LAWS, "draws_per_independent_set": 64,
        "sets": 2, "row_batch": 4, "draw_chunk": 32, "state_seeds": [202610140, 202610141, 202610142],
        "future_seeds": [202610150, 202610151, 202610152], "host_head_pairing": list(zip(SEEDS, HEAD_SEEDS)),
        "rng": "Philox seed from SHA256(base_seed,condition,sample/instance,set,stream); independent fields",
        "common_random_numbers": "same linear normal, mode uniform, Laplace uniform across compatible laws",
        "student_xy": "single shared chi-square radial factor per XY draw; no truncation/redraw",
        "primary_population": "official_clean", "primary_contrast": "AG-AH", "mechanistic_contrast": "AM-AH",
        "multiplicity": "one primary official AG-AH; AM-AH explanatory; pointwise intervals, no simultaneous claim across populations/endpoints",
        "bootstrap_seed": 20260805, "bootstrap_repeats": 10000, "unit": "paired scene, then average three paired runs",
        "risk_label": "same-run A0 predictive mixture-mean FDE > 2m", "risk_rank": "final log moment ellipse area",
        "risk_coverage": "fractions 0.10 through 1.00 step 0.01", "safe_fraction": .8,
        "source_sha256": sources, "prospective_full_forward_cost_seconds_sum": estimate,
        "cost_is_extrapolation": True, "failure_policy": "affected comparison invalid; preserve row and evidence",
        "MC_sensitivity": "256 per set on first 64 SHA256-ranked clean-dev row identities, same random prefixes"})


def check_freeze():
    config = json.loads((OUT / "freeze.json").read_text())
    for name, digest in config["source_sha256"].items():
        assert sha(HERE / name) == digest, name
    return config


def support_probe(replicate, physical):
    from run_host import hashed_rows
    device = gpu(SEEDS[replicate], physical)
    model = load_model(replicate, device)
    dataset = ContextDataset("clean_s0")
    rows = hashed_rows(dataset, 8)
    heads = parameters(dataset, replicate, device)
    identities = [str(dataset.arrays["sample_token"][r]) + "/" + str(dataset.arrays["instance_token"][r]) for r in rows]
    records = []
    hooks = []
    extrema = {}

    def inspect(name):
        def hook(_module, _inputs, value):
            assert torch.isfinite(value).all(), name
            extrema[name] = max(extrema.get(name, 0.), float(value.abs().max()))
        return hook

    for name in ("local_encoder", "global_interactor"):
        hooks.append(getattr(model, name).register_forward_hook(inspect(name)))
    try:
        context = collate([dataset[int(r)] for r in rows]).to(device)
        no_map, no_neighbors = context.clone(), context.clone()
        no_map.lane_actor_index = no_map.lane_actor_index[:, :0]
        no_map.lane_actor_vectors = no_map.lane_actor_vectors[:0]
        no_neighbors.edge_index = no_neighbors.edge_index[:, :0]
        with torch.no_grad():
            normal = forward(model, context)[0]
            map_difference = float((forward(model, no_map)[0] - normal).abs().max())
            neighbor_difference = float((forward(model, no_neighbors)[0] - normal).abs().max())
        assert map_difference > 0 and neighbor_difference > 0
        for law in ("AG", "AH", "AM"):
            maximum_state = 0.
            for draw_set in (0, 1):
                random = randoms(replicate, "clean_s0", identities, draw_set, 64)
                current = states(heads, law, rows, random)
                maximum_state = max(maximum_state, float(np.abs(current).max()))
                for begin in range(0, len(rows), 4):
                    selected = rows[begin:begin + 4]
                    graphs = [dataset[int(r)] for r in selected]
                    repeated = collate([g for g in graphs for _ in range(32)]).to(device)
                    for draw in (0, 32):
                        changed = perturb(repeated, torch.tensor(current[begin:begin + 4, draw:draw + 32].reshape(-1, 2), dtype=torch.float32, device=device))
                        with torch.no_grad():
                            output = forward(model, changed)
                        assert all(torch.isfinite(v).all() for v in output)
            records.append({"law": law, "state_max_absolute_xy": maximum_state, "targets": len(rows), "draws_per_set": 64})
    finally:
        for hook in hooks:
            hook.remove()
    write(HOST / ("run_" + str(replicate)) / "support_probe.json", {"status": "PASS", "rows": rows.tolist(),
        "laws": records, "key_activation_max_absolute": extrema, "future_labels_in_features": False,
        "trained_host_map_ablation_max_mean_difference": map_difference,
        "trained_host_neighbor_ablation_max_mean_difference": neighbor_difference,
        "scope": "fixed hash dev probe; formal outputs still checked independently", "physical_gpu": physical})


def row_rng(seed, condition, identity, draw_set, stream):
    key = "{}|{}|{}|{}|{}".format(seed, condition, identity, draw_set, stream)
    value = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], "little")
    return np.random.Generator(np.random.Philox(value))


def randoms(replicate, condition, identities, draw_set, count):
    state_seed, future_seed = 202610140 + replicate, 202610150 + replicate
    fields = {"z": [], "radial": [], "mode": [], "laplace": []}
    for identity in identities:
        fields["z"].append(row_rng(state_seed, condition, identity, draw_set, "linear_normal").standard_normal((count, 8)))
        fields["radial"].append(row_rng(state_seed, condition, identity, draw_set, "radial").random(count))
        fields["mode"].append(row_rng(future_seed, condition, identity, draw_set, "mode").random(count))
        u = row_rng(future_seed, condition, identity, draw_set, "laplace").random((count, 6, 2)) - .5
        fields["laplace"].append(-np.sign(u) * np.log1p(-2 * np.abs(u)))
    return {name: np.asarray(value) for name, value in fields.items()}


def parameters(dataset, replicate, device):
    path = HERE / "recovery_v1/checkpoints" / ("seed_{}.pt".format(HEAD_SEEDS[replicate]))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    a = dataset.arrays
    result = {}
    for name, model_name in (("AG", "M1"), ("AH", "M2")):
        head = ProbabilisticQueryHead().to(device).eval()
        head.load_state_dict(payload["models"][model_name])
        values = {k: [] for k in ("mean_state", "scale", "degrees_of_freedom", "yaw_mean", "yaw_scale", "yaw_kappa")}
        with torch.no_grad():
            for begin in range(0, len(dataset), 4096):
                stop = min(begin + 4096, len(dataset))
                inputs = [torch.as_tensor(a[k][begin:stop])[:, None].to(device) for k in ("feature", "logits", "box", "seed")]
                output = head(*inputs, torch.zeros(stop - begin, 1, device=device))
                for key in values:
                    values[key].append(output[key][:, 0].cpu().numpy())
        result[name] = {key: np.concatenate(value) for key, value in values.items()}
        np.testing.assert_array_equal(result[name]["mean_state"][:, :2], a["target_xy"][:, -1])
    df = result["AH"]["degrees_of_freedom"].reshape(-1).astype(np.float64)
    assert np.all(df > 2) and all(np.isfinite(v).all() for value in result.values() for v in value.values())
    htc_scale = result["AH"]["scale"].astype(np.float64)
    mmg = htc_scale * np.sqrt(df / (df - 2))[:, None]
    assert np.isfinite(mmg).all()
    np.testing.assert_allclose(mmg ** 2, htc_scale ** 2 * (df / (df - 2))[:, None], rtol=1e-12, atol=0)
    result["AM"] = dict(result["AH"], scale=mmg)
    return result


def states(parameters, law, rows, random):
    p = parameters[law]
    mean = p["mean_state"][rows, :2].astype(np.float64)
    scale = p["scale"][rows, :2].astype(np.float64)
    z = random["z"][..., :2]
    if law == "AH":
        df = p["degrees_of_freedom"][rows].reshape(-1).astype(np.float64)
        radial = np.sqrt(df[:, None] / chi2.ppf(random["radial"], df[:, None]))
        z = z * radial[..., None]
    value = mean[:, None] + scale[:, None] * z
    if not np.isfinite(value).all():
        raise FloatingPointError("Non-finite current XY draw")
    return value


def samples_moments(output, factor, mode_uniform, laplace, future):
    means, scales, probability, inverse, _ = [v.detach().cpu().numpy().astype(np.float64) for v in output]
    scales *= factor
    cdf = probability.cumsum(axis=1)
    cdf[:, -1] = 1.
    mode = (mode_uniform[:, None] >= cdf).sum(axis=1)
    row = np.arange(len(means))
    residual = scales[row, mode] * laplace
    sample = means[row, mode] + np.einsum("bti,bij->btj", residual, inverse)
    mean = np.einsum("bm,bmti->bti", probability, means)
    covariance = np.einsum("bdi,bmtd,bdj->bmtij", inverse, 2 * scales ** 2, inverse)
    second = np.einsum("bm,bmtij->btij", probability, covariance + means[..., :, None] * means[..., None, :])
    local_residual = np.einsum("bmti,bji->bmtj", future[:, None] - means, inverse)
    logpdf = (-np.log(2 * scales) - np.abs(local_residual) / scales).sum(axis=(-1, -2))
    logpdf = logsumexp(np.log(probability) + logpdf, axis=1)
    mode_error = np.linalg.norm(means - future[:, None], axis=-1)
    return sample, mean, second, logpdf, mode_error.mean(-1).min(1), mode_error[:, :, -1].min(1)


def metrics(samples_a, samples_b, means, seconds, logpdfs, future, min_ade, min_fde):
    n = len(future)
    energy = np.linalg.norm((samples_a - future[:, None]).reshape(n, len(samples_a[0]), 12), axis=-1).mean(1)
    energy -= .5 * np.linalg.norm((samples_a - samples_b).reshape(n, len(samples_a[0]), 12), axis=-1).mean(1)
    mean = means.mean(1)
    covariance = seconds.mean(1) - mean[..., :, None] * mean[..., None, :]
    covariance = (covariance + covariance.swapaxes(-1, -2)) / 2
    determinant = np.linalg.det(covariance)
    if not np.all(determinant > 0):
        raise FloatingPointError("Non-positive propagated moment covariance")
    inverse = np.linalg.inv(covariance)
    residual = future - mean
    maha = np.einsum("nti,ntij,ntj->nt", residual, inverse, residual)
    error = np.linalg.norm(residual, axis=-1)
    area = np.pi * chi2.ppf(.9, 2) * np.sqrt(determinant[:, -1])
    result = dict(energy=energy, mean_ade=error.mean(1), mean_fde=error[:, -1],
                  nll=-logsumexp(logpdfs, axis=1) + np.log(logpdfs.shape[1]),
                  min_ade=min_ade.min(1), min_fde=min_fde.min(1),
                  final_maha2=maha[:, -1], final_det=determinant[:, -1],
                  area90=area, log_area90=np.log(area))
    for level in (.5, .8, .9, .95):
        result["coverage_" + str(int(level * 100))] = (maha <= chi2.ppf(level, 2)).mean(1)
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError("Non-finite trajectory metric")
    return result


def evaluate_batch(dataset, rows, model, heads, replicate, condition, factor, count, device, draw_chunk):
    a = dataset.arrays
    identities = [str(a["sample_token"][r]) + "/" + str(a["instance_token"][r]) for r in rows]
    random = [randoms(replicate, condition, identities, draw_set, count) for draw_set in (0, 1)]
    graphs = [dataset[int(r)] for r in rows]
    future = dataset.future[rows]
    base = collate(graphs).to(device)
    with torch.no_grad():
        fixed = forward(model, base)
    a0_samples = []
    m = s = lp = amin = fmin = None
    for draw_set in (0, 1):
        repeated_output = tuple(v.repeat_interleave(count, dim=0) for v in fixed)
        value = samples_moments(repeated_output, factor, random[draw_set]["mode"].reshape(-1),
            random[draw_set]["laplace"].reshape(-1, 6, 2), np.repeat(future, count, axis=0))
        sample, m, s, lp, amin, fmin = [v.reshape(len(rows), count, *v.shape[1:]) for v in value]
        a0_samples.append(sample)
        m, s, lp, amin, fmin = [v[:, 0] for v in (m, s, lp, amin, fmin)]
    result = {"A0": metrics(*a0_samples, m[:, None], s[:, None], lp[:, None], future, amin[:, None], fmin[:, None])}
    repeated = collate([g for g in graphs for _ in range(draw_chunk)]).to(device)
    for law in ("AG", "AH", "AM"):
        all_samples, all_means, all_seconds, all_logpdf, all_minade, all_minfde = [], [], [], [], [], []
        for draw_set in (0, 1):
            state = states(heads, law, rows, random[draw_set])
            samples, means, seconds, logpdf, minade, minfde = [], [], [], [], [], []
            for begin in range(0, count, draw_chunk):
                stop = begin + draw_chunk
                current = torch.as_tensor(state[:, begin:stop].reshape(-1, 2), dtype=torch.float32, device=device)
                changed = perturb(repeated, current)
                with torch.no_grad():
                    output = forward(model, changed)
                repeat_future = np.repeat(future, draw_chunk, axis=0)
                value = samples_moments(output, factor, random[draw_set]["mode"][:, begin:stop].reshape(-1),
                    random[draw_set]["laplace"][:, begin:stop].reshape(-1, 6, 2), repeat_future)
                containers = (samples, means, seconds, logpdf, minade, minfde)
                for container, item in zip(containers, value):
                    container.append(item.reshape(len(rows), draw_chunk, *item.shape[1:]))
            collections = [np.concatenate(v, axis=1) for v in (samples, means, seconds, logpdf, minade, minfde)]
            for container, item in zip((all_samples, all_means, all_seconds, all_logpdf, all_minade, all_minfde), collections):
                container.append(item)
        result[law] = metrics(all_samples[0], all_samples[1], np.concatenate(all_means, 1), np.concatenate(all_seconds, 1),
                              np.concatenate(all_logpdf, 1), future, np.concatenate(all_minade, 1), np.concatenate(all_minfde, 1))
    return result


def evaluate(replicate, physical, sensitivity=False):
    config = check_freeze()
    device = gpu(SEEDS[replicate], physical)
    model = load_model(replicate, device)
    factor = json.loads((HOST / ("run_" + str(replicate)) / "calibration.json").read_text())["scale_factor"]
    for condition in (CONDITIONS[:1] if sensitivity else CONDITIONS):
        directory = OUT / ("sensitivity" if sensitivity else "main") / ("run_" + str(replicate)) / condition
        directory.mkdir(parents=True, exist_ok=False)
        dataset = ContextDataset(condition, labels=True)
        heads = parameters(dataset, replicate, device)
        rows = np.asarray(config["rows"][condition])
        if sensitivity:
            from run_host import hashed_rows
            rows = hashed_rows(dataset, 64)
        count = 256 if sensitivity else config["draws_per_independent_set"]
        assert count % config["draw_chunk"] == 0
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        records = []
        first_batch_seconds = None
        for begin in range(0, len(rows), config["row_batch"]):
            selected = rows[begin:begin + config["row_batch"]]
            batch_started = time.perf_counter()
            try:
                values = evaluate_batch(dataset, selected, model, heads, replicate, condition, factor, count,
                                        device, config["draw_chunk"])
            except (FloatingPointError, RuntimeError) as error:
                write(directory / ("failure_{:05d}.json".format(begin)), {"rows": selected.tolist(), "message": str(error),
                    "type": type(error).__name__, "affected_comparison": "all stochastic comparisons in this batch",
                    "invalid_not_zero_effect": True})
                raise
            if begin == 0:
                first_batch_seconds = time.perf_counter() - batch_started
            path = directory / ("part-{:05d}.npz".format(len(records)))
            with path.open("xb") as stream:
                np.savez(stream, row=selected, **{law + "_" + key: value for law, group in values.items() for key, value in group.items()})
            records.append({"file": path.name, "sha256": sha(path), "rows": len(selected)})
            if len(records) % 25 == 0 or begin == 0:
                print(json.dumps({"replicate": replicate, "condition": condition, "rows": begin + len(selected),
                                  "total": len(rows), "seconds": time.time() - started}), flush=True)
        write(directory / "complete.json", {"replicate": replicate, "condition": condition, "rows": len(rows),
            "parts": records, "seconds": time.time() - started, "physical_gpu": physical,
            "peak_allocation_bytes": torch.cuda.max_memory_allocated(), "draws_per_set": count,
            "gpu_name": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "row_batch": config["row_batch"], "draw_chunk": config["draw_chunk"],
            "native_forward_calls": len(records) * (1 + 3 * 2 * (count // config["draw_chunk"])),
            "mean_state_conditional_target_inputs": len(rows),
            "stochastic_conditional_target_inputs": len(rows) * 3 * 2 * count,
            "first_batch_all_four_laws_compute_seconds": first_batch_seconds,
            "first_batch_target_rows": min(config["row_batch"], len(rows)),
            "cost_scope": "conditional host inference, graph batching, sampling, metrics and metric I/O; excludes detector/head inference and data/model loading",
            "freeze_sha256": sha(OUT / "freeze.json"), "head_seed": HEAD_SEEDS[replicate], "host_seed": SEEDS[replicate]})


def self_check():
    r64 = randoms(0, "clean_s0", ["token/instance"], 0, 64)
    r256 = randoms(0, "clean_s0", ["token/instance"], 0, 256)
    for key in r64:
        np.testing.assert_array_equal(r64[key], r256[key][:, :64])
    assert not np.array_equal(r64["z"], randoms(0, "clean_s0", ["token/instance"], 1, 64)["z"])
    means = torch.zeros(1, 2, 6, 2)
    means[:, 1, :, 0] = 2
    b = torch.ones_like(means)
    p = torch.tensor([[.5, .5]])
    inverse = torch.tensor([[[0., 1.], [-1., 0.]]])
    output = (means, b, p, inverse, means)
    sample, mean, second, pdf, _, _ = samples_moments(output, 1, np.array([.2]), np.ones((1, 6, 2)), np.zeros((1, 6, 2)))
    np.testing.assert_allclose(sample[0, 0], [-1, 1])
    np.testing.assert_allclose(mean[0, 0], [1, 0])
    cov = second - mean[..., :, None] * mean[..., None, :]
    np.testing.assert_allclose(cov[0, 0], [[3, 0], [0, 2]])
    assert np.isfinite(pdf).all()
    c = np.sqrt(.5)
    inverse = torch.tensor([[[c, c], [-c, c]]], dtype=torch.float64)
    b = torch.ones(1, 1, 6, 2, dtype=torch.float64)
    b[..., 0] = 2
    zero = torch.zeros_like(b)
    rotated = samples_moments((zero, b, torch.ones(1, 1), inverse, zero), 1,
                              np.array([.1]), np.zeros((1, 6, 2)), np.zeros((1, 6, 2)))
    np.testing.assert_allclose(rotated[2][0, 0], [[5, 3], [3, 5]])
    repeated = tuple(v.repeat_interleave(64, dim=0) for v in output)
    vectorized = samples_moments(repeated, 1, r64["mode"].reshape(-1),
                                 r64["laplace"].reshape(-1, 6, 2), np.zeros((64, 6, 2)))
    scalar = [samples_moments(output, 1, r64["mode"][:, i], r64["laplace"][:, i], np.zeros((1, 6, 2)))[0]
              for i in range(64)]
    np.testing.assert_array_equal(vectorized[0].reshape(1, 64, 6, 2), np.stack(scalar, axis=1))
    print("PASS: independent streams, batch/count invariant RNG, rotation and total mixture covariance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "probe", "evaluate", "sensitivity", "self-check"))
    parser.add_argument("--replicate", type=int, choices=range(3))
    parser.add_argument("--gpu", type=int, choices=(0, 2, 3))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "self-check":
        self_check()
    elif args.command == "probe":
        support_probe(args.replicate, args.gpu)
    else:
        evaluate(args.replicate, args.gpu, args.command == "sensitivity")
