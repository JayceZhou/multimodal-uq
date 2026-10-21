"""Re-score the existing CV/MLP with recovered full-state heads on identical vehicles."""
import argparse
import json
from pathlib import Path
import sys
import time

from evaluate import CONDITIONS, HERE, HEAD_SEEDS, OUT as EVALUATION, check_freeze, parameters, randoms, row_rng
from hivt_runtime import ContextDataset
from run_host import SEEDS, gpu, sha, write
sys.path.insert(0, str(HERE.parent / "tools"))
import riskprop3d_r0 as r0
import numpy as np
import torch
from scipy.special import i0e, i1e
from scipy.stats import chi2

OUT = HERE / "controls_v1"


def evaluate(replicate, physical):
    config = check_freeze()
    device = gpu(SEEDS[replicate], physical)
    predictor, calibration = r0.load_predictor(device)
    started = time.time()
    for condition in CONDITIONS:
        dataset = ContextDataset(condition)
        a = dataset.arrays
        manifest = json.loads((HERE / "data_v1" / condition / "manifest.json").read_text())
        with np.load(HERE.parent / manifest["source_tracks"], allow_pickle=False) as source:
            original = a["original_row"]
            inputs = source["input"][original]
            future, ego = source["future"][original], source["ego_future"][original]
        head = parameters(dataset, replicate, device)
        htc = head["AH"]
        kappa = htc["yaw_kappa"].astype(np.float64)
        resultant = i1e(kappa) / i0e(kappa)
        mmg_yaw_sigma = np.sqrt(-2 * np.log(resultant))
        assert np.isfinite(mmg_yaw_sigma).all()
        rows = np.asarray(config["rows"][condition])
        directory = OUT / ("run_" + str(replicate)) / condition
        directory.mkdir(parents=True, exist_ok=False)
        records = []
        for begin in range(0, len(rows), 32):
            selected = rows[begin:begin + 32]
            identities = [str(a["sample_token"][r]) + "/" + str(a["instance_token"][r]) for r in selected]
            streams = [randoms(replicate, condition, identities, draw_set, 64) for draw_set in (0, 1)]
            result = {}
            for host in ("CV", "MLP"):
                noise = []
                chol = np.linalg.cholesky(calibration["covariance"][host])
                for draw_set in (0, 1):
                    standard = np.asarray([row_rng(202610150 + replicate, condition, key, draw_set, "old_process_normal").standard_normal((64, 6, 2)) for key in identities])
                    noise.append(np.einsum("bmti,tji->bmtj", standard, chol))
                for law in ("A0", "AG", "AH", "AM"):
                    centers = []
                    for draw_set in (0, 1):
                        if law == "A0":
                            current = inputs[selected, None]
                        else:
                            p = head[law]
                            z = streams[draw_set]["z"]
                            if law == "AH":
                                df = p["degrees_of_freedom"][selected].reshape(-1).astype(np.float64)
                                radial = np.sqrt(df[:, None] / chi2.ppf(streams[draw_set]["radial"], df[:, None]))
                                z = z * radial[..., None]
                            linear = p["mean_state"][selected, None] + p["scale"][selected, None] * z
                            yaw = []
                            for index, (row, identity) in enumerate(zip(selected, identities)):
                                rng = row_rng(202610140 + replicate, condition, identity, draw_set, "yaw")
                                if law == "AH":
                                    yaw.append(rng.vonmises(float(p["yaw_mean"][row]), float(p["yaw_kappa"][row]), 64))
                                else:
                                    sigma = mmg_yaw_sigma[row] if law == "AM" else p["yaw_scale"][row]
                                    yaw.append(p["yaw_mean"][row] + sigma * rng.standard_normal(64))
                            yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
                            current = r0.replace_current(inputs[selected], linear, yaw)
                        centers.append(r0.predict_centers(host, current, predictor, calibration, device))
                    values = r0.distribution_metrics(centers[0], centers[1], future[selected], ego[selected],
                                                     calibration["covariance"][host], noise[0], noise[1])
                    result.update({host + "_" + law + "_" + key: value for key, value in values.items()})
            path = directory / ("part-{:05d}.npz".format(len(records)))
            with path.open("xb") as stream:
                np.savez(stream, row=selected, **result)
            records.append({"file": path.name, "sha256": sha(path), "rows": len(selected)})
            if len(records) % 50 == 0 or begin == 0:
                print(json.dumps({"condition": condition, "replicate": replicate, "rows": begin + len(selected), "total": len(rows)}), flush=True)
        write(directory / "complete.json", {"rows": len(rows), "parts": records, "replicate": replicate,
            "physical_gpu": physical, "seconds_since_start": time.time() - started,
            "source_predictor_sha256": sha(r0.OUT / "predictor.pt"), "head_seed": HEAD_SEEDS[replicate],
            "prediction_law": "original CV/MLP residual Gaussian kernels; full-state/yaw interface",
            "same_vehicle_row_ids_as_HiVT": True, "old_reports_rewritten": False,
            "evaluation_freeze_sha256": sha(EVALUATION / "freeze.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate", type=int, choices=range(3), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 2, 3), required=True)
    args = parser.parse_args()
    evaluate(args.replicate, args.gpu)
