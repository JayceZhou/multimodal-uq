#!/usr/bin/env python3
"""Serialization-only shim for the frozen matched-law finalizer.

The frozen evaluator's HDF5 packager passed NumPy Unicode arrays directly to
h5py, which has no conversion path for ``<U`` dtypes.  This shim changes only
the serialization representation of identifiers to fixed-width UTF-8 bytes;
all scientific estimators, bootstrap code, gates, and endpoint logic remain
the frozen evaluator's implementations.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import riskprop3d_level1_matched_law as frozen


def h5_compatible(value):
    array = np.asarray(value)
    if array.dtype.kind == "U":
        return array.astype(f"S{array.dtype.itemsize // 4}")
    return array


def create_h5_outputs(runs):
    paired_path = frozen.OUT / "paired_track_metrics.h5"
    scene_path = frozen.OUT / "scene_level_metrics.h5"
    if paired_path.exists() or scene_path.exists():
        raise FileExistsError("HDF5 output already exists")
    paired_tmp = paired_path.with_suffix(".h5.partial")
    scene_tmp = scene_path.with_suffix(".h5.partial")
    with h5py.File(paired_tmp, "w") as paired, h5py.File(scene_tmp, "w") as scene_file:
        paired.attrs["layout"] = "replicate/condition grouped arrays; law and probe dimensions stored as attributes"
        paired.attrs["laws"] = json.dumps(frozen.LAW_IDS)
        paired.attrs["probes"] = json.dumps(frozen.r0.PREDICTORS)
        paired.attrs["serialization_shim"] = "NumPy Unicode identifiers converted to fixed-width UTF-8 bytes"
        for replicate, run in enumerate(runs):
            for source_name in frozen.CONDITIONS:
                group = paired.create_group(f"replicate_{replicate}/{source_name}")
                for name in ("scene_token", "sample_token", "instance_token", "row_id"):
                    group.create_dataset(
                        name,
                        data=h5_compatible(run[f"{source_name}__common__{name}"]),
                        compression="gzip",
                    )
                upstream = np.stack([
                    np.stack([
                        np.asarray(run[f"{source_name}__upstream__{law}__{metric}"], dtype=np.float32)
                        for metric in frozen.UPSTREAM_METRICS
                    ], axis=1)
                    for law in frozen.LAW_IDS
                ], axis=1)
                group.create_dataset("upstream", data=upstream, compression="gzip", compression_opts=4)
                downstream = np.stack([
                    np.stack([
                        np.stack([
                            np.asarray(run[f"{source_name}__{host}__{law}__{metric}"], dtype=np.float32)
                            for metric in frozen.DOWNSTREAM_METRICS
                        ], axis=1)
                        for law in frozen.LAW_IDS
                    ], axis=1)
                    for host in frozen.r0.PREDICTORS
                ], axis=1)
                group.create_dataset("downstream", data=downstream, compression="gzip", compression_opts=4)
                group.attrs["upstream_metrics"] = json.dumps(frozen.UPSTREAM_METRICS)
                group.attrs["downstream_metrics"] = json.dumps(frozen.DOWNSTREAM_METRICS)
                group.attrs["head_seed"] = int(run["meta__head_seed"])

                scenes = np.asarray(run[f"{source_name}__common__scene_token"])
                unique = np.unique(scenes)
                sg = scene_file.create_group(f"replicate_{replicate}/{source_name}")
                sg.create_dataset("scene_token", data=h5_compatible(unique))
                energy = np.empty(
                    (len(unique), len(frozen.r0.PREDICTORS), len(frozen.LAW_IDS)),
                    dtype=np.float32,
                )
                for host_index, host in enumerate(frozen.r0.PREDICTORS):
                    for law_index, law in enumerate(frozen.LAW_IDS):
                        tokens, values = frozen.scene_means(
                            run[f"{source_name}__{host}__{law}__energy"], scenes
                        )
                        if not np.array_equal(tokens, unique):
                            raise ValueError("scene aggregation mismatch")
                        energy[:, host_index, law_index] = values
                sg.create_dataset("trajectory_energy", data=energy, compression="gzip")
    paired_tmp.replace(paired_path)
    scene_tmp.replace(scene_path)
    return paired_path, scene_path


def main():
    frozen.verify_protocol()
    audit = frozen.OUT / "packaging_shim.json"
    frozen.atomic_json(audit, {
        "scope": "serialization only",
        "reason": "frozen HDF5 packager supplied NumPy Unicode identifiers to h5py",
        "change": "convert identifier arrays from NumPy Unicode to fixed-width UTF-8 bytes",
        "scientific_estimators_changed": False,
        "bootstrap_changed": False,
        "gates_or_endpoint_logic_changed": False,
        "frozen_evaluator_sha256": frozen.sha256(Path(frozen.__file__)),
        "shim_sha256": frozen.sha256(Path(__file__)),
    })
    frozen.create_h5_outputs = create_h5_outputs
    frozen.finalize()


if __name__ == "__main__":
    main()
