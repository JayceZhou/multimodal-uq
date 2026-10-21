#!/usr/bin/env python3
"""Check release integrity and retained result anchors without data or GPUs."""

import ast
import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    paths = [record["path"] for record in manifest["files"]]
    assert len(paths) == len(set(paths)), "Duplicate manifest paths"
    for record in manifest["files"]:
        path = ROOT / record["path"]
        assert path.is_file() and not path.is_symlink(), path
        assert path.stat().st_size == record["bytes"], path
        assert sha256(path) == record["sha256"], path

    files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    python_files = [p for p in files if p.suffix == ".py"]
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    json_files = [p for p in files if p.suffix == ".json"]
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))

    with (ROOT / "BEVFusion/riskprop3d_hivt_extension/results_v1/contrasts.csv").open(newline="") as stream:
        rows = {r["contrast"]: r for r in csv.DictReader(stream) if r["condition"] == "official_clean"}
    for name, expected in (("AG-AH", -0.0006341651320278898), ("AM-AH", 0.023107993422298828)):
        assert abs(float(rows[name]["difference"]) - expected) < 1e-12
        assert int(rows[name]["rows"]) == 38644 and int(rows[name]["scenes"]) == 150
    assert json.loads(rows["AG-AH"]["ci95"])[0] < 0 < json.loads(rows["AG-AH"]["ci95"])[1]
    assert json.loads(rows["AM-AH"]["ci95"])[0] > 0

    original = (ROOT / "manuscript/attachment_original.tex").read_text()
    main_tex = (ROOT / "manuscript/main.tex").read_text()
    assert main_tex == original.replace(
        r"\graphicspath{{../../../figures/}}",
        r"\graphicspath{{../BEVFusion/riskprop3d_paper/figures/}}",
    ), "Unexpected manuscript edits"
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", main_tex)
    missing = [name for name in figures if not (ROOT / "BEVFusion/riskprop3d_paper/figures" / name).is_file()]
    assert missing == ["figure_3_uncertainty_transport.pdf"], "New manuscript asset gap: " + repr(missing)
    assert missing[0] in (ROOT / "KNOWN_GAPS.md").read_text()
    assert (ROOT / "BEVFusion/LICENSE").is_file()
    assert (ROOT / "BEVFusion/riskprop3d_hivt_extension/third_party/MapUncertaintyPrediction/HiVT_modified/LICENSE").is_file()
    print("PASS: {} source hashes, {} Python files, {} JSON files, and HiVT result anchors".format(
        len(paths), len(python_files), len(json_files)))
    print("Documented manuscript gap: " + ", ".join(missing))


if __name__ == "__main__":
    main()
