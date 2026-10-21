# multimodal-uq

Research materials for **Where Does 3D Perception Uncertainty Add Value to Motion Forecasting Under Sensor Degradation?**

This repository collects the manuscript, implementation, retained checkpoints,
experiment protocols, result summaries, and figure source data for the
perception-to-forecast uncertainty study. It covers stochastic propagation
through CV/MLP and HiVT, learned uncertainty representations and probability
tokens, and distribution-preserving training.

## Start here

| Material | Location |
| --- | --- |
| Manuscript supplied for this release | [manuscript/main.tex](manuscript/main.tex) |
| Unmodified attachment | [manuscript/attachment_original.tex](manuscript/attachment_original.tex) |
| Reproduction instructions and prerequisites | [REPRODUCING.md](REPRODUCING.md) |
| Paper-to-code and evidence map | [PAPER_CONTENTS.md](PAPER_CONTENTS.md) |
| Missing assets and historical version differences | [KNOWN_GAPS.md](KNOWN_GAPS.md) |
| Original paper figures and panel source data | [BEVFusion/riskprop3d_paper/figures](BEVFusion/riskprop3d_paper/figures) |
| Original CV/MLP final-validation report | [BEVFusion/riskprop3d_paper/FINAL_CONFIRMATION_REPORT.md](BEVFusion/riskprop3d_paper/FINAL_CONFIRMATION_REPORT.md) |
| HiVT extension report | [BEVFusion/riskprop3d_hivt_extension/REPORT.md](BEVFusion/riskprop3d_hivt_extension/REPORT.md) |
| HiVT numerical results | [BEVFusion/riskprop3d_hivt_extension/results_v1](BEVFusion/riskprop3d_hivt_extension/results_v1) |

The supplied manuscript is the current paper version. Earlier Markdown drafts,
supplementary material, and status reports are preserved as historical evidence;
their claims and completion status may predate the HiVT extension.

## Verify the package

From the repository root, using Python 3.8 or newer:

```bash
python check_package.py
```

This checks copied-file hashes, Python syntax, JSON readability, and the two
official-clean HiVT energy-score contrasts against the retained results.
It does not load checkpoints or start training.

## Results and scope

The trained HTC law improves CV/MLP stochastic propagation in the original
study. In the HiVT extension, the primary trained-Gaussian-minus-HTC energy
contrast is approximately **-0.000634 m** with a scene CI spanning zero.
The moment-matched-Gaussian-minus-HTC contrast is approximately **0.023108 m**
with a positive scene CI. Learned-interface and preservation results require
their matched comparators; this is a controlled empirical study, not a claim
of universal uncertainty benefit.

Raw nuScenes data, derived per-target archives, detector weights, compiled
extensions, and installed Python environments are not distributed here.
The excluded derived artifacts are inventoried in [EXCLUDED_ARTIFACTS.csv](EXCLUDED_ARTIFACTS.csv).
Retained study checkpoints are included at their original relative paths.
Full reproduction requires the prerequisites and missing original upstream
artifacts described in the reproduction guide.

## Provenance and third-party code

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) records source paths, byte sizes,
SHA-256 hashes, and the parent working tree's base commit. Files were copied
from the working tree; this is not a clean checkout of that base commit.
The original research tree and its Git history are separate from this release.

BEVFusion and the modified HiVT reference retain their existing licenses.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This packaging step does
not assign a new license to the manuscript or original research contributions.
