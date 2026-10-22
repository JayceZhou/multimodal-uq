# multimodal-uq

Code and research materials for **Where Does 3D Perception Uncertainty Add Value to Motion Forecasting Under Sensor Degradation?**

The paper studies how uncertainty from Camera–LiDAR 3D perception affects
motion forecasting under sensor degradation. It compares stochastic
propagation, learned uncertainty representations, and distribution-preserving
training using CV/MLP predictors and a map- and interaction-aware HiVT host.
The results show that uncertainty benefits depend on the forecasting host,
integration interface, and comparison used.

## Quick start

Clone the repository, download the Git LFS assets, and check the retained code and results:

```bash
git clone https://github.com/JayceZhou/multimodal-uq.git
cd multimodal-uq
git lfs install --local
git lfs pull
python check_package.py
```

The package check needs no GPU or dataset. To regenerate the historical paper
figures from the included summaries (requires NumPy and Matplotlib; overwrites
the existing figure files):

```bash
python BEVFusion/riskprop3d_paper/plot_manuscript_figures.py
```

For training and evaluation, first follow the environment and data setup in
[REPRODUCING.md](REPRODUCING.md). The experiment entry points include:

```bash
cd BEVFusion
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/riskprop3d_r0.py --help
python riskprop3d_hivt_extension/evaluate.py --help
```

## Directory structure

```text
multimodal-uq/
├── manuscript/                   # Paper LaTeX source
├── BEVFusion/
│   ├── mmdet3d/                  # Perception and uncertainty implementation
│   ├── configs/                  # Model and experiment configurations
│   ├── tools/                    # Training, evaluation, and analysis scripts
│   ├── probset3d/                 # Probabilistic object-state heads
│   ├── riskprop3d_r*/             # Original CV/MLP experiment protocols
│   ├── riskprop3d_paper/          # Drafts, figures, and figure source data
│   ├── riskprop3d_hivt_extension/ # HiVT code, checkpoints, and results
│   ├── outputs/                  # Retained results and study checkpoints
│   ├── environment/              # Environment specification
│   └── tests/                    # Existing implementation tests
├── REPRODUCING.md                 # Reproduction instructions
├── PAPER_CONTENTS.md              # Paper-to-code mapping
└── SOURCE_MANIFEST.json           # File provenance and checksums
```
