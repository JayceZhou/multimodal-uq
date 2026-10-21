# Reproducing and inspecting the study

Model checkpoints, TIFF figures, and the upstream demo GIF use Git LFS.
After cloning, run `git lfs install --local` and `git lfs pull` to retrieve them.

## Inspect the retained evidence

Run `python check_package.py` from the root. No GPU or dataset is needed.
The original result JSON files, HiVT CSV tables, figure source CSV files,
protocols, and audit records can be inspected directly.

The original figure generator reads the included summary JSON files:

```bash
python BEVFusion/riskprop3d_paper/plot_manuscript_figures.py
```

This command needs NumPy and Matplotlib and regenerates the **historical**
figure bundle in place. Use a disposable checkout to preserve release hashes.
It does not generate the missing newer manuscript Figure 3.

## Environment

The retained BEVFusion environment specification is
`BEVFusion/environment/bev-stage1.yml`. It records Python 3.8, PyTorch 2.4.1,
CUDA 12.4, and the detector's Python packages. BEVFusion requires compiled
CUDA operators and matching MMCV/MMDetection dependencies; follow its
retained `BEVFusion/README.md` and `setup.py` for those build steps.

Additional analysis packages used by the study include SciPy, scikit-learn,
Matplotlib and h5py. The HiVT preflight records these extension versions:

| Package | Recorded version |
| --- | --- |
| torch-geometric | 2.5.3 |
| torch-scatter | 2.1.2+pt24cu124 |
| torch-sparse | 0.6.18+pt24cu124 |
| pytorch-lightning | 2.3.3 |
| torchmetrics | 1.4.3 |
| lightning-utilities | 0.11.9 |

The installed `python/` dependency directory was deliberately excluded;
install compatible packages for your own PyTorch/CUDA environment. No new
environment installation or clean-environment reproduction was attempted
during packaging.

## Data and artifacts

Obtain nuScenes data and maps through the dataset's own access process.
The repository does not redistribute the dataset, sensor images, point clouds,
GT label archives, or the pretrained detector. Place the frozen detector at
`BEVFusion/pretrained/bevfusion-det.pth` and prepare the info files and derived
archives required by each runner's protocol. Original paths, schemas and
hashes remain in its manifests. The inventory describes derived artifacts
retained locally at the source paths, not downloadable files in this repo.

Full replay of the original CV/MLP study additionally needs the missing
original upstream head/authority files, as explained in `KNOWN_GAPS.md`.
The independently restored heads can support a new, clearly labeled run;
they cannot establish a byte-identical replay of those original results.

## Experiment entry points

Run research scripts from `BEVFusion/` with that directory on `PYTHONPATH`.
Use each entry point's `--help` and its protocol for the execution order:

```bash
cd BEVFusion
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/riskprop3d_r0.py --help
python tools/riskprop3d_r2_core.py --help
python tools/riskprop3d_r2d.py --help
python tools/riskprop3d_r3p.py --help
python tools/riskprop3d_final_confirmation.py --help
python tools/riskprop3d_level1_matched_law.py --help
python riskprop3d_hivt_extension/run_host.py --help
python riskprop3d_hivt_extension/evaluate.py --help
```

Frozen scripts often refuse to overwrite results. Use a separate checkout
and an explicit new-run directory plan; do not remove retained evidence just
to make an old command run again. Recovered-head and original-head outputs
must remain distinct.

Before GPU work, inspect current use with `nvidia-smi`, select available
devices, and explicitly set `CUDA_VISIBLE_DEVICES`. The historical HiVT
runner accepts physical GPUs 0, 2 and 3; its `--gpu` value must match that
assignment. This package creation did not run any GPU jobs.

## Manuscript

`manuscript/attachment_original.tex` preserves the supplied text byte for byte.
`manuscript/main.tex` changes only `graphicspath` for this directory layout.
An IEEEtran LaTeX installation is required. Once the correct missing Figure 3
has been restored, compilation is:

```bash
cd manuscript
latexmk -pdf main.tex
```

No publication PDF is claimed to be complete: the updated Figure 3 and
supplementary manuscript remain outstanding.
