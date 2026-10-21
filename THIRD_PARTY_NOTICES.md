# Third-party notices

- **BEVFusion / MMDetection3D:** runtime source and configuration snapshot in
  `BEVFusion/mmdet3d/`, `BEVFusion/configs/` and related build files. The
  existing Apache 2.0 license is retained at `BEVFusion/LICENSE`; original
  source notices and README remain present. Upstream:
  https://github.com/mit-han-lab/bevfusion.
- **MapUncertaintyPrediction / modified HiVT:** reference source snapshot at
  `BEVFusion/riskprop3d_hivt_extension/third_party/MapUncertaintyPrediction/HiVT_modified/`.
  Upstream: https://github.com/alfredgu001324/MapUncertaintyPrediction,
  commit `0a11ee4b925e4967f2f889079d4086b663d396dc`. Its Apache 2.0 license is
  retained in that directory. Local runtime compatibility modifications are
  documented in `BEVFusion/riskprop3d_hivt_extension/runtime_compatibility.patch`.
- Other Python/CUDA dependencies are installed separately and retain their
  respective licenses; installed environments and compiled wheels are not
  bundled.

No repository-wide license is inferred for the manuscript, original research
scripts, or trained weights by copying these third-party licenses.
