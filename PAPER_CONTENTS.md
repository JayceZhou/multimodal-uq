# Paper contents and implementation map

The directory layout keeps `BEVFusion/` relative paths intact because the
experiment runners resolve configurations, checkpoints, and evidence through
those paths.

| Paper component | Implementation | Protocols and retained evidence |
| --- | --- | --- |
| Frozen Camera–LiDAR detector, exports, and corruptions | `BEVFusion/mmdet3d/`, `BEVFusion/configs/`, `BEVFusion/tools/probset3d_f0_export.py`, `BEVFusion/tools/probset3d_f1_authority.py` | `BEVFusion/environment/bev-stage1.yml` and experiment manifests |
| Gaussian and Student-t/von Mises current-state laws | `BEVFusion/probset3d/f1/implementation/probabilistic_query.py`, `BEVFusion/tools/probset3d_v3_formal.py` | HiVT restored heads: `BEVFusion/riskprop3d_hivt_extension/recovery_v1/checkpoints/` |
| CV/MLP stochastic propagation | `BEVFusion/tools/riskprop3d_r0.py` | `BEVFusion/riskprop3d_r0/`, `BEVFusion/outputs/riskprop3d_r0/` |
| Structured distributional encoders | `BEVFusion/tools/riskprop3d_r1.py`, `riskprop3d_r15.py`, `riskprop3d_r2_core.py` | Corresponding `riskprop3d_r1`, `riskprop3d_r15`, `riskprop3d_r2_core` directories and outputs |
| Distribution-preserving student | `BEVFusion/tools/riskprop3d_r2d.py` | `BEVFusion/riskprop3d_r2d/`, `BEVFusion/outputs/riskprop3d_r2d/` |
| Direct probability tokens | `BEVFusion/tools/riskprop3d_r3p.py` | `BEVFusion/riskprop3d_r3p/`, `BEVFusion/outputs/riskprop3d_r3p/` |
| External degradation and original final validation | `BEVFusion/tools/riskprop3d_final_confirmation.py`, `riskprop3d_confirmation_export.py` | `BEVFusion/outputs/riskprop3d_final_confirmation/{external,official}/result.json` |
| Moment-matched shape control | `BEVFusion/tools/riskprop3d_level1_matched_law.py`, `riskprop3d_level1_matched_law_packaging.py` | `BEVFusion/outputs/riskprop3d_level1_matched_law/` |
| Output-noise sensitivity | `BEVFusion/tools/riskprop3d_process_noise_sensitivity.py` | `BEVFusion/outputs/riskprop3d_process_noise_sensitivity/` |
| Upstream/downstream transport analysis | `BEVFusion/tools/export_riskprop3d_upstream_downstream.py` | `BEVFusion/outputs/riskprop3d_upstream_downstream_{export,analysis}/` |
| Map/neighbor context and HiVT adaptation | `BEVFusion/riskprop3d_hivt_extension/{context_export,build_context,hivt_runtime,run_host}.py` | `DATA_CONTRACT.md`, `SUPPLEMENTARY_PLAN.md`, `data_v1/audit.json`, `host_v1/` in that directory |
| Four-law HiVT propagation and same-target CV/MLP controls | `BEVFusion/riskprop3d_hivt_extension/{evaluate,evaluate_controls,summarize}.py` | `evaluation_v1/`, `controls_v1/`, `results_v1/`, `REPORT.md` |
| Figures and original supplementary appendix | `BEVFusion/riskprop3d_paper/plot_manuscript_figures.py` | `BEVFusion/riskprop3d_paper/{figures,evidence_exports}/`, `SUPPLEMENTARY_APPENDIX.md` |

Existing checkpoints include the CV/MLP predictor, learned representation and
student replicates, three restored probability-head pairs, and three HiVT
hosts. Checkpoint files are copied as opaque artifacts, without deserialization.
Large track/context/per-target archives remain in the source research tree.
