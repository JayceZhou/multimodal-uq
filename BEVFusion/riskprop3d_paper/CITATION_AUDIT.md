# Closest-related-work citation audit

**Search date:** 2026-08-11

**Scope:** primary literature most directly addressing imperfect object-state or association inputs to motion forecasting; no Nature/CNS restriction because the target venues are T-ITS and TR-C.

## Claim-to-reference mapping

### S001 — Perceptual state uncertainty can be propagated through trajectory forecasting

**Manuscript use:** Section 1 and Section 2.2.

- B. Ivanovic, Y. Lin, S. Shrivastava, P. Chakravarty, and M. Pavone, “Propagating State Uncertainty Through Trajectory Forecasting,” ICRA 2022, doi:10.1109/ICRA46639.2022.9811776.
- **Support:** strong. The work directly incorporates perceptual state uncertainty into trajectory forecasting with a statistical-distance objective and evaluates predictive calibration.
- **Four-axis positioning:** (1) its primary question is how to incorporate state uncertainty, whereas this paper separates stochastic propagation value, representation utility, and objective necessity; (2) this paper adds matched mean, confidence, Gaussian, capacity/initialization, and identical GT-only necessity controls; (3) this paper derives the object-state laws from Camera--LiDAR 3D perception and evaluates six sensor-degradation families; and (4) this paper's target result is an applicability boundary containing positive and null evidence rather than a uniformly positive uncertainty-aware method.
- **Boundary:** [23] supports the learned-propagation research gap, but not this paper's Camera--LiDAR degradation, matched-necessity controls, or three-level conclusions.
- **Primary source checked:** NVIDIA Research publication page and arXiv:2110.03267, both accessed 2026-08-11. Their abstract explicitly identifies the statistical-distance loss, perceptual state uncertainty, simulation/real-world evaluation, and calibration result.

### S002 — Tracking association error is a separate perception-to-forecast problem

**Manuscript use:** Section 2.2 and Section 6.4.

- X. Weng, B. Ivanovic, K. Kitani, and M. Pavone, “Whose Track Is It Anyway? Improving Robustness to Tracking Errors With Affinity-Based Trajectory Prediction,” CVPR 2022, pp. 6563--6572.
- **Support:** strong for the complementary boundary. The paper replaces hard track association with affinity-based prediction to improve robustness to tracking errors.
- **Boundary:** it does not support state-distribution effects conditional on correct identity; it motivates stating that association uncertainty is outside the present scope.

### S003 — Forecasting can operate directly on uncertain detection associations

**Manuscript use:** Section 2.2 and Section 6.4.

- P. Zhang, L. Bai, Y. Wang, J. Fang, J. Xue, N. Zheng, and W. Ouyang, “Towards Trajectory Forecasting From Detection,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 45, no. 10, pp. 12550--12561, 2023, doi:10.1109/TPAMI.2023.3274686.
- **Support:** strong for the complementary association boundary. The journal article forecasts from detections and uses affinity uncertainty rather than preformed hard-associated trajectories.
- **Boundary:** it addresses noisy association and detection-to-forecast modeling, not calibrated object-state predictive laws conditional on correct identity. The earlier arXiv title was replaced by the verified TPAMI publication in manuscript v0.4.

### S004 — Online-map uncertainty is a complementary downstream interface

**Manuscript use:** Section 1 and Section 2.3.

- X. Gu, G. Song, I. Gilitschenski, M. Pavone, and B. Ivanovic, “Producing and Leveraging Online Map Uncertainty in Trajectory Prediction,” CVPR 2024, pp. 14521--14530.
- **Support:** strong for map uncertainty; background/contrast support for the present object-state interface.
- **Boundary:** it should not be cited as direct evidence for Camera--LiDAR object-state uncertainty.

### S005 — Perception-information uncertainty can be encoded in an interaction-aware vehicle predictor

**Manuscript use:** Section 1 and Section 2.2.

- M. Cao, C. Wang, W. Zhao, and Z. Zhang, “PU-TSI: Interactive Vehicle Trajectory Prediction Considering Perception Information Uncertainty,” *IEEE Internet of Things Journal*, vol. 12, no. 15, pp. 30518--30532, 2025, doi:10.1109/JIOT.2025.3570741.
- **Support:** direct and current. PU-TSI integrates perception-information uncertainty into trajectory feature encoding and interaction modeling.
- **Positioning:** PU-TSI asks how an uncertainty-aware predictor can improve accuracy and stability. The present study instead compares stochastic-law value, matched representation utility, and objective necessity, and retains the failure of learned necessity controls as part of its conclusion.
- **Boundary:** the present manuscript should not imply that PU-TSI used the same Camera--LiDAR object-state laws, proper-score endpoint, or degradation protocol.
- **Source basis checked:** IEEE Xplore bibliographic record (document 11006072) and DOI metadata; the Bi-GRU/data-confidence-attention description is limited to the article abstract reproduced by bibliographic indexing services. No claim is made about unreported controls.

### S006 — Recent joint detection--forecasting systems strengthen the interface context

**Manuscript use:** Section 2.3.

- S. Papais, L. Wang, B. Cheong, and S. L. Waslander, “ForeSight: Multi-View Streaming Joint Object Detection and Trajectory Forecasting,” ICCV 2025, pp. 25474--25484.
- **Support:** strong for the current joint-interface landscape. ForeSight shares streaming query memory between 3D detection and trajectory forecasting and avoids explicit tracking association.
- **Boundary:** it is not an explicit probabilistic object-state uncertainty method and therefore does not replace [23] or [26] as the closest uncertainty-specific comparisons.

### S007 — Explicit perceptual-uncertainty indications can alter downstream automated-driving behavior

**Manuscript use:** Section 2.4.

- N. Grabowsky, A. Mütze, and M. Rottmann, “Does Knowledge About Perceptual Uncertainty Help an Agent in Automated Driving?” CVPR Workshops 2026, pp. 4219--4229.
- **Support:** relevant but task-adjacent. The work evaluates whether an explicit uncertainty indication changes behavior in a proxy driving task.
- **Boundary:** it concerns downstream decision behavior rather than object trajectory forecasting, so it supports the need to separate interface questions but not any forecasting result in this manuscript.

## 2025--2026 currency screen

The update screen covered explicit object-state uncertainty, perception-informed vehicle trajectory prediction, joint detection/forecasting, online-map uncertainty, and downstream automated-driving decisions. It identified PU-TSI as the most directly relevant new peer-reviewed object-state paper and ForeSight as the most relevant new joint-interface paper. Two 2026 map-uncertainty preprints were also screened--“Uncertainty-Aware Trajectory Prediction: A Unified Framework Harnessing Positional and Semantic Uncertainties” (arXiv:2603.29362) and “Uncertainty Matters: Structured Probabilistic Online Mapping for Motion Prediction in Autonomous Driving” (arXiv:2603.20076). They were not added to the main reference list because they address online-map rather than object-state uncertainty and remain preprints at the search date. SHIFT and other forecasting-output uncertainty methods were excluded because they estimate uncertainty inside the trajectory predictor rather than propagate an upstream perception distribution.

## Export

Six closest or updated contextual references are available in `CLOSEST_RELATED_WORK.ris`. In EndNote, import it using the Reference Manager (RIS) option.

The non-ranking design comparison used in Supplementary Table S1 is limited to
[23] and [26]. Its cells describe each paper's stated research question and
entry point for uncertainty; “not isolated” is used instead of a negative
checkmark when a necessity contrast is not the paper's stated question.
