# Figure contract — process-noise sensitivity

- **Core conclusion:** Within the controlled CV/MLP probes, the HTC-over-G
  trajectory-energy contrast remains positive across 0–2× the fitted
  process-noise covariance and is therefore not an artifact of one downstream
  noise magnitude.
- **Archetype:** quantitative grid.
- **Target/output:** T-ITS double-column figure; editable PDF/SVG plus PNG and
  600-dpi TIFF.
- **Backend:** Python/matplotlib only.
- **Final size:** 183 mm wide, approximately 67 mm high.
- **Panel a:** development clean effect curve.
- **Panel b:** final-validation effect curve.
- **Hero evidence:** paired G-minus-HTC trajectory-energy point estimates.
- **Robustness evidence:** four prespecified covariance multipliers and two
  structurally distinct probes.
- **Statistics:** three paired head/propagation seeds; target differences
  averaged within scene and replicate, then across replicates; 10,000 paired
  scene-bootstrap resamples; 95% intervals; positive favors HTC.
- **Source data:** `figure_source_data.csv` generated from retained per-target
  metric archives.
- **Integrity:** 1× output must reproduce the existing retained clean and
  final-validation trajectory-energy arrays within 1e-6 before any result is
  accepted.
- **Reviewer risk:** covariance multiplier scales covariance (standard
  deviation changes by its square root); the audit tests magnitude
  sensitivity only and does not validate a different process-noise family.
