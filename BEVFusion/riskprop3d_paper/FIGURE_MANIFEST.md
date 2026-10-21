# Manuscript figure manifest

The manuscript contains five main figures and one supplementary figure. All
quantitative panels are generated directly from the frozen machine-readable
evidence; no values are manually transcribed into the plotting code.

## Main figures

| Figure | Scientific role | Panels | Source data |
|---|---|---|---|
| `figure_1_interface_taxonomy` | Defines the controlled perception-to-forecast interface taxonomy and scope boundary | Stochastic propagation value; representation utility; objective necessity | Conceptual schematic; no quantitative source table |
| `figure_2_stochastic_propagation` | Establishes the primary stochastic-propagation result on the development evaluation | HTC-over-Gaussian energy; HTC-over-mean energy; equal-coverage sharpness; high-FDE AUPRC | `source_data/figure_2.csv` |
| `figure_3_upstream_downstream_transfer` | Relates trained upstream-law differences to downstream energy transfer | XY NLL, natural coverage, and equal-coverage sharpness versus CV/MLP energy | `source_data/figure_3.csv` |
| `figure_4_learned_interfaces` | Contrasts learned representation and preservation-objective results | Structured encoding; direct token conditioning; teacher fidelity; downstream contrasts; runtime | `source_data/figure_4.csv` |
| `figure_5_interface_stability` | Visualizes the central cross-population stability asymmetry | Stochastic effects across populations; learned-interface outcomes across populations and controls | `source_data/figure_5.csv` |

## Supplementary figure

| Figure | Scientific role | Panels | Source data |
|---|---|---|---|
| `figure_s1_risk_boundary` | Separates forecast-error ranking from average trajectory proper-score quality | Development token effects; final-validation token effects; final-validation stochastic risk effects | `source_data/figure_s1.csv` |

## Deliverables

Each figure is exported as:

- editable `SVG`;
- vector `PDF`;
- 600-dpi LZW-compressed `TIFF` for submission;
- 240-dpi `PNG` for Markdown preview and drafting.

The PNG previews are embedded in the English manuscript, Chinese manuscript,
and supplementary appendix. The quantitative source-data tables are retained
under `figures/source_data/`.

## Reproduction

From `BEVFusion/riskprop3d_paper` in the `bev` environment:

```bash
MPLCONFIGDIR=figures/.mplconfig \
  /home/zhousc/miniconda3/envs/bev/bin/python plot_manuscript_figures.py
```

The script validates preregistered anchor values before plotting and exits on
any mismatch with the frozen evidence files.

## Quality-control record

- Scientific conclusion and panel logic were specified before plotting in
  `FIGURE_CONTRACTS.md`.
- All quantitative values are traced to frozen JSON evidence and exported to
  figure-specific CSV files.
- SVG exports retain editable text nodes.
- TIFF exports were verified at 600 dpi.
- Color and grayscale previews were inspected for legibility and marker
  redundancy.
- Captions define effect orientation, confidence intervals, bootstrap unit,
  sample size, and the limits of the risk proxy where applicable.
