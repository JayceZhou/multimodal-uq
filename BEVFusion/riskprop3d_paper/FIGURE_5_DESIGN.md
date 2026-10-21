# Figure 5 production specification

## Scientific conclusion

Within the controlled study, the trained-law contrast is stable under explicit
stochastic propagation, whereas learned-interface effects are heterogeneous
across comparator, seed, and evaluation population.

The figure must communicate this contrast within five seconds. It must not
suggest a universal stochastic-greater-than-learned theorem or compare effect
magnitudes across unmatched model audits.

## Evidence source

Use only the frozen values summarized in Main Sections 5.1--5.4, Main Table 3,
Supplementary Tables S2--S3, and their referenced machine-readable evidence archives. Do not
recompute a new pooled endpoint. Blank cells must remain blank when an
interface was not evaluated in a stage.

## Layout

Use two horizontally aligned panels with a shared column order:

1. **Development/original degradations**
2. **Held-out degradations**
3. **Final validation**

### Panel (a): stochastic propagation

Rows:

- HTC versus Gaussian, CV probe;
- HTC versus Gaussian, MLP probe.

Within the development and transfer columns, show the relevant degradation
families as compact subcolumns. The final-validation column contains one cell
per probe. Because every cell uses the same HTC-minus-Gaussian trajectory-
energy orientation, point estimates and 95% confidence intervals may be shown
on a common horizontal scale.

### Panel (b): learned interfaces

Row groups:

- full structured encoder versus mean-matched control;
- full structured encoder versus Gaussianized control;
- HTC token versus mean-only;
- HTC token versus confidence;
- HTC token versus Gaussian token.

Use a categorical effect matrix rather than a shared numerical scale because
the models and audits are not matched across every row and population. Show
only the populations in which each contrast was evaluated.

## Visual encoding

Do not rely on red/green alone. Use color and shape together:

- dark blue filled upward triangle: positive point estimate with a 95%
  interval wholly above zero;
- light blue open upward triangle: positive point estimate with an interval
  spanning zero;
- orange filled downward triangle: negative point estimate with a 95%
  interval wholly below zero;
- orange open downward triangle: negative point estimate with an interval
  spanning zero;
- white cell with diagonal hatch: not evaluated;
- small split-ring overlay: at least one paired seed reverses the aggregate
  direction.

Add a thin outline around the entire stochastic block to make its consistency
visually immediate. Do not add a similar outline to the learned block.

## Annotation

Use one short callout above the panels:

> Stable trained-law contrast under stochastic propagation; conditional value
> under learned interfaces.

Add a footnote:

> Cells summarize pre-specified paired contrasts. A split ring marks seed
> reversal; hatching marks an unevaluated contrast. Learned-interface cell
> magnitudes are not compared across unmatched audits.

## Export and accessibility

- Target a two-column IEEE width of approximately 7.16 inches.
- Minimum final text size: 7.5 pt.
- Export vector PDF and SVG; also retain a 600-dpi TIFF fallback.
- Use an Okabe--Ito-compatible blue/orange/gray palette.
- Check legibility in grayscale and under common color-vision-deficiency
  simulations.
- The caption must state “observed within this controlled study” and must not
  use “universal,” “intrinsic,” or “causal mechanism.”
