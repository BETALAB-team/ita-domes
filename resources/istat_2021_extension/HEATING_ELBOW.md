# Heating groups: diminishing returns

Run `run_heating_elbow.py` in your IDE using the repository's `.venv` interpreter.
Outputs go to `revised_paper_outputs/heating_elbow/2021`. Earlier results and
published 2013 figures are not overwritten.

## Connection to the original workflow

- `01_PCA_Exploratory.py`, lines 76 onward: k-means WCSS versus cluster count
  for demographic clustering, with visual elbow inspection.
- `07_ANOVA_analysis.py`, lines 476–517: heating combinations of region,
  dwelling type and archetype are iteratively merged using Tukey comparisons.
  This is a different stopping rule, not an elbow analysis.

The new script adapts the original WCSS elbow idea to the 48 heating factor-cell
means. It solves the one-dimensional weighted k-means objective exactly, using
training cell population weights. It does not cluster individual households
directly by their heating hours: held-out assignments use factor cells only.
Insulation is neither a predictor nor an eligibility restriction. Existing
demographic archetypes and train/tuning/test splits are retained.

## Interpretation

The curve reports both between-cell WCSS and total household training MSE.
Explained between-cell variation must not be described as explained household
variation. Tuning RMSE is a separate predictive diagnostic.

A normalized endpoint-chord heuristic identifies an exploratory elbow, not a
uniquely established optimum. With maximum k of 9 or 12 it selects 2; with 16
it selects 3, and with 24 it selects 4. All these candidate mappings and a
comparison plot are saved. `ELBOW_MAX_K` controls the main display and heuristic.
The current result therefore supports examining 2–4 groups rather than
claiming an unequivocal optimum. Four groups capture about 96% of the training
between-cell variation; the fifth adds about 1.58 percentage points.

The original unweighted Tukey stopping rule is also run on training data as
a diagnostic and retains eight groups. Its adaptive training p-values are not
confirmatory evidence. Test-set pairwise comparisons are reported for the
main elbow candidate, but never used to choose the count. Survey weights are
used for the new elbow and descriptive summaries; pairwise standard errors
do not incorporate an unavailable full complex-survey design.

Key outputs: `elbow_curve.csv`, `results.json`, `heating_assignment.csv`,
`candidate_2_assignment.csv` (and other sensitivity candidates),
`heating_robust_pairs.csv`, and the PNG/SVG/PDF files in `figures`.
