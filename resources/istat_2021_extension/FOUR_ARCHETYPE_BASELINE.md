# Four-archetype baseline

`archetype_pipeline.py` reconstructs the four household archetypes directly
from `istat_microdata_2013.csv`. It is isolated from the older exploratory
scripts and does not overwrite `family_trees.csv` or `clusters.csv`.

The pipeline performs the following operations:

1. converts the survey age classes to deterministic pseudo-random ages;
2. calculates minimum age, maximum age, household size, and education index;
3. applies the two PCA equations published in the article;
4. min-max normalizes PC1, PC2, and education;
5. fits four clusters using deterministic k-means++ and Lloyd iteration;
6. assigns stable semantic archetype numbers; and
7. validates the results against Table 3 and Figure 5 of the article.

Run it from this folder with a Python environment containing NumPy and pandas:

```powershell
python -m pip install -r requirements-archetypes.txt
python archetype_pipeline.py
```

The outputs are written to `baseline_outputs/`:

- `household_archetypes.csv`: one row per eligible household;
- `archetype_summary.csv`: archetype statistics, published values, and deltas;
- `preprocessing_exclusions.csv`: excluded IDs and explicit reasons;
- `archetype_model.json`: formulas, scaling limits, centers, and diagnostics;
- `validation_report.json`: machine-readable pass/fail checks.

The default run is strict and returns a nonzero exit code if the result drifts
beyond the declared validation tolerances. Seeds can be overridden from the
command line for sensitivity checks, but doing so may intentionally fail the
published-baseline validation.

The effective education recoding is stated explicitly in the source. This is
important because the original in-place recoding changed education category 3
twice, ultimately mapping it to 12 years. Preserving that effective behavior is
necessary to reproduce the published archetype statistics.
