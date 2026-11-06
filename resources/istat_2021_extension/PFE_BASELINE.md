# Windowed dwelling-to-archetype probabilities

`pfe_pipeline.py` replaces the Naive Bayes probability chain with the
windowed empirical probability method described in the article. It uses the
restored assignments in `baseline_outputs/household_archetypes.csv` and the
dwelling classes in the raw ISTAT data.

Run both restored stages from this folder:

```powershell
python archetype_pipeline.py
python pfe_pipeline.py
```

The default PFE configuration is deterministic:

- dwelling-class randomization seed: 42;
- building-age grid: 20 to 140 years in 20-year steps;
- dwelling-area grid: 40 to 360 m² in 40 m² steps;
- centered building-age windows: ±6 years;
- centered dwelling-area windows: ±8 m²; and
- minimum-support flag: five observations.

The main output is
`pfe_outputs/dwelling_archetype_probabilities.csv`. Every row contains the
building-age and area coordinates, window bounds, total sample count, four
archetype counts, four empirical probabilities, and their sum. Low-support
windows remain visible and are flagged rather than silently discarded.

`pfe_outputs/validation_report.json` verifies that counts are conserved,
probabilities are finite and bounded, every probability equals its empirical
count ratio, and each row sums to one.
