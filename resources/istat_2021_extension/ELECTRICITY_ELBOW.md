# Electricity groups: diminishing returns

Run `run_electricity_elbow.py` in your IDE with the repository `.venv`
interpreter. It calls the same analysis implementation as the heating runner.
Outputs are saved separately in `revised_paper_outputs/electricity_elbow/2021`.
No prior outputs, published 2013 results or appliance-conversion assumptions
are changed.

The analysis uses the saved 2021 modelled appliance electricity values, existing
archetypes and train/tuning/test splits. Insulation columns are removed without
excluding households for missing insulation. The 48 location x dwelling type x
archetype combinations are clustered by their training weighted mean electricity
using exact one-dimensional weighted k-means. Test assignment uses these factors,
not household electricity values.

The main normalized-chord elbow over k=1..12 selects three groups. Sensitivity
to the upper bound gives two for 9, three for 12, and four for 16 or 24. All
candidate mappings and comparison boxplots are saved. Four groups capture 96.29%
of between-cell training variation; the fifth adds 1.05 percentage points.
This is not the percentage of total household variation explained, and the
elbow is exploratory rather than a uniquely established optimum.

The legacy Tukey diagnostic retains the original electricity factors (location
and dwelling type only), unlike the three-factor elbow analysis; it returns five
training groups. It is therefore not a controlled comparison of stopping rules
alone. Its adaptive training p-values are not confirmatory tests.

Key outputs: `elbow_curve.csv`, `results.json`, `electricity_assignment.csv`,
`electricity_groups.csv`, `electricity_robust_pairs.csv`, candidate mappings, and
PNG/SVG/PDF plots in `figures`. Main-result files correspond to the automatic
three-group candidate; the four-group mapping is `candidate_4_assignment.csv`.
`ELBOW_MAX_K` in `run_heating_elbow.py` controls the shared main elbow range.
