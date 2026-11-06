# 2021 article-method inputs

`analysis_input_2021.csv` preserves all 54,069 released household IDs and all numeric source variables. Missing source values remain missing. This packet does not refit or modify the 2013 analysis.

The official `PESO` weight is retained as `weight`. Original `RIP` has five categories; `macroregion_code` merges South and Islands into the article’s fourth location. The all-household file includes a separate size-1–5 eligibility flag.

## Dwelling categories

| Code | 2013 meaning | 2021 meaning | Valid use |
|---|---|---|---|
| 1 | Single-family house | Single-family house | Direct correspondence |
| 2 | Multi-family or terraced house | Multi-family or terraced house | Direct correspondence |
| 3 | Apartment in building with fewer than 10 apartments | Ground, basement or raised apartment | No direct correspondence |
| 4 | Apartment in building with 10–27 apartments | Intermediate-floor apartment | No direct correspondence |
| 5 | Apartment in building with 28+ apartments | Top-floor apartment | No direct correspondence |

A new 2021 analysis can use all five 2021 categories with their correct labels. A shared analysis can collapse codes 3–5 into Apartment. Exact application of published five-type Appendix A rows is limited to types 1 and 2; apartment rows require new 2021 group estimation or the shared three-category design. These restrictions do not prevent a 2021 H/E grouping based on dwelling type, macroregion and new archetype assignments.

## Energy and missing variables

Heating hours sum the three reported 0–8-hour blocks only when all are present and valid. The primary comparable column requires frequency 1 or 2. Cooling has both the three-frequency 2021 column and a common-frequency 1/2 column. Missing hours are not zero. Inventory presence and reported use remain separate.

The original subjective insulation rating has no released 2021 equivalent. `insulation` and `insulation_level` explicitly say unavailable; raw recent-investment variables remain separate. They must not replace the rating in the original ANOVA. Terms involving that rating cannot be estimated; an available-factor ANOVA must disclose the omitted terms.

The reused appliance model is an ownership/use estimate with reference 2013 parameters. It is not metered electricity or total household energy. The saved audit reproduces 2013 component totals within 1.82e-12 kWh. All 2021 IDs match, all totals are finite and positive, and component sums reproduce totals within the same numerical tolerance. Component columns and lower/upper assumption scenarios are retained.

## Age and area proxies

Source construction and floor-area intervals are retained as bounds and midpoints. Construction “unknown” (6,128 households) remains missing. Before-1950 age is capped at 150 years; area below 40 has assumed lower bound 10 m² and area 300+ has assumed upper bound 400 m². These are proxy assumptions. Separate seeded inclusive uniform draws are included (age seed 42; area seed 43; full raw row order), without claiming to recover the original unseeded random stream. The probability workflow may use the bounds with its declared sampling procedure.

`input_provenance.json` records source hashes, original-model assumptions, counts and validations. `build_inputs.py` writes only this new input directory.
