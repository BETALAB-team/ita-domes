# Replicability audit and 2013/2021 comparison

The article is **partly numerically reproducible**, and its broad workflow can be rebuilt. It is not currently an exact, scientifically verified raw-data reproduction. The strongest positive evidence is that the saved original probability table reproduces the first three Table 4 regressions at the published F-statistic precision. The strongest concern is that the matching demographic reconstruction preserves confirmed age and education errors. Reproducing a faulty calculation and producing a defensible new analysis are different objectives.

This audit reran both raw releases and the 2013 energy stage in this folder's `reconstructed/` directory. It added a comparison with consistent coding, official survey weights, a common household-size restriction, and a frozen 2013 reference model. Existing production scripts, configurations, Word reports, and outputs were not overwritten.

## Evidence and provenance

- [Article, all 15 pages](<C:/Users/khajmoh18975/Downloads/1-s2.0-S0378778825006541-main (1).pdf>), especially Sections 3.1-3.3 and 4.1-4.4, Equations 1-3, Tables 2-7, and Figures 2-9.
- [Public code cited in the article](https://github.com/BETALAB-team/ita-domes/tree/1b5972b1e052f00a2fdd9fa30d46bb442205c83d/resources/istat_data/UserClusters/istat_data), inspected at commit `1b5972b1e052f00a2fdd9fa30d46bb442205c83d`. Its household and clustering scripts contain the same defects found locally; this is not evidence of the exact unpublished execution that created every paper result.
- Local git history: `e7094f4` (2024-10-03) contains `classification_analysis_bayes.py`, including the older empirical window routine. `e07cd69` is the 2025 analysis snapshot; `a129009` contains the 2026 reconstruction. Later local changes were already present before this audit.
- ISTAT's supplied 2013/2021 classification HTML files and questionnaires define the age, education, household-size, construction, floor-area, and heating-question mappings. The 2021 methodological note identifies the survey frame, mixed CAWI/CATI collection, final weights, and 2020 reference year for consumption.
- Input paths and SHA-256 hashes, assumptions, results, and output hashes are recorded in [audit_results.json](C:/Works/GRINS/GRINS-Tool/resources/istat_data/replicability_outputs/audit_results.json) and [manifest.json](C:/Works/GRINS/GRINS-Tool/resources/istat_data/replicability_outputs/manifest.json). No private survey records were sent to a remote service.

## What can be recovered

### Four archetypes

The current 2013 reconstruction retains 17,855 of 20,000 households and satisfies all 40 declared Table 3/Figure 5 tolerances. Its unweighted shares are 40.129%, 26.777%, 22.767%, and 10.328%, close to the paper's 40.1%, 26.9%, 22.9%, and 10.1%. This is approximate numerical recovery under legacy assumptions, not exact agreement or proof of methodological validity.

The effect of changing only preprocessing, while keeping the published component equations and the same k-means seed, is:

| Scenario | Retained | Paper checks / 40 | Reassigned after best label alignment (%) |
| --- | --- | --- | --- |
| Current legacy coding | 17855 | 40 | 0 |
| Education recoding fixed | 17855 | 20 | 19 |
| Age intervals fixed | 17855 | 16 | 26.92 |
| Both recodings fixed | 17855 | 20 | 22.38 |

The age-only and both-recoding scenarios use the full published age intervals and an explicit upper limit of 100 for the open-ended 75+ group. Those scenarios retain the legacy class-code education denominator to isolate recoding effects. They are diagnostic alternatives, not a uniquely identified correction. Label agreement uses the best permutation of labels on shared household IDs.

![Preprocessing sensitivity](C:/Works/GRINS/GRINS-Tool/resources/istat_data/replicability_outputs/preprocessing_sensitivity.svg)

*Figure 1. Unweighted 2013 cluster shares under alternative preprocessing. The dashed line is the published Figure 5 reference; cluster membership and meaning may change between scenarios.*

### Probability regressions: a substantial recovery, with a clear discrepancy

The file `probablities_only.csv` contains 350 empirical windows. The saved analysis restricts it to `total_number >= 5`, `age < 150`, and `area < 150`, giving 125 observations on a 25-by-5 grid. Fitting the sparse terms printed in Table 4 gives:

| archetype | f_statistic | paper_f | adjusted_r2 | paper_adjusted_r2 | rmse |
| --- | --- | --- | --- | --- | --- |
| 1 | 648.9 | 648.9 | 0.94 | 0.94 | 0.02048 |
| 2 | 571 | 571 | 0.9324 | 0.93 | 0.01694 |
| 3 | 282.8 | 282.8 | 0.8721 | 0.87 | 0.008563 |
| 4 | 326.7 | 219.7 | 0.8874 | 0.84 | 0.009601 |

The first three F-statistics reproduce the paper at its printed precision. For A4, the printed terms do not reproduce its coefficients or F-statistic. The corresponding recovered predictions are included separately; the fourth model has not been relabelled as an exact match.

The archived routine uses age centers spaced 5 years apart, area centers spaced 24 m² apart, and **half windows of 15 years and 40 m²**. The paper instead describes 20-year/40-m² bins and window widths of 6 years/8 m². The reconstruction adopted a 63-window grid with half widths 6 and 8, then fitted a sample-count-weighted full quadratic model. These are materially different datasets and regressions. My earlier claim that the reconstructed probability graphs replicated the paper was too strong.

Using the older grid directly with raw 2013 dwelling data, saved original cluster labels, and a new fixed random seed produces 350 windows. Its mean absolute probability difference from the saved table is 0.0280. That is a partial raw-data recovery, not an exact rerun: the source random realization is not recorded, and the original cluster assignments remain an input. Results are in [historical_regression_recovery.csv](C:/Works/GRINS/GRINS-Tool/resources/istat_data/replicability_outputs/historical_regression_recovery.csv).

![Recovered original probability regressions](C:/Works/GRINS/GRINS-Tool/resources/istat_data/replicability_outputs/historical_probability_recovery.svg)

*Figure 2. Fits to the original saved 125-window subset. A1-A3 recover the paper's reported F-statistics. These are in-sample fits to overlapping, smoothed proportions, not independent household predictions; high R² is not evidence of equivalent out-of-sample predictive accuracy.*

### Energy analysis

The rerun agrees with 25 of 28 Table 5 significance decisions. It yields 8 heating groups and 4 electricity groups. The paper reports seven heating groups, while its text says six electricity groups and Table 6 lists five. The current deterministic grouping is computationally reproducible but does not reproduce all published group definitions.

The electricity response is model-derived appliance kWh from `results_grins.csv`, not metered household consumption. This audit verifies its existing ID join; it does not regenerate MODENA's appliance estimates from the full external model/database chain. The 2021 data do not supply the same modeled response, so no 2013-versus-2021 kWh comparison has been manufactured.

## Problems in the original version and the reconstruction

| Layer | Finding | Consequence |
| --- | --- | --- |
| Original demographic code | In-place education recoding changes primary code 3 to 5, then changes all 5s to 12. Primary schooling is effectively assigned 12 years. | The error changes the education geometry and roughly 19% of assignments when corrected alone. The reconstruction explicitly preserved it. |
| Original age conversion | The 15-24 band is sampled only at 20-24; the under-15 class only at 11-14; the open 75+ band only at 80-84. | Systematic age distortion and falsely narrow ages, especially in the elderly cluster. |
| Education denominator and exclusions | The old denominator receives age-class codes rather than realized ages. The current reconstruction excludes 2,050 households for undefined education and 95 for zero members. | Sample selection is consequential. The 2021 path uses a different denominator and excludes 6+ households instead. |
| Original executable chain | Public `02_Families_Analysis.py` uses `.at[i]` on an ID-indexed frame starting at 1 and never writes the required `family_trees.csv`; `03_PCA_Clustering.py` requests 20 clusters and consumes that saved file. | The numbered files are not a clean, complete executable record of the four-cluster paper. Earlier scripts contain alternate paths, so these defects alone do not identify the exact run used for publication. |
| Naive Bayes replacement | `05_02_PFE_Probablity.py` writes `probabilities_only.csv` with held-out `Prob_*` columns, while the next script reads `probablities_only.csv` and expects age, area, counts, and `prob0`-`prob3`. | Both a filename mismatch and a changed statistical target broke the chain. |
| PCA and clustering | Published equations are hard-coded for clustering; the reported exploratory PCA is calculated separately. Seeds, full loadings, centering, and initialization details are not fully specified in the paper. | Applying the equations is reproducible; calling it a refit of the paper's exact PCA is not justified. In Figure 2, the final cumulative values shown for all listed components are below 100%, another item requiring the original plotting data. |
| Probability reconstruction | The 63-window grid differs from the recovered 350-window source; 12 of its 2013 windows and 9 of its 2021 windows have fewer than 5 observations. Algorithm 1's iterative categorical-variable removal and regrouping is absent. | It cannot substantiate the paper's variable-selection result or replicate its reported fit merely because its probabilities sum to one. |
| Probability outputs | The empirical count probabilities sum to one, but the Word plotting regression clips each fitted probability independently. | The 2013 fitted sums deviate from one by as much as 0.01334. Counts and regression outputs need separate validation. |
| Previous figure replica | The standards-comparison figure places energy-group boxes at evenly spaced positions on a dwelling-area axis. These positions are not observed group areas. | Visual similarity is not a data-faithful reproduction of a relationship with dwelling size. |
| Multi-year comparison | The existing runner changes education coding, household eligibility, min-max limits, and independently fits centers for each wave, then gives them the same labels. | Its concatenated summary cannot be read directly as temporal change in fixed archetypes. |
| Weighting | The paper explicitly reports unweighted sample proportions in Section 4.2. The two surveys have different frames and modes. | Unweighted changes do not establish national demographic change. Applying official weights helps but cannot fully remove mode/frame differences. |
| Inference | Overlapping probability windows reuse households. Repeated data-dependent Tukey merging uses the same response for selection and subsequent significance testing. | Nominal regression p-values and one-round Tukey family-wise guarantees do not validate the complete adaptive process. |

## Inconsistencies in the article itself

Equation 3 gives A4 an age-squared coefficient of 0.152, while Table 4 prints 0.02. At building age 100 years and area 100 m², Equation 3 gives probabilities `[0.525, 0.173, 0.204, 0.238]`, summing to **1.14**. At age 20 and area 360 m² it gives a negative A1 probability. The latter is outside Figure 6's displayed domain and demonstrates that the polynomial must not be extrapolated unrestrictedly.

Table 4's F-statistics can partly be recovered as shown above, but A4 remains unresolved. Figure 6's horizontal axis is dwelling area even though its caption describes age changing. Table 5's electricity interaction decisions and F-values conflict with the surrounding text; Table 6 repeats the same example for H6 and H7; the text says six electricity groups while the table lists five. Table 7 prints a -60% ISO deviation for E5, but `(1717-2895)/2895` is -40.69%. These are reporting inconsistencies, not discrepancies that code should conceal by tuning results.

### Regional effects depend on the unit of analysis

Applying the Section 3.1 description literally at the household level (`PC1 + PC2 + education ~ region`) gives:

| Statistic | Value | Num DF | Den DF | F Value | Pr > F |
| --- | --- | --- | --- | --- | --- |
| Wilks' lambda | 0.9663 | 57 | 5.317e+04 | 10.8 | 4.706e-94 |
| Roy's greatest root | 0.0271 | 19 | 1.784e+04 | 25.44 | 3.482e-89 |

For comparison, applying the same test to four fitted centroids per region, with and without conditioning on archetype, gives:

| Statistic | model | Value | Num DF | Den DF | F Value | Pr > F |
| --- | --- | --- | --- | --- | --- | --- |
| Wilks' lambda | centroids_region_only | 0.8402 | 57 | 173.8 | 0.1833 | 1 |
| Roy's greatest root | centroids_region_only | 0.1146 | 19 | 60 | 0.3618 | 0.9918 |
| Wilks' lambda | centroids_region_adjusted_for_archetype | 0.5296 | 57 | 164.8 | 0.687 | 0.9488 |
| Roy's greatest root | centroids_region_adjusted_for_archetype | 0.4873 | 19 | 57 | 1.462 | 0.1359 |

These are diagnostic, unweighted tests, not survey-design-adjusted national inference. They test different hypotheses. A non-significant test that pools four very different archetype centroids within each region does not establish that household distributions or archetype prevalences are geographically identical. The exact Table 2 input table is needed to settle its intended interpretation.

## Comparison of the two actual datasets

The releases contain 20,000 and 54,069 records. The common analysis retains 19,688 and 53,520 households of size 1-5, respectively. The 2021 release exposes only five members and top-codes the sixth size category as 6+. Official `coef_red` and `PESO` weights are retained. Coverage and exclusions are in [sample_comparison.csv](C:/Works/GRINS/GRINS-Tool/resources/istat_data/replicability_outputs/sample_comparison.csv). Household IDs are local to each release; this is a repeated cross-section, not a panel of matched families.

### Direct observable comparisons

These survey-weighted quantities do not depend on randomized age values or fitted cluster boundaries. Percentages are conditional on the shared 1-5-member sample; qualification and construction measures use known-answer denominators, recorded in the CSV. Construction unknowns are reported separately, not assigned random ages. The open upper floor-area category has a small boundary ambiguity at exactly 150 m² in the 2013 label.

| measure | 2013 | 2021 |
| --- | --- | --- |
| construction_before_1950_pct | 17.3 | 13.95 |
| construction_unknown_pct | 4.191 | 14.52 |
| dwelling_at_least_150m2_pct | 10.5 | 12.38 |
| households_1_member_pct | 36.79 | 32.21 |
| households_2_member_pct | 23.76 | 30.43 |
| households_3_member_pct | 17.68 | 18.75 |
| households_4_member_pct | 18.1 | 14.76 |
| households_5_member_pct | 3.669 | 3.843 |
| households_with_child_under_15_pct | 22.01 | 18.5 |
| households_with_tertiary_educated_member_pct | 22.99 | 25.1 |
| mean_household_size | 2.281 | 2.276 |

These are descriptive differences, not causal estimates. The rise in unknown construction year affects the selection of households usable for dwelling-conditioned probabilities.

### Fixed-reference archetypes

The reference uses age-class midpoints; common education categories are assigned 5, 8, 12, and 18 years in both waves. No qualification and primary schooling must be combined because 2021 combines them; five years is an upper-bound assumption. Members with zero potential schooling are omitted from the education denominator. The 2013 unweighted model is chosen from ten starts by minimum inertia, then its scaling and centers are applied unchanged to 2021. There are 2,983 2021 households outside at least one 2013 min-max bound; their transformed values are retained without clipping.

| archetype | 2013 | 2021 | Change (percentage points) |
| --- | --- | --- | --- |
| 1 | 26.98 | 25.06 | -1.92 |
| 2 | 36.01 | 37.52 | 1.508 |
| 3 | 20.95 | 23.64 | 2.683 |
| 4 | 16.05 | 13.78 | -2.271 |

The A1-A4 names here describe this new reference model, not guaranteed one-to-one replicas of the paper's archetypes. The existing workflow's A1 comparison is 40.13% to 25.68% (unweighted and independently refitted); this controlled example is 26.98% to 25.06% (weighted and fixed). The difference between those comparisons combines coding, eligibility, weighting, and model-definition effects and must not be attributed to one correction alone.

![Archetype comparison](C:/Works/GRINS/GRINS-Tool/resources/istat_data/replicability_outputs/archetype_comparison.svg)

*Figure 3. Existing unweighted separate fits versus the common weighted fixed-reference analysis. Each panel has its own model definitions.*

Across ten age/education scenarios, after aligning reference labels on shared 2013 IDs, the changes span:

| archetype | min | max |
| --- | --- | --- |
| 1 | -2.708 | -1.521 |
| 2 | -1.855 | 5.919 |
| 3 | -2.494 | 5.299 |
| 4 | -2.271 | 1.667 |

*These are scenario ranges in percentage points, not confidence intervals. A2 reverses direction between midpoints and random draws; A3 reverses under the open-age-cap scenario; A4 reverses under alternative no/primary education assumptions. Precise temporal archetype trends are not robust enough to claim from this exercise.*

![Assumption sensitivity](C:/Works/GRINS/GRINS-Tool/resources/istat_data/replicability_outputs/comparison_assumption_sensitivity.svg)

*Figure 4. Scenario sensitivity of the cross-wave share changes. Blue ranges are not sampling uncertainty intervals.*

### Dwelling probabilities and prediction across waves

The new comparison uses 24 common construction-period/floor-area cells. It uses observed category membership, rather than generating precise ages and areas from differently sized intervals. Each cell records sample count, effective sample size, four weighted probabilities, and a support flag. Construction periods hold building vintages comparable; the '1990+' group naturally also includes post-2013 buildings in 2021. A missing construction year is excluded from this conditional analysis.

![Dwelling-conditioned probability changes](C:/Works/GRINS/GRINS-Tool/resources/istat_data/replicability_outputs/dwelling_probability_changes.svg)

*Figure 5. Weighted changes in conditional archetype probability, 2021 minus 2013. Asterisks identify cells below 30 records or effective observations in either wave. All cells use the midpoint reference model; the age/education sensitivity still applies.*

As an independent prediction check, the conditional probability table is trained on 80% of eligible households and scored on the remaining 20%, then the 2013 table is transferred to 2021. Five average-weight pseudo-households shrink each cell towards the training marginal. Brier scores sum the squared errors of all four probabilities; lower is better. Survey weights are used in fitting cell proportions and evaluating scores.

| evaluation | area_and_construction_period | training_marginal_only | Brier improvement (%) |
| --- | --- | --- | --- |
| 2013_household_holdout | 0.6927 | 0.7293 | 5.012 |
| 2013_to_2021_external_wave | 0.721 | 0.7277 | 0.9212 |
| 2021_household_holdout | 0.7073 | 0.7271 | 2.717 |

This tests prediction from the common dwelling categories conditional on a frozen archetype definition. The clustering itself used the 2013 demographic sample, so this is not nested validation of the entire discovery pipeline. It is a simple probabilistic comparator, not a reproduction of Algorithm 1. A modest gain over the marginal baseline is more informative about practical predictive value than R² on overlapping fitted proportions.

### Heating hours

The previous runner skips all 2021 energy analysis, but the raw release contains `v3_8a`, `v3_8b`, and `v3_8c`, corresponding to the three eight-hour bands in 2013 `q_3_11A/B/C`. The routing differs: 2013 asks hours for frequency categories 1/2, 2021 for 1/2/3. This comparison restricts both to 1/2, with all three hours present and in 0-8. It has 18,332 and 49,545 observations. Row 0 below is the overall eligible group.

| archetype | 2013 | 2021 | Change (hours) |
| --- | --- | --- | --- |
| 0 | 7.899 | 8.545 | 0.6459 |
| 1 | 8.311 | 9.045 | 0.7333 |
| 2 | 7.575 | 8.323 | 0.7474 |
| 3 | 7.705 | 7.957 | 0.2517 |
| 4 | 8.208 | 9.184 | 0.9759 |

![Heating comparison](C:/Works/GRINS/GRINS-Tool/resources/istat_data/replicability_outputs/heating_comparison.svg)

*Figure 6. Weighted mean winter-day operating hours for the matched frequent-use sample and fixed reference archetypes. These are neither annual heating energy nor means over every surveyed household. The 2021 answers explicitly reference winter 2020; pandemic conditions, survey mode/frame, climate, and equipment changes are not controlled.*

## Recommended next implementation

1. Preserve a documented legacy-reproduction mode, including the recovered probability intermediates and exact historical settings. Label paper matches and unresolved discrepancies separately.
2. Implement a scientifically revised preprocessing mode using the codebooks, explicit missing-value rules, shared education categories, configurable open-ended age assumptions, saved transformations, and stable reference centers. Do not silently change the legacy baseline.
3. Use valid joint probabilities and household-level validation for the dwelling model. Carry uncertainty from coarse inputs and sparse cells. Revisit the regional hypothesis and adaptive energy grouping with explicit analysis units and validation data.

The present comparison is a concrete audit and reproducible prototype of those choices. It does not claim that a single repaired implementation has been scientifically validated or that the paper's conclusions have all been replicated.

## Re-run and outputs

From `resources/istat_data`:

```powershell
.\.venv\Scripts\python.exe audit_replicability.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The runner accepts `--config` and `--output-dir`; all new results default to `replicability_outputs/`. Main tables include `archetype_comparison.csv`, `direct_survey_comparison.csv`, `heating_comparison.csv`, `dwelling_probability_comparison.csv`, `heldout_probability_scores.csv`, `historical_regression_recovery.csv`, `comparison_sensitivity.csv`, and `preprocessing_sensitivity.csv`. All six plots have SVG and PNG versions. This report is regenerated from the audit outputs.
