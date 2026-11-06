# Post hoc v2 peak context: review 05 reporting correction

This diagnostic was added after R5-02, whose reported counts were known before the independent author-side recomputation. It reads all 144 scientific raw hourly files directly: 128 held-out respondent/site files and 16 component/site files, 1,261,440 rows. It reads no producer endpoint summaries to locate maxima and imports neither the engine nor the frozen analysis code. The eight technical repeat/warm-up cases are excluded. It runs no annual simulation and changes no input, endpoint, maximum, primary result or frozen code.

Under Bolzano exposure, **27 of the 32 held-out respondents** and **two of the four shared components** attain their individual annual maximum hourly heating load on the prescribed restart day, model hours **6912–6935** (16 October in the study calendar). **18 of the 32 respondents** and **both of those components** peak exactly at **hour 6912**. The other five held-out respondents and two components have their annual maximum in January. The denominators remain 32 respondents and four components; they are not combined into a 36-person population.

This concerns the new heterogeneous archetype campaign v2. It is distinct from the earlier v5 single-respondent timing pilot's three equal Bolzano maxima. All primary annual maxima remain intact. The mixture endpoint still means the expectation of individual annual maxima from separately simulated components, not the maximum of an averaged load curve.

| Coordinate TMY | Held-out maxima on restart day | Held-out maxima exactly at restart hour | Component maxima on restart day | Component maxima exactly at restart hour |
|---|---:|---:|---:|---:|
| Padova | 2 / 32 | 2 / 32 | 0 / 4 | 0 / 4 |
| Bolzano | 27 / 32 | 18 / 32 | 2 / 4 | 2 / 4 |
| Roma | 0 / 32 | 0 / 32 | 0 / 4 | 0 / 4 |
| Palermo | 0 / 32 | 0 / 32 | 0 / 4 | 0 / 4 |

Each raw case has one strictly positive annual maximum with no exact numerical ties. Therefore first-maximum, any-tied-maximum and all-tied-maxima classifications agree. The script nevertheless preserves all exactly equal positive maximum hours and handles a hypothetical all-zero heating case without inventing a peak. No tolerance or alternate peak-selection threshold was fitted to the result.

The calendar is 2023 nonleap with fixed UTC+1 and no daylight saving. The imposed climate-E seasonal profile resumes at model hour 6912, day index 288; the preceding hour has the standardized off-season lower setpoint. This is descriptive coincidence with an imposed recovery boundary. It does not quantify a causal restart effect, establish local regulatory operation, remove winter maxima, validate observed operation or support equipment sizing. The Bolzano peak-loss and subgroup results should be interpreted with this fixed-season context adjacent to them.

New evidence is separate from the frozen primary analysis:

- `04_analysis/archetype_extension/posthoc_peak_context_v1/scientific_peak_context.csv`: every scientific file's independently found maximum, hour, model calendar, source role, temperature context and exact-tie classification.
- `04_analysis/archetype_extension/posthoc_peak_context_v1/cohort_counts.csv`: counts with explicit 32-test/four-component denominators for each site.
- `04_analysis/archetype_extension/posthoc_peak_context_v1/input_file_hashes.csv`: all raw scientific input byte hashes and frozen role/selection hashes.
- `04_analysis/archetype_extension/posthoc_peak_context_v1/diagnostic_manifest.json`: source/plan/output hashes, exact command, scope and post hoc status.
- `03_methods/archetype_extension/posthoc_peak_context_plan.md`: written before this recomputation, explicitly acknowledging prior knowledge of the review's counts.

The claim author's separate Bolzano raw-hourly check reports the same four counts (27/32, 2/4, 18/32 and 2/4). That agreement is an arithmetic cross-check, not empirical validation. Current authoritative executed-v2 gates are in validation_matrix.md; its retained v1 table is explicitly historical, superseded and unexecuted.
