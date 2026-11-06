# Version 0.3 acceptance checks — 29 September 2026

This release implements the one-reference EnergyPlus + ISTAT annual-ratio hybrid. No RC solver is called by its run engine.

## Executed checks

- 15 unit tests passed: the nine existing app/plot tests and six ISTAT-transfer tests.
- Source and packaged GUI self-tests passed (five tabs and parameter round-trip).
- Source full experiment: `energy_studio_runs/20260929_165015_232816`.
- Rebuilt EXE full experiment: `energy_studio_runs/20260929_170608_5853c1`; exit 0, manifest complete, 36 EnergyPlus runs, zero EnergyPlus warnings.
- Default reference: Palermo / Multi-family / A2; H4 without insulation, E5 with cooling-system presence; 100 m² and 36 configured scenarios.
- Reference-only integration: `energy_studio_runs/20260929_165208_ef4f35`; exactly one EP run, all 36 hybrid estimates, deliberately nonexistent target EPW paths accepted because target weather is not used.
- Source full/reference-only frozen predictions were byte-identical: SHA-256 `665e29cc28dccb24111ae234125634922312b06a3c8c70bafc86378a0240c01d`.
- Changed-configuration integration: `energy_studio_runs/20260929_165157_4912c1`; three refitted archetypes, three H/E groups, PCA, 120 m², no EnergyPlus. Thermal kWh remained missing.
- Tests confirm missing cooling hours are excluded, overrides cannot alter survey indices, unsupported cohorts have unavailable indices, zero denominators are flagged, and changing target benchmark outputs cannot change predictions.
- Legacy RC reports/configurations are rejected rather than relabelled.

## Annual benchmark accuracy

35 non-reference cases; sensible ideal thermal energy, not utility consumption.

| End use | MAE kWh/year | RMSE kWh/year | CVRMSE |
| --- | ---: | ---: | ---: |
| Heating | 1787.2 | 2500.9 | 110.3% |
| Cooling | 402.2 | 487.2 | 41.0% |

These substantial errors are retained. Daily-hour ratios alone do not capture annual climate severity or physical exposure effects. Appliance agreement is not independent validation because the same reconstructed input supplies the detailed model.

## Aggregation diagnostic

The errors above are annual, not hourly. Pooled signed bias and case-level WAPE are saved together in aggregation_metrics.csv. Results use equally weighted configured scenarios, not a representative national building-stock mixture.

| Scope | End use | Pooled bias | Case-level WAPE |
| --- | --- | ---: | ---: |
| All 35 non-reference cases | Heating | −76.0% | 78.8% |
| All 35 non-reference cases | Cooling | +21.6% | 33.9% |
| Palermo (11 cases) | Heating | −9.9% | 49.6% |
| Palermo (11 cases) | Cooling | −10.0% | 16.4% |
| Single-family across climates (12 cases) | Cooling | +3.1% | 21.9% |

Promising pooled cooling errors do not establish accurate individual predictions. The Palermo heating comparison particularly illustrates cancellation. These post-hoc subgroup checks are exploratory.

## PowerPoint

The aggregation report contains 26 slides, 37 native editable charts and eight native tables. All slides were rendered and visually inspected; the aggregation and interpretation slides were inspected at full size. Package integrity and geometry checks reported zero findings and zero layout warnings. Point-only scatter chart styles and empty speaker notes were checked. Native PowerPoint application rendering was not tested.

The source report revision is `energy_studio_runs/20260929_165015_232816/reports/20260929_170033_543b1f`. The actual packaged run generates the same report structure from its own outputs.

## Exploratory degree-day check (not enabled in version 0.3)

In response to the follow-up climate question, the existing annual predictions were multiplied by target/reference degree days calculated from the same EPW daily-mean temperatures. Simple HDD18 reduced heating CVRMSE from 110.3% to 80.2%; simple CDD22 reduced cooling CVRMSE from 41.0% to 33.5%. These use sum(max(18 − Tdaily, 0)) and sum(max(Tdaily − 22, 0)); they are not Eurostat's threshold definitions or observed survey-year weather.

Other fixed bases were also checked to expose sensitivity, not to select an optimum on the benchmark. This extension is not the shipped baseline and has not been independently validated. A future climate-adjusted option should separate weather severity from climate-related behaviour already embedded in survey cohorts.

No commits were made. EnergyPlus and the configured local PowerPoint runtime remain external dependencies.
