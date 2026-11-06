# GRINS Energy Studio 0.5

The new **independent schedule benchmark** gives EnergyPlus household schedules that do not come from ISTAT. It separately tests transfer with known aggregates and transfer with ISTAT-predicted profiles. Built-in households are synthetic, not measured validation cases. See [INDEPENDENT_BENCHMARK.md](INDEPENDENT_BENCHMARK.md) for configuration, equations, results and hourly CSV import. The v0.5 EXE opens this new configuration; the shared-ISTAT mode and its earlier outputs remain available.

Estimate Palermo from one EnergyPlus run in Milan, using target weather, exposure settings and ISTAT household information. The default hybrid is a reference-fitted energy signature, not a 1R1C simulation.

## Click and run

Launch GRINS_Energy_Studio.exe with its _internal folder alongside it.

For the independent benchmark, keep the v0.5 packaged defaults and follow INDEPENDENT_BENCHMARK.md. The steps below describe the earlier workflow: select `shared_istat` under Benchmark household inputs first.

1. Data & execution: choose the household CSV and EnergyPlus 25.1 executable. Keep the hybrid method climate_exposure_gains.
2. Keep “Simulate every scenario for benchmarking” unchecked to predict with one reference simulation. This is now the default. Check it only to obtain target EnergyPlus comparison results.
3. Weather & exposures: select the EPWs, exposed walls, roof/floor boundaries and reference case. Default reference: **Milano / Multi-family / A2**, 100 m².
4. Clustering: choose input factors, H/E group counts and fixed/elbow selection. Defaults: H4 without insulation; E5 retaining cooling-system presence.
5. Save configuration and click Run experiment. Each run gets a new folder.
6. To use an existing compatible simulation, choose its completed run folder in “Reuse reference”. The program verifies the exact reference model and weather before reusing it. With benchmarking off, this makes zero new EnergyPlus simulation runs.

Target weather is still required when target simulations are off. It supplies climate features, not target simulation results.

## Three methods

- **EnergyPlus:** simulate each configured case, if benchmarking is enabled.
- **Hybrid:** fit the weather response of one reference simulation; adjust its balance temperatures using ISTAT operating hours and household/appliance gains; apply the target climate and exposure. Output annual sensible heating/cooling kWh. Appliance electricity uses the separate survey-ratio transfer.
- **ISTAT only:** report supported heating/cooling hours and reconstructed appliance kWh. Thermal kWh remain unavailable because hours alone do not identify energy.

The older pure ISTAT-ratio hybrid is retained as a clearly labelled baseline. It can also be selected explicitly. Neither hybrid runs a target thermal solver. Neither creates target hourly or monthly load profiles.

See HYBRID_METHOD.md for the equations, data boundaries and caveats.

## Local IDE and command line

Use the repository virtual environment:

~~~powershell
.\.venv\Scripts\python.exe run_energy_studio.py
.\.venv\Scripts\python.exe run_energy_studio.py --write-default my_config.json
.\.venv\Scripts\python.exe run_energy_studio.py --config my_config.json --run
.\.venv\Scripts\python.exe run_energy_studio.py --refresh-report PATH_TO_COMPLETED_RUN --with-powerpoint
~~~

The EXE accepts the same arguments. Report refresh creates a new revision under reports/ without rerunning simulations or changing saved scientific results. Schema-1/2 configurations must be reset to schema-3 defaults and settings reapplied; old RC/ratio reports are not relabelled.

## Where to change things

| File / setting | Purpose |
| --- | --- |
| config.py / default_config.json | Defaults and validation; all parameters are in saved JSON |
| GUI Clustering / analysis.group_factors | Variables defining H/E grouping cells |
| analysis.electricity_include_cooling | Independently retain cooling-system presence for E groups |
| analysis.heating_groups / electricity_groups | Fixed group counts; group_selection selects fixed or elbow |
| GUI Data & execution / hybrid.adjust_* | Switch exposure, hours, internal-gain and solar corrections |
| hybrid.*balance* / solar_step / solar_coefficient_max | Reference fitting search bounds and resolution, in Advanced JSON |
| analysis.py | Weighted training-only archetypes, H/E clustering and supported joint-cohort indices |
| signature.py | Reference fit, adjusted degree sums, annual thermal transfer and component checks |
| transfer.py | Original ratio baseline and appliance transfer |
| reuse.py | Exact-model/weather compatibility checks for saved reference runs |
| engine.py | Reference-first workflow, frozen predictions, optional benchmark and provenance |
| energyplus.py | Detailed single-zone geometry, schedules and ideal sensible loads |
| physics.py | Weather, geometry and solar helpers; legacy RC function is never used by the application |
| plots.py / report_deck.mjs | Scientific figure data, PNG/SVG/CSV and editable PowerPoint |
| gui.py | Controls, run/cancel and configuration handling |

To exclude insulation, leave insulation_improvement unchecked. Missing values of selected grouping factors are excluded. The insulation field describes reported improvements, not measured current U-values. Changing archetype count requires refit mode; new labels do not automatically match published A1–A4.

Archetype overrides change the simulation inputs, not the recorded survey indices. For the hybrid, the reference fit uses the actual reference settings, while targets use supported ISTAT indices. Changing the reference model invalidates reuse; clear the reuse folder and simulate a new reference.

## Saved evidence

Open report.html or presentation/run_report.pptx. Each run includes:

- annual_results.csv; comparison_metrics.csv; aggregation_metrics.csv.
- scenario_profiles.csv with separate simulation inputs and ISTAT indices/support.
- signature_fit.json and reference_signature_daily.csv: reference-only fit and diagnostics.
- signature_transfer_audit.csv: every balance shift, gain, conductance and degree ratio.
- transfer_audit.csv: baseline/appliance ratios and coverage.
- hybrid_predictions_frozen.csv and signature_ablations_frozen.csv, saved before target benchmarking and checked by hash.
- signature_component_metrics.csv: sequential weather/exposure/hours/gains comparisons, evaluated only after predictions.
- config.json and manifest.json: paths, input hashes, assumptions, timing and reuse provenance.
- plots/: comparison figures, points-only error scatterplots, Palermo/Roma plots, reference fit and components; PNG, SVG and CSV.
- energyplus/: model and hourly results for cases actually simulated or the reused reference.

Each added correction is also evaluated in a fixed order with the same reference fit. These comparisons are order-dependent, not causal effect estimates.

## Boundaries and dependencies

Results are annual sensible ideal zone demand, not purchased fuel/electricity. They omit HVAC efficiency, latent conditioning, DHW and cooking. Ground temperature, daily schedules, adiabatic neighbours and solar projection are assumptions. All scenarios share the configured floor area. Appliance kWh are reconstructed per household, not metered.

The reference fit needs hourly output (aggregated to daily); one annual reference total alone cannot identify it. Reuse currently supports completed Energy Studio runs, not arbitrary third-party EP exports.

The Italian IGDG EPWs come from the EnergyPlus weather database; they are not observed 2021 weather. Survey locations are macro-regions, not city observations. Cooling indices describe reporting users with valid hours.

The scientific engine and GUI are bundled. EnergyPlus 25.1 remains a separate installation. PowerPoint export uses this workstation's configured Node/artifact-tool/Python runtime; that runtime is **not bundled for redistribution**. Disable PowerPoint for HTML/CSV-only operation elsewhere, or configure a compatible presentation runtime. Update file paths when moving the app.

The reported benchmark is exploratory model-to-model agreement for this scenario matrix, not validation against real buildings or a previously untouched external test set. Target EP outcomes are excluded from fitting and prediction; the matrix was examined during method development. Appliance agreement is not independent validation because the models share its reconstructed input.

No files are committed automatically. Existing runs, earlier packages and the main branch are left unchanged.
