# Version 0.1 acceptance checks — 29 September 2026

**Historical record only.** These checks describe the earlier RC-based implementation, not the version 0.3 ISTAT-ratio hybrid. Do not use the errors below as evidence for the new method. See VALIDATION_V0_3.md for current checks.

The packaged Windows EXE was executed, not only its Python sources.

## Default experiment

Run: `../energy_studio_runs/20260929_140859_a59dda`

- Packaged GUI self-test: exit 0 (five tabs, parameter round-trip).
- Packaged complete experiment: exit 0; manifest status `complete`.
- 100 m², three climates, three exposure configurations, four archetypes.
- Heating: four groups; electricity: five groups, including cooling presence.
- 36 annual EnergyPlus 25.1 simulations; zero EnergyPlus warnings.
- Annual/hourly/monthly results, configuration, input/weather hashes and model files saved.
- Eight-slide PPTX: four editable charts and four native table slides.
- PPTX package and layout validators: zero findings, zero layout warnings.
- Rendered report slides visually inspected; no clipping or overlapping content found.
- Total experiment/report wall time on this workstation: approximately 87 seconds.

## Comparison with EnergyPlus

The 35 non-reference scenarios give:

| Method | End use | RMSE, kWh/year | CVRMSE |
| --- | --- | ---: | ---: |
| One-run hybrid | Sensible heating | 448.9 | 19.9% |
| One-run hybrid | Sensible cooling | 148.4 | 12.4% |
| Survey + physics | Sensible heating | 1264.8 | 56.1% |
| Survey + physics | Sensible cooling | 1031.6 | 86.1% |

This is an exploratory model-to-model comparison. The shared appliance electricity
input is not independent validation. No claims about measured-building accuracy,
optimal clusters, legal compliance or transferability to arbitrary buildings follow.

## Other checks

- Seven unit tests passed, covering exact partition optimality, annual appliance
  conservation, geometry/exposure, schedules, disabled cooling, model structure,
  electricity-only cooling grouping and invalid inputs.
- Source GUI self-test passed.
- A refit using three archetypes, three demographic features, two PCA components,
  altered grouping factors and 120 m² completed in strict-data-only mode. Thermal
  kWh were missing as intended, appliance results present, zero EnergyPlus runs.
  Run: `../energy_studio_runs/20260929_140908_65d9a2`.
- Reference-only hybrid mode completed four scenarios with exactly one EnergyPlus
  run. Run: `../energy_studio_runs/20260929_140917_0d8913`.

## Packaging boundary

The scientific Python runtime is bundled. The repository's household CSV and
downloaded EPWs are referenced by configuration. EnergyPlus 25.1 and the local
PowerPoint-generation runtime remain external dependencies. This package is
configured for this workstation, not a dependency-free installer for other PCs.

Earlier trial run folders remain for traceability; the completed default run
identified above is the acceptance example. No repository commits were made.
