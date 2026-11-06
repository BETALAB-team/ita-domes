# Newly executed EUReCA results — 20 October 2026

**Provisional technical outputs:** independent review raised a potential recursive state-update defect in the timestep dispatcher after these runs. Scientific interpretation is on hold pending the reviewer-owned diagnostic. See `05_simulations/SCIENTIFIC_USE_HOLD.md`. Do not treat the following values as settled scientific findings.

All results are conditional simulation outputs for one assumed 80 m² dwelling and one actual 2013 Veneto response. No observed hourly schedules, meter validation, stock weights, retrofit scenarios or final-energy conversions enter this campaign. The three reconstructions preserve4/4/1 daily block hours.

20 new annual official-weather runs completed:12 primary (4 coordinate cases × 3 schedules),4 exact-repeat checks,4 longer-warm-up checks. The preceding local-weather Padova pilot adds 5 annual runs, for25 total new annual runs. Analytical network stepping is a separate verification task, not an annual case.

| Site | Reconstruction | Annual sensible heating kWh | Peak hourly sensible heating kW | Annual change vs same-site seeded % | Peak change % |
|---|---|---:|---:|---:|---:|
|Padova|seeded|6521.186|9.0053|0.0000|0.0000|
|Padova|early|6376.550|9.1512|-2.2179|1.6202|
|Padova|late|6412.556|8.8805|-1.6658|-1.3854|
|Bolzano|seeded|11334.015|10.8078|0.0000|0.0000|
|Bolzano|early|10979.062|10.7379|-3.1318|-0.6469|
|Bolzano|late|11283.141|10.6021|-0.4489|-1.9036|
|Roma|seeded|3868.069|7.9088|-0.0000|0.0000|
|Roma|early|3891.382|8.5613|0.6027|8.2506|
|Roma|late|3646.067|8.3200|-5.7394|5.1999|
|Palermo|seeded|2547.399|7.2019|-0.0000|0.0000|
|Palermo|early|2723.579|7.6123|6.9161|5.6989|
|Palermo|late|2434.124|7.3613|-4.4467|2.2135|

Actual summed engine-case runtime was 14.715 s, excluding Python imports, weather solar preprocessing and acquisition; wall-clock campaign runtime is not this sum. Full-precision data and source paths are in paired_schedule_contrasts_v2.csv. Independent standard-library CSV integration, peak recomputation and saved-file hash checks pass for all 20 outputs; integration discrepancy is below 1e-9 kWh. Every declared numerical gate passed, all repeats were exact, and all outputs had 8760 finite rows and zero active cooling.

The results show sensitivity to allowed within-block timing for this model/input combination. Early/late are neither observed alternatives nor bounds on every possible schedule. Same reported hours do not identify one deterministic annual load or peak. No schedule is shown to be more accurate, and differences across coordinate cases are conditional on fixed climate-E operating season and the same assumed dwelling. High summer free-floating temperatures arise because cooling is intentionally disabled; no comfort performance is established.

The analytical 1C network tests pass zero-gradient equilibrium (maximum error 9.71e−13), steady convective heating (3327.272727 W, solver−analytic absolute error 2.28e−12 W), and independently recomputed transient air/surface/mass residuals (largest <6e−12 W). This checks network algebra, not external benchmark performance, envelope characterization or empirical accuracy. A complete annual heat-balance residual from real campaign states and an independently specified building benchmark remain unperformed.

Provider files are TMY scenarios for 2005–2023, not historical-year matched validation. JSON reports elevations Padova 25 m, Bolzano 272 m, Roma 72 m, Palermo 29 m; these are provider coordinate/grid estimates, not surveyed building altitudes. Grid/horizon and source-period limitations remain. Local-v1weather provenance is retained separately; official v2 Padova numerical outputs happen to match it, which does not prove the original file retrieval history.

Reproduction: execute 04_analysis/audit_inputs.py,04_analysis/fetch_climate_inputs.py (network access),04_analysis/audit_climates.py, then04_analysis/run_climate_campaign_v2.py SITE with the recorded EUReCA .venv Python with -B, and finally 04_analysis/summarize_campaign.py. Config/design/protocol/adapter snapshots, complete source-package hashes, weather checksums, commands, logs and completion manifests are under 05_simulations. Do not rerun in place if preserving existing immutable results; copy the campaign configuration to a new run ID.
