# Corrected new EUReCA simulation outputs — v3

Status: adapter-local correction passed declared transition checks and full annual step-balance checks. Independent scientific review of the correction remains pending. No empirical validation has been completed. All original 25 v1/v2 annual outputs remain quarantined because the original recursive dispatcher violates timestep state balances. Use only v3 tables for current conditional simulation claims.

Twenty corrected annual runs were executed: twelve primary cases (four coordinate TMYs × three placements), four repeated seeded cases and four longer-warm-up seeded cases. Corrected Padova completed first; independent balance/integration checks passed before the other three sites launched, recorded by expansion_gate.json. Across all attempts, 45 new annual simulations were executed: 25 quarantined and 20 corrected.

The assumed 80 m² dwelling, physical parameters, one 2013 Veneto response, source-derived 22/19°C mapping, 4/4/1 high-setpoint hours, season indices and weather inputs match v2. The local correction changes only state stepping: free/constrained trials use the same old state, and the selected solution is committed once. Moisture is uncoupled and not stepped/reported; the reported boundary remains useful sensible space-heating demand with no final-energy, latent-load, DHW, carrier, emission, comfort or retrofit result.

| Coordinate case | Placement | Annual useful sensible heating kWh | Peak hourly sensible heating kW | Annual change % | Peak change % |
|---|---|---:|---:|---:|---:|
|Padova|seeded|5947.725|8.8696|0.0000|0.0000|
|Padova|early|6078.714|9.0185|2.2023|1.6791|
|Padova|late|5988.123|8.6560|0.6792|-2.4078|
|Bolzano|seeded|10613.614|10.7792|0.0000|0.0000|
|Bolzano|early|10657.376|10.6289|0.4123|-1.3943|
|Bolzano|late|10632.361|11.0257|0.1766|2.2865|
|Roma|seeded|3325.654|7.7044|0.0000|0.0000|
|Roma|early|3521.934|8.3336|5.9020|8.1673|
|Roma|late|3373.496|7.9127|1.4386|2.7031|
|Palermo|seeded|2321.091|7.0785|0.0000|0.0000|
|Palermo|early|2475.231|7.5110|6.6408|6.1104|
|Palermo|late|2363.074|7.1023|1.8088|0.3372|

All 175,200 exported corrected hours close each air/surface/mass step balance within 1e-6 W; maximum absolute residual across nodes and hours is 2.024e-10 W. Exported old-state continuity is exact. All twenty files have 8,760 finite rows; all repeats are exact, warm-up comparisons meet the declared limits, active heating meets its setpoint, and cooling is zero. Independent standard-library CSV integration and peak checks agree within 1e-9 kWh and 1e-12 kW. These are numerical/physics verification results, not empirical accuracy.

The three placements preserve durations assigned the high setpoint, not guaranteed heater runtime. At the 19°C setback the ideal heater may remain active. The explicitly post-hoc partition below quantifies that property from corrected files; it is descriptive and did not replace predeclared endpoints. Early/late are neither observed schedules nor bounds over every allowable schedule. No placement is shown more accurate. A few coordinate TMYs and one assumed dwelling cannot establish household, municipal, national or universal transferability.

An independently specified whole-building benchmark, measured building validation, calibrated envelope/equipment characterization and historical-weather/meter matching remain unperformed. Candidate ANACI annual fields need provenance, meter boundaries and calibration-independence verification before becoming a validation target. Primary climate descriptors are unaffected by the correction.

Reproduction and source evidence: correction_protocol_v3.md; single_commit_1c.py; test_single_commit_1c.py; run_climate_campaign_v3.py; check_full_step_balances_v3.py; summarize_corrected_campaign_v3.py. Immutable per-site correction/protocol/config/design/adapter copies and hashes, source-package hashes, raw forcing/states, parameters, logs and full_step_balance_checks.json reside under 05_simulations/climate_campaign_v3. Avoid rerunning into an existing immutable run directory.