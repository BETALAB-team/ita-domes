# Corrected forcing and stepping outputs — v4

Status: adapter-local correction passed declared transition checks and full annual step-balance checks. Independent scientific review of the correction remains pending. No empirical validation has been completed. All earlier 45 annual outputs are quarantined: v1/v2 have dispatcher and sky-forcing defects; v3 fixes stepping but retained invalid opaque-sky forcing. Use only v4 tables for current conditional simulation claims. The valid horizontal IR now supplies effective sky temperature and the native annual-mean longwave scalar; no cloud filling is used.

Twenty corrected annual runs were executed: twelve primary cases (four coordinate TMYs × three placements), four repeated seeded cases and four longer-warm-up seeded cases. Corrected Padova completed first; independent balance/integration checks passed before the other three sites launched, recorded by expansion_gate.json. Across all attempts, 66 annual runs were executed: 45 quarantined, 20 current campaign cases, and one separately identified genuine seeded replay in a fresh directory. Replay matches the current Padova seeded hourly file byte-for-byte and is not an extra scientific sample.

The assumed 80 m² dwelling, physical parameters, one 2013 Veneto response, source-derived 22/19°C mapping, 4/4/1 high-setpoint hours, season indices and weather inputs match v3. The documented forcing correction replaces the invalid cloud-derived scalar with the mean air-minus-effective-sky temperature derived from valid horizontal IR. Native annual-mean longwave and solar-time approximations remain limitations. The retained local stepping correction uses: free/constrained trials use the same old state, and the selected solution is committed once. Moisture is uncoupled and not stepped/reported; the reported boundary remains useful sensible space-heating demand with no final-energy, latent-load, DHW, carrier, emission, comfort or retrofit result.

| Coordinate case | Placement | Annual useful sensible heating kWh | Peak hourly sensible heating kW | Annual change % | Peak change % |
|---|---|---:|---:|---:|---:|
|Padova|seeded|9789.026|9.9760|0.0000|0.0000|
|Padova|early|9870.890|10.1728|0.8363|1.9730|
|Padova|late|9819.111|9.7655|0.3073|-2.1094|
|Bolzano|seeded|15940.680|17.5978|0.0000|0.0000|
|Bolzano|early|15928.677|17.5978|-0.0753|0.0000|
|Bolzano|late|15943.412|17.5978|0.0171|0.0000|
|Roma|seeded|6870.281|8.8718|0.0000|0.0000|
|Roma|early|6987.843|9.5803|1.7112|7.9855|
|Roma|late|6899.130|9.1631|0.4199|3.2828|
|Palermo|seeded|5368.087|8.2425|0.0000|0.0000|
|Palermo|early|5537.048|8.6897|3.1475|5.4250|
|Palermo|late|5408.765|8.3322|0.7578|1.0885|

Active forcing fields and IR/JSON mapping passed strict QC. Palermo has one preserved invalid wind record, unused in this guarded fixed-ACH pathway; the weather file is not claimed universally valid. Native annual-mean longwave and unadjusted solar-time offsets are explicit model assumptions. No empirical validation follows.
All 175,200 exported corrected hours close each air/surface/mass step balance within 1e-6 W; maximum absolute residual across nodes and hours is 1.746e-10 W. Exported old-state continuity is exact. All twenty files have 8,760 finite rows; all repeats are exact, warm-up comparisons meet the declared limits, active heating meets its setpoint, and cooling is zero. Independent standard-library CSV integration and peak checks agree within 1e-9 kWh and 1e-12 kW. These are numerical/physics verification results, not empirical accuracy.

The three placements preserve durations assigned the high setpoint, not guaranteed heater runtime. At the 19°C setback the ideal heater may remain active. The explicitly post-hoc partition below quantifies that property from corrected files; it is descriptive and did not replace predeclared endpoints. Early/late are neither observed schedules nor bounds over every allowable schedule. No placement is shown more accurate. A few coordinate TMYs and one assumed dwelling cannot establish household, municipal, national or universal transferability.

An independently specified whole-building benchmark, measured building validation, calibrated envelope/equipment characterization and historical-weather/meter matching remain unperformed. Candidate ANACI annual fields need provenance, meter boundaries and calibration-independence verification before becoming a validation target. Primary climate descriptors are unaffected by the correction.

Reproduction and source evidence: correction_protocol_v3.md; single_commit_1c.py; test_single_commit_1c.py; run_climate_campaign_v4.py; check_full_step_balances_v3.py; summarize_corrected_campaign_v3.py. Immutable per-site correction/protocol/config/design/adapter copies and hashes, source-package hashes, raw forcing/states, parameters, logs and full_step_balance_checks.json reside under 05_simulations/climate_campaign_v4. Avoid rerunning into an existing immutable run directory.