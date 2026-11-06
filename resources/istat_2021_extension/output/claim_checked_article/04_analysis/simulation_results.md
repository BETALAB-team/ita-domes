# Candidate independent-surface simulation outputs — v5

Status: reviewed conditional model illustration. Local surface, sky and stepping corrections passed declared internal checks; independent technical review 03 supplement passed; fresh full-claim review 04 and final v04 closure passed for the bounded conditional claims (07_reviews/final_closure.md). No empirical validation has been completed. All earlier 66 annual executions are quarantined: v1/v2 have dispatcher defects, v1/v2/v3 have invalid cloud forcing, and v4 still borrowed preceding-wall forcing for the zero-glazed roof. Current v5 independently initializes each surface and retains valid-IR and single-commit corrections. These conditional results have passed the bounded claim-review gate; no empirical accuracy is claimed.

Twenty corrected annual runs were executed: twelve primary cases (four coordinate TMYs × three placements), four repeated seeded cases and four longer-warm-up seeded cases. Corrected Padova completed first; independent balance/integration checks passed before the other three sites launched, recorded by expansion_gate.json. Across all attempts, 87 annual runs were executed: 66 quarantined, 20 candidate v5 campaign cases, and one separately identified genuine v5 seeded replay in a fresh directory. Replay matches the current Padova seeded hourly file byte-for-byte and is not an extra scientific sample.

The assumed 80 m² dwelling, physical parameters, one 2013 Veneto response, source-derived 22/19°C mapping, 4/4/1 high-setpoint hours, season indices and weather inputs match v4. The new local surface preprocessor independently computes each wall/roof plane, opaque and glazed area, shortwave and longwave term. It retains the valid-IR mean-sky correction without cloud imputation. Native annual-mean longwave and solar-time approximations remain limitations. The retained local stepping correction uses: free/constrained trials use the same old state, and the selected solution is committed once. Moisture is uncoupled and not stepped/reported; the reported boundary remains useful sensible space-heating demand with no final-energy, latent-load, DHW, carrier, emission, comfort or retrofit result.

| Coordinate case | Placement | Annual useful sensible heating kWh | Peak hourly sensible heating kW | Annual change % | Peak change % |
|---|---|---:|---:|---:|---:|
|Padova|seeded|9574.018|10.0495|0.0000|0.0000|
|Padova|early|9659.633|10.2535|0.8942|2.0293|
|Padova|late|9606.296|9.8382|0.3371|-2.1032|
|Bolzano|seeded|15851.214|17.8086|0.0000|0.0000|
|Bolzano|early|15835.683|17.8086|-0.0980|0.0000|
|Bolzano|late|15855.464|17.8086|0.0268|0.0000|
|Roma|seeded|6602.229|8.9099|0.0000|0.0000|
|Roma|early|6736.446|9.6518|2.0329|8.3260|
|Roma|late|6635.240|9.2112|0.5000|3.3814|
|Palermo|seeded|5078.837|8.2925|0.0000|0.0000|
|Palermo|early|5275.293|8.7875|3.8681|5.9691|
|Palermo|late|5128.684|8.4189|0.9815|1.5242|

All 175,200 exported site-surface-hours across four sites passes separate opaque-shortwave, longwave and window-shortwave reconstruction; all 175,200 solver hours match reconstructed node forcing. Reverse/rotate surface order preserves node forcing within 1e-8 W. Fixtures include a windowless zero-glazed roof. The preprocessing correction does not change physical areas, U-values, absorptance, glazing or network parameters.
Active forcing fields and IR/JSON mapping passed strict QC. Palermo has one preserved invalid wind record, unused in this guarded fixed-ACH pathway; the weather file is not claimed universally valid. Native annual-mean longwave and unadjusted solar-time offsets are explicit model assumptions. No empirical validation follows.
All 175,200 exported corrected hours close each air/surface/mass step balance within 1e-6 W; maximum absolute residual across nodes and hours is 1.769e-10 W. Exported old-state continuity is exact. All twenty files have 8,760 finite rows; all repeats are exact, warm-up comparisons meet the declared limits, active heating meets its setpoint, and cooling is zero. Independent standard-library CSV integration and peak checks agree within 1e-9 kWh and 1e-12 kW. These are numerical/physics verification results, not empirical accuracy.

The three placements preserve durations assigned the high setpoint, not guaranteed heater runtime. At the 19°C setback the ideal heater may remain active. The explicitly post-hoc partition below quantifies that property from corrected files; it is descriptive and did not replace predeclared endpoints. Early/late are neither observed schedules nor bounds over every allowable schedule. No placement is shown more accurate. A few coordinate TMYs and one assumed dwelling cannot establish household, municipal, national or universal transferability.

An independently specified whole-building benchmark, measured building validation, calibrated envelope/equipment characterization and historical-weather/meter matching remain unperformed. Candidate ANACI annual fields need provenance, meter boundaries and calibration-independence verification before becoming a validation target. Primary climate descriptors are unaffected by the correction.

Reproduction and source evidence: correction_protocol_v3.md; single_commit_1c.py; test_single_commit_1c.py; run_climate_campaign_v5.py; check_full_step_balances_v4.py; summarize_corrected_campaign_v5.py. Immutable per-site correction/protocol/config/design/adapter copies and hashes, source-package hashes, raw forcing/states, parameters, logs and full_step_balance_checks.json reside under 05_simulations/climate_campaign_v5. Avoid rerunning into an existing immutable run directory.
Technical-review disposition: the independent reviewer 03 v5 supplement rechecked the source, per-surface forcing and all 20 campaign outputs and passed its technical checks. The fresh full-claim review 04 and final v04 closure passed; four complete cycles permit an explicitly incomplete prose first draft. Empirical validation remains absent. See `07_reviews/v5_verification_supplement.md`.

Post-review status annotation by the orchestrator: the saved final closure supersedes procedural pending-review labels in historical sections. Frozen protocols and run snapshots remain unchanged.
