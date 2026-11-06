# Methods execution and provenance — v5 candidate

**Current scientific status: v5 CANDIDATE.** All 66 earlier annual executions are quarantined. The current campaign has 20 annual cases (12 primary, 4 repeat and 4 longer-warm-up checks), plus one separate genuine seeded replay that is not another scientific sample. Total execution count is 87. Independent technical review 03 supplement passed; fresh full-claim review 04 remains pending; no empirical validation is claimed. Exact frozen pre-run protocols/configs/source snapshots remain unchanged under each run directory. This canonical index was updated after execution and does not replace those pre-run records.

| Execution group | Disposition | Annual runs |
|---|---|---:|
| v1 local pilot | Quarantined |5|
| v2 original official campaign | Quarantined |20|
| v3 single-commit campaign | Quarantined: invalid cloud forcing remained |20|
| v4 IR-corrected campaign + separate replay | Quarantined: zero-glazed roof borrowed wall forcing |21|
| v5 current candidate campaign | 12 primary + 4 repeat + 4 warm-up checks |20|
| v5 separate seeded replay | Fresh directory; not another scientific sample |1|
| Total | 66 quarantined + 20 current candidates + 1 separate replay |87|

Original reports, manifests, raw data and failed diagnostics remain preserved. Current candidate results use only v5. No calibration, physical area/U/absorptance/glazing/equipment change, new household assumption or endpoint change accompanied the source corrections. The findings concern the inspected local working source and adapter, not an identified published conference release. External checkouts/environments were not modified.

The three scoped local modules are `single_commit_1c.py`, `ir_sky_forcing.py` and `surface_forcing_1c.py`. The surface module separately initializes each external wall/roof plane, areas, construction and radiative coefficient; zero-glazed surfaces do not require a fictitious window. It computes opaque shortwave, longwave and window shortwave separately, then retains original 1C node weights. It exports all physical inputs/components and enforces identical forcing across schedule cases.

Padova's five new cases passed independent surface component, ordering, actual solver-forcing, transition, weather and state-balance checks before expansion. The other coordinates then passed the same checks. The independent checkers reconstruct every surface-hour and compare forcing with every solver-hour. Source/config/input/protocol hashes, pre-run fixture/readiness evidence and chronology are preserved in each run and `v5_pre_run_readiness.json`.

The current reported energy boundary is useful sensible space-heating demand; final energy, carriers, DHW, emissions, retrofit savings, measured hourly accuracy and comfort compliance were not computed or validated. High-setpoint hours are not realized heater-runtime hours. Native annual-mean longwave and unadjusted irradiance-time offsets remain explicit limitations. One negative Palermo wind value remains a failed but unused source-variable QC item, protected by a runtime guard; no whole-file universal weather-validity claim is made.

An independent benchmark, physical-input characterization, compatible independent measured data, historical-weather matching and any calibration/holdout design remain pending. The ANACI audit/access plan is retained. V5 is a candidate conditional illustration pending independent scientific review, not an empirically validated results paper.

Practical non-destructive replay uses SITE plus a required fresh campaign ID and refuses existing hourly/completed outputs. Exact v5 commands and argv include executable, source, SITE, campaign and optional replay flag. The separate seeded replay is byte-identical to Padova seeded; the refusal test leaves all existing file hashes unchanged. See `v5_reproduction.md`, `05_simulations/recorded_command_addendum.md` and `v5_replay_interface_checks.json`. Old omitted-SITE records were not overwritten.

Current data: `04_analysis/paired_schedule_contrasts_v5.csv`, `posthoc_setpoint_partition_v5.csv`, `independent_output_checks_v5.json`, `independent_surface_checks_v5.json`, `05_simulations/run_inventory.csv` and `run_inventory_v5.csv`. The candidate table/report in `04_analysis/simulation_results.md` is the current numerical handoff. Legacy outputs remain quarantined even where their arithmetic or limited numerical checks passed.

Current maximum full-step residual: 1.769e-10 W. Maximum independent surface component error: 1.364e-12 W. Maximum surface-order forcing error: 1.819e-12 W. These are numerical verification, not measured accuracy.
