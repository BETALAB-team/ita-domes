from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'03_methods';S=ROOT/'05_simulations';R=S/'climate_campaign_v5'
sites=['Padova','Bolzano','Roma','Palermo']
balances=[json.loads((R/s/'full_step_balance_checks.json').read_text()) for s in sites]
surfaces=[json.loads((R/s/'independent_surface_forcing_checks.json').read_text()) for s in sites]
assert all(x['pass'] for x in balances+surfaces)
max_res=max(v for x in balances for c in x['cases'] for v in c['maximum_air_surface_mass_abs_residual_W'])
max_component=max(v for x in surfaces for c in x['surfaces'].values() for v in c['max_SW_LW_window_reconstruction_error_W'])
max_order=max(v for x in surfaces for c in x['case_checks'] for v in c['surface_order_checks']['reverse_rotate_max_node_forcing_error_W'])
with (S/'run_inventory.csv').open(newline='') as f:inventory=list(csv.DictReader(f))
assert len(inventory)==87
replay=json.loads((S/'v5_replay_interface_checks.json').read_text());assert replay['pass']
status='''**Current scientific status: v5 CANDIDATE.** All 66 earlier annual executions are quarantined. The current campaign has 20 annual cases (12 primary, 4 repeat and 4 longer-warm-up checks), plus one separate genuine seeded replay that is not another scientific sample. Total execution count is 87. Independent scientific review of v5 remains pending; no empirical validation is claimed. Exact frozen pre-run protocols/configs/source snapshots remain unchanged under each run directory. This canonical index was updated after execution and does not replace those pre-run records.
'''
protocol='''# Current simulation protocol index — v5 candidate

'''+status+'''
The question is conditional sensitivity of annual useful sensible heating and maximum hourly sensible heating load to within-block schedule placement for one assumed dwelling. The selected 2013 Veneto response has 4/4/1 reported hours; the adapter maps them to high-setpoint durations using the actual GRINS source rule. These are not observed hourly schedules or guaranteed heater-runtime durations. Seeded, early and late placements preserve the block totals, source indicator-based 22/19°C mapping and climate-E season indices 105/288. Cooling has a fixed 50°C upper bound and remains inactive.

The assumed dwelling is 10×8×3 m, 80 m² floor and 240 m³ volume, with exposed walls, roof and ground floor, 20% wall glazing, wall/roof/floor U-values 1.0/0.8/0.8 W m⁻² K⁻¹ and windows U2.8/g0.65/frame0.15. Heavy constructions, GRINS internal-mass multipliers, constant 240 W electric load with fraction-to-zone0.99, and fixed 0.3 ACH are retained. No measured-building, city-stock or population-weight inference is made. No retrofit or final-energy conversion is included.

All sites use the same physical inputs and fixed season. Official shared-source PVGIS SARAH3/ERA5 TMYs for 2005–2023 are calendar-coerced to 2023, hourly, fixed UTC+1 with no DST/leap day. The three original candidate corrections are now explicit study-local modules: `single_commit_1c.py` commits one old-state solution per hour; `ir_sky_forcing.py` derives effective sky temperature from valid horizontal IR and retains the native annual-mean longwave scalar; `surface_forcing_1c.py` initializes every external surface's own area, plane irradiance and construction before shortwave/longwave/glazing calculation. These concerns relate to the inspected local working source and adapter, not an established conference release. A publication-to-code commit mapping is unavailable.

Pre-run records are `simulation_protocol_v2_addendum.md`, `correction_protocol_v3.md`, `weather_correction_protocol_v4.md` and `surface_correction_protocol_v5.md`, with exact immutable copies under `05_simulations/climate_campaign_v5/SITE`. The original v1 pre-run content is in `padova_pilot_v1/protocol_pre_run_snapshot.md`. The v5 pre-run readiness file records code and weather hashes before any v5 annual run.

Synthetic component/transition fixtures are implementation checks, not observations. Every annual case exports states and forcing for independent node residual checks. Per-site exports contain all 8,760 hourly components for five external surfaces; reverse and rotate all eight surfaces to test order invariance. Independent per-surface reconstruction and comparison with the actual solver forcing must pass before scientific interpretation. Padova's five cases and all forcing/state checks passed before the other three sites launched, documented in `climate_campaign_v5/expansion_gate.json`.

Limits retained: one assumed dwelling/household response, deterministic alternatives rather than probabilistic uncertainty bounds, fixed season outside its local context, annual-mean longwave and unadjusted irradiance-time offsets, and one invalid but guarded unused Palermo wind record. No general EPW quality, empirical hourly accuracy, comfort compliance or universal transfer claim follows. See `validation_protocol.md`, `validation_matrix.md`, `methods_execution_report.md` and `v5_reproduction.md` for current dispositions and reproducibility.
'''
(M/'simulation_protocol.md').write_text(protocol,encoding='utf-8')
validation='''# Current validation strategy — v5 candidate

'''+status+'''
Implementation and physics verification covers units/calendar, source block totals, independent opaque shortwave/longwave/window fixtures, windowless-roof handling, surface standalone/order invariance, mode transitions, source-schedule mapping, all-hour component and node-forcing reconstruction, single-step balances, state continuity, deterministic repetition and warm-up. Passing these checks does not establish correspondence with measured energy. Earlier successful exits and limited algebra tests missed material bugs; their original outputs remain quarantined.

A matched independent building benchmark has not been run. Candidate VDI workbooks are leads, not passed test evidence. Any future named standard requires verified primary specification and matched boundary conditions. Inter-model agreement would be a benchmark comparison, not empirical truth.

Usable independent Padova measurements remain not established. The ANACI GIS audit found candidate annual energy attributes and calibration fields; their units, provenance, meter boundary, actual interval, zero codes and fit/holdout history are unknown. `padova_measurement_access_plan.md` specifies the minimum acquisition/metadata and prevents dismissing GIS values or declaring them measured automatically. No data-holder contact has been made.

Future empirical evaluation requires compatible building/area/equipment metadata, carrier/end-use and calorific conventions, timestamp/missingness definitions and matching historical weather. TMY scenario comparisons must not masquerade as historical validation. Freeze fitting/tuning/untouched-evaluation splits before calibration. Report signed/absolute errors, temporal/peak/comfort metrics only at supported scales. NMBE/CVRMSE require explicit interval, mean denominator, n, fitted-parameter count and primary-source acceptance rationale; no threshold is transplanted here.

Regional/national checks require aligned year, weather treatment, population/weights, sector, carriers, end uses and units. This one-dwelling useful sensible demand cannot be compared with a regional final-energy total. The chapter's national BEN comparison remains published historical evidence, not a new disaggregated validation result. A few modeled climates without observations are conditional transfer scenarios.
'''
(M/'validation_protocol.md').write_text(validation,encoding='utf-8')
matrix='''# Current validation matrix — v5 candidate

'''+status+f'''
| Output/claim | Reference and independence | Scale/boundary | Metric and rationale | Current disposition / unvalidated scope |
|---|---|---|---|---|
| Schedule reconstruction | Verified survey fields and hashed GRINS rule; no observed hourly schedule | One response, three 8 h blocks | Exact 4/4/1 high-setpoint durations | Passed; heater runtime is not constrained to those totals |
| Surface physics | Independent synthetic scalar fixtures, windowless roof and standalone contributions | Opaque SW/LW, glazed SW separately | Component error ≤1e-8 W | Passed; not empirical envelope characterization |
| Actual surface forcing | Exported plane/input/area/SHGC components; separate reconstruction | 175,200 site-surface-hours across four sites | Error ≤1e-8 W; max {max_component:.3e} W | Passed; retained model radiation approximations remain |
| Surface order | Reverse/rotate actual eight-surface lists | All five cases per site | Node forcing error ≤1e-8 W; max {max_order:.3e} W | Passed; correct geometry assumed, not surveyed |
| Solver forcing | Independent component/node sum versus annual exported forcing | All 175,200 campaign hours | Node difference ≤1e-8 W | Passed; no measured truth target |
| Dispatcher and balances | Synthetic transitions plus exported old/current states | Hourly useful sensible heat | Every node residual ≤1e-6 W; max {max_res:.3e} W; exact continuity | Passed; bounded sensible-only adapter, not general HVAC validation |
| Integration and peaks | Standard-library summation from raw hourly files | 20 annual files | <1e-9 kWh and <1e-12 kW discrepancies | Passed; useful demand only |
| Repetition/warm-up | Same model/input duplicate and 168/336 h warm-up | Four sites | Exact repeats; annual ≤0.5%, peak ≤1% | Passed; numerical criterion, not empirical tolerance |
| Weather forcing | Primary EPW/PVGIS definitions, exact JSON record matching | Active fields, valid horizontal IR | Sentinel/range checks; IR reconstruction <1e-9 W/m² | Passed active-path gate; raw invalid unused wind explicitly retained |
| Reproduction interface | Separate genuine seeded replay, completed-destination refusal | Fresh run directory | Byte-identical hourly replay; unchanged existing files | Passed; not another scientific sample |
| Whole-building benchmark | Independently specified matched case | Matching input/time/energy boundary required | No acceptance score invented | Missing |
| Padova empirical accuracy | Untouched compatible meters/building metadata | Matching historical weather and meter/end-use boundary | Bias/MAE/peaks as data permit | Not established; ANACI provenance/access plan pending |
| Aggregate accuracy | Independent official totals with aligned scope | Carrier/year/sector/population/weights | Boundary-aligned discrepancy and uncertainty | Not tested by illustrative pilot |
| Climate sensitivity | Same model/input across four coordinate TMYs | Fixed assumed dwelling and season | Paired conditional differences | Candidate result only; no empirical transfer accuracy |

Independent scientific review of v5 is pending. Current result paths are `04_analysis/paired_schedule_contrasts_v5.csv`, `independent_output_checks_v5.json`, `independent_surface_checks_v5.json` and per-site raw/check files. All 66 earlier executions are quarantined.
'''
(M/'validation_matrix.md').write_text(matrix,encoding='utf-8')
report='''# Methods execution and provenance — v5 candidate

'''+status+'''
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
'''
report+=f'\nCurrent maximum full-step residual: {max_res:.3e} W. Maximum independent surface component error: {max_component:.3e} W. Maximum surface-order forcing error: {max_order:.3e} W. These are numerical verification, not measured accuracy.\n'
(M/'methods_execution_report.md').write_text(report,encoding='utf-8')
hold='''# Scientific quarantine and current candidate pointer

All 66 pre-v5 annual executions, including both v4 campaign and its replay, are quarantined for scientific energy-effect claims. Current candidate outputs are the 20 v5 cases and one separately identified seeded replay, for 87 total annual executions. Independent scientific review of v5 remains pending; empirical validation is absent.

The successive defects in the inspected local working source were recursive timestep state updates, missing cloud code 99 entering sky-temperature forcing, and a windowless roof reusing the preceding wall's area/solar plane. Neither old algebraic agreement nor repeatability cured those physical errors. Original datasets, reports, sources, failed tests and manifests are preserved.

V5 uses explicit local single-commit stepping, valid-IR native mean-sky correction and per-surface shortwave/longwave/window preprocessing. It passes component fixtures, standalone/order tests, all-hour surface and node forcing reconstruction, state balances, repetition/warm-up and replay/refusal checks. These establish limited implementation credibility only. Current canonical methods and numerical tables point to v5; no v1/v2/v3/v4 energy number may be promoted as an accepted scientific result. See `03_methods/methods_execution_report.md`, `surface_correction_protocol_v5.md` and `04_analysis/simulation_results.md`.
'''
(S/'SCIENTIFIC_USE_HOLD.md').write_text(hold,encoding='utf-8')
old=(M/'climate_selection.md').read_text(encoding='utf-8')
start=old.index('| Coordinate label |');end=old.index('\n\n',start)
table=old[start:end]
climate='''# Selected coordinate climates — current v5 candidate

'''+status+'''
Sites were selected from source/QC and degree/solar/humidity descriptors before the original expanded thermal runs. Subsequent implementation corrections did not change the sites, TMY data or selection. Shared JRC PVGIS v5_3 inputs use SARAH3 radiation, ERA5 meteorology, 2005–2023 and a DEM horizon. The immutable pre-run choice is in `simulation_protocol_v2_addendum.md`; complete acquisition metadata and QC are in `05_simulations/climate_inputs_v2` and `weather_forcing_v4`.

'''+table+'''

HDD18 and CDD26 use daily mean temperatures and fixed bases 18/26°C; they are not legally defined degree days. This coordinate set spans 3.09-fold HDD18 and annual GHI 1,455–1,728 kWh/m². Padova has the largest CDD26 among the files, so no simple north–south cooling ranking is asserted. Provider grid/elevation/horizon values do not establish complete municipal climates or local building-stock representativeness.

The controlled comparison holds the assumed dwelling, one Veneto response, high-setpoint durations, schedule alternatives, equipment boundary and climate-E season fixed while changing weather. It is not a locally representative case comparison or empirical climate-transfer validation. Native annual-mean longwave, irradiance-time interpretation and the explicitly guarded unused invalid wind value remain limitations. Separate official-source municipal climate-class context is in `01_plan/climate_classification_context.md`; it does not change timing or establish legal compliance.
'''
(M/'climate_selection.md').write_text(climate,encoding='utf-8')
repro=(M/'v4_reproduction.md').read_text(encoding='utf-8').replace('v4','v5').replace('current campaign is `climate_campaign_v5`','current candidate campaign is `climate_campaign_v5`')
repro=repro.replace('weather_correction_protocol_v5.md','surface_correction_protocol_v5.md').replace('check_full_step_balances_v5.py','check_full_step_balances_v4.py')
repro=repro.replace('The twenty campaign cases and one replay are listed separately','The twenty candidate campaign cases and one replay are listed separately')
(M/'v5_reproduction.md').write_text(repro,encoding='utf-8')
f=S/'climate_campaign_v5/verification_source_snapshots';f.mkdir(exist_ok=True)
scripts=['check_surface_forcing_v5.py','check_full_step_balances_v4.py','summarize_corrected_campaign_v5.py','verify_v5_replay_interface.py']
hashes={}
for name in scripts:
    content=(ROOT/'04_analysis'/name).read_bytes();(f/name).write_bytes(content);hashes[name]=hashlib.sha256(content).hexdigest()
(R/'verification_source_manifest.json').write_text(json.dumps({'created_after_checks':True,'purpose':'exact analysis/checker source copies; model pre-run snapshots are separate','sha256':hashes,'maximum_full_step_residual_W':max_res,'maximum_surface_component_error_W':max_component,'maximum_order_error_W':max_order},indent=2))
print(json.dumps({'current_status':'v5 candidate; independent review pending','all_annual_executions':87,'quarantined':66,'current_campaign':20,'separate_replay':1,'max_full_step_residual_W':max_res,'max_component_error_W':max_component,'max_order_error_W':max_order},indent=2))
