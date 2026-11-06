from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[1]
methods=ROOT/'03_methods'
report='''# Methods and simulation execution report — current v4

Current outputs use a local single-commit 1C dispatcher and an IR-based correction of the native annual-mean air-minus-sky scalar. Both corrections preserve the selected household response, reported-hour mapping, assumed dwelling, fixed climate-E season, equipment boundary, sites, weather files and predeclared endpoints. No calibration or external checkout/environment modification occurred. Independent scientific review of v4 remains pending; no empirical validation is claimed.

| Campaign | Disposition | Annual executions |
|---|---|---:|
| Local-weather Padova pilot v1 | Quarantined: dispatcher and sky-forcing defects |5|
| Official four-site campaign v2 | Quarantined: dispatcher and sky-forcing defects |20|
| Single-commit campaign v3 | Quarantined: missing cloud code still entered sky forcing |20|
| Current forcing/stepping campaign v4 | 12 primary cases + 4 repeats + 4 longer-warm-up checks |20|
| Separate genuine v4 seeded replay | Fresh output directory; not another scientific sample |1|
| Total executed | 45 quarantined + 20 current + 1 replay |66|

All original outputs and manifests remain preserved. V3 passes numerical step conservation but has invalid physical forcing; it cannot support scientific energy claims. The current twelve primary results are in `04_analysis/paired_schedule_contrasts_v4.csv`, with a clearly post-hoc setpoint partition in `posthoc_setpoint_partition_v4.csv`. The three reconstructions preserve high-setpoint durations, not actual ideal-heater runtime; heating remains possible at the assumed 19°C setback.

The v4 Padova cases completed first and passed active-forcing, longwave-sign, transition, repeatability, warm-up and independent full-step balances before the other three coordinate cases ran. `climate_campaign_v4/expansion_gate.json` records that chronology. All 175,200 current campaign hours pass air/surface/mass residuals ≤1e-6 W; maximum observed residual is 1.746e-10 W. Old-state continuity and repeats are exact. Peak/integration checks agree within 1e-12 kW and 1e-9 kWh. The largest warm-up annual relative difference is 1.03e-8. These verify numerical implementation, not empirical accuracy.

Valid horizontal IR was transformed using the documented effective radiative temperature relation, then averaged as required by the native scalar longwave formulation. Mean air-minus-sky differences are 12.775656, 17.620782, 13.551708 and 14.073727 K at the four coordinates. No missing cloud value is filled or clipped. Field-specific sentinel/range checks and exact EPW/JSON record matching are saved in `weather_forcing_v4/weather_forcing_audit.json`. One Palermo wind value is physically invalid in the source; it is retained and explicitly excluded from the active-forcing gate because fixed ACH and zero natural/mechanical ventilation/PV make it unused. A runtime dictionary guard raises on any later wind-speed or opaque-sky-cover read, so successful runs establish this scoped exception. The EPWs are not claimed universally valid.

Native annual-mean longwave and unadjusted irradiance-time offsets remain limitations. The hour-ending/local-standard export convention and its relation to JSON labels are audited without inventing a new absolute-time shift. No exact solar-phase accuracy, hourly empirical prediction or comfort compliance follows. An independent preprocessing-only check compares native node forcing with corrected sky versus a zero-sky reference; it adds no annual thermal simulation.

Remaining evidence layers are independent review of v4, a matched externally specified building benchmark, independently characterized physical inputs, compatible measured building/meter data, historical-weather matching and calibration/untouched-validation separation if fitting becomes necessary. Candidate ANACI annual attributes need source/meter-boundary/calibration-independence checks before being used as observations. Aggregate final energy is not comparable with this one-dwelling ideal useful sensible demand.

The v4 interface requires SITE and a fresh campaign ID under this artifact root and refuses existing hourly/completed outputs. Frozen commands now contain executable, script, SITE, campaign, optional replay flag and exact argv. Historical missing-SITE records are transparently corrected by addendum, never overwritten. The separate replay is byte-identical to current seeded Padova; the refusal test leaves hashes of all 33 existing files unchanged. See `v4_reproduction.md`, `05_simulations/recorded_command_addendum.md` and `v4_replay_interface_checks.json`.

Complete inventories, immutable protocol/config/design/source snapshots, input/software hashes, weather derivations, failed QC logs, raw state/forcing outputs and independent check files remain under `05_simulations`. Use only v4 tables for current conditional result claims; v1/v2/v3 stay quarantined.
'''
(methods/'methods_execution_report.md').write_text(report,encoding='utf-8')
status='''

## Current v4 disposition (supersedes earlier execution-status notes)

Review 02 confirmed that missing cloud code 99 entered the native sky-temperature calculation. All 45 prior annual outputs, including v3, are quarantined. `weather_correction_protocol_v4.md` predeclared an IR-derived effective sky temperature and retained native annual-mean longwave approximation. Twenty v4 cases and one separate genuine seeded replay completed after active-forcing and Padova-first gates. All current full-step balances pass (maximum 1.746e-10 W), state continuity/repeats are exact, and the replay is byte-identical in a fresh destination. There are 66 total annual executions: 45 quarantined, 20 current and one non-sample replay. Independent review of v4 and empirical validation remain outstanding. Native irradiance-time offsets, annual-mean longwave and the explicitly unused invalid Palermo wind record remain documented limitations. Current results are `04_analysis/simulation_results.md` and `paired_schedule_contrasts_v4.csv`; older numerical findings must not be promoted.
'''
for name in ['simulation_protocol.md','validation_protocol.md','feasibility_audit.md','climate_selection.md']:
    p=methods/name;text=p.read_text(encoding='utf-8')
    if '## Current v4 disposition' not in text:p.write_text(text+status,encoding='utf-8')
p=methods/'validation_matrix.md';text=p.read_text(encoding='utf-8')
text=text.replace('The 20 new v3 runs use the documented local single-commit correction and pass transition/full-step checks.', 'The 20 v3 runs also have invalid sky forcing and remain quarantined. The 20 current v4 cases retain single-commit stepping and use the documented valid-IR sky correction.')
text=text.replace('completed checks refer to v3','completed checks refer to v4').replace('20 v3 files','20 v4 files').replace('20 v3 runs','20 v4 runs').replace('2.024e-10','1.746e-10').replace('8.16e-5','1.03e-8').replace('independent_output_checks_v3.json','independent_output_checks_v4.json')
text+='''

V4 forcing gate: active fields pass finite/sentinel/range and EPW/JSON record checks; effective sky temperature reconstructs horizontal IR within 1e-9 W/m²; the native longwave term has loss sign. The invalid Palermo wind record stays a failed unused-variable QC item, with a runtime guard preventing active consumption. This is not universal EPW validation. A separate seeded replay and completed-destination refusal test pass; neither establishes measured accuracy. See `weather_correction_protocol_v4.md` and `methods_execution_report.md`.
'''
p.write_text(text,encoding='utf-8')
p=methods/'climate_selection.md';text=p.read_text(encoding='utf-8');text+='\nContext only: official-source municipal climate classifications are separately audited in `01_plan/climate_classification_context.md`; they do not alter the selected coordinates, fixed season or scenario definitions, and do not establish legal compliance.\n';p.write_text(text,encoding='utf-8')
hold='''# Scientific quarantine — v1/v2/v3

All 45 earlier annual article runs remain quarantined. Reviewer 01 confirmed recursive dispatcher state-update errors in v1/v2. Reviewer 02 independently confirmed that v3 retained the missing opaque-sky code 99 in sky forcing, despite passing hourly state balances. Neither their demand values nor their timing contrasts may support scientific energy claims. Old reports, datasets, source hashes and logs are preserved.

Current v4 retains the local single-commit dispatcher and replaces only the native mean air-minus-sky scalar with an effective temperature derived from valid horizontal IR. Twenty current cases pass active-forcing, transition, full-balance and repeat/warm-up gates. One additional seeded replay in a fresh output directory matches byte-for-byte; it is not an extra scientific sample. Independent review of v4 remains pending, and empirical validation is absent. Current data paths are in `03_methods/methods_execution_report.md` and `04_analysis/simulation_results.md`.

The correction retains native annual-mean longwave and solar-time approximations. Palermo's one negative source wind value is preserved as an explicitly unused-variable QC failure; runtime guards prohibit using it or invalid cloud cover in the active pathway. There is no cloud imputation or general claim that every EPW variable is valid. No external checkout/environment was modified.
'''
(ROOT/'05_simulations/SCIENTIFIC_USE_HOLD.md').write_text(hold,encoding='utf-8')
with (ROOT/'05_simulations/run_inventory.csv').open(newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==66
print('Finalized v4 records; 66 annual executions, 45 quarantined, 20 current, one separate replay')
