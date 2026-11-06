"""Recompute corrected primary endpoints and descriptive diagnostics from v3."""
from pathlib import Path
import csv,json,math,hashlib
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'05_simulations/climate_campaign_v3'
def read_csv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def save_csv(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
primary=[];partition=[];inventory=[];verification=[]
for site in ['Padova','Bolzano','Roma','Palermo']:
    directory=RUN/site
    balance=json.loads((directory/'full_step_balance_checks.json').read_text());assert balance['pass']
    summary=json.loads((directory/'summary.json').read_text())
    for s in summary:
        case=s['case']; p=directory/f'{case}_hourly.csv'; rows=read_csv(p)
        q=[float(r['sensible_load_W']) for r in rows]
        annual=math.fsum(max(v,0.) for v in q)/1000
        peak=max(max(v,0.) for v in q)/1000
        assert len(rows)==8760 and abs(annual-s['annual_sensible_heating_kWh'])<1e-9 and abs(peak-s['peak_sensible_heating_kW'])<1e-12
        digest=hashlib.sha256(p.read_bytes()).hexdigest(); assert digest==s['hourly_sha256']
        inventory.append({'campaign':'climate_campaign_v3','site':site,'case':case,'purpose':'primary_contrast' if case in ['seeded','early','late'] else 'technical_check','warmup_hours':s['warmup_h'],'hourly_path':str(p),'hourly_sha256':digest,'annual_sensible_heating_kWh':annual,'peak_sensible_heating_kW':peak,'scientific_disposition':'corrected full-step verification passed; independent review pending'})
        verification.append({'site':site,**next(x for x in balance['cases'] if x['case']==case)})
        if case not in ['seeded','early','late']:continue
        base=summary[0]
        primary.append({'site':site,'schedule':case,'annual_sensible_heating_kWh':annual,'peak_sensible_heating_kW':peak,'delta_annual_kWh':0. if case=='seeded' else annual-base['annual_sensible_heating_kWh'],'delta_annual_percent':0. if case=='seeded' else 100*(annual/base['annual_sensible_heating_kWh']-1),'delta_peak_kW':0. if case=='seeded' else peak-base['peak_sensible_heating_kW'],'delta_peak_percent':0. if case=='seeded' else 100*(peak/base['peak_sensible_heating_kW']-1),'hourly_path':str(p),'run_id':f'climate_campaign_v3/{site}/{case}'})
        for sp in [0.,19.,22.]:
            subset=[r for r in rows if float(r['heating_setpoint_C'])==sp]
            values=[float(r['sensible_load_W']) for r in subset]
            energy=math.fsum(max(v,0.) for v in values)/1000
            partition.append({'site':site,'schedule':case,'setpoint_C':sp,'calendar_hours_at_setpoint':len(values),'positive_heating_hours_gt_1e_minus6W':sum(v>1e-6 for v in values),'positive_sensible_heating_kWh':energy,'share_annual_sensible_heating_percent':100*energy/annual,'positive_load_tolerance_W':1e-6})
save_csv(ROOT/'04_analysis/paired_schedule_contrasts_v3.csv',primary)
save_csv(ROOT/'04_analysis/posthoc_setpoint_partition_v3.csv',partition)
save_csv(ROOT/'05_simulations/run_inventory_v3.csv',inventory)
old_path=ROOT/'05_simulations/run_inventory.csv'
old=read_csv(old_path)
if len(old)==25:
    (ROOT/'05_simulations/original_run_inventory_v1_v2.csv').write_bytes(old_path.read_bytes())
else:old=read_csv(ROOT/'05_simulations/original_run_inventory_v1_v2.csv')
all_inventory=[{**x,'scientific_disposition':'QUARANTINED: confirmed original recursive dispatcher defect'} for x in old]+inventory
assert len(all_inventory)==45
save_csv(ROOT/'05_simulations/run_inventory.csv',all_inventory)
(ROOT/'04_analysis/independent_output_checks_v3.json').write_text(json.dumps(verification,indent=2))
interactions=[]
for r in primary:
    if r['site']=='Padova' or r['schedule']=='seeded':continue
    ref=next(x for x in primary if x['site']=='Padova' and x['schedule']==r['schedule'])
    interactions.append({'site':r['site'],'schedule':r['schedule'],'difference_of_delta_annual_kWh':r['delta_annual_kWh']-ref['delta_annual_kWh'],'difference_of_delta_peak_kW':r['delta_peak_kW']-ref['delta_peak_kW']})
save_csv(ROOT/'04_analysis/climate_schedule_interactions_v3.csv',interactions)
old_primary=read_csv(ROOT/'04_analysis/paired_schedule_contrasts_v2.csv')
correction=[]
for r in primary:
    o=next(x for x in old_primary if x['site']==r['site'] and x['schedule']==r['schedule'])
    correction.append({'site':r['site'],'schedule':r['schedule'],'quarantined_v2_heating_kWh':float(o['annual_sensible_heating_kWh']),'corrected_v3_heating_kWh':r['annual_sensible_heating_kWh'],'v3_minus_v2_heating_kWh':r['annual_sensible_heating_kWh']-float(o['annual_sensible_heating_kWh']),'role':'implementation-correction diagnostic; v2 is not scientific evidence'})
save_csv(ROOT/'04_analysis/dispatcher_correction_diagnostic_v2_v3.csv',correction)
max_residual=max(v for x in verification for v in x['maximum_air_surface_mass_abs_residual_W'])
report=['# Corrected new EUReCA simulation outputs — v3','','Status: adapter-local correction passed declared transition checks and full annual step-balance checks. Independent scientific review of the correction remains pending. No empirical validation has been completed. All original 25 v1/v2 annual outputs remain quarantined because the original recursive dispatcher violates timestep state balances. Use only v3 tables for current conditional simulation claims.','','Twenty corrected annual runs were executed: twelve primary cases (four coordinate TMYs × three placements), four repeated seeded cases and four longer-warm-up seeded cases. Corrected Padova completed first; independent balance/integration checks passed before the other three sites launched, recorded by expansion_gate.json. Across all attempts, 45 new annual simulations were executed: 25 quarantined and 20 corrected.','','The assumed 80 m² dwelling, physical parameters, one 2013 Veneto response, source-derived 22/19°C mapping, 4/4/1 high-setpoint hours, season indices and weather inputs match v2. The local correction changes only state stepping: free/constrained trials use the same old state, and the selected solution is committed once. Moisture is uncoupled and not stepped/reported; the reported boundary remains useful sensible space-heating demand with no final-energy, latent-load, DHW, carrier, emission, comfort or retrofit result.','','| Coordinate case | Placement | Annual useful sensible heating kWh | Peak hourly sensible heating kW | Annual change % | Peak change % |','|---|---|---:|---:|---:|---:|']
for r in primary:report.append(f"|{r['site']}|{r['schedule']}|{r['annual_sensible_heating_kWh']:.3f}|{r['peak_sensible_heating_kW']:.4f}|{r['delta_annual_percent']:.4f}|{r['delta_peak_percent']:.4f}|")
report.extend(['',f'All 175,200 exported corrected hours close each air/surface/mass step balance within 1e-6 W; maximum absolute residual across nodes and hours is {max_residual:.3e} W. Exported old-state continuity is exact. All twenty files have 8,760 finite rows; all repeats are exact, warm-up comparisons meet the declared limits, active heating meets its setpoint, and cooling is zero. Independent standard-library CSV integration and peak checks agree within 1e-9 kWh and 1e-12 kW. These are numerical/physics verification results, not empirical accuracy.', '',
'The three placements preserve durations assigned the high setpoint, not guaranteed heater runtime. At the 19°C setback the ideal heater may remain active. The explicitly post-hoc partition below quantifies that property from corrected files; it is descriptive and did not replace predeclared endpoints. Early/late are neither observed schedules nor bounds over every allowable schedule. No placement is shown more accurate. A few coordinate TMYs and one assumed dwelling cannot establish household, municipal, national or universal transferability.','','An independently specified whole-building benchmark, measured building validation, calibrated envelope/equipment characterization and historical-weather/meter matching remain unperformed. Candidate ANACI annual fields need provenance, meter boundaries and calibration-independence verification before becoming a validation target. Primary climate descriptors are unaffected by the correction.','',
'Reproduction and source evidence: correction_protocol_v3.md; single_commit_1c.py; test_single_commit_1c.py; run_climate_campaign_v3.py; check_full_step_balances_v3.py; summarize_corrected_campaign_v3.py. Immutable per-site correction/protocol/config/design/adapter copies and hashes, source-package hashes, raw forcing/states, parameters, logs and full_step_balance_checks.json reside under 05_simulations/climate_campaign_v3. Avoid rerunning into an existing immutable run directory.'])
report_path=ROOT/'04_analysis/simulation_results.md'
if not (ROOT/'05_simulations/quarantined_v2_results_report_snapshot.md').exists():
    (ROOT/'05_simulations/quarantined_v2_results_report_snapshot.md').write_bytes(report_path.read_bytes())
report_path.write_text('\n'.join(report),encoding='utf-8')
note=['# Post-hoc corrected setpoint partition — v3','','Descriptive diagnostic requested after the original primary endpoints were complete, now recomputed only from corrected v3 files. No extra annual simulation or primary test was added. Positive heating hours use load >1e-6 W; energy is the one-hour sum of max(load,0). The 4/4/1 reported on-hour inputs become 22°C high-setpoint durations under the source mapping, not constraints enforcing the same actual heater runtime. All schedules assign the same calendar hours to 19/22/0°C; positive load hours and energy can differ.','','| Coordinate case | Placement | Heating at 19°C kWh | Positive hours at 19°C | Heating at 22°C kWh | Positive hours at 22°C | Heating at 0°C kWh |','|---|---|---:|---:|---:|---:|---:|']
for site in ['Padova','Bolzano','Roma','Palermo']:
    for mode in ['seeded','early','late']:
        d={r['setpoint_C']:r for r in partition if r['site']==site and r['schedule']==mode}
        note.append(f"|{site}|{mode}|{d[19.]['positive_sensible_heating_kWh']:.3f}|{d[19.]['positive_heating_hours_gt_1e_minus6W']}|{d[22.]['positive_sensible_heating_kWh']:.3f}|{d[22.]['positive_heating_hours_gt_1e_minus6W']}|{d[0.]['positive_sensible_heating_kWh']:.3f}|")
(ROOT/'04_analysis/posthoc_setpoint_partition_v3.md').write_text('\n'.join(note),encoding='utf-8')
print('\n'.join(report[:25]))
