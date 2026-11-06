"""Independent CSV integration and paired contrasts; no reused preexisting results."""
from pathlib import Path
import csv,math,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'05_simulations/climate_campaign_v2'
rows=[];checkrows=[];runtimes=0.
for site in ['Padova','Bolzano','Roma','Palermo']:
    summary=json.loads((RUN/site/'summary.json').read_text())
    gate=json.loads((RUN/site/'checks.json').read_text());assert gate['pass']
    for s in summary:
        p=RUN/site/f"{s['case']}_hourly.csv"
        with p.open(newline='') as f: hourly=list(csv.DictReader(f))
        q=[float(x['sensible_load_W']) for x in hourly]
        annual=math.fsum(max(x,0) for x in q)/1000
        peak=max(max(x,0) for x in q)/1000
        assert len(hourly)==8760 and abs(annual-s['annual_sensible_heating_kWh'])<1e-9 and abs(peak-s['peak_sensible_heating_kW'])<1e-12
        assert hashlib.sha256(p.read_bytes()).hexdigest()==s['hourly_sha256']
        checkrows.append({'site':site,'case':s['case'],'n':len(hourly),'integration_abs_error_kWh':annual-s['annual_sensible_heating_kWh'],'peak_abs_error_kW':peak-s['peak_sensible_heating_kW'],'hourly_hash_matches':True,'run_gate_pass':True})
        runtimes+=s['runtime_seconds']
        if s['case'] in ['seeded','early','late']:
            base=summary[0]
            rows.append({'site':site,'schedule':s['case'],'annual_sensible_heating_kWh':annual,'peak_sensible_heating_kW':peak,'delta_annual_kWh':annual-base['annual_sensible_heating_kWh'],'delta_annual_percent':100*(annual/base['annual_sensible_heating_kWh']-1),'delta_peak_kW':peak-base['peak_sensible_heating_kW'],'delta_peak_percent':100*(peak/base['peak_sensible_heating_kW']-1),'hourly_path':str(p),'run_id':f'climate_campaign_v2/{site}/{s["case"]}'})
for name,data in [('paired_schedule_contrasts_v2.csv',rows),('independent_output_checks_v2.csv',checkrows)]:
    with (ROOT/'04_analysis'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
interactions=[]
for r in rows:
    if r['site']!='Padova' and r['schedule']!='seeded':
        ref=next(x for x in rows if x['site']=='Padova' and x['schedule']==r['schedule'])
        interactions.append({'site':r['site'],'schedule':r['schedule'],'difference_of_delta_annual_kWh':r['delta_annual_kWh']-ref['delta_annual_kWh'],'difference_of_delta_peak_kW':r['delta_peak_kW']-ref['delta_peak_kW']})
with (ROOT/'04_analysis/climate_schedule_interactions_v2.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=interactions[0].keys());w.writeheader();w.writerows(interactions)
report=['# Newly executed EUReCA results — 20 October 2026','','All results are conditional simulation outputs for one assumed80m²dwelling and one actual2013Veneto response. No observed hourly schedules, meter validation, stock weights, retrofit scenarios or final-energy conversions enter this campaign. The three reconstructions preserve4/4/1daily blockhours.','','20new annual official-weather runs completed:12primary (4coordinatecases×3schedules),4exact-repeat checks,4longer-warm-up checks. The preceding local-weather Padova pilot adds5annual runs, for25totalnew annual runs. Analytical network stepping is a separate verification task, not an annual case.','','| Site | Reconstruction | Annual sensible heating kWh | Peak hourly sensible heating kW | Annual change vs same-site seeded % | Peak change % |','|---|---|---:|---:|---:|---:|']
for r in rows:report.append(f"|{r['site']}|{r['schedule']}|{r['annual_sensible_heating_kWh']:.3f}|{r['peak_sensible_heating_kW']:.4f}|{r['delta_annual_percent']:.4f}|{r['delta_peak_percent']:.4f}|")
report.extend(['',f'Actual summed engine-case runtime was{runtimes:.3f}s, excluding Python imports, weather solar preprocessing and acquisition; wall-clock campaign runtime is not this sum. Full-precision data and source paths are in paired_schedule_contrasts_v2.csv. Independent standard-library CSV integration, peak recomputation and saved-file hash checks pass for all20outputs; integration discrepancy is below1e-9kWh. Every declared numerical gate passed, all repeats were exact, and all outputs had8760finite rows and zero active cooling.','',
'The results show sensitivity to allowed within-block timing for this model/input combination. Early/late are neither observed alternatives nor bounds on every possible schedule. Same reported hours do not identify one deterministic annual load or peak. No schedule is shown to be more accurate, and differences across coordinate cases are conditional on fixed climate-E operating season and the same assumed dwelling. High summer free-floating temperatures arise because cooling is intentionally disabled; no comfort performance is established.','',
'The analytical1Cnetwork tests pass zero-gradient equilibrium (maximumerror9.71e−13), steady convective heating (3327.272727W, solver−analytic absoluteerror2.28e−12W), and independently recomputed transient air/surface/mass residuals (largest<6e−12W). This checks network algebra, not external benchmark performance, envelope characterization or empirical accuracy. A complete annual heat-balance residual from real campaign states and an independently specified building benchmark remain unperformed.','',
'Provider files are TMYscenarios2005–2023, not historical-year matched validation. JSON reports elevationPadova25m, Bolzano272m, Roma72m, Palermo29m; these are provider coordinate/grid estimates, not surveyed building altitudes. Grid/horizon and source-period limitations remain. Local-v1weather provenance is retained separately; official-v2Padova numerical outputs happen to match it, which does not prove the original file retrieval history.','',
'Reproduction: execute04_analysis/audit_inputs.py,04_analysis/fetch_climate_inputs.py (networkaccess),04_analysis/audit_climates.py, then04_analysis/run_climate_campaign_v2.py SITE with the recorded EUReCA.venvPython-B, and finally04_analysis/summarize_campaign.py. Config/design/protocol/adapter snapshots, complete source-package hashes, weatherchecksums, commands, logs and completion manifests are under05_simulations. Do not rerun in place if preserving existing immutable results; copy the campaign configuration to a new runID.'])
(ROOT/'04_analysis/simulation_results.md').write_text('\n'.join(report),encoding='utf-8')
print('\n'.join(report[:24]))
