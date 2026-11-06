"""Independent standard-library conservation checks from exported hourly states."""
from pathlib import Path
import csv,json,sys,math,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
SITE=sys.argv[1]
RUN=ROOT/'05_simulations/climate_campaign_v3'/SITE
out=[]
for case in ['seeded','early','late','repeat_seeded','warmup_seeded']:
    with (RUN/f'{case}_hourly.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    p=json.loads((RUN/f'{case}_network_parameters.json').read_text())
    his=p['Htr_is'];hw=p['Htr_w'];hms=p['Htr_ms'];hem=p['Htr_em'];cm=p['Cm'];ca=p['_air_thermal_capacity'];dt=p['timestep_seconds']
    residuals=[];qs=[];consecutive_error=[]
    for i,raw in enumerate(rows):
        r={k:float(v) for k,v in raw.items()}
        q=r['sensible_load_W'];ta=r['air_temperature_C'];ts=r['surface_C'];tm=r['mass_C'];ext=r['outdoor_temperature_C'];hi=r['infiltration_H_W_K']
        residuals.append([q*r['sigma_convective']+r['phi_air_W']+his*(ts-ta)+hi*(ext-ta)-ca*(ta-r['old_air_C'])/dt,
        q*r['sigma_radiant']+r['phi_surface_W']+his*(ta-ts)+hw*(ext-ts)+hms*(tm-ts),
        r['phi_mass_W']+hms*(ts-tm)+hem*(ext-tm)-cm*(tm-r['old_mass_C'])/dt])
        qs.append(q)
        if i:
            consecutive_error.append(max(abs(r['old_air_C']-float(rows[i-1]['air_temperature_C'])),abs(r['old_mass_C']-float(rows[i-1]['mass_C']))))
    max_by_node=[max(abs(v[j]) for v in residuals) for j in range(3)]
    bad=sum(any(abs(x)>1e-6 for x in r) for r in residuals)
    baseline=next(s for s in json.loads((RUN/'summary.json').read_text()) if s['case']==case)
    annual=math.fsum(max(q,0.) for q in qs)/1000
    peak=max(max(q,0.) for q in qs)/1000
    entry={'case':case,'n':len(rows),'n_residual_failure_gt_1e_minus6W':bad,'maximum_air_surface_mass_abs_residual_W':max_by_node,'maximum_old_state_continuity_error_K':max(consecutive_error),'annual_integration_error_kWh':annual-baseline['annual_sensible_heating_kWh'],'peak_error_kW':peak-baseline['peak_sensible_heating_kW'],'hourly_sha256':hashlib.sha256((RUN/f'{case}_hourly.csv').read_bytes()).hexdigest()}
    entry['pass']=len(rows)==8760 and bad==0 and max(consecutive_error)<1e-12 and abs(entry['annual_integration_error_kWh'])<1e-9 and abs(entry['peak_error_kW'])<1e-12 and entry['hourly_sha256']==baseline['hourly_sha256']
    out.append(entry)
gate={'site':SITE,'checked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'checker':'independent CSV arithmetic; no engine import','cases':out,'prior_numerical_checks_pass':json.loads((RUN/'checks.json').read_text())['pass'],'transition_gate_pass':json.loads((RUN/'transition_checks_pre_run.json').read_text())['pass']}
gate['pass']=all(x['pass'] for x in out) and gate['prior_numerical_checks_pass'] and gate['transition_gate_pass']
(RUN/'full_step_balance_checks.json').write_text(json.dumps(gate,indent=2))
print(json.dumps(gate,indent=2))
assert gate['pass']
