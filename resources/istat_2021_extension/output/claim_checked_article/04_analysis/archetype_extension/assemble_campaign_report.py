"""Inventory and gate summary assembled from finished outputs; no model run."""
from pathlib import Path
import json,csv,hashlib,time
ROOT=Path(__file__).resolve().parents[2];CAM=ROOT/'05_simulations/archetype_campaign_v2';OUT=ROOT/'04_analysis/archetype_extension/results_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[];sites=[];node_max=[0.,0.,0.];source=[]
for site in ['Padova','Bolzano','Roma','Palermo']:
    run=CAM/site;gate=json.loads((run/'site_gate.json').read_text());assert gate['pass'];design=json.loads((run/'design.json').read_text());summary=json.loads((run/'summary.json').read_text());balance=json.loads((run/'full_step_balance_checks.json').read_text());forcing=json.loads((run/'independent_surface_forcing_checks.json').read_text());checks=json.loads((run/'checks.json').read_text());frozen=json.loads((run/'frozen_manifest.json').read_text())
    assert len(design['cases'])==len(summary)==len(balance['cases'])==38
    assert frozen['script_sha256']==sha(ROOT/'04_analysis/archetype_extension/run_mixture_campaign.py')
    for c in balance['cases']:
        node_max=[max(x,y) for x,y in zip(node_max,c['maximum_air_surface_mass_abs_residual_W'])]
    for s in summary:
        rows.append({'site':site,'case':s['case'],'profile_id':s['profile_id'],'role_category':'technical' if s['case'] in ['repeat_component','warmup_component'] else 'scientific','warmup_h':s['warmup_h'],'rows':s['rows'],'runtime_seconds':s['runtime_seconds'],'hourly_sha256':s['hourly_sha256']})
    sites.append({'site':site,'scientific_executions':36,'technical_executions':2,'rows':sum(s['rows'] for s in summary),'site_gate_sha256':sha(run/'site_gate.json'),'pass':True,'created_utc':gate['created_utc'],'repeat_max_difference':checks['repeat_max_difference'],'warmup_annual_relative_difference':checks['warmup_annual_sensible_heating_kWh']['relative_difference'],'warmup_peak_relative_difference':checks['warmup_peak_sensible_heating_kW']['relative_difference'],'maximum_state_continuity_error_K':max(c['maximum_old_state_continuity_error_K'] for c in balance['cases']),'max_solver_forcing_reconstruction_error_W':max(max(c['solver_forcing_max_air_surface_mass_error_W']) for c in forcing['case_checks']),'command':frozen['command'],'frozen_manifest_sha256':sha(run/'frozen_manifest.json')})
    for filename in ['run.log','full_balance_check.log','surface_check.log','site_gate.log','design.json','frozen_manifest.json','completion_manifest.json','summary.json','checks.json','full_step_balance_checks.json','independent_surface_forcing_checks.json','profile_preservation_checks.json','site_gate.json']:
        source.append({'path':str((run/filename).relative_to(ROOT)),'sha256':sha(run/filename)})
    (run/'site_gate_checker_snapshot.py').write_bytes(Path(__file__).with_name('save_site_gate.py').read_bytes())
with (OUT/'execution_inventory.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
analysis=json.loads((OUT/'analysis_manifest.json').read_text());assert analysis['independent_hourly_integration_pass']
result={'status':'candidate technical campaign complete; fresh full-claim review pending','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'campaign':'archetype_campaign_v2','test_respondents':32,'shared_real_components':4,'unique_profiles':36,'scientific_executions':144,'technical_executions':8,'annual_executions_total':len(rows),'exported_hourly_rows':sum(r['rows'] for r in rows),'primary_profile_rows':sum(r['rows'] for r in rows if r['role_category']=='scientific'),'maximum_air_surface_mass_abs_residual_W':node_max,'all_site_gates_pass':True,'sites':sites,'analysis_manifest_sha256':sha(OUT/'analysis_manifest.json'),'selection_manifest_sha256':sha(CAM/'pre_run/selection_manifest.json'),'runtime_manifest_sha256':sha(CAM/'pre_run/runtime_source_manifest.json'),'input_only_v1_annual_executions':0,'preflight_diagnostic_annual_executions':0,'source_records':source,'inventory_sha256':sha(OUT/'execution_inventory.csv'),'assembler_sha256':sha(__file__)}
(CAM/'campaign_completion_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:result[k] for k in ['status','annual_executions_total','exported_hourly_rows','maximum_air_surface_mass_abs_residual_W','all_site_gates_pass']},indent=2))
