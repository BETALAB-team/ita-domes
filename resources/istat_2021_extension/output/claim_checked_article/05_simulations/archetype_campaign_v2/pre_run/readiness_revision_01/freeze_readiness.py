"""Input/source-only readiness freeze; executes no annual thermal timesteps."""
from pathlib import Path
import json,hashlib,ast,sys,time,subprocess
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PRE=ROOT/'05_simulations/archetype_campaign_v2/pre_run'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
original=ROOT/'04_analysis/run_climate_campaign_v5.py';runner=HERE/'run_mixture_campaign.py'
def zone_ast(path):
    tree=ast.parse(path.read_text());return ast.dump(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='zone'),include_attributes=False)
assert zone_ast(original)==zone_ast(runner),'Physical zone setup changed'
files=[runner,HERE/'profile_bridge.py',ROOT/'04_analysis/single_commit_1c.py',ROOT/'04_analysis/ir_sky_forcing.py',ROOT/'04_analysis/surface_forcing_1c.py',ROOT/'04_analysis/check_full_step_balances_v4.py',ROOT/'04_analysis/check_surface_forcing_v5.py',Path(r'C:\Works\EUReCA\EUReCA\eureca_building\config.json')]
files+=sorted(Path(r'C:\Works\EUReCA\EUReCA\eureca_building').glob('*.py'))
for site in ['Padova','Bolzano','Roma','Palermo']:
    files.extend([ROOT/f'05_simulations/climate_inputs_v2/{site}.epw',ROOT/f'05_simulations/climate_inputs_v2/{site}.json'])
for p in files:
    if p.suffix=='.py':compile(p.read_text(encoding='utf-8'),str(p),'exec')
manifest={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'source/runtime freeze before any v2 annual simulation','sha256':{str(p):sha(p) for p in files},'physical_zone_AST_exactly_matches_reviewed_v5':True,'python':sys.version,'executable':sys.executable}
(PRE/'runtime_source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
for p in [runner,HERE/'freeze_readiness.py']:(PRE/p.name).write_bytes(p.read_bytes())
r=subprocess.run([sys.executable,'-B',str(runner),'Padova','--campaign','archetype_campaign_v2','--preflight'],capture_output=True,text=True)
(PRE/'preflight.log').write_text(r.stdout+r.stderr,encoding='utf-8');assert r.returncode==0
roles=pd.read_csv(PRE/'role_manifest.csv');test=roles[roles.role=='test'];assert len(test)==32 and test.groupby(['archetype_id','control_branch']).size().eq(2).all()
assert abs(test.overall_empirical_weight.sum()-1)<1e-12
assert all(abs(v-1)<1e-12 for v in test.groupby('archetype_id').within_group_weight.sum())
fixtures=json.loads((HERE/'profile_fixture_checks.json').read_text());assert fixtures['pass']
result={'pass':True,'annual_execution':False,'selection_manifest_sha256':sha(PRE/'selection_manifest.json'),'runtime_manifest_sha256':sha(PRE/'runtime_source_manifest.json'),'runner_sha256':sha(runner),'protocol_sha256':sha(PRE/'selection_protocol.md'),'profile_fixture_sha256':sha(HERE/'profile_fixture_checks.json'),'preflight_stdout':r.stdout.strip(),'role_n':len(roles),'test_n':len(test),'unique_profiles_n':36,'four_site_primary_executions':144,'four_site_technical_executions':8,'total_planned':152,'source_drift_guard':True,'same_physical_zone_AST_as_reviewed_v5':True,'expected_wall_seconds':'approximately230 annual-case seconds from prior1.1–1.6s per case, plus shared weather loading and independent checks; bounded to a few minutes'}
(PRE/'readiness.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
