"""Independent v2 frozen input verification; no annual solver execution."""
import runpy,json,hashlib,ast,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
v=runpy.run_path(str(HERE/'independent_input_check.py'))
ROOT=v['ROOT'];pre=ROOT/'05_simulations/archetype_campaign_v2/pre_run'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((pre/'selection_manifest.json').read_text())
for name,digest in m['output_sha256'].items():assert sha(pre/name)==digest,name
assert sha(ROOT/'03_methods/archetype_extension/mixture_protocol_v2.md')==m['protocol_sha256']
assert sha(pre/'original_v1_selection_manifest.json')==sha(v['PRE']/'selection_manifest.json')
assert sha(pre/'test_respondents.csv')==sha(v['PRE']/'test_respondents.csv')
prob=json.loads((pre/'mixture_probabilities.json').read_text())
fit=json.loads((pre/'component_fit_details.json').read_text())
train=v['train'];idx=v['indexed']
for c in fit['components']:
    t=train[train.branch==c['branch']]
    mat=np.array([idx[i] for i in t.household_id]);mu=mat.mean(axis=0);dist=((mat-mu)**2).sum(axis=1)
    ties=t.household_id.to_numpy()[abs(dist-dist.min())<=1e-10]
    assert int(min(ties))==c['household_id'] and len(ties)==c['tie_n']
    b=c['branch']
    assert abs(prob['pooled_P_branch'][b]-len(t)/len(train))<1e-15
    for g in [1,2,3,4]:
        assert abs(prob['training_P_branch_given_group'][str(g)][b]-sum(t.archetype_id==g)/sum(train.archetype_id==g))<1e-15
roles=pd.read_csv(pre/'role_manifest.csv')
assert len(roles)==36 and roles.role_id.is_unique and roles.profile_sha256.nunique()==36
assert set(roles[roles.role=='test'].household_id)==set(v['picked'])
assert not set(roles[roles.role=='component'].household_id)&set(v['picked'])
for _,r in roles.iterrows():
    day=idx[r.household_id];full=np.r_[np.tile(day,105),np.zeros(183*24),np.tile(day,77)]
    csv=pd.read_csv(pre/f'{r.profile_id}.csv')
    assert np.array_equal(csv.lower_C,full) and np.array_equal(csv.hour,np.arange(8760))
    assert hashlib.sha256(full.astype('<f8').tobytes()).hexdigest()==r.profile_sha256
runtime=json.loads((pre/'runtime_source_manifest.json').read_text())
for name,digest in runtime['sha256'].items():assert sha(name)==digest,name
runner=ROOT/'04_analysis/archetype_extension/run_mixture_campaign.py'
assert sha(runner)==sha(pre/'run_mixture_campaign.py')
def zone_ast(p):
    return ast.dump(next(n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='zone'),include_attributes=False)
assert zone_ast(runner)==zone_ast(ROOT/'04_analysis/run_climate_campaign_v5.py')
analysis_path=ROOT/'04_analysis/archetype_extension/analyze_mixtures.py'
analysis=runpy.run_path(str(analysis_path))
loss=analysis['loss']
fixture=[{'w':0.2,'detailed_reference':0.,'conditional_prediction':3.,'pooled_prediction':1.},
         {'w':0.8,'detailed_reference':5.,'conditional_prediction':3.,'pooled_prediction':1.}]
calc=loss(fixture,'w')
assert abs(calc['conditional_MSE']-5.)<1e-14 and abs(calc['pooled_MSE']-13.)<1e-14
assert abs(calc['paired_MSE_difference_conditional_minus_pooled']+8.)<1e-14
assert abs(calc['conditional_MAE']-2.2)<1e-14 and abs(calc['conditional_signed_deviation']+1.)<1e-14
assert loss([{'w':1.,'detailed_reference':0.,'conditional_prediction':0.,'pooled_prediction':0.}],'w')['conditional_RMSE']==0.
command=[sys.executable,'-B',str(runner),'Padova','--campaign','archetype_campaign_v2','--preflight']
run=subprocess.run(command,capture_output=True,text=True)
assert run.returncode==0,run.stdout+run.stderr
out={'pass':True,'annual_execution':False,'command':command,'preflight_stdout':run.stdout.strip(),
     'protocol_sha256':m['protocol_sha256'],'selection_manifest_sha256':sha(pre/'selection_manifest.json'),
     'runner_sha256':sha(runner),'runtime_manifest_sha256':sha(pre/'runtime_source_manifest.json'),
     'fit_script_sha256':m['code_sha256']['selection_script'],'role_manifest_sha256':sha(pre/'role_manifest.csv'),
     'analyzer_sha256':sha(analysis_path),'analytical_loss_fixture_pass':True,
     'probability_manifest_sha256':sha(pre/'mixture_probabilities.json'),
     'checks':['all frozen selection outputs and runtime sources','v1 holdout unchanged','training-only branch medoids and ties','independent training probabilities','36 full source-rule profiles independently reconstructed','unchanged physical-zone AST','actual runner preflight']}
(HERE/'v2_frozen_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
