"""Frozen pre-output v2 mixture fit; original v1 files remain read-only."""
from pathlib import Path
import json,hashlib,time
import pandas as pd
import numpy as np
from profile_bridge import make_profile,FIELDS
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DATA=ROOT.parents[1];PRE=ROOT/'05_simulations/archetype_campaign_v1/pre_run'
OUT=ROOT/'05_simulations/archetype_campaign_v2/pre_run'
if (OUT/'selection_manifest.json').exists():raise FileExistsError('Preserving frozen v2 selection')
OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for source,target in [(ROOT/'03_methods/archetype_extension/mixture_protocol_v2.md','selection_protocol.md'),(PRE/'selection_protocol.md','original_v1_protocol.md'),(PRE/'selection_manifest.json','original_v1_selection_manifest.json'),(PRE/'test_respondents.csv','test_respondents.csv'),(PRE/'eligibility.csv','eligibility.csv'),(PRE/'stratum_counts.csv','stratum_counts.csv'),(PRE/'archived_four_archetype_assignments.csv','archived_four_archetype_assignments.csv'),(PRE/'assignment_mapping_schema.json','assignment_mapping_schema.json'),(PRE/'survey_control_eligibility_audit.json','survey_control_eligibility_audit.json'),(HERE/'profile_bridge.py','profile_bridge.py'),(HERE/'profile_fixture_checks.json','profile_fixture_checks.json'),(Path(__file__),'select_mixture_inputs.py')]:
    (OUT/target).write_bytes(source.read_bytes())
roles=pd.read_csv(PRE/'role_manifest.csv');test=set(roles[roles.role=='test'].household_id)
e=pd.read_csv(PRE/'eligibility.csv');e=e[e.eligible]
s=pd.read_csv(DATA/'istat_microdata_2013.csv',sep=';')
d=e.merge(s[FIELDS],left_on='household_id',right_on='id',validate='1:1');d=d[~d.household_id.isin(test)].sort_values('household_id').reset_index(drop=True)
rows=[];vectors=[]
for _,r in d.iterrows():
    p=make_profile(r);vectors.append(p['lower_C'][:24]);rows.append({'household_id':int(r.id),'archetype_id':int(r.archetype_id),'branch':p['control_branch']})
t=pd.DataFrame(rows);x=np.array(vectors)
components=[];centroids={}
for branch in sorted(t.branch.unique(),key=lambda z:tuple(map(float,z.split('/')))):
    idx=np.flatnonzero(t.branch==branch);mat=x[idx];c=mat.mean(axis=0);cost=((mat-c)**2).sum(axis=1)
    ties=idx[np.flatnonzero(np.isclose(cost,cost.min(),atol=1e-10,rtol=0))];chosen=min(ties,key=lambda i:int(t.iloc[i].household_id));raw=d.iloc[chosen];p=make_profile(raw)
    components.append({'branch':branch,'household_id':int(raw.id),'training_n':len(idx),'profile_sha256':p['profile_sha256'],'daily_high_setpoint':p['daily_high_setpoint'].tolist(),'on_C':p['on_C'],'off_C':p['off_C'],'min_distance_K2':float(cost.min()),'tie_n':len(ties),'response':p['response']})
    centroids[branch]=c.tolist()
counts=pd.crosstab(t.archetype_id,t.branch).reindex(columns=[p['branch'] for p in components]);priors=counts.div(counts.sum(axis=1),axis=0)
heldhash=set(roles[roles.role=='test'].profile_sha256)
unique=heldhash|{c['profile_sha256'] for c in components}
result={'status':'post-input/pre-output design assessment, no annual simulations','training_n':len(d),'test_ids_unchanged':sorted(test),'components':components,'counts':counts.to_dict(orient='index'),'training_P_branch_given_group':priors.to_dict(orient='index'),'pooled_P_branch':(counts.sum(axis=0)/len(d)).to_dict(),'component_centroids':centroids,'unique_profiles_if_shared_components':len(unique),'primary_executions_four_sites':4*len(unique),'checks_four_sites':8,'total_executions':4*len(unique)+8,'v1_selection_manifest_sha256':hashlib.sha256((PRE/'selection_manifest.json').read_bytes()).hexdigest()}

# Save actual roles and full profiles, never simulate centroids or mean curves.
newroles=roles[roles.role=='test'].copy().drop(columns=['profile_id','profile_sha256']).to_dict(orient='records')
for c in components:
    row={k:v for k,v in c['response'].items() if k!='Fascia_clima'}
    newroles.append({'role_id':'component_'+c['branch'].replace('/','_'),'role':'component','household_id':c['household_id'],'archetype_id':int(t[t.household_id==c['household_id']].iloc[0].archetype_id),'control_branch':c['branch'],**row})
allraw=pd.read_csv(DATA/'istat_microdata_2013.csv',sep=';').set_index('id',drop=False)
profiles={}
for role in newroles:
    p=make_profile(allraw.loc[int(role['household_id'])]);pid='profile_'+p['profile_sha256'][:16]
    role['profile_id']=pid;role['profile_sha256']=p['profile_sha256']
    if pid not in profiles:
        profiles[pid]={'profile_id':pid,'profile_sha256':p['profile_sha256'],'response':p['response'],'on_C':p['on_C'],'off_C':p['off_C'],'control_branch':p['control_branch'],'daily_high_setpoint':p['daily_high_setpoint'].tolist(),'equivalent_role_ids':[],'source_household_ids':[]}
        pd.DataFrame({'hour':np.arange(8760),'lower_C':p['lower_C']}).to_csv(OUT/f'{pid}.csv',index=False)
    assert profiles[pid]['profile_sha256']==p['profile_sha256']
    profiles[pid]['equivalent_role_ids'].append(role['role_id']);profiles[pid]['source_household_ids'].append(role['household_id'])
roleframe=pd.DataFrame(newroles);assert len(roleframe)==36 and len(profiles)==36
roleframe.to_csv(OUT/'role_manifest.csv',index=False)
roleframe[roleframe.role=='component'].to_csv(OUT/'component_manifest.csv',index=False)
probabilities={'branch_order':[c['branch'] for c in components],'training_n':len(d),'group_branch_training_counts':counts.to_dict(orient='index'),'training_P_branch_given_group':priors.to_dict(orient='index'),'pooled_P_branch':(counts.sum(axis=0)/len(d)).to_dict(),'component_role_ids':{c['branch']:'component_'+c['branch'].replace('/','_') for c in components},'technical_reference_role_id':'component_22_19','free_probability_parameters':{'pooled':3,'conditional':12}}
for g in probabilities['training_P_branch_given_group']:assert abs(sum(probabilities['training_P_branch_given_group'][g].values())-1)<1e-12
assert abs(sum(probabilities['pooled_P_branch'].values())-1)<1e-12
(OUT/'mixture_probabilities.json').write_text(json.dumps(probabilities,indent=2),encoding='utf-8')
(OUT/'profile_manifest.json').write_text(json.dumps(profiles,indent=2),encoding='utf-8')
(OUT/'component_fit_details.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
base=json.loads((PRE/'selection_manifest.json').read_text())
manifest={'status':'v2 post-input/pre-output amendment frozen; no new thermal output or launch authorization','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol_sha256':sha(OUT/'selection_protocol.md'),'original_v1_selection_manifest_sha256':sha(PRE/'selection_manifest.json'),'input_sha256':base['input_sha256'],'code_sha256':{'selection_script':sha(__file__),'profile_bridge':sha(HERE/'profile_bridge.py')},'seed_for_preserved_split':20261020,'assigned_n':17855,'eligible_n':13624,'training_n':13592,'test_n':32,'component_roles_n':4,'role_profiles_n':36,'unique_thermal_profiles_n':36,'site_n':4,'scientific_thermal_executions_planned':144,'repeat_warmup_executions_planned':8,'total_campaign_executions_planned':152,'component_ids':{c['branch']:c['household_id'] for c in components},'test_ids':sorted(test),'python':__import__('sys').version,'numpy':np.__version__,'pandas':pd.__version__,'output_sha256':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='selection_manifest.json'}}
(OUT/'selection_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({k:manifest[k] for k in ['status','protocol_sha256','code_sha256','component_ids','total_campaign_executions_planned']},indent=2))
