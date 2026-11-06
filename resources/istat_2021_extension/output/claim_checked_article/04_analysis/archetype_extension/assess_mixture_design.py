"""Post-selection/pre-output assessment; v1 artifacts are read-only."""
from pathlib import Path
import json,hashlib,time
import pandas as pd
import numpy as np
from profile_bridge import make_profile,FIELDS
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DATA=ROOT.parents[1];PRE=ROOT/'05_simulations/archetype_campaign_v1/pre_run'
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
(HERE/'mixture_design_assessment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['training_n','components','training_P_branch_given_group','pooled_P_branch','unique_profiles_if_shared_components','total_executions']},indent=2))
