"""Independent input arithmetic only; does not import or execute thermal model."""
from pathlib import Path
import hashlib,json,random
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2];DATA=ROOT.parents[1]
PRE=ROOT/'05_simulations/archetype_campaign_v1/pre_run'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((PRE/'selection_manifest.json').read_text())
for name,h in m['output_sha256'].items():assert sha(PRE/name)==h,name
assert sha(ROOT/'03_methods/archetype_extension/simulation_protocol.md')==m['protocol_sha256']
for key,name in [('selection_script','select_inputs.py'),('profile_bridge','profile_bridge.py')]:
    assert sha(ROOT/'04_analysis/archetype_extension'/name)==m['code_sha256'][key]
a=pd.read_csv(DATA/'clusters.csv');mapping=pd.read_csv(PRE/'archived_four_archetype_assignments.csv')
assert dict(zip(a.identi,a.segment+1))==dict(zip(mapping.household_id,mapping.archetype_id))
s=pd.read_csv(DATA/'istat_microdata_2013.csv',sep=';')
r=pd.read_csv(DATA/'istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt',sep='\t',low_memory=False)
d=mapping[['household_id','archetype_id']].merge(s,left_on='household_id',right_on='id',validate='one_to_one')
bands=['q_3_11A','q_3_11B','q_3_11C'];controls=['q_3_13_2','q_3_13_3','q_3_19_6'];fires=['q_2_32C','q_2_32D']
for c in bands+controls+fires+['q_3_0']:r[c]=pd.to_numeric(r[c],errors='coerce')
def eligible(t):
    b=t[bands]
    return t.q_3_0.isin([1,2])&b.notna().all(axis=1)&b.ge(0).all(axis=1)&b.le(8).all(axis=1)&b.mod(1).eq(0).all(axis=1)&b.sum(axis=1).gt(0)
e=d[eligible(d)].sort_values('household_id').copy()
r=r.rename(columns={'nprog':'household_id'}).merge(mapping[['household_id','archetype_id']],on='household_id',validate='one_to_one')
assert set(r.loc[eligible(r),'household_id'])==set(e.household_id)
assert len(e)==13624
raw=r.set_index('household_id').loc[e.household_id]
for fields in [bands]:
    assert np.array_equal(raw[fields].to_numpy(),e[fields].to_numpy())
for fields in [controls,fires]:
    assert np.array_equal(raw[fields].eq(1).any(axis=1).to_numpy(),e[fields].eq(1).any(axis=1).to_numpy())
def vector(row):
    blocks=[]
    for name in bands:
        n=int(row[name]);start=random.Random(int(row['id'])).randint(0,8-n)
        blocks.append(np.r_[np.zeros(start),np.ones(n),np.zeros(8-start-n)])
    mt,pm,nt=blocks
    daily=np.r_[nt[-5:],mt,pm,nt[:3]]
    on=22+2*any(row[k]==1 for k in fires)
    off=11+8*any(row[k]==1 for k in controls)+2*any(row[k]==1 for k in fires)
    return off+(on-off)*daily,f'{on}/{off}'
vs=[vector(row) for _,row in e.iterrows()]
x=np.array([v[0] for v in vs]);e['branch']=[v[1] for v in vs]
indexed=dict(zip(e.household_id,x))
test=pd.read_csv(PRE/'test_respondents.csv');proto=pd.read_csv(PRE/'prototype_manifest.csv')
rng=np.random.Generator(np.random.PCG64(20261020));picked=[]
for (g,b),cell in e.groupby(['archetype_id','branch'],sort=True):
    ids=sorted(rng.choice(cell.household_id.to_numpy(),2,replace=False).tolist())
    actual=test[(test.archetype_id==g)&(test.control_branch==b)]
    assert ids==sorted(actual.household_id.tolist())
    assert np.allclose(actual.overall_empirical_weight,len(cell)/(2*len(e)),rtol=0,atol=1e-15)
    assert np.allclose(actual.within_group_weight,len(cell)/(2*sum(e.archetype_id==g)),rtol=0,atol=1e-15)
    picked+=ids
train=e[~e.household_id.isin(picked)]
centroids=json.loads((PRE/'training_input_centroids.json').read_text());medoids={}
for g in [1,2,3,4,0]:
    t=train if g==0 else train[train.archetype_id==g]
    matrix=np.array([indexed[i] for i in t.household_id]);mu=matrix.mean(axis=0)
    dist=np.sum((matrix-mu)**2,axis=1)
    ties=t.household_id.to_numpy()[abs(dist-dist.min())<=1e-10];key=int(min(ties))
    p=proto[proto.prototype_for_group==g].iloc[0]
    assert key==p.household_id and len(ties)==p.tied_minimum_candidates_n
    assert np.allclose(mu,centroids[str(g)]['centroid_24h_C'],rtol=0,atol=1e-12)
    medoids[g]={'id':key,'ties':len(ties),'distance':float(dist.min())}
roles=pd.read_csv(PRE/'role_manifest.csv')
for _,role in roles.iterrows():
    day=indexed[role.household_id]
    full=np.r_[np.tile(day,105),np.zeros(183*24),np.tile(day,77)]
    actual=pd.read_csv(PRE/f'{role.profile_id}.csv')
    assert np.array_equal(actual.hour,np.arange(8760))
    assert np.array_equal(actual.lower_C,full)
    assert hashlib.sha256(full.astype('<f8').tobytes()).hexdigest()==role.profile_sha256
assert len(set(picked))==32 and not set(picked)&set(proto.household_id)
assert proto.profile_sha256.nunique()==1
out={'pass':True,'checks':['all frozen output hashes','live protocol and selection/bridge hashes','archived mapping exact','raw/compact eligible ID equality','all eligible raw hour and branch predicates','independent seeded draws','weight formulas','training-only centroid squared-distance medoids and tie counts','independent full-year profiles for every role'],
     'eligible_n':len(e),'training_n':len(train),'test_n':len(picked),'unique_profiles_n':roles.profile_sha256.nunique(),
     'medoids':medoids,'distinct_prototype_profiles':int(proto.profile_sha256.nunique()),
     'prototype_collapse_sha256':str(proto.profile_sha256.iloc[0]),
     'protocol_sha256':m['protocol_sha256'],'code_sha256':m['code_sha256'],'selection_manifest_sha256':sha(PRE/'selection_manifest.json'),
     'reviewer_script_sha256':sha(__file__)}
(HERE/'independent_input_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
