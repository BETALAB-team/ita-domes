"""Frozen input-only split and medoids. No thermal energy outputs are read."""
from pathlib import Path
import json,hashlib,shutil,sys,time
import numpy as np
import pandas as pd
from profile_bridge import make_profile,FIELDS,source_sha256
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DATA=ROOT.parents[1]
OUT=ROOT/'05_simulations/archetype_campaign_v1/pre_run'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def integer_dict(row):return {k:int(row[k]) for k in FIELDS}
def main():
    if (OUT/'selection_manifest.json').exists():raise FileExistsError('Preserving existing frozen selection')
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=ROOT/'03_methods/archetype_extension/simulation_protocol.md'
    (OUT/'selection_protocol.md').write_bytes(protocol.read_bytes())
    for p in [Path(__file__),HERE/'profile_bridge.py',HERE/'profile_fixture_checks.json']:(OUT/p.name).write_bytes(p.read_bytes())
    assignment=ROOT/'00_context/archetype_extension/archived_four_archetype_assignments.csv'
    schema=ROOT/'00_context/archetype_extension/assignment_mapping_schema.json'
    raw_audit=ROOT/'00_context/archetype_extension/survey_control_eligibility_audit.json'
    for p in [assignment,schema,raw_audit]:(OUT/p.name).write_bytes(p.read_bytes())
    source=DATA/'istat_microdata_2013.csv';survey=pd.read_csv(source,sep=';')
    columns=FIELDS+['q_3_10']
    a=pd.read_csv(assignment);d=a[['household_id','archetype_id']].merge(survey[columns],left_on='household_id',right_on='id',validate='1:1')
    assert len(d)==17855 and a['household_id'].is_unique and survey['id'].is_unique
    bands=['q_3_11A','q_3_11B','q_3_11C']
    valid=d[bands].notna().all(axis=1)&d[bands].ge(0).all(axis=1)&d[bands].le(8).all(axis=1)&(d[bands]%1==0).all(axis=1)
    d['eligible']=valid&d['q_3_0'].isin([1,2])&(d[bands].sum(axis=1)>0)
    d['exclusion_reason']=['eligible' if ok else ('invalid_blocks' if not v else 'main_system_outside_1_2' if main not in [1,2] else 'zero_total_hours') for ok,v,main in zip(d['eligible'],valid,d['q_3_0'])]
    d[['household_id','archetype_id','eligible','exclusion_reason']].to_csv(OUT/'eligibility.csv',index=False)
    e=d[d['eligible']].copy().sort_values('household_id');assert len(e)==13624 and e['q_3_10'].isin([1,2]).all()
    profiles={};vectors=[];branches=[]
    for _,row in e.iterrows():
        p=make_profile(row);key=int(row['id']);profiles[key]=p;vectors.append(p['lower_C'][:24]);branches.append(p['control_branch'])
    e['control_branch']=branches
    x=np.asarray(vectors,dtype=float);assert x.shape==(13624,24)
    rng=np.random.Generator(np.random.PCG64(20261020))
    tests=[];counts=[];N=len(e)
    for group in [1,2,3,4]:
        g=e[e['archetype_id']==group];Ng=len(g)
        for branch in sorted(g['control_branch'].unique(),key=lambda z:tuple(map(float,z.split('/')))):
            b=g[g['control_branch']==branch].sort_values('household_id');Ngb=len(b)
            picked=rng.choice(b['household_id'].to_numpy(dtype=int),size=2,replace=False)
            counts.append({'archetype_id':group,'control_branch':branch,'eligible_n':Ngb,'group_n':Ng,'domain_n':N})
            for draw,key in enumerate(sorted(picked),1):
                row=b[b['household_id']==key].iloc[0]
                tests.append({'role_id':f'test_g{group}_b{branch.replace("/","_")}_{draw}','role':'test','household_id':int(key),'archetype_id':group,'control_branch':branch,'stratum_N':Ngb,'stratum_test_n':2,'inclusion_probability':2/Ngb,'within_group_weight':Ngb/(2*Ng),'overall_empirical_weight':Ngb/(2*N),**integer_dict(row)})
    test_ids={r['household_id'] for r in tests};assert len(test_ids)==32
    train=e[~e['household_id'].isin(test_ids)].copy();assert len(train)==13592
    id_to_index={int(key):i for i,key in enumerate(e['household_id'])}
    prototypes=[];centroids={}
    for group in [1,2,3,4,0]:
        t=train if group==0 else train[train['archetype_id']==group]
        matrix=x[[id_to_index[int(key)] for key in t['household_id']]]
        centroid=matrix.mean(axis=0);cost=((matrix-centroid)**2).sum(axis=1)
        tied=t.iloc[np.flatnonzero(np.isclose(cost,cost.min(),rtol=0,atol=1e-10))].sort_values('household_id')
        row=tied.iloc[0];key=int(row['household_id']);role='pooled_medoid' if group==0 else 'group_medoid'
        prototypes.append({'role_id':'prototype_pooled' if group==0 else f'prototype_g{group}','role':role,'household_id':key,'archetype_id':int(row['archetype_id']),'prototype_for_group':group,'control_branch':profiles[key]['control_branch'],'training_n':len(t),'squared_input_distance_K2':float(((profiles[key]['lower_C'][:24]-centroid)**2).sum()),'tied_minimum_candidates_n':len(tied),**integer_dict(row)})
        centroids[str(group)]={'training_n':len(t),'centroid_24h_C':centroid.tolist(),'minimum_squared_distance_K2':float(cost.min()),'selected_household_id':key}
    assert not(test_ids & {p['household_id'] for p in prototypes})
    roles=tests+prototypes;unique={}
    for role in roles:
        p=profiles[role['household_id']];digest=p['profile_sha256'];pid='profile_'+digest[:16]
        role['profile_id']=pid;role['profile_sha256']=digest
        if pid not in unique:
            unique[pid]={'profile_id':pid,'profile_sha256':digest,'response':p['response'],'on_C':p['on_C'],'off_C':p['off_C'],'control_branch':p['control_branch'],'daily_high_setpoint':p['daily_high_setpoint'].tolist(),'equivalent_role_ids':[],'source_household_ids':[]}
            pd.DataFrame({'hour':np.arange(8760),'lower_C':p['lower_C']}).to_csv(OUT/f'{pid}.csv',index=False)
        assert unique[pid]['profile_sha256']==digest
        unique[pid]['equivalent_role_ids'].append(role['role_id']);unique[pid]['source_household_ids'].append(role['household_id'])
    pd.DataFrame(roles).to_csv(OUT/'role_manifest.csv',index=False)
    pd.DataFrame(tests).to_csv(OUT/'test_respondents.csv',index=False)
    pd.DataFrame(prototypes).to_csv(OUT/'prototype_manifest.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/'stratum_counts.csv',index=False)
    (OUT/'profile_manifest.json').write_text(json.dumps(unique,indent=2),encoding='utf-8')
    (OUT/'training_input_centroids.json').write_text(json.dumps(centroids,indent=2),encoding='utf-8')
    frame=pd.DataFrame(tests)
    weight_sums={int(g):float(t['within_group_weight'].sum()) for g,t in frame.groupby('archetype_id')}
    assert all(abs(v-1)<1e-12 for v in weight_sums.values()) and abs(frame['overall_empirical_weight'].sum()-1)<1e-12
    assert frame.groupby(['archetype_id','control_branch']).size().eq(2).all()
    manifest={'status':'input selection frozen; no new thermal outputs read or annual run authorized','seed':20261020,'rng':'NumPy Generator PCG64','numpy_version':np.__version__,'pandas_version':pd.__version__,'python':sys.version,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol_sha256':sha(OUT/'selection_protocol.md'),'input_sha256':{'assignment':sha(assignment),'assignment_schema':sha(schema),'compact_survey':sha(source),'raw_control_audit':sha(raw_audit),'EndUse.py':source_sha256()},'code_sha256':{'selection_script':sha(Path(__file__)),'profile_bridge':sha(HERE/'profile_bridge.py')},'assigned_n':17855,'eligible_n':13624,'training_n':13592,'test_n':32,'prototype_roles_n':5,'role_profiles_n':37,'unique_thermal_profiles_n':len(unique),'site_n':4,'scientific_thermal_executions_planned':4*len(unique),'repeat_warmup_executions_planned':8,'total_campaign_executions_planned':4*len(unique)+8,'within_group_weight_sums':weight_sums,'overall_empirical_weight_sum':float(frame['overall_empirical_weight'].sum()),'split_no_overlap':True,'prototype_ids':[p['household_id'] for p in prototypes],'test_ids':sorted(test_ids),'output_sha256':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='selection_manifest.json'}}
    (OUT/'selection_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['eligible_n','training_n','test_n','prototype_roles_n','unique_thermal_profiles_n','total_campaign_executions_planned','prototype_ids','protocol_sha256','code_sha256']},indent=2))
if __name__=='__main__':main()
