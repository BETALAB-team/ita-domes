"""Reviewer05 independent raw-file calculations; no production/checker imports."""
from pathlib import Path
import hashlib,json,random
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parents[3]
O=Path(__file__).resolve().parent
P=R/'05_simulations/archetype_campaign_v2/pre_run'
seen={}
def record(p):
    seen[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return p
def csv(p,**kw):return pd.read_csv(record(p),**kw)
def js(p):return json.loads(record(p).read_text())
for p in [R/'06_claim_drafts/article_claims_v05.md',R/'06_claim_drafts/claim_ledger_v05.json']:
    record(p)
assign=csv(R/'00_context/archetype_extension/archived_four_archetype_assignments.csv')
survey=csv(R.parents[1]/'istat_microdata_2013.csv',sep=';')
d=assign.merge(survey,left_on='household_id',right_on='id',validate='1:1')
hours=['q_3_11A','q_3_11B','q_3_11C']
ok=d.q_3_0.isin([1,2]) & d[hours].apply(lambda x:x.between(0,8)&(x%1==0)).all(axis=1)&(d[hours].sum(axis=1)>0)
d=d[ok].copy()
control=d[['q_3_13_2','q_3_13_3','q_3_19_6']].eq(1).any(axis=1)
fire=d[['q_2_32C','q_2_32D']].eq(1).any(axis=1)
d['branch']=[f'{22+2*f}/{11+8*c+2*f}' for c,f in zip(control.astype(int),fire.astype(int))]
roles=csv(P/'role_manifest.csv')
tests=roles[roles.role=='test']; comps=roles[roles.role=='component']
assert len(d)==13624 and len(tests)==32 and tests.household_id.nunique()==32
train=d[~d.household_id.isin(tests.household_id)].copy()
assert not set(comps.household_id)&set(tests.household_id)
counts=pd.crosstab(d.archetype_id,d.branch)
tc=pd.crosstab(train.archetype_id,train.branch)
prob=js(P/'mixture_probabilities.json')
for g in tc.index:
    for b in tc.columns:
        assert tc.loc[g,b]==prob['group_branch_training_counts'][str(g)][b]
        assert abs(tc.loc[g,b]/tc.loc[g].sum()-prob['training_P_branch_given_group'][str(g)][b])<1e-14
def daily(row):
    seg=[]
    for k in hours:
        n=int(row[k]); start=random.Random(int(row.id)).randint(0,8-n)
        a=np.zeros(8);a[start:start+n]=1;seg.append(a)
    a=np.r_[seg[2][-5:],seg[0],seg[1],seg[2][:3]]
    f=int(any(row[k]==1 for k in ['q_2_32C','q_2_32D']))
    c=int(any(row[k]==1 for k in ['q_3_13_2','q_3_13_3','q_3_19_6']))
    lo=11+8*c+2*f;hi=22+2*f
    return lo+(hi-lo)*a
vectors=np.stack([daily(r) for _,r in train.iterrows()])
selected={}
for b in tc.columns:
    mask=(train.branch==b).to_numpy()
    x=vectors[mask];rows=train[mask];cost=((x-x.mean(axis=0))**2).sum(axis=1)
    ids=rows.loc[np.isclose(cost,cost.min(),atol=1e-10,rtol=0),'id']
    chosen=int(ids.min())
    assert chosen==int(comps[comps.control_branch==b].household_id.iloc[0])
    selected[b]=chosen
for _,r in roles.iterrows():
    row=d[d.id==r.household_id].iloc[0]
    full=np.tile(daily(row),365);full[105*24:288*24]=0
    saved=csv(P/f'{r.profile_id}.csv').lower_C.to_numpy()
    assert np.array_equal(full,saved)
    assert hashlib.sha256(full.astype('<f8').tobytes()).hexdigest()==r.profile_sha256
    if r.role=='test':
        assert abs(r.overall_empirical_weight-counts.loc[r.archetype_id,r.control_branch]/(2*len(d)))<1e-14
        assert abs(r.within_group_weight-counts.loc[r.archetype_id,r.control_branch]/(2*counts.loc[r.archetype_id].sum()))<1e-14
metrics=[];runchecks=[];endpoints=[];maxres=0.
published=csv(R/'04_analysis/archetype_extension/results_v2/weighted_metrics.csv')
for site in ['Padova','Bolzano','Roma','Palermo']:
    root=R/'05_simulations/archetype_campaign_v2'/site
    vals={}
    forcing=csv(root/'surface_forcing_hourly.csv')
    for path in sorted(root.glob('*_hourly.csv')):
        if path.name=='surface_forcing_hourly.csv':continue
        h=csv(path);assert len(h)==8760
        q=h.sensible_load_W.to_numpy(); a=h.air_temperature_C.to_numpy()
        s=h.surface_C.to_numpy(); m=h.mass_C.to_numpy();e=h.outdoor_temperature_C.to_numpy()
        par=js(root/(path.name.replace('_hourly.csv','_network_parameters.json')))
        ra=h.phi_air_W+h.sigma_convective*q+par['Htr_is']*(s-a)+h.infiltration_H_W_K*(e-a)-par['_air_thermal_capacity']*(a-h.old_air_C)/3600
        rs=h.phi_surface_W+h.sigma_radiant*q+par['Htr_is']*(a-s)+par['Htr_w']*(e-s)+par['Htr_ms']*(m-s)
        rm=h.phi_mass_W+par['Htr_ms']*(s-m)+par['Htr_em']*(e-m)-par['Cm']*(m-h.old_mass_C)/3600
        residual=max(abs(ra).max(),abs(rs).max(),abs(rm).max())
        maxres=max(maxres,float(residual));assert residual<1e-6
        assert np.array_equal(h.old_air_C.to_numpy()[1:],a[:-1])
        assert np.array_equal(h.old_mass_C.to_numpy()[1:],m[:-1])
        for col in ['phi_air_W','phi_surface_W','phi_mass_W']:assert np.max(np.abs(h[col]-forcing[col]))<1e-8
        key=path.name.replace('_hourly.csv','')
        vals[key]=[float(np.maximum(q,0).sum()/1000),float(np.maximum(q,0).max()/1000)]
        runchecks.append({'site':site,'case':key,'rows':len(h),'residual_W':float(residual),'peak_hour':int(q.argmax())})
        if key.startswith('profile_'):
            inp=csv(P/f'{key}.csv');assert np.array_equal(inp.lower_C,h.heating_setpoint_C)
            endpoints.append({'site':site,'profile_id':key,'annual_kWh':vals[key][0],'maximum_kW':vals[key][1]})
    for j,endpoint in enumerate(['annual_sensible_heating_kWh','peak_sensible_heating_kW']):
        bybranch={r.control_branch:vals[r.profile_id][j] for _,r in comps.iterrows()}
        pooled=sum(tc[b].sum()/len(train)*v for b,v in bybranch.items())
        cond={g:sum(tc.loc[g,b]/tc.loc[g].sum()*v for b,v in bybranch.items()) for g in tc.index}
        for g in [0,1,2,3,4]:
            sub=tests if g==0 else tests[tests.archetype_id==g]
            w=sub.overall_empirical_weight.to_numpy() if g==0 else sub.within_group_weight.to_numpy()
            refs=np.array([vals[r.profile_id][j] for _,r in sub.iterrows()])
            pred=np.array([cond[r.archetype_id] for _,r in sub.iterrows()])
            cm=float(np.dot(w,(pred-refs)**2));pm=float(np.dot(w,(pooled-refs)**2))
            domain='all_eligible_empirical_groups' if g==0 else f'archetype_{g}'
            old=published[(published.site==site)&(published.endpoint==endpoint)&(published.domain==domain)].iloc[0]
            assert abs(cm-old.conditional_MSE)<1e-7 and abs(pm-old.pooled_MSE)<1e-7
            metrics.append({'site':site,'endpoint':endpoint,'domain':domain,'conditional_RMSE':cm**.5,'pooled_RMSE':pm**.5,'difference_MSE':cm-pm})
pd.DataFrame(metrics).to_csv(O/'metrics.csv',index=False)
pd.DataFrame(endpoints).to_csv(O/'endpoints.csv',index=False)
pd.DataFrame(runchecks).to_csv(O/'run_checks.csv',index=False)
(O/'audit.json').write_text(json.dumps({'assignment_n':len(assign),'eligible_n':len(d),'training_n':len(train),'selected_components':selected,'annual_files':len(runchecks),'hours':sum(x['rows'] for x in runchecks),'max_node_residual_W':maxres,'all_40_metrics_match':True,'checks':'own calculations; no imported production or reviewer helper','sha256':seen},indent=2))
print(json.dumps({'files':len(runchecks),'max_residual_W':maxres,'overall':[x for x in metrics if x['domain']=='all_eligible_empirical_groups'],'worse_subgroups':[x for x in metrics if x['domain']!='all_eligible_empirical_groups' and x['difference_MSE']>0]},indent=2))
