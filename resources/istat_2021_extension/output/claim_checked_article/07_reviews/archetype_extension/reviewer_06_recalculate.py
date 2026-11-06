"""Independent review06 calculations; no imports of study analysis or engine."""
from pathlib import Path
import json, hashlib, random, math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
PRE=ROOT/'05_simulations/archetype_campaign_v2/pre_run'
DATA=ROOT.parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
report={}
report['input_hashes']={n:sha(ROOT/'06_claim_drafts'/n) for n in ['article_claims_v06.md','claim_ledger_v06.json','claim_evidence_ledger_v06.json']}
report['registry_sha256']=sha(ROOT/'02_evidence/archetype_extension/local_sources.json')
runtime=read(PRE/'runtime_source_manifest.json')
report['runtime_hash_count']=len(runtime['sha256'])
report['runtime_hash_mismatches']=[p for p,h in runtime['sha256'].items() if sha(p)!=h]
sel=read(PRE/'selection_manifest.json')
report['frozen_selection_mismatches']=[p for p,h in sel['output_sha256'].items() if sha(PRE/p)!=h]
survey=pd.read_csv(DATA/'istat_microdata_2013.csv',sep=';').set_index('id',drop=False)
assigned=pd.read_csv(PRE/'archived_four_archetype_assignments.csv')
raw=pd.read_csv(DATA/'istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt',sep='\t',low_memory=False)
report['join']={'assigned':len(assigned),'unique_assigned':assigned.household_id.nunique(),'compact_rows':len(survey),'raw_rows':len(raw),'compact_missing':len(set(assigned.household_id)-set(survey.index)),'raw_missing':len(set(assigned.household_id)-set(raw.nprog))}
d=assigned[['household_id','archetype_id']].merge(survey.reset_index(drop=True),left_on='household_id',right_on='id',validate='1:1').sort_values('id')
bands=['q_3_11A','q_3_11B','q_3_11C']
eligible=d[d.q_3_0.isin([1,2]) & d[bands].notna().all(axis=1) & d[bands].ge(0).all(axis=1) & d[bands].le(8).all(axis=1) & (d[bands]%1==0).all(axis=1) & d[bands].sum(axis=1).gt(0)].copy()
eligible['on']=22+2*((eligible.q_2_32C==1)|(eligible.q_2_32D==1))
eligible['off']=11+8*((eligible.q_3_13_2==1)|(eligible.q_3_13_3==1)|(eligible.q_3_19_6==1))+2*((eligible.q_2_32C==1)|(eligible.q_2_32D==1))
eligible['branch']=[f'{a}/{b}' for a,b in zip(eligible.on,eligible.off)]
counts=eligible.groupby(['archetype_id','branch']).size()
report['eligible_counts']={str(k):int(v) for k,v in counts.items()}
roles=pd.read_csv(PRE/'role_manifest.csv'); tests=roles[roles.role=='test']; components=roles[roles.role=='component']
rng=np.random.Generator(np.random.PCG64(20261020))
picks=[]
for (g,b), frame in eligible.groupby(['archetype_id','branch'],sort=True):
    picks.extend(rng.choice(frame.id.to_numpy(),2,replace=False).tolist())
report['split_exact']=set(picks)==set(tests.household_id)
train=eligible[~eligible.id.isin(picks)]
tc=train.groupby(['archetype_id','branch']).size().unstack()
prob=read(PRE/'mixture_probabilities.json')
report['probability_error']=max(abs(float(tc.loc[int(g),b]/tc.loc[int(g)].sum())-v) for g,ps in prob['training_P_branch_given_group'].items() for b,v in ps.items())
report['weight_error']=max(abs(float(r.overall_empirical_weight)-counts.loc[r.archetype_id,r.control_branch]/(2*len(eligible))) for r in tests.itertuples())
def daily(r):
    seq=[]
    for h in [int(r[k]) for k in bands]:
        start=random.Random(int(r['id'])).randint(0,8-h)
        seq.append([0]*start+[1]*h+[0]*(8-start-h))
    mt,pm,nt=seq
    x=np.array(nt[-5:]+mt+pm+nt[:3],dtype=float)
    on=22+2*(r['q_2_32C']==1 or r['q_2_32D']==1)
    off=11+8*any(r[k]==1 for k in ['q_3_13_2','q_3_13_3','q_3_19_6'])+2*(r['q_2_32C']==1 or r['q_2_32D']==1)
    return off+(on-off)*x
report['medoids']={}
for branch,frame in train.groupby('branch'):
    arr=np.stack([daily(r) for _,r in frame.iterrows()])
    cost=((arr-arr.mean(axis=0))**2).sum(axis=1)
    chosen=int(frame.iloc[np.flatnonzero(abs(cost-cost.min())<=1e-10)].id.min())
    report['medoids'][branch]={'id':chosen,'minimum_distance_K2':float(cost.min())}
profile_errors=[]
for r in roles.itertuples():
    profile=np.tile(daily(survey.loc[r.household_id]),365);profile[105*24:288*24]=0
    frozen=pd.read_csv(PRE/(r.profile_id+'.csv')).lower_C.to_numpy()
    if not np.array_equal(profile,frozen) or hashlib.sha256(profile.astype('<f8').tobytes()).hexdigest()!=r.profile_sha256:profile_errors.append(r.role_id)
report['profile_errors']=profile_errors
report['sites']={}
for site in ['Padova','Bolzano','Roma','Palermo']:
    run=ROOT/'05_simulations/archetype_campaign_v2'/site
    design=read(run/'design.json')
    summ={s['case']:s for s in read(run/'summary.json')}
    endpoints={};peak_hours={};maxres=np.zeros(3);state_error=0;forcing_error=0;load_at_wrong_setpoint=0;hash_fail=[];rows=0;cool=0
    forcing=pd.read_csv(run/'surface_forcing_hourly.csv')
    surf=read(run/'surface_forcing_parameters.json')
    solar=np.zeros(8760);max_surface_error=0
    for name,s in surf['surfaces'].items():
        total=forcing[name+'::irradiance_global_W_m2'].to_numpy()
        direct=forcing[name+'::irradiance_direct_W_m2'].to_numpy()
        sw=s['shading']*total*s['absorptance']*s['Rse_m2_K_W']*s['U_net_W_m2_K']*s['opaque_area_m2']
        lw=-s['sky_view_factor']*s['Rse_m2_K_W']*s['U_net_W_m2_K']*s['opaque_area_m2']*s['h_r_W_m2_K']*s['mean_air_minus_sky_K']
        glass=0
        if s['glazed_area_m2']>0:
            gd=forcing[name+'::SHGC_direct'].to_numpy();gf=forcing[name+'::SHGC_diffuse'].to_numpy()
            factor=s['window_shading_external']*s['window_shading_internal']*(1-s['frame_factor'])*s['glazed_area_m2']
            glass=(s['shading']*direct*gd+(total-direct)*gf)*factor
        max_surface_error=max(max_surface_error,float(np.max(abs(sw-forcing[name+'::opaque_shortwave_W']))),float(np.max(abs(lw-forcing[name+'::opaque_longwave_W']))),float(np.max(abs(glass-forcing[name+'::window_shortwave_W']))))
        solar+=sw+lw+glass
    for case,warmup in design['cases']:
        p=run/(case+'_hourly.csv');h=pd.read_csv(p);rows+=len(h)
        assert len(h)==8760 and np.array_equal(h.hour,np.arange(8760)) and np.isfinite(h.to_numpy()).all()
        q=h.sensible_load_W.to_numpy();cool+=float(-np.minimum(q,0).sum())
        endpoints[case]=[math.fsum(np.maximum(q,0))/1000,float(np.maximum(q,0).max()/1000)]
        peak_hours[case]=np.flatnonzero(q==q.max()).tolist()
        if sha(p)!=summ[case]['hourly_sha256']:hash_fail.append(case)
        par=read(run/(case+'_network_parameters.json'))
        ta=h.air_temperature_C.to_numpy();tm=h.mass_C.to_numpy();ts=h.surface_C.to_numpy();te=h.outdoor_temperature_C.to_numpy()
        ra=par['_air_thermal_capacity']/3600*(ta-h.old_air_C)-par['Htr_is']*(ts-ta)-h.infiltration_H_W_K*(te-ta)-h.phi_air_W-q*h.sigma_convective
        rs=par['Htr_is']*(ta-ts)+par['Htr_w']*(te-ts)+par['Htr_ms']*(tm-ts)+h.phi_surface_W+q*h.sigma_radiant
        rm=par['Cm']/3600*(tm-h.old_mass_C)-par['Htr_em']*(te-tm)-par['Htr_ms']*(ts-tm)-h.phi_mass_W
        maxres=np.maximum(maxres,[abs(ra).max(),abs(rs).max(),abs(rm).max()])
        state_error=max(state_error,float(np.max(abs(ta[:-1]-h.old_air_C.to_numpy()[1:]))),float(np.max(abs(tm[:-1]-h.old_mass_C.to_numpy()[1:]))))
        expected=pd.read_csv(PRE/(design['case_to_profile_id'][case]+'.csv')).lower_C
        assert np.array_equal(h.heating_setpoint_C,expected)
        load_at_wrong_setpoint=max(load_at_wrong_setpoint,float(np.max(abs(ta[q>1e-6]-h.heating_setpoint_C[q>1e-6]))))
        for col in ['phi_air_W','phi_surface_W','phi_mass_W']:
            forcing_error=max(forcing_error,float(np.max(abs(h[col]-forcing[col]))))
    metrics=[]
    for j,label in enumerate(['annual_kWh','peak_kW']):
        values={r.control_branch:endpoints[r.profile_id][j] for r in components.itertuples()}
        pool=sum(train.branch.value_counts()[b]/len(train)*v for b,v in values.items())
        cp={g:sum(tc.loc[g,b]/tc.loc[g].sum()*v for b,v in values.items()) for g in tc.index}
        for group in [0,1,2,3,4]:
            subset=tests if group==0 else tests[tests.archetype_id==group]
            w=np.array([counts.loc[r.archetype_id,r.control_branch]/(2*(len(eligible) if group==0 else (eligible.archetype_id==group).sum())) for r in subset.itertuples()])
            y=np.array([endpoints[r.profile_id][j] for r in subset.itertuples()])
            c=np.array([cp[r.archetype_id] for r in subset.itertuples()])
            mse_c=float(sum(w*(c-y)**2));mse_p=float(sum(w*(pool-y)**2))
            metrics.append({'endpoint':label,'group':group,'pooled_RMSE':math.sqrt(mse_p),'conditional_RMSE':math.sqrt(mse_c),'paired_MSE_difference':mse_c-mse_p,'reference_min':float(y.min()),'reference_max':float(y.max()),'conditional_min':min(cp.values()),'conditional_max':max(cp.values()),'pooled_prediction':pool})
    peaks={}
    for role in ['test','component']:
        hh=[peak_hours[r.profile_id] for r in roles[roles.role==role].itertuples()]
        peaks[role]={'n':len(hh),'restart_day':sum(any(6912<=t<6936 for t in a) for a in hh),'restart_hour':sum(6912 in a for a in hh),'january':sum(any(t<744 for t in a) for a in hh),'tied_maxima':sum(len(a)>1 for a in hh)}
    report['sites'][site]={'rows':rows,'max_air_surface_mass_residual_W':maxres.tolist(),'state_continuity_error_K':state_error,'solver_forcing_error_W':forcing_error,'surface_component_error_W':max_surface_error,'active_setpoint_error_K':load_at_wrong_setpoint,'cooling_Wh':cool,'hourly_hash_failures':hash_fail,'metrics':metrics,'peak_counts':peaks}
    print(site,rows,maxres.tolist(),peaks,flush=True)
legacy={}
for site in ['Padova','Bolzano','Roma','Palermo']:
    run=ROOT/'05_simulations/climate_campaign_v5'/site
    legacy[site]={}
    for p in run.glob('*_hourly.csv'):
        h=pd.read_csv(p)
        if 'sensible_load_W' not in h:continue
        q=h.sensible_load_W.clip(lower=0)
        legacy[site][p.name]={'annual_kWh':float(q.sum()/1000),'peak_kW':float(q.max()/1000),'peak_hour':int(q.idxmax()),'max_air_C':float(h.air_temperature_C.max())}
        if site=='Padova' and p.name=='seeded_hourly.csv':
            mask=h.heating_setpoint_C==19
            legacy[site][p.name]['setback']={'calendar_hours':int(mask.sum()),'positive_heat_hours':int((mask&(q>0)).sum()),'heat_kWh':float(q[mask].sum()/1000)}
report['legacy_v5']=legacy
(OUT/'reviewer_06_calculations.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('SAVED',OUT/'reviewer_06_calculations.json')
