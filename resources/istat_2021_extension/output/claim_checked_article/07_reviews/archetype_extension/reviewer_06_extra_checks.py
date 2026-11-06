"""Additional independent source/response, technical and report consistency checks."""
from pathlib import Path
import json, re, hashlib
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT.parents[1];OUT=Path(__file__).parent
PRE=ROOT/'05_simulations/archetype_campaign_v2/pre_run'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=pd.read_csv(DATA/'clusters.csv');assign=pd.read_csv(PRE/'archived_four_archetype_assignments.csv')
report={'archive_hash':sha(DATA/'clusters.csv'),'assignment_hash':sha(PRE/'archived_four_archetype_assignments.csv')}
joined=a.merge(assign,left_on='identi',right_on='household_id',validate='1:1')
report['archive_labels_match']=bool((joined.segment+1==joined.archetype_id).all())
report['archive_demographic_summaries']={}
for g,frame in a.groupby('segment'):
    report['archive_demographic_summaries'][str(g+1)]={'n':len(frame),'household_percent':len(frame)/len(a)*100,'person_percent':frame.number.sum()/a.number.sum()*100,
      'means':frame[['min_age','max_age','number','studio_ind']].mean().to_dict(),
      'SD':frame[['min_age','max_age','number','studio_ind']].std().to_dict()}
raw=pd.read_csv(DATA/'istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt',sep='\t',dtype=str).set_index('nprog')
compact=pd.read_csv(DATA/'istat_microdata_2013.csv',sep=';').set_index('id')
raw=raw.apply(lambda col: pd.to_numeric(col.str.strip(),errors='coerce'))
raw.index=raw.index.astype(int)
bands=['q_3_11A','q_3_11B','q_3_11C']
report['raw_band_columns']=[c for c in raw if '3_11' in c]
if not set(bands)<=set(raw): raw=raw.rename(columns={c:c.replace('q_3_11_','q_3_11') for c in raw if '3_11_' in c})
r=raw.loc[assign.household_id]
e=r[r.q_3_0.isin([1,2]) & r[bands].notna().all(axis=1) & r[bands].ge(0).all(axis=1) & r[bands].le(8).all(axis=1) & (r[bands]%1==0).all(axis=1) & r[bands].sum(axis=1).gt(0)]
report['raw_eligible_n']=len(e)
frozen=pd.read_csv(PRE/'eligibility.csv')
report['raw_eligible_ids_exact']=set(e.index)==set(frozen.loc[frozen.eligible,'household_id'])
controls=['q_3_13_2','q_3_13_3','q_3_19_6','q_2_32C','q_2_32D']
report['raw_to_compact_fields']={}
for c in bands+controls:
    u=e[c];v=compact.loc[e.index,c]
    report['raw_to_compact_fields'][c]={'observed_mismatch_n':int((u.notna()&(u!=v)).sum()),'blank_to_zero_n':int((u.isna()&(v==0)).sum()),'affirmative_predicate_mismatch_n':int(((u==1)!=(v==1)).sum())}
calc=read(OUT/'reviewer_06_calculations.json')
metrics=pd.read_csv(ROOT/'04_analysis/archetype_extension/results_v2/weighted_metrics.csv')
maxerr=0
for s,d in calc['sites'].items():
    for m in d['metrics']:
        ep='annual_sensible_heating_kWh' if m['endpoint']=='annual_kWh' else 'peak_sensible_heating_kW'
        domain='all_eligible_empirical_groups' if m['group']==0 else 'archetype_'+str(m['group'])
        x=metrics[(metrics.site==s)&(metrics.endpoint==ep)&(metrics.domain==domain)].iloc[0]
        for key in ['conditional_RMSE','pooled_RMSE']:maxerr=max(maxerr,abs(m[key]-x[key]))
        maxerr=max(maxerr,abs(m['paired_MSE_difference']-x.paired_MSE_difference_conditional_minus_pooled))
report['derived_metrics_max_abs_error']=maxerr
report['site_extra']={}
for s in ['Padova','Bolzano','Roma','Palermo']:
    run=ROOT/'05_simulations/archetype_campaign_v2'/s;design=read(run/'design.json')
    pid=design['technical_reference_profile_id']
    a=pd.read_csv(run/(pid+'_hourly.csv'));b=pd.read_csv(run/'repeat_component_hourly.csv');c=pd.read_csv(run/'warmup_component_hourly.csv')
    warm={k:float(f(c.sensible_load_W)/f(a.sensible_load_W)-1) for k,f in [('annual',lambda q:q.clip(lower=0).sum()),('peak',lambda q:q.clip(lower=0).max())]}
    epw=pd.read_csv(ROOT/f'05_simulations/climate_inputs_v2/{s}.epw',skiprows=8,header=None)
    mean=float((epw[6]-((epw[12]/5.6697e-8)**.25-273.15)).mean())
    params=read(run/'surface_forcing_parameters.json');sf=pd.read_csv(run/'surface_forcing_hourly.csv')
    shgcerror=0
    for name,p in params['surfaces'].items():
        if 'shgc_spline' in p:
            spline=p['shgc_spline'];g=BSpline(spline['knots'],spline['coefficients'],spline['degree'])
            shgcerror=max(shgcerror,float(np.max(abs(g(sf[name+'::AOI_degree'])-sf[name+'::SHGC_direct']))),float(np.max(abs(g(70)-sf[name+'::SHGC_diffuse']))))
    report['site_extra'][s]={'repeat_max_numeric_error':float(np.max(abs(a.to_numpy()-b.to_numpy()))),'warmup_relative_changes':warm,'IR_mean_delta_K':mean,'IR_to_forcing_delta_error':max(abs(p['mean_air_minus_sky_K']-mean) for p in params['surfaces'].values()),'SHGC_spline_max_error':shgcerror,'cloud_99_n':int((epw[23]==99).sum()),'negative_wind_n':int((epw[21]<0).sum()),'weather_metadata':read(ROOT/f'05_simulations/climate_inputs_v2/{s}.json')['inputs']}
p=ROOT/'06_claim_drafts';md=(p/'article_claims_v06.md').read_text(encoding='utf8')
blocks={b.split(' — ')[0]:dict(re.findall(r'^- ([^:]+): (.*)$',b,re.M)) for b in md.split('### ')[1:]}
ledger=read(p/'claim_ledger_v06.json');evidence=read(p/'claim_evidence_ledger_v06.json')
report['claim_count']=len(blocks);report['claim_ledger_content_mismatches']=[]
for c in ledger['claims']:
    for a,b in [('exact_claim','Exact claim'),('permitted_manuscript_wording','Permitted manuscript wording')]:
        if c[a]!=blocks[c['id']][b] or c[a]!=evidence['claims'][c['id']][a]:report['claim_ledger_content_mismatches'].append([c['id'],a])
(OUT/'reviewer_06_extra_checks.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
