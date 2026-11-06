from pathlib import Path
import json,hashlib
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DATA=ROOT.parents[1]
mapping=ROOT/'00_context/archetype_extension/archived_four_archetype_assignments.csv'
a=pd.read_csv(mapping);s=pd.read_csv(DATA/'istat_microdata_2013.csv',sep=';')
columns=['id','reg','q_3_0','q_3_11A','q_3_11B','q_3_11C','q_3_13_2','q_3_13_3','q_3_19_6','q_2_32C','q_2_32D']
d=a.merge(s[columns],left_on='household_id',right_on='id',validate='1:1')
bands=['q_3_11A','q_3_11B','q_3_11C']
valid=d[bands].notna().all(axis=1)&d[bands].ge(0).all(axis=1)&d[bands].le(8).all(axis=1)&(d[bands]%1==0).all(axis=1)
main=d['q_3_0'].isin([1,2]);positive=d[bands].sum(axis=1)>0
d['timer_or_valve_source_branch']=d[['q_3_13_2','q_3_13_3','q_3_19_6']].eq(1).any(axis=1)
d['fireplace_source_branch']=main&d[['q_2_32C','q_2_32D']].eq(1).any(axis=1)
d['source_on_C']=22+2*d['fireplace_source_branch']
d['source_off_C']=11+8*d['timer_or_valve_source_branch']+2*d['fireplace_source_branch']
d['provisional_domain']=valid&main&positive
out={'status':'pre-run input-domain audit; not a frozen eligibility choice or new thermal result','assignment_path':str(mapping),'assignment_sha256':hashlib.sha256(mapping.read_bytes()).hexdigest(),'survey_sha256':hashlib.sha256((DATA/'istat_microdata_2013.csv').read_bytes()).hexdigest(),'all_archived_n':len(d),'eligible_if_main12_positive_integer_blocks':int(d['provisional_domain'].sum()),'groups':[]}
for g,allg in d.groupby('archetype_id'):
    e=allg[allg['provisional_domain']]
    rows=[]
    for (on,off),b in e.groupby(['source_on_C','source_off_C']):rows.append({'on_C':int(on),'off_C':int(off),'n':len(b),'fraction':len(b)/len(e)})
    out['groups'].append({'archetype':int(g),'archived_n':len(allg),'valid_block_n':int(valid.loc[allg.index].sum()),'main12_n':int(main.loc[allg.index].sum()),'positive_n':int(positive.loc[allg.index].sum()),'eligible_n':len(e),'source_joint_temperature_branches':rows,'source_field_value_counts':{c:{str(k):int(v) for k,v in e[c].value_counts(dropna=False).items()} for c in columns if c not in ['id','reg']},'block_means':{c:float(e[c].mean()) for c in bands},'total_block_hours_sd':float(e[bands].sum(axis=1).std(ddof=1))})
(HERE/'archived_operating_domain_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:out[k] for k in ['all_archived_n','eligible_if_main12_positive_integer_blocks']},indent=2))
for g in out['groups']:print(json.dumps({k:g[k] for k in ['archetype','archived_n','eligible_n','source_joint_temperature_branches','block_means','total_block_hours_sd']},indent=2))
