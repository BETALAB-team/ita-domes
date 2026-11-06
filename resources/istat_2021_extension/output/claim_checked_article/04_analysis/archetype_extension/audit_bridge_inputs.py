"""Read-only archetype/survey join and operational-input feasibility audit."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=ROOT.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    sources={'survey':DATA/'istat_microdata_2013.csv','reconstructed_assignment':DATA/'baseline_outputs/household_archetypes.csv','original_clustered_lead':DATA/'Clustered.csv','pipeline':DATA/'archetype_pipeline.py','validation_lead':DATA/'baseline_outputs/validation_report.json'}
    survey=pd.read_csv(sources['survey'],sep=';'); assignment=pd.read_csv(sources['reconstructed_assignment']);old=pd.read_csv(sources['original_clustered_lead'])
    assert survey['id'].is_unique and assignment['id'].is_unique
    joined=assignment.merge(survey,on='id',how='left',validate='1:1',indicator=True)
    blocks=['q_3_11A','q_3_11B','q_3_11C']
    valid=joined[blocks].notna().all(axis=1)&joined[blocks].ge(0).all(axis=1)&joined[blocks].le(8).all(axis=1)
    integer=(joined[blocks].fillna(-999)%1==0).all(axis=1)
    controls=['q_3_13_2','q_3_13_3','q_3_19_6','q_3_0','q_2_32C','q_2_32D']
    rows=[]
    for key,g in joined.groupby('archetype'):
        good=g.loc[valid&integer].copy()
        row={'archetype':int(key),'assignment_n':len(g),'matched_survey_n':int((g['_merge']=='both').sum()),'valid_block_integer_n':len(good),'positive_total_n':int((good[blocks].sum(axis=1)>0).sum()),'unique_hours_control_patterns':len(good[blocks+controls].drop_duplicates())}
        for c in blocks:
            a=good[c].to_numpy(float);row[c]={'mean':float(a.mean()),'sd':float(a.std(ddof=1)),'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max())}
        row['controls_values']={c:{str(k):int(v) for k,v in good[c].value_counts(dropna=False).items()} for c in controls}
        row['regions']={str(k):int(v) for k,v in good['reg'].value_counts().items()}
        rows.append(row)
    weight_candidates=[c for c in survey.columns if any(x in c.lower() for x in ['coef','peso','weight'])]
    comparisons={}
    if 'identi' in old and 'segment' in old:
        old=old[['identi','segment']].copy();old['old_group']=old['segment'].astype(str).str.extract(r'(\d+)')[0].astype('Int64')
        merged=assignment.merge(old,left_on='id',right_on='identi',how='inner')
        table=pd.crosstab(merged['archetype'],merged['old_group']).rename(columns=str).reset_index()
        comparisons={'old_rows':len(old),'old_id_unique':bool(old['identi'].is_unique),'old_labels':old['segment'].value_counts().to_dict(),'joined_n':len(merged),'raw_number_agreement':float((merged['archetype']==merged['old_group']).mean()),'contingency':table.to_dict(orient='records')}
    report={'status':'preparatory feasibility; membership provenance awaits context audit; no clustering rerun or thermal outputs','source_paths_sha256':{k:{'path':str(p),'sha256':sha(p)} for k,p in sources.items()},'survey_n':len(survey),'assignment_n':len(assignment),'all_joined':bool((joined['_merge']=='both').all()),'assignment_group_counts':assignment['archetype'].value_counts().sort_index().to_dict(),'weight_candidates':{c:{'nonmissing_n':int(survey[c].notna().sum()),'min':float(survey[c].min()),'max':float(survey[c].max())} for c in weight_candidates},'heating_hours_valid_n':int(valid.sum()),'integer_block_hours_n':int((valid&integer).sum()),'by_group':rows,'original_clustered_comparison':comparisons}
    HERE.mkdir(parents=True,exist_ok=True)
    (HERE/'bridge_input_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['survey_n','assignment_n','all_joined','assignment_group_counts','weight_candidates','heating_hours_valid_n','integer_block_hours_n','original_clustered_comparison']},indent=2))
if __name__=='__main__':main()
