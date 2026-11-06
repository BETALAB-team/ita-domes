from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_simulations/climate_inputs_v2'
rows=[]; metadata={}
for name in ['Padova','Bolzano','Roma','Palermo']:
    p=OUT/f'{name}.epw'; df=pd.read_csv(p,skiprows=8,header=None)
    obj=json.loads((OUT/f'{name}.json').read_text())
    metadata[name]={'inputs':obj.get('inputs'),'meta':obj.get('meta'),'tmy_months_selected':obj.get('outputs',{}).get('months_selected'),'epw_header':p.read_text().splitlines()[:8]}
    assert len(df)==8760 and len(df[[1,2,3]].drop_duplicates())==8760
    qc={}
    for col,lo,hi in [(6,-70,70),(8,0,100),(13,-.1,1500),(14,-.1,1500),(15,-.1,1500)]:
        qc[col]={'min':float(df[col].min()),'max':float(df[col].max()),'finite':bool(np.isfinite(df[col]).all()),'range_pass':bool(df[col].between(lo,hi).all())}
    metadata[name]['numeric_QC']=qc
    assert all(x['finite'] and x['range_pass'] for x in qc.values())
    daily=df[6].to_numpy().reshape(365,24).mean(axis=1)
    rows.append({'site':name,'lat':obj['inputs']['location']['latitude'],'lon':obj['inputs']['location']['longitude'],'n':len(df),'HDD18_daily_Kd':float(np.clip(18-daily,0,None).sum()),'CDD26_daily_Kd':float(np.clip(daily-26,0,None).sum()),'mean_RH_percent':float(df[8].mean()),'annual_GHI_kWh_m2':float(df[13].sum()/1000),'T_min_C':float(df[6].min()),'T_max_C':float(df[6].max()),'RH100_hours':int((df[8]==100).sum()),'epw_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
pd.DataFrame(rows).to_csv(ROOT/'04_analysis/climate_descriptors_v2.csv',index=False)
(OUT/'audited_metadata.json').write_text(json.dumps(metadata,indent=2))
print(pd.DataFrame(rows).to_string(index=False))
print(json.dumps(metadata['Padova']['inputs'],indent=2))
