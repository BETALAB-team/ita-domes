from pathlib import Path
import pandas as pd,numpy as np
out=Path('output/temporal_extension_2013_2021_20261104/02_analysis');old=pd.read_csv('istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt',sep='\t',low_memory=False);new=pd.read_csv('istat/Microdati21/500Tj00000M4IEcIAN-1_00250525-1/MICRODATI/ISTAT_MFR_ECH_Microdati_2021_DELIMITED.txt',sep='^',low_memory=False)
rows=[];regional=[]
for y,z,nn,w,ac,rg in [(2013,old,'q_1_1_sq1','coef_red',[f'cletad_{i}' for i in range(1,13)],'reg'),(2021,new,'ncomp_mfr','PESO',[f'eta_mfr{i}' for i in range(1,6)],'REGIO')]:
 size=pd.to_numeric(z[nn],errors='coerce');valid=size.between(1,5);z=z[valid].copy();size=size[valid];weight=pd.to_numeric(z[w].astype(str).str.replace(',','.',regex=False),errors='coerce');ages=z[ac].apply(pd.to_numeric,errors='coerce');active=np.arange(len(ac))[None,:]<size.to_numpy()[:,None];under=((ages.to_numpy()<=3) if y==2021 else (ages.to_numpy()==0));under=(under&active).any(axis=1)
 metrics={'household_size_mean':size.to_numpy(),'has_under15_pct':under.astype(float)*100}
 for n in range(1,6):metrics[f'household_size{n}_pct']=(size.to_numpy()==n)*100
 # Heating/cooling hour conditional outcomes are analyzed in a separate table with source-verified routing.
 for kind,f,cs in [('heating','q_3_10' if y==2013 else 'v3_7',['q_3_11A','q_3_11B','q_3_11C'] if y==2013 else ['v3_8a','v3_8b','v3_8c']),('cooling','q_5_7' if y==2013 else 'v5_6',['q_5_8A','q_5_8B','q_5_8C'] if y==2013 else ['v5_7a','v5_7b','v5_7c'])]:
  freq=pd.to_numeric(z[f],errors='coerce');blocks=z[cs].apply(pd.to_numeric,errors='coerce');ok=freq.isin([1,2])&blocks.notna().all(axis=1)&blocks.ge(0).all(axis=1)&blocks.le(8).all(axis=1);h=blocks.sum(axis=1,min_count=3)
  for region in [0]+sorted(z[rg].unique()):
   rr=np.ones(len(z),dtype=bool) if region==0 else (z[rg].to_numpy()==region);m=ok.to_numpy()&rr
   regional.append(dict(year=y,region=int(region),metric=f'{kind}_conditional_hours',domain_n=int(rr.sum()),valid_n=int(m.sum()),unweighted=float(h[m].mean()),surveyweighted=float(np.average(h[m],weights=weight[m])) if m.any() else None))
 for metric,v in metrics.items():
  rows.append(dict(year=y,metric=metric,denominator=len(z),unweighted=float(np.mean(v)),surveyweighted=float(np.average(v,weights=weight)),weight_sum=weight.sum(),scope='new full size1–5 raw-release descriptive diagnostic; not published archetype replacement'))
  for region in sorted(z[rg].unique()):
   rr=z[rg].to_numpy()==region;regional.append(dict(year=y,region=int(region),metric=metric,domain_n=int(rr.sum()),valid_n=int(rr.sum()),unweighted=float(np.mean(v[rr])),surveyweighted=float(np.average(v[rr],weights=weight[rr]))))
pd.DataFrame(rows).to_csv(out/'full_size1_5_direct_survey.csv',index=False);pd.DataFrame(regional).to_csv(out/'full_size1_5_direct_regional.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
