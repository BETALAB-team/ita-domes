from pathlib import Path
import pandas as pd,numpy as np,json
out=Path('output/temporal_extension_2013_2021_20261104/02_analysis');raw=pd.read_csv('istat/Microdati21/500Tj00000M4IEcIAN-1_00250525-1/MICRODATI/ISTAT_MFR_ECH_Microdati_2021_DELIMITED.txt',sep='^',low_memory=False);old=pd.read_csv('istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt',sep='\t',low_memory=False);arc=pd.read_csv('clusters.csv');d=pd.read_csv(out/'assigned_2021_proxy.csv')
a13=arc[['identi','segment']].merge(old,left_on='identi',right_on='nprog',validate='one_to_one');a13['archetype']=a13.segment+1;a13['size']=pd.to_numeric(a13.q_1_1_sq1,errors='coerce');a13=a13[a13['size'].between(1,5)];d21=d.merge(raw,left_on='id',right_on='ID_RECORD',validate='one_to_one')
rows=[]
for year,z,w,types in [(2013,a13,'coef_red',[('heating','q_3_10',['q_3_11A','q_3_11B','q_3_11C']),('cooling','q_5_7',['q_5_8A','q_5_8B','q_5_8C'])]),(2021,d21,'PESO',[('heating','v3_7',['v3_8a','v3_8b','v3_8c']),('cooling','v5_6',['v5_7a','v5_7b','v5_7c'])])]:
 for kind,f,cs in types:
  freq=pd.to_numeric(z[f],errors='coerce');blocks=z[cs].apply(pd.to_numeric,errors='coerce');eligible=freq.isin([1,2])&blocks.notna().all(axis=1)&blocks.ge(0).all(axis=1)&blocks.le(8).all(axis=1);hours=blocks.sum(axis=1,min_count=3);weights=pd.to_numeric(z[w].astype(str).str.replace(',','.',regex=False),errors='coerce')
  for a in range(1,5):
   m=(z.archetype==a)&eligible;den=int((z.archetype==a).sum());n=int(m.sum());h=hours[m];ww=weights[m];rows.append(dict(year=year,outcome=kind,archetype=a,cohort_n=den,conditional_valid_n=n,coverage_pct=100*n/den,mean_hours_unweighted=h.mean(),sd_hours_unweighted=h.std(),mean_hours_surveyweighted=np.average(h,weights=ww) if n else None,effective_n=ww.sum()**2/(ww**2).sum() if n else None,estimand='conditional frequency1/2 and complete valid0–8 hour blocks; separate new archive diagnostic; not published H/E means',reference='typical winter/summer day' if year==2013 else 'cold/hot months of2020'))
pd.DataFrame(rows).to_csv(out/'conditional_heating_cooling_new_diagnostic.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
