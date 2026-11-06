from pathlib import Path
import pandas as pd,numpy as np,json
out=Path('output/temporal_extension_2013_2021_20261104/02_analysis');r=pd.read_csv('istat/Microdati21/500Tj00000M4IEcIAN-1_00250525-1/MICRODATI/ISTAT_MFR_ECH_Microdati_2021_DELIMITED.txt',sep='^',low_memory=False);d=pd.read_csv(out/'assigned_2021_proxy.csv');p=json.loads((out.parent/'00_baseline/archive_transfer_parameters.json').read_text());c=np.array(list(p['centers'].values()));z=d[['id','number']].merge(r,left_on='id',right_on='ID_RECORD',validate='one_to_one');ranges={1:(0,5),2:(6,10),3:(11,14),4:(15,19),5:(20,29),6:(30,39),7:(40,49),8:(50,59),9:(60,69),10:(70,79),11:(80,100)}
mp={}
for code,(lo,hi) in ranges.items():
 t=np.arange(lo,hi+1);cl=np.where(t<15,0,np.minimum(75,15+10*np.floor((t-15)/10)));rep=np.where(cl==0,12.5,cl+7);mp[code]=rep.mean()
A=z[[f'eta_mfr{i}' for i in range(1,6)]].apply(pd.to_numeric,errors='coerce').replace(mp).to_numpy().copy();mask=np.arange(5)[None,:]<z.number.to_numpy()[:,None];A[~mask]=np.nan;mn=np.nanmin(A,axis=1);mx=np.nanmax(A,axis=1);X=np.column_stack(((.25*mn+.75*mx-10*d.number+40.5)/114.5,(.33*mn+.66*mx+10*d.number-30.79)/91.05,d.studio_ind));lab=((X[:,None,:]-c[None,:,:])**2).sum(2).argmin(1)+1
rows=[dict(scenario='review_triggered_archive_age_imputation_alignment',archetype=a,share_pct=100*(lab==a).mean(),n=len(d),changed_pct=100*(lab!=d.archetype.to_numpy()).mean(),education='held fixed at primary proxy') for a in range(1,5)];pd.DataFrame(rows).to_csv(out/'review_triggered_age_alignment.csv',index=False)
# Cooling broad union affirmative inventory. Unknown remains explicit; no blank-as-no inference.
old=pd.read_csv('istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt',sep='\t',low_memory=False);rows=[]
for year,q,nc,w,cols in [(2013,old,'q_1_1_sq1','coef_red',[f'q_2_37{v}' for v in 'ABCDEF']),(2021,r,'ncomp_mfr','PESO',[f'v2_11{v}' for v in 'abcde'])]:
 q=q[pd.to_numeric(q[nc],errors='coerce').between(1,5)].copy();M=q[cols].apply(pd.to_numeric,errors='coerce');pos=M.eq(1).any(axis=1);explicit_no=M.eq(2).all(axis=1);unknown=~pos&~explicit_no;ww=pd.to_numeric(q[w].astype(str).str.replace(',','.',regex=False),errors='coerce')
 for label,mask in [('affirmative_inventory',pos),('all_items_explicit_no',explicit_no),('unknown_no_affirmative',unknown)]:rows.append(dict(year=year,status=label,n=int(mask.sum()),denominator=len(q),unweighted_pct=100*mask.mean(),surveyweighted_pct=100*ww[mask].sum()/ww.sum(),interpretation='affirmative union comparable broad cooling availability; unknown retained, never inferred absent'))
pd.DataFrame(rows).to_csv(out/'full_size1_5_cooling_inventory.csv',index=False)
rg=pd.read_csv(out/'full_size1_5_direct_regional.csv');rg['raw_region_code']=rg.region;rg['region_code_system']=np.where(rg.year==2013,'2013 canonical','2021 source REGIO');m={i*10:i for i in range(1,21)};m.update({41:4,42:4});rg['canonical_region']=np.where(rg.year==2013,rg.region,rg.region.map(m));rg.to_csv(out/'full_size1_5_direct_regional.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False));print(mp)

