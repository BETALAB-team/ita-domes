from pathlib import Path
import sys,json,hashlib,itertools,os
import numpy as np,pandas as pd
ROOT=Path.cwd();OUT=ROOT/'output/article_method_2021_20261104/02_probabilities';sys.path.insert(0,str(ROOT))
from pfe_pipeline import build_dwelling_samples,calculate_window_probabilities
RAW=ROOT/'istat/Microdati21/500Tj00000M4IEcIAN-1_00250525-1/MICRODATI/ISTAT_MFR_ECH_Microdati_2021_DELIMITED.txt'
oldparams=ROOT/'output/temporal_extension_2013_2021_20261104/00_baseline/archive_transfer_parameters.json'
source_files=[RAW,ROOT/'clusters.csv',ROOT/'02_Families_Analysis.py',ROOT/'03_PCA_Clustering.py',ROOT/'01_PCA_Exploratory.py',ROOT/'05_00_PFE_Exploratory.py',ROOT/'05_01_PFE_Variable_Analysis.py',ROOT/'05_02_PFE_Probablity.py',ROOT/'05_03_PFE_Analysis.py',ROOT/'pfe_pipeline.py'];before_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
raw=pd.read_csv(RAW,sep='^',low_memory=False);originalhash=hashlib.sha256((ROOT/'clusters.csv').read_bytes()).hexdigest()
ranges={1:(0,5),2:(6,10),3:(11,14),4:(15,19),5:(20,29),6:(30,39),7:(40,49),8:(50,59),9:(60,69),10:(70,79),11:(80,100)}
def features(primary_years=5,diploma_years=12,seed=42):
 rng=np.random.RandomState(seed);recs=[]
 for r in raw.to_dict('records'):
  n=int(r['ncomp_mfr']);reason='';ages=[];ed=[];routed=0
  if not 1<=n<=5:reason='count6plus_exceeds_five_member_release'
  if not reason:
   for i in range(1,n+1):
    ac=int(r[f'eta_mfr{i}']);ec=str(r[f'istr_mfr{i}']).strip()
    if ac not in ranges:reason='invalid_active_age';break
    lo,hi=ranges[ac];age=int(rng.randint(lo,hi+1));ages.append(age)
    if hi<11:routed+=1;continue
    if ec not in ['1','3','4','6']:reason='missing_schooling_age11plus';break
    years={'1':primary_years,'3':8,'4':diploma_years,'6':18}[ec]
    potential=0 if age<10 else 5 if age<15 else 8 if age<19 else 12 if age<23 else 16 if age<25 else 18 if age<29 else 21
    ed.append(min(years/potential,1) if potential else (1 if years>0 else np.nan))
   if not ed and not reason:reason='no_measured_schooling_member'
  recs.append(dict(id=int(r['ID_RECORD']),eligible=not bool(reason),exclusion_reason=reason,number=n,min_age=min(ages) if ages else np.nan,max_age=max(ages) if ages else np.nan,studio_ind=np.nanmean(ed) if ed else np.nan,schooling_observed_members=len(ed),schooling_routed_under11_members=routed,weight=float(r['PESO']),region_raw=int(r['REGIO']),macroregion=(4 if int(r['RIP'])==5 else int(r['RIP'])),macroregion_raw=int(r['RIP'])))
 return pd.DataFrame(recs)
dall=features();d=dall[dall.eligible].copy();d['loneliness']=.25*d.min_age+.75*d.max_age-10*d.number;d['generation']=.33*d.min_age+.66*d.max_age+10*d.number
lo=d[['loneliness','generation']].min().to_numpy();span=np.ptp(d[['loneliness','generation']].to_numpy(),axis=0);X=np.column_stack(((d[['loneliness','generation']].to_numpy()-lo)/span,d.studio_ind))
# A full orthogonal PCA preserves all Euclidean distances.
mean=X.mean(0);u,s,v=np.linalg.svd(X-mean,full_matrices=False);pcs=(X-mean)@v.T
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,silhouette_score
model=KMeans(n_clusters=4,init='k-means++',n_init=20,random_state=42,max_iter=500,tol=1e-8).fit(X)
def archive_metric(df):return np.column_stack(((.25*df.min_age+.75*df.max_age-10*df.number+40.5)/114.5,(.33*df.min_age+.66*df.max_age+10*df.number-30.79)/91.05,df.studio_ind))
ref=np.array(list(json.loads(oldparams.read_text())['centers'].values()));M=archive_metric(d);cm=np.array([M[model.labels_==k].mean(0) for k in range(4)]);cost=((cm[:,None,:]-ref[None,:,:])**2).sum(2);perm=min(itertools.permutations(range(4)),key=lambda p:sum(cost[i,p[i]] for i in range(4)));d['archetype']=np.array(perm)[model.labels_]+1;d['normal_l']=X[:,0];d['normal_g']=X[:,1];d['normal_s']=X[:,2]
for j in range(3):d[f'PC{j+1}']=pcs[:,j]
dall=dall.merge(d[['id','archetype','normal_l','normal_g','normal_s','PC1','PC2','PC3']],on='id',how='left',validate='one_to_one');dall.to_csv(OUT/'household_archetypes_2021.csv',index=False);d.to_csv(OUT/'eligible_household_features_2021.csv',index=False)
params=dict(seed=42,n_init=20,minmax_min=lo.tolist(),minmax_span=span.tolist(),centers_raw_labels=model.cluster_centers_.tolist(),raw_label_to_aligned_archetype={str(k):int(perm[k]+1) for k in range(4)},alignment_centers_archive_metric=cm.tolist(),alignment_cost=cost.tolist(),pca_mean=mean.tolist(),pca_loadings=v.tolist(),pca_explained_ratio=(s*s/(s*s).sum()).tolist(),full_pca_max_distance_error=float(np.max(np.abs(np.sum((X[:100,None]-X[None,:100])**2,axis=2)-np.sum((pcs[:100,None]-pcs[None,:100])**2,axis=2)))))
(OUT/'archetype_fit_parameters.json').write_text(json.dumps(params,indent=2));print('LABELS_READY',len(d),'of',len(dall),flush=True)
rows=[]
for a,z in d.groupby('archetype'):
 rows.append(dict(archetype=a,n=len(z),household_share=100*len(z)/len(d),weighted_household_share=100*z.weight.sum()/d.weight.sum(),person_share=100*z.number.sum()/d.number.sum(),min_age_mean=z.min_age.mean(),min_age_sd=z.min_age.std(),max_age_mean=z.max_age.mean(),max_age_sd=z.max_age.std(),number_mean=z.number.mean(),number_sd=z.number.std(),education_mean=z.studio_ind.mean(),education_sd=z.studio_ind.std()))
pd.DataFrame(rows).to_csv(OUT/'archetype_summary_2021.csv',index=False)
# verified2021 construction/area intervals; unknown age excluded, not invented.
AR={1:(0,11),2:(12,21),3:(22,31),4:(32,41),5:(42,51),6:(52,61),7:(62,71),8:(72,150)}
SR={1:(10,39),2:(40,49),3:(50,59),4:(60,69),5:(70,79),6:(80,89),7:(90,99),8:(100,109),9:(110,119),10:(120,129),11:(130,149),12:(150,169),13:(170,199),14:(200,249),15:(250,299),16:(300,400)}
canon=raw[['ID_RECORD','v1_2_new','v1_7_mfr']].rename(columns={'ID_RECORD':'id','v1_2_new':'q_2_4_ric','v1_7_mfr':'q_2_7_class'});canon[['q_2_4_ric','q_2_7_class']]=canon[['q_2_4_ric','q_2_7_class']].apply(pd.to_numeric,errors='coerce');pfeids=canon[canon.q_2_4_ric.isin(AR)&canon.q_2_7_class.isin(SR)].merge(d[['id','archetype']],on='id',validate='one_to_one')
pfeelig=canon.merge(dall[['id','eligible']],on='id',validate='one_to_one');pfeelig['construction_known']=pfeelig.q_2_4_ric.isin(AR);pfeelig['area_valid']=pfeelig.q_2_7_class.isin(SR);pfeelig['pfe_eligible']=pfeelig.eligible & pfeelig.construction_known & pfeelig.area_valid;pfeelig.to_csv(OUT/'pfe_eligibility_all_ids.csv',index=False)
samples=build_dwelling_samples(pfeids[['id','q_2_4_ric','q_2_7_class']],pfeids[['id','archetype']],42,AR,SR);samples.to_csv(OUT/'dwelling_samples_2021.csv',index=False);probs,unsupported,contributing=calculate_window_probabilities(samples,5);probs.to_csv(OUT/'probability_grid_empirical.csv',index=False);unsupported.to_csv(OUT/'probability_grid_empty_windows.csv',index=False)
from scipy.stats import t as tdist
basis=['intercept','age/100','area/100','age*area/10000','age^2/10000','area^2/10000']
def design(df):
 a=df.building_age_years.to_numpy()/100;s=df.dwelling_area_m2.to_numpy()/100;return np.column_stack((np.ones(len(df)),a,s,a*s,a*a,s*s))
def backward(A,y):
 cols=list(range(6));removed=[]
 while True:
  Z=A[:,cols];b,res,rank,s=np.linalg.lstsq(Z,y,rcond=None);e=y-Z@b;df=len(y)-len(cols);cov=np.linalg.pinv(Z.T@Z)*(e@e/df);se=np.sqrt(np.diag(cov));pv=2*tdist.sf(np.divide(abs(b),se,out=np.zeros_like(b),where=se>0),df);candidates=[i for i in range(len(cols)) if cols[i]!=0];worst=max(candidates,key=lambda j:pv[j]) if candidates else None
  if worst is None or pv[worst]<=.05:break
  removed.append(dict(term=basis[cols[worst]],p=float(pv[worst])));cols.pop(worst)
 out=np.zeros(6);out[cols]=b;return out,cols,removed,dict(residual_df=df,rank=int(rank),condition_number=float(np.linalg.cond(Z)))
fit=probs[probs.meets_minimum_support&(probs.building_age_years<150)&(probs.dwelling_area_m2<150)].copy();A=design(fit);B=[];diagnostics=[];fitout=fit.copy()
for a in range(1,5):
 y=fit[f'prob_archetype_{a}'].to_numpy();b,cols,removed,extra=backward(A,y);pred=A@b;B.append(b);sse=((y-pred)**2).sum();sst=((y-y.mean())**2).sum();r2=1-sse/sst;diagnostics.append(dict(archetype=a,n_windows=len(y),terms=[basis[j] for j in cols],removed=removed,RMSE=float(np.sqrt(sse/len(y))),R2=float(r2),adjusted_R2=float(1-(1-r2)*(len(y)-1)/(len(y)-len(cols))),**extra));fitout[f'fitted_archetype_{a}']=pred
B=np.array(B);pd.DataFrame(B,index=[1,2,3,4],columns=basis).rename_axis('archetype').to_csv(OUT/'polynomial_coefficient_matrix.csv');fitout.to_csv(OUT/'empirical_vs_fitted_training.csv',index=False);(OUT/'polynomial_fit_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
full=pd.DataFrame([(a,s) for s in range(40,361,40) for a in range(20,141,20)],columns=['building_age_years','dwelling_area_m2']);pred=design(full)@B.T
for a in range(4):full[f'prob_archetype_{a+1}']=pred[:,a]
full['probability_sum']=pred.sum(1);full['outside_0_1']=((pred<0)|(pred>1)).any(1);full['sum_not_one']=abs(pred.sum(1)-1)>1e-8;full['outside_primary_fit_domain']=full.dwelling_area_m2>=150;full.to_csv(OUT/'polynomial_full_grid_predictions_flags.csv',index=False)
# Separate survey-weighted window probabilities and new secondary OLS coefficient matrix.
weighted_samples=samples.merge(d[['id','weight']],on='id',validate='one_to_one');weighted_probs=probs.copy()
for ix,r in weighted_probs.iterrows():
 selected=weighted_samples.building_age_years.between(r.age_window_lower,r.age_window_upper)&weighted_samples.dwelling_area_m2.between(r.area_window_lower,r.area_window_upper);ww=weighted_samples[selected];total=ww.weight.sum();weighted_probs.loc[ix,'survey_weight_sum']=total;weighted_probs.loc[ix,'effective_n']=total**2/(ww.weight**2).sum()
 for a in range(1,5):weighted_probs.loc[ix,f'prob_archetype_{a}']=ww.loc[ww.archetype==a,'weight'].sum()/total
weighted_probs.to_csv(OUT/'probability_grid_surveyweighted_sensitivity.csv',index=False);wf=weighted_probs[weighted_probs.meets_minimum_support&(weighted_probs.building_age_years<150)&(weighted_probs.dwelling_area_m2<150)];WB=[];wd=[]
for a in range(1,5):
 y=wf[f'prob_archetype_{a}'].to_numpy();b,cols,removed,extra=backward(design(wf),y);WB.append(b);pr=design(wf)@b;wd.append(dict(archetype=a,n_windows=len(wf),RMSE=float(np.sqrt(np.mean((y-pr)**2))),terms=[basis[j] for j in cols],removed=removed,qualification='OLS on official survey-weighted window ratios; separate sensitivity not original unweighted estimand'))
pd.DataFrame(WB,index=[1,2,3,4],columns=basis).rename_axis('archetype').to_csv(OUT/'polynomial_matrix_surveyweighted_sensitivity.csv');(OUT/'surveyweighted_polynomial_diagnostics.json').write_text(json.dumps(wd,indent=2))
# New-household holdout, retaining fitted new archetype representation for a conditional probability validation.
order=np.random.RandomState(2021).permutation(len(samples));cut=int(.7*len(samples));tr=samples.iloc[order[:cut]];te=samples.iloc[order[cut:]];ptr,_,_=calculate_window_probabilities(tr,5);pte,_,_=calculate_window_probabilities(te,5);ptr=ptr[ptr.meets_minimum_support&(ptr.dwelling_area_m2<150)];pte=pte[pte.meets_minimum_support&(pte.dwelling_area_m2<150)];hold=[]
for a in range(1,5):
 b,_,_,_=backward(design(ptr),ptr[f'prob_archetype_{a}'].to_numpy());y=pte[f'prob_archetype_{a}'].to_numpy();pr=design(pte)@b;hold.append(dict(archetype=a,train_households=len(tr),test_households=len(te),train_windows=len(ptr),test_windows=len(pte),RMSE=float(np.sqrt(np.mean((y-pr)**2))),MAE=float(np.mean(abs(y-pr))),qualification='new-household PFE holdout conditional on all-household demographic fit; not external validation'))
pd.DataFrame(hold).to_csv(OUT/'household_holdout_probability_scores.csv',index=False)
# Bound sensitivity of schooling ambiguity and seeded ages; new2021fits only.
sens=[]
for label,py,dy,seed in [('schooling_code1_0',0,12,42),('diploma13',5,13,42),('age_seed0',5,12,0),('age_seed2021',5,12,2021)]:
 dd=features(py,dy,seed);dd=dd[dd.eligible].copy();LL=np.column_stack((.25*dd.min_age+.75*dd.max_age-10*dd.number,.33*dd.min_age+.66*dd.max_age+10*dd.number));ll=LL.min(0);sp=np.ptp(LL,axis=0);xx=np.column_stack(((LL-ll)/sp,dd.studio_ind));mm=KMeans(n_clusters=4,n_init=20,random_state=42,max_iter=500,tol=1e-8).fit(xx);cc=np.array([archive_metric(dd)[mm.labels_==k].mean(0) for k in range(4)]);co=((cc[:,None]-ref[None])**2).sum(2);pp=min(itertools.permutations(range(4)),key=lambda p:sum(co[i,p[i]] for i in range(4)));labels=np.array(pp)[mm.labels_]+1;ari=adjusted_rand_score(d.archetype,labels);nmi=normalized_mutual_info_score(d.archetype,labels)
 for a in range(1,5):sens.append(dict(scenario=label,archetype=a,n=len(dd),share=100*(labels==a).mean(),ARI_vs_primary=float(ari),NMI_vs_primary=float(nmi)))
pd.DataFrame(sens).to_csv(OUT/'archetype_sensitivities.csv',index=False)
manifest=dict(raw_n=len(raw),eligible_n=len(d),pfe_n=len(samples),window_contributing_n=len(contributing),nonempty_windows=len(probs),minimum5_windows=int(probs.meets_minimum_support.sum()),fit_windows=len(fit),age_seed=42,dwelling_seed=42,unknown_construction_excluded=True,probability_flags=dict(outside_0_1=int(full.outside_0_1.sum()),sum_not_one=int(full.sum_not_one.sum())),source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RAW,ROOT/'clusters.csv',ROOT/'03_PCA_Clustering.py',ROOT/'01_PCA_Exploratory.py',ROOT/'05_00_PFE_Exploratory.py',ROOT/'05_01_PFE_Variable_Analysis.py',ROOT/'05_02_PFE_Probablity.py',ROOT/'05_03_PFE_Analysis.py',ROOT/'pfe_pipeline.py']},archived_labels_unchanged=originalhash==hashlib.sha256((ROOT/'clusters.csv').read_bytes()).hexdigest(),versions=dict(numpy=np.__version__,pandas=pd.__version__))
manifest['source_before_hashes']=before_hashes;manifest['all_source_hashes_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in before_hashes.items());assert manifest['all_source_hashes_unchanged'];(OUT/'results.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)




