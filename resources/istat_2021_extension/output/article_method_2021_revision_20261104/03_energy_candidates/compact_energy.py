"""Count-weighted optimal 1D partition; household-fold CV, frozen demographic labels."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path.cwd(); OLD=ROOT/'output/article_method_2021_20261104'; OUT=ROOT/'output/article_method_2021_revision_20261104/03_energy_candidates';sys.path.insert(0,str(OLD/'.scientific'))
import pandas as pd,numpy as np
FACTORS=['location','dwelling_type','archetype'];KMAX=15;FOLDS=5
SPEC={'registered_before_computation':True,'rule':'Smallest K with mean fold RMSE <= minimum mean fold RMSE + SE of RMSE at that minimum AND full-data between-cell mean variance retained >=0.95','fidelity_floor':.95,'K_candidates':list(range(1,KMAX+1)),'folds':5,'fold_seed':20261104,'objective':'Exact contiguous partition of training cell means sorted by mean minimizing sum n_cell*(mean_cell-group_mean)^2','unit':'Households, unweighted original-style group estimation; survey-weighted group summaries separate','leakage':'Fold training-only cell means, ranks, partitions and group predictions; same cell features map heldout households. Unseen cell fallback training global mean. Demographic labels previously fitted and fixed; CV validates energy compression conditional on those labels, not entire demographic workflow.','standard_errors':'Sample SD of five fold RMSEs /sqrt(5); folds are dependent training sets, not population confidence intervals','comparison':'K7H/K5E and K9H/K13E DP resolutions are separate from original Tukey membership. Existing original Tukey full-data summaries retained as descriptive comparisons only, never used for fold predictions.'}
(OUT/'selection_spec.json').write_text(json.dumps(SPEC,indent=2))
source=OLD/'03_energy/analysis_households_2021.csv';d=pd.read_csv(source,usecols=['id','weight',*FACTORS,'heating_daily_hours','electricity_annual_kwh','H_group','E_group']);d.archetype=d.archetype.astype(int);d['cell']=d[FACTORS].astype(str).agg('|'.join,axis=1)
# Assign folds independently of outcomes, once for all households.
ids=d.id.sort_values().to_numpy();rng=np.random.RandomState(20261104);perm=rng.permutation(len(ids));fold=np.empty(len(ids),int);fold[perm]=np.arange(len(ids))%FOLDS;foldmap=dict(zip(ids,fold));d['fold']=d.id.map(foldmap)
d[['id','fold']].to_csv(OUT/'household_folds.csv',index=False)
def partition(z,response):
 c=z.groupby('cell')[response].agg(n='size',mean='mean',sd='std').reset_index().sort_values(['mean','cell']).reset_index(drop=True);n=c.n.to_numpy(float);y=c['mean'].to_numpy(float);N=len(c)
 a=np.r_[0,np.cumsum(n)];b=np.r_[0,np.cumsum(n*y)];e=np.r_[0,np.cumsum(n*y*y)];cost=np.zeros((N+1,N+1))
 for j in range(1,N+1):
  ii=np.arange(j);cost[ii,j]=np.maximum(0,e[j]-e[ii]-(b[j]-b[ii])**2/(a[j]-a[ii]))
 dp=np.full((KMAX+1,N+1),np.inf);prev=np.full((KMAX+1,N+1),-1,int);dp[0,0]=0
 for k in range(1,min(KMAX,N)+1):
  for j in range(k,N+1):
   vals=dp[k-1,k-1:j]+cost[k-1:j,j];i=k-1+int(np.argmin(vals));dp[k,j]=vals[i-(k-1)];prev[k,j]=i
 maps={}
 for k in range(1,min(KMAX,N)+1):
  j=N;labels=np.empty(N,int);bounds=[]
  for g in range(k,0,-1):
   i=prev[g,j];labels[i:j]=g;bounds.append((i,j));j=i
  cc=c.copy();cc['group_number']=labels;cc['group_prediction']=cc.groupby('group_number').apply(lambda g:np.average(g['mean'],weights=g.n),include_groups=False).reindex(cc.group_number).to_numpy();cc['K']=k;cc['cell_rank']=np.arange(1,N+1);maps[k]=cc
 return maps,dp[:,N]
summary={};allcurves=[];allfolds=[];alllook=[];assign=d[['id',*FACTORS,'weight','fold']].copy();appendix=d[FACTORS].drop_duplicates().sort_values(FACTORS);selected_stats=[];uncompressed=[]
for pref,response,oldK,origK in [('H','heating_daily_hours',7,9),('E','electricity_annual_kwh',5,13)]:
 z=d[d[response].notna()].copy();z=z[np.isfinite(z[response])];maps,cost=partition(z,response);total=float(cost[1]);foldrows=[]
 for ff in range(FOLDS):
  train=z[z.fold.ne(ff)];test=z[z.fold.eq(ff)];tm,tc=partition(train,response);base=train[response].mean()
  cm=train.groupby('cell')[response].mean();pred=test.cell.map(cm).fillna(base).to_numpy();err=np.square(test[response].to_numpy()-pred);uncompressed.append({'response':pref,'fold':ff,'n_test':len(test),'rmse':float(np.sqrt(err.mean())),'mse':float(err.mean()),'unseen_test_cells':int((~test.cell.isin(cm.index)).sum())})
  for k in range(1,KMAX+1):
   mp=tm[k].set_index('cell').group_prediction;pred=test.cell.map(mp).fillna(base).to_numpy();err=np.square(test[response].to_numpy()-pred);foldrows.append({'response':pref,'fold':ff,'K':k,'n_train':len(train),'n_test':len(test),'rmse':float(np.sqrt(err.mean())),'mse':float(err.mean()),'unseen_test_cells':int((~test.cell.isin(mp.index)).sum())})
 ft=pd.DataFrame(foldrows);cv=ft.groupby('K').agg(cv_rmse_mean=('rmse','mean'),cv_rmse_sd=('rmse','std'),cv_mse_mean=('mse','mean')).reset_index();cv['cv_rmse_se']=cv.cv_rmse_sd/np.sqrt(FOLDS);best=int(cv.loc[cv.cv_rmse_mean.idxmin(),'K']);bestrow=cv[cv.K.eq(best)].iloc[0];threshold=float(bestrow.cv_rmse_mean+bestrow.cv_rmse_se)
 cv['between_cell_variance_retained']=[max(0,1-cost[k]/total) for k in cv.K];cv['within_one_se']=cv.cv_rmse_mean.le(threshold);cv['meets_fidelity_floor']=cv.between_cell_variance_retained.ge(.95);feasible=cv[cv.within_one_se&cv.meets_fidelity_floor];selected=int(feasible.K.min()) if len(feasible) else None
 if selected is None:raise RuntimeError('No K meets preregistered rule')
 cv['selected']=cv.K.eq(selected);cv['response']=pref;cv['cv_rmse_delta_vs_best']=cv.cv_rmse_mean-float(bestrow.cv_rmse_mean)
 for k in cv.K:
  aa=ft[ft.K.eq(k)].sort_values('fold');bb=ft[ft.K.eq(best)].sort_values('fold');delta=aa.mse.to_numpy()-bb.mse.to_numpy();cv.loc[cv.K.eq(k),'paired_mse_delta_vs_best']=delta.mean();cv.loc[cv.K.eq(k),'paired_mse_delta_se']=delta.std(ddof=1)/np.sqrt(FOLDS)
 cv.to_csv(OUT/f'{pref}_candidate_curve.csv',index=False);allcurves.append(cv);allfolds.append(ft)
 for k,lookup in maps.items():
  lo=lookup.copy();lo['response']=pref;lo['group']=pref+lo.group_number.astype(str);lo['selected']=k==selected;alllook.append(lo)
 lookup=maps[selected].copy();lookup['group']=pref+lookup.group_number.astype(str);lookup=lookup.merge(z[['cell',*FACTORS]].drop_duplicates(),on='cell',validate='one_to_one');lookup.to_csv(OUT/f'{pref}_selected_cell_lookup.csv',index=False)
 zz=z[['id','cell',response,'weight']].merge(lookup[['cell','group']],on='cell',validate='many_to_one');assign=assign.merge(zz[['id','group']].rename(columns={'group':pref+'_compact_group'}),on='id',how='left',validate='one_to_one');appendix=appendix.merge(lookup[FACTORS+['group']].rename(columns={'group':pref+'_group'}),on=FACTORS,how='left',validate='one_to_one')
 st=zz.groupby('group')[response].agg(n='size',mean='mean',sd='std',minimum='min',maximum='max').reset_index();st['response']=pref;st['K']=selected;st['cell_count']=st.group.map(lookup.groupby('group').size());st['weighted_mean']=st.group.map(zz.groupby('group').apply(lambda g:np.average(g[response],weights=g.weight),include_groups=False));st.to_csv(OUT/f'{pref}_selected_group_summary.csv',index=False);selected_stats.append(st)
 # Descriptive original Tukey fidelity on precisely same full domain.
 gm=z.groupby(pref+'_group')[response].mean();cell=z.groupby('cell')[response].agg(n='size',mean='mean');og=z[['cell',pref+'_group']].drop_duplicates().set_index('cell')[pref+'_group'];sse=float((cell.n*(cell['mean']-og.map(gm))**2).sum());oldactual=1-sse/total
 summary[pref]={'response':response,'households':len(z),'initial_cells':len(lookup),'selected_K':selected,'best_cv_K':best,'one_se_threshold':threshold,'selected_curve':cv[cv.K.eq(selected)].to_dict('records')[0],'original_resolution_sensitivity':cv[cv.K.eq(oldK)].to_dict('records')[0],'previous_2021_resolution_DP_sensitivity':cv[cv.K.eq(origK)].to_dict('records')[0],'previous_actual_Tukey_groups':origK,'previous_actual_Tukey_fidelity':oldactual,'selected_segments':selected,'cells_absorbed':len(lookup)-selected}
 print(pref,json.dumps(summary[pref]),flush=True)
pd.concat(allcurves).to_csv(OUT/'candidate_curves.csv',index=False);pd.concat(allfolds).to_csv(OUT/'fold_losses.csv',index=False);pd.concat(alllook).to_csv(OUT/'all_K_cell_lookups.csv',index=False);pd.DataFrame(uncompressed).to_csv(OUT/'uncompressed_cell_cv.csv',index=False);pd.concat(selected_stats).to_csv(OUT/'selected_group_summaries.csv',index=False);assign.to_csv(OUT/'household_compact_assignments.csv',index=False);appendix.to_csv(OUT/'appendix_A_compact_long.csv',index=False);appendix.pivot(index=['location','dwelling_type'],columns='archetype',values=['H_group','E_group']).to_csv(OUT/'appendix_A_compact_wide.csv');(OUT/'results.json').write_text(json.dumps({'spec':SPEC,'results':summary,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2));print('DONE',flush=True)
