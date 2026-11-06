from pathlib import Path
import json,numpy as np,pandas as pd
O=Path('output/article_method_2021_revision_20261104/02_probability_candidates');s=(O/'run_candidates.py').read_text();exec(s[:s.index('\ncandidates=[(k')]);coarse=(O/'coarse21_diagnostic/run_candidates.py').read_text();chunk=coarse[coarse.index('def grid('):coarse.index('def softmax(')].replace('def grid(','def coarse_grid(');exec(chunk)
rows=[];details=[]
for seed in [11,42,101,2021,2026]:
 order=np.random.default_rng(seed).permutation(len(S));cut=int(.7*len(S));trainS=S.iloc[order[:cut]];testS=S.iloc[order[cut:]];test=grid(testS);test=test[test.dwelling_area_m2<150];Y=test[['p1','p2','p3','p4']].to_numpy()
 for name,train,kind in [('old21_backward',coarse_grid(trainS),'current_backward'),('new125_full_quadratic',grid(trainS),'full_quadratic'),('new125_backward',grid(trainS),'current_backward')]:
  train=train[train.dwelling_area_m2<150];B,_=fit(train,kind);pr=predict(test,B,kind);err=pr-Y;w=test.n.to_numpy();rows.append(dict(seed=seed,candidate=name,train_windows=len(train),test_windows=len(test),RMSE=float(np.sqrt(np.mean(err**2))),MAE=float(np.mean(abs(err))),countweighted_RMSE=float(np.sqrt(np.average(np.mean(err**2,axis=1),weights=w))),max_simplex_error=float(abs(pr.sum(1)-1).max())))
  for j in range(4):details.append(dict(seed=seed,candidate=name,archetype=j+1,RMSE=float(np.sqrt(np.mean(err[:,j]**2))),MAE=float(np.mean(abs(err[:,j])))))
D=pd.DataFrame(rows);D.to_csv(O/'fair_old21_vs_new125_same_holdout.csv',index=False);pd.DataFrame(details).to_csv(O/'fair_old21_vs_new125_by_archetype.csv',index=False);D.groupby('candidate').agg(RMSE_mean=('RMSE','mean'),RMSE_sd=('RMSE','std'),MAE_mean=('MAE','mean'),countweighted_RMSE_mean=('countweighted_RMSE','mean'),max_simplex_error=('max_simplex_error','max')).reset_index().to_csv(O/'fair_geometry_validation_summary.csv',index=False)
# Match empirical and predicted vintage profiles by IDENTICAL equal age-node weights.
g=grid(S);g=g[g.dwelling_area_m2<150].copy();g['vintage']=np.select([g.building_age_years>=72,g.building_age_years>=52,g.building_age_years>=32],['Before1950','1950–1969','1970–1989'],default='1990onward');B,_=fit(g,'full_quadratic');pd.DataFrame(B.T,index=[1,2,3,4],columns=terms).rename_axis('archetype').to_csv(O/'recommended_full_quadratic_matrix.csv');obs=[];curves=[]
for vintage,z in g.groupby('vintage'):
 ages=np.sort(z.building_age_years.unique())
 for area in [40,64,88,112,136]:
  zz=z[z.dwelling_area_m2.eq(area)];assert np.array_equal(np.sort(zz.building_age_years.unique()),ages);pr=predict(zz,B,'full_quadratic').mean(0);emp=zz[['p1','p2','p3','p4']].mean().to_numpy();obs.append(dict(vintage=vintage,area=area,age_nodes=len(zz),window_n_sum=int(zz.n.sum()),**{f'observed_p{j+1}':emp[j] for j in range(4)},**{f'predicted_p{j+1}':pr[j] for j in range(4)}))
 for area in range(40,137):
  E=pd.DataFrame({'building_age_years':ages,'dwelling_area_m2':area});pr=predict(E,B,'full_quadratic').mean(0);curves.append(dict(vintage=vintage,area=area,age_nodes=len(ages),**{f'p{j+1}':pr[j] for j in range(4)}))
pd.DataFrame(obs).to_csv(O/'matched_vintage_observed_fitted_nodes.csv',index=False);pd.DataFrame(curves).to_csv(O/'matched_vintage_area_curves.csv',index=False)
# Apparent diagnostics and dense rectangular support validity for selected matrix.
pr=predict(g,B,'full_quadratic');Y=g[['p1','p2','p3','p4']].to_numpy();ap=[]
for j in range(4):
 e=Y[:,j]-pr[:,j];r2=1-(e@e)/np.sum((Y[:,j]-Y[:,j].mean())**2);ap.append(dict(archetype=j+1,n_windows=len(g),RMSE=float(np.sqrt(np.mean(e**2))),R2=float(r2),adjusted_R2=float(1-(1-r2)*(len(g)-1)/(len(g)-6))))
pd.DataFrame(ap).to_csv(O/'recommended_apparent_fit_statistics.csv',index=False);E=pd.DataFrame([(a,s) for a in range(15,136) for s in range(40,137)],columns=['building_age_years','dwelling_area_m2']);pr=predict(E,B,'full_quadratic');E[[f'p{j+1}' for j in range(4)]]=pr;E['invalid_range']=((pr<0)|(pr>1)).any(1);E['sum_error']=abs(pr.sum(1)-1);E.to_csv(O/'recommended_dense_support_predictions.csv',index=False)
result=dict(recommendation='Full six-term quadratic probability matrix on recovered historical125-node primary domain',matrix_basis_order=terms,fit_windows=len(g),domain=dict(age_centers=[15,135],area_centers=[40,136],window_halfwidth_age=15,window_halfwidth_area=40),dense_support_invalid_n=int(E.invalid_range.sum()),dense_support_max_simplex_error=float(E.sum_error.max()),apparent=ap,household_validation_fair=D.groupby('candidate')[['RMSE','countweighted_RMSE']].mean().to_dict('index'),vintage_matching='Equal weighting of same historical age-centre nodes in each source-era group; empirical and predicted points use identical weights. Overlapping window counts are not independent household denominators.',selection_limits='No nested final untouched split; repeated splits compare candidates conditionally on frozen household labels; differences are descriptive, no independent-window inference.')
(O/'recommendation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
