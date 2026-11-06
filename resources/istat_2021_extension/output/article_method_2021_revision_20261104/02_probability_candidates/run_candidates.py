from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.stats import t
ROOT=Path.cwd();O=ROOT/'output/article_method_2021_revision_20261104/02_probability_candidates';P=ROOT/'output/article_method_2021_20261104/02_probabilities';S=pd.read_csv(P/'dwelling_samples_2021.csv');hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'household_archetypes_2021.csv',P/'dwelling_samples_2021.csv',ROOT/'clusters.csv',ROOT/'05_03_PFE_Analysis.py']}
terms=['intercept','age/100','area/100','age*area/10000','age^2/10000','area^2/10000']
def X(df):
 a=df.building_age_years.to_numpy()/100;s=df.dwelling_area_m2.to_numpy()/100;return np.column_stack((np.ones(len(df)),a,s,a*s,a*a,s*s))
def grid(s):
 rows=[]
 for ar in range(40,361,24):
  for age in range(15,136,5):
   z=s[s.building_age_years.between(age-15,age+15)&s.dwelling_area_m2.between(ar-40,ar+40)];n=len(z)
   if n<5:continue
   co=np.bincount(z.archetype.to_numpy().astype(int),minlength=5)[1:];rows.append([age,ar,n,*list(co/n)])
 return pd.DataFrame(rows,columns=['building_age_years','dwelling_area_m2','n','p1','p2','p3','p4'])
def softmax(z):
 z=z-z.max(axis=1,keepdims=True);v=np.exp(z);return v/v.sum(1,keepdims=True)
def fit(g,kind):
 A=X(g);Y=g[['p1','p2','p3','p4']].to_numpy()
 if kind=='joint_softmax_quadratic':
  # Equal-window multinomial crossentropy; three log odds against A4.
  def fun(v):
   B=v.reshape(6,3);Z=np.column_stack((A@B,np.zeros(len(A))));pr=softmax(Z);loss=-np.mean(np.sum(Y*np.log(np.maximum(pr,1e-15)),axis=1))+1e-4*np.sum(B[1:]**2);grad=A.T@(pr[:,:3]-Y[:,:3])/len(A);grad[1:]+=2e-4*B[1:];return loss,grad.ravel()
  opt=minimize(fun,np.zeros(18),jac=True,method='L-BFGS-B',options={'maxiter':3000,'ftol':1e-12,'gtol':1e-8});return opt.x.reshape(6,3),dict(converged=bool(opt.success),objective=float(opt.fun))
 if kind=='full_quadratic':return np.linalg.lstsq(A,Y,rcond=None)[0],{}
 B=np.zeros((6,4))
 for j in range(4):
  cols=({0:[0,1,4,5],1:[0,1,2,4],2:[0,1,2,4],3:[0,1,3,4]}[j] if kind=='paper_sparse_terms' else [0,1,2,4] if kind=='legacy_restricted' else list(range(6)))
  while True:
   Z=A[:,cols];b=np.linalg.lstsq(Z,Y[:,j],rcond=None)[0]
   if kind!='current_backward':break
   e=Y[:,j]-Z@b;df=len(A)-len(cols);se=np.sqrt(np.diag(np.linalg.pinv(Z.T@Z))*(e@e/df));p=2*t.sf(np.divide(abs(b),se,out=np.zeros_like(b),where=se>0),df);candidates=[k for k in range(len(cols)) if cols[k]!=0]
   if not candidates:break
   k=max(candidates,key=lambda k:p[k])
   if p[k]<=.05:break
   cols.pop(k)
  B[cols,j]=b
 return B,{}
def predict(g,B,kind):
 z=X(g)@B
 return softmax(np.column_stack((z,np.zeros(len(z))))) if kind=='joint_softmax_quadratic' else z
candidates=[(k,domain) for domain in ['primary_area_lt150','all_supported_areas'] for k in ['current_backward','legacy_restricted','paper_sparse_terms','full_quadratic','joint_softmax_quadratic']]
rows=[];allpred=[];fits=[]
for seed in [11,42,101,2021,2026]:
 order=np.random.default_rng(seed).permutation(len(S));cut=int(.7*len(S));tr=grid(S.iloc[order[:cut]]);te=grid(S.iloc[order[cut:]])
 for kind,domain in candidates:
  train=tr[tr.dwelling_area_m2<150] if domain=='primary_area_lt150' else tr
  B,extra=fit(train,kind);pred=predict(te,B,kind);Y=te[['p1','p2','p3','p4']].to_numpy()
  for evaluation in ['primary_area_lt150','all_supported_areas']:
   mask=(te.dwelling_area_m2<150).to_numpy() if evaluation=='primary_area_lt150' else np.ones(len(te),dtype=bool);er=pred[mask]-Y[mask];w=te.loc[mask,'n'].to_numpy();rows.append(dict(seed=seed,candidate=kind,fit_domain=domain,evaluation_domain=evaluation,train_windows=len(train),test_windows=int(mask.sum()),RMSE=float(np.sqrt(np.mean(er**2))),MAE=float(np.mean(abs(er))),countweighted_RMSE=float(np.sqrt(np.average(np.mean(er**2,axis=1),weights=w))),range_violation_windows=int(((pred[mask]<0)|(pred[mask]>1)).any(1).sum()),max_simplex_error=float(abs(pred[mask].sum(1)-1).max()),**extra))
  for k,(_,r) in enumerate(te.iterrows()):allpred.append(dict(seed=seed,candidate=kind,fit_domain=domain,building_age_years=r.building_age_years,dwelling_area_m2=r.dwelling_area_m2,n=r.n,**{f'observed_A{a+1}':Y[k,a] for a in range(4)},**{f'predicted_A{a+1}':pred[k,a] for a in range(4)}))
 print('split',seed,'complete',flush=True)
metrics=pd.DataFrame(rows);metrics.to_csv(O/'repeated_household_holdout_metrics.csv',index=False);pd.DataFrame(allpred).to_csv(O/'repeated_holdout_predictions.csv',index=False);summary=metrics.groupby(['candidate','fit_domain','evaluation_domain']).agg(RMSE_mean=('RMSE','mean'),RMSE_sd=('RMSE','std'),MAE_mean=('MAE','mean'),countweighted_RMSE_mean=('countweighted_RMSE','mean'),range_violations_sum=('range_violation_windows','sum'),max_simplex_error=('max_simplex_error','max')).reset_index();summary.to_csv(O/'validation_summary.csv',index=False)
fullgrid=pd.DataFrame([(a,s) for s in range(40,361,24) for a in range(15,136,5)],columns=['building_age_years','dwelling_area_m2']);g=grid(S);out=[];coeff=[];models={}
for kind,domain in candidates:
 train=g[g.dwelling_area_m2<150] if domain=='primary_area_lt150' else g;B,extra=fit(train,kind);models[(kind,domain)]=B;pr=predict(fullgrid,B,kind)
 for j in range(B.shape[1]):
  for i,term in enumerate(terms):coeff.append(dict(candidate=kind,fit_domain=domain,response=f'logodds_A{j+1}_vs_A4' if kind=='joint_softmax_quadratic' else f'probability_A{j+1}',term=term,coefficient=B[i,j]))
 for k,r in fullgrid.iterrows():out.append(dict(candidate=kind,fit_domain=domain,building_age_years=r.building_age_years,dwelling_area_m2=r.dwelling_area_m2,**{f'p{a+1}':pr[k,a] for a in range(4)},probability_sum=pr[k].sum(),invalid_range=bool(((pr[k]<0)|(pr[k]>1)).any()),**extra))
pd.DataFrame(coeff).to_csv(O/'candidate_coefficients.csv',index=False);pd.DataFrame(out).to_csv(O/'candidate_grid_predictions.csv',index=False);g.to_csv(O/'observed_supported_windows.csv',index=False)
# Original Figure6 visual orientation, four source-observed construction-era groups.
S['vintage']=np.select([S.q_2_4_ric.eq(8),S.q_2_4_ric.isin([6,7]),S.q_2_4_ric.isin([4,5]),S.q_2_4_ric.isin([1,2,3])],['Before1950','1950–1969','1970–1989','1990onward'],default='Unknown');curve=[];obs=[]
for vintage,z in S.groupby('vintage'):
 if vintage=='Unknown':continue
 ages,counts=np.unique(z.building_age_years,return_counts=True);weights=counts/counts.sum()
 for area in range(20,151,2):
  evalgrid=pd.DataFrame({'building_age_years':ages,'dwelling_area_m2':area})
  for (kind,domain),B in models.items():
   p=weights@predict(evalgrid,B,kind);curve.append(dict(vintage=vintage,area=area,candidate=kind,fit_domain=domain,age_standardization_n=len(z),**{f'p{a+1}':p[a] for a in range(4)}))
 for area in [40,64,88,112,136]:
  zz=z[z.dwelling_area_m2.between(area-40,area+40)];n=len(zz)
  if n<5:continue
  p=np.bincount(zz.archetype.to_numpy().astype(int),minlength=5)[1:]/n;obs.append(dict(vintage=vintage,area=area,n=n,**{f'p{a+1}':p[a] for a in range(4)}))
pd.DataFrame(curve).to_csv(O/'vintage_area_candidate_curves.csv',index=False);pd.DataFrame(obs).to_csv(O/'vintage_area_observed_points.csv',index=False)
# Area contrasts in the original-style vintage curves.
c=pd.DataFrame(curve);contrasts=[]
for keys,z in c.groupby(['candidate','fit_domain','vintage']):
 lo=z.loc[z.area.eq(40),['p1','p2','p3','p4']].to_numpy()[0];hi=z.loc[z.area.eq(120),['p1','p2','p3','p4']].to_numpy()[0]
 for j in range(4):contrasts.append(dict(candidate=keys[0],fit_domain=keys[1],vintage=keys[2],archetype=j+1,change40to120_pp=100*(hi[j]-lo[j])))
pd.DataFrame(contrasts).to_csv(O/'area_trend_contrasts.csv',index=False)
result=dict(n_fixedlabel_households=len(S),split_seeds=[11,42,101,2021,2026],training_fraction=.7,primary_basis=terms,source_hashes=hashes,source_hashes_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()),window_geometry=dict(age_centers='15:5:135',area_centers='40:24:352',halfwidth_age=15,halfwidth_area=40,source='e7094f4:resources/istat_data/classification_analysis_bayes.py; saved probablities_only.csv'),selection='Compare matched repeated household holdout metrics; no recommendation chosen by appearance',fig6_source_note='Actual figure has dwellingarea x-axis and vintage lines; caption contradicts actual figure.2021 vintage before1950 used rather than inventing within-bin before1945 distinction.',softmax_note='Equal-window crossentropy, quadratic three log odds vsA4, fixedridge1e-4 for nonintercept coefficients, no tuning ontest splits. Fit onlytraining windows; fixed demographic labels external.')
(O/'results.json').write_text(json.dumps(result,indent=2));print(summary.to_string(index=False))


