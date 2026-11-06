"""Independent dense-window count, matrix and same-heldout-target verification."""
from pathlib import Path
import sys,json,itertools
here=Path(__file__).resolve().parent;base=here.parent.parent/"article_method_2021_20261104";sys.path.insert(0,str(base/".scientific"));pc=here.parent/"02_probability_candidates"
import numpy as np,pandas as pd,statsmodels.api as sm
s=pd.read_csv(base/"02_probabilities/dwelling_samples_2021.csv")
eq=lambda a,b:np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-9)
def grid(ix,coarse=False):
    age=s.building_age_years.to_numpy()[ix];area=s.dwelling_area_m2.to_numpy()[ix];lab=s.archetype.to_numpy()[ix]
    hist=np.zeros((152,402,4),int);np.add.at(hist,(age+1,area+1,lab-1),1);cs=hist.cumsum(0).cumsum(1)
    coord=np.array(list(itertools.product([40,80,120] if coarse else [40,64,88,112,136],range(20,141,20) if coarse else range(15,136,5))))
    ss,aa=coord.T; h,w=(6,8) if coarse else (15,40)
    a0=np.maximum(aa-h,0);s0=np.maximum(ss-w,0);a1=aa+h+1;s1=ss+w+1
    counts=cs[a1,s1]-cs[a0,s1]-cs[a1,s0]+cs[a0,s0]
    mask=counts.sum(1)>=5;n=counts[mask].sum(1)
    aa,ss=aa[mask]/100,ss[mask]/100
    deg=np.stack([np.ones_like(aa),aa,ss,aa*ss,aa**2,ss**2],axis=1)
    return deg,counts[mask]/n[:,None],n
def betas(A,Y,kind):
    betas=[]
    for y in Y.T:
        cols=list(range(6))
        while True:
            m=sm.OLS(y,A[:,cols]).fit()
            if kind=="full" or len(cols)==1:break
            mx=1+np.argmax(m.pvalues[1:])
            if m.pvalues[mx]<=.05:break
            del cols[mx]
        b=np.zeros(6);b[cols]=m.params;betas.append(b)
    return np.stack(betas)
A,Y,n=grid(np.arange(len(s)));G=pd.read_csv(pc/"observed_supported_windows.csv");G=G[G.dwelling_area_m2<150];eq(Y,G[["p1","p2","p3","p4"]]);eq(n,G.n)
B=betas(A,Y,"full");eq(B,pd.read_csv(pc/"recommended_full_quadratic_matrix.csv").iloc[:,1:].to_numpy())
sc=pd.read_csv(pc/"fair_old21_vs_new125_same_holdout.csv");scores={}
for seed in [11,42,101,2021,2026]:
    rng=np.random.default_rng(seed);seq=rng.permutation(len(s));cut=int(.7*len(s));te,y,nt=grid(seq[cut:])
    for k,coarse,model in [("old21_backward",True,"back"),("new125_backward",False,"back"),("new125_full_quadratic",False,"full")]:
        tr,yt,_=grid(seq[:cut],coarse=coarse);beta=betas(tr,yt,model);err=te@beta.T-y
        m=np.sqrt(np.mean(err**2));wm=np.sqrt(np.average(np.mean(err**2,axis=1),weights=nt))
        rr=sc[(sc.seed==seed)&(sc.candidate==k)].iloc[0];eq([m,wm],[rr.RMSE,rr.countweighted_RMSE]);scores.setdefault(k,[]).append(m)
# Check extrema analytically over the continuous application rectangle, including edge extrema.
def extreme(b):
    a0,a1=.15,1.35;s0,s1=.40,1.36
    pts=list(itertools.product([a0,a1],[s0,s1]))
    for ar in [a0,a1]:
        sx=-(b[2]+ar*b[3])/(2*b[5])
        if s0<=sx<=s1:pts.append((ar,sx))
    for sx in [s0,s1]:
        ar=-(b[1]+sx*b[3])/(2*b[4])
        if a0<=ar<=a1:pts.append((ar,sx))
    ar,sx=np.linalg.solve([[2*b[4],b[3]],[b[3],2*b[5]]],-b[1:3])
    if a0<=ar<=a1 and s0<=sx<=s1:pts.append((ar,sx))
    val=[b@np.array([1,a,u,a*u,a*a,u*u]) for a,u in pts]
    return [min(val),max(val)]
ext=[extreme(b) for b in B];eq(B.sum(axis=0),[1,0,0,0,0,0])
vnames=np.select([A[:,1]>=.72,A[:,1]>=.52,A[:,1]>=.32],["Before1950","1950–1969","1970–1989"],default="1990onward");nodes=pd.read_csv(pc/"matched_vintage_observed_fitted_nodes.csv")
for r in nodes.itertuples():
    ii=(vnames==r.vintage)&(np.isclose(A[:,2],r.area/100));eq(Y[ii].mean(0),[getattr(r,"observed_p"+str(k)) for k in range(1,5)])
    eq((A[ii]@B.T).mean(0),[getattr(r,"predicted_p"+str(k)) for k in range(1,5)])
areas=pd.read_csv(pc/"candidate_grid_predictions.csv");outside=areas[(areas.candidate=="full_quadratic")&(areas.fit_domain=="primary_area_lt150")&(areas.dwelling_area_m2>=150)]
res={"n":len(s),"primary_windows":len(A),"unique_ids":s.id.nunique(),"nodes_total_counts_including_repeats":int(n.sum()),"node_n_range":[int(n.min()),int(n.max())],"same_target_average_RMSE":{k:np.mean(x) for k,x in scores.items()},"same_target_change_pct":100*(1-np.mean(scores["new125_full_quadratic"])/np.mean(scores["old21_backward"])),"selected_beats_old_in_each_split":bool((np.array(scores["new125_full_quadratic"])<np.array(scores["old21_backward"])).all()),"continuous_domain_probability_ranges":ext,"fullgrid_extrapolation_nodes":len(outside),"fullgrid_extrapolation_outside01":int(outside.invalid_range.sum())}
(here/"probability_check.json").write_text(json.dumps(res,indent=2));print(res)
