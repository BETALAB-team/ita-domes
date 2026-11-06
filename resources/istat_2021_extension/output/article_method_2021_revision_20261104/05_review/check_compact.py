"""Independent verification from response-level 2021 data and the registered fold IDs."""
import sys,json
from pathlib import Path
from functools import lru_cache
here=Path(__file__).resolve().parent; base=here.parent.parent/"article_method_2021_20261104";sys.path.insert(0,str(base/".scientific")); c=here.parent/"03_energy_candidates"
import numpy as np,pandas as pd
raw=pd.read_csv(base/"03_energy/analysis_households_2021.csv",usecols=["id","location","dwelling_type","archetype","weight","heating_daily_hours","electricity_annual_kwh","H_group","E_group"]).set_index("id",verify_integrity=True)
fld=pd.read_csv(c/"household_folds.csv").set_index("id",verify_integrity=True)
ids=raw.index.sort_values(); shuffled=np.random.RandomState(20261104).permutation(ids.to_numpy())
assert np.array_equal(fld.loc[shuffled,"fold"],np.arange(len(ids))%5)
raw=raw.join(fld);raw["cell"]=raw.location+"|"+raw.dwelling_type+"|"+raw.archetype.astype(int).astype(str)
losses=pd.read_csv(c/"fold_losses.csv");curve=pd.read_csv(c/"candidate_curves.csv")
allmaps=pd.read_csv(c/"all_K_cell_lookups.csv");ap=pd.read_csv(c/"appendix_A_compact_long.csv");ap["cell"]=ap.location+"|"+ap.dwelling_type+"|"+ap.archetype.astype(str)
assigned=pd.read_csv(c/"household_compact_assignments.csv").set_index("id",verify_integrity=True);stats=pd.read_csv(c/"selected_group_summaries.csv")
eq=lambda x,y:np.testing.assert_allclose(x,y,rtol=1e-9,atol=1e-7)
def dp(z):
    desc=z.groupby("cell").y.agg(["mean","size"]).reset_index().sort_values(["mean","cell"])
    y=desc["mean"].to_numpy();n=desc["size"].to_numpy();cells=desc.cell.to_numpy();ends=len(y)
    # Direct residual sum-of-squares for each possible segment. Top-down optimal recursion.
    cost={};mn={}
    for a in range(ends):
        for b in range(a+1,ends+1):
            mu=np.average(y[a:b],weights=n[a:b]);mn[a,b]=mu;cost[a,b]=n[a:b]@((y[a:b]-mu)**2)
    @lru_cache(None)
    def best(k,b):
        if k==1:return cost[0,b],(0,)
        return min((best(k-1,j)[0]+cost[j,b],best(k-1,j)[1]+(j,)) for j in range(k-1,b))
    fits={}
    for k in range(1,16):
        s,starts=best(k,ends);gr={};pred={}
        for j,(a,b) in enumerate(zip(starts,starts[1:]+(ends,)),1):
            for name in cells[a:b]:gr[name]=j;pred[name]=mn[a,b]
        fits[k]=(s,gr,pred)
    return fits
summ={}
for pref,col in [("H","heating_daily_hours"),("E","electricity_annual_kwh")]:
    z=raw.dropna(subset=[col]).copy();z["y"]=z[col]
    L=np.empty((5,15))
    for i in range(5):
        tr=z[z.fold!=i];te=z[z.fold==i];fits=dp(tr)
        for k,(_,g,pred) in fits.items():
            v=te.cell.map(pred);assert v.notna().all()
            L[i,k-1]=np.sqrt(np.mean((te.y-v)**2))
    ref=losses[losses.response==pref].pivot(index="fold",columns="K",values="rmse").to_numpy();eq(ref,L)
    fits=dp(z);S=np.array([a[0] for a in fits.values()]);r=1-S/S[0]
    for k,(_,g,pred) in fits.items():
        oo=allmaps[(allmaps.response==pref)&(allmaps.K==k)];eq(oo.cell.map(g),oo.group_number);eq(oo.cell.map(pred),oo.group_prediction)
    ave=L.mean(0);best=int(ave.argmin());lim=ave[best]+L[:,best].std(ddof=1)/np.sqrt(5);con=ave<=lim;pick=np.flatnonzero(con&(r>=.95))[0]; k=int(pick+1)
    o=curve[curve.response==pref];eq(o.between_cell_variance_retained,r);eq(o.cv_rmse_mean,ave)
    mapping={s:pref+str(j) for s,j in fits[k][1].items()};zz=z.cell.map(mapping)
    assert (ap.cell.map(mapping)==ap[pref+"_group"]).all();assert (zz==assigned.loc[z.index,pref+"_compact_group"]).all()
    for g,jj in z.groupby(zz):
        rr=stats[(stats.response==pref)&(stats.group==g)].iloc[0]
        eq([len(jj),jj.y.mean(),jj.y.std(),np.average(jj.y,weights=jj.weight)],[rr.n,rr["mean"],rr.sd,rr.weighted_mean])
    firstse=int(1+np.flatnonzero(con)[0]);first95=int(1+np.flatnonzero(r>=.95)[0])
    legacy=int(7 if pref=="H" else 5)
    summ[pref]={"best_k":best+1,"selected_k":k,"one_standard_error_only":firstse,"fidelity_only":first95,"rmse_best":ave[best],"rmse_threshold":lim,"rmse_selected":ave[pick],"delta_best":ave[pick]-ave[best],"pct_loss_vs_best":100*(ave[pick]/ave[best]-1),"fidelity":r[pick],"previous_k_fidelity":r[pick-1],"paper_resolution_k":legacy,"paper_resolution_rmse":ave[legacy-1],"paper_resolution_fidelity":r[legacy-1],"one_group":ave[0],"choice_fidelity_sensitivity":{str(f):int(1+np.flatnonzero(con&(r>=f))[0]) for f in [.9,.95,.975,.99]}}
print(summ);(here/"compact_check.json").write_text(json.dumps(summ,indent=2))
