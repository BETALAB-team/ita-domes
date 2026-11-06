"""Independent audit from native releases, archived membership and released result tables.
Run from the istat_data directory. Does not import or run the analytical pipeline.
"""
from pathlib import Path
import hashlib, itertools, json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "02_analysis"
ROOT = HERE.parents[2]
read = lambda name: pd.read_csv(OUT / name)
def native(path, sep, cols):
    return pd.read_csv(path, sep=sep, dtype=str, usecols=cols).apply(lambda col: pd.to_numeric(col.str.strip().str.replace(",", "."), errors="coerce"))
def table(k):
    return pd.read_csv(OUT / k)
def nn(x, c):
    dist = np.linalg.norm(x[:, None] - c, axis=2)
    return 1 + dist.argmin(1), np.sort(dist, axis=1)
def shares(lab, w):
    return np.bincount(lab, weights=w, minlength=5)[1:] * 100 / np.sum(w)
def agreement(x, y):
    mat = pd.crosstab(x, y).to_numpy()
    n = mat.sum()
    row, col = mat.sum(1), mat.sum(0)
    cp = lambda a: np.sum(a * (a - 1) / 2)
    exp = cp(row) * cp(col) / cp(n)
    ari = (cp(mat) - exp) / ((cp(row) + cp(col)) / 2 - exp)
    p = mat / n
    nz = p > 0
    info = (p[nz] * np.log(p[nz] / (np.outer(row, col) / n**2)[nz])).sum()
    ent = lambda a: -np.sum(a/n * np.log(a/n))
    return ari, 2 * info / (ent(row) + ent(col))
def check_est(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-8)
def dump(data):
    return json.dumps(data, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())

a = pd.read_csv(ROOT / "clusters.csv")
b = json.loads((HERE.parent / "00_baseline/baseline_freeze.json").read_text(encoding="utf-8-sig"))
for f in b["sources"]:
    assert hashlib.sha256(Path(f["path"]).read_bytes()).hexdigest() == f["sha256"]
acols = ["normal_l","normal_g","normal_s"]
c = a.groupby("segment")[acols].mean().to_numpy()
assert (nn(a[acols].to_numpy(),c)[0] == a.segment+1).all()
r = native(next((ROOT/"istat/Microdati21").rglob("*Microdati*DELIMITED.txt")), "^", lambda s: s in {"PESO","ID_RECORD","REGIO","ncomp_mfr"} or s.startswith(("eta_mfr","istr_mfr")))
o = native(next((ROOT/"istat/ECH_2013_IT/MICRODATI").glob("*.txt")), "\t", lambda s: s in {"nprog","reg","q_1_1_sq1","coef_red"} or s.startswith(("cletad_","titstu_")))
assert r.ID_RECORD.is_unique and o.nprog.is_unique
assert (r.PESO>0).all() and (o.coef_red>0).all()
m = o.merge(a, left_on="nprog", right_on="identi", validate="one_to_one")
assert len(m) == len(a) and (m.q_1_1_sq1==m.number).all()
m = m[m.number<=5].copy()
oldage=m.filter(regex="^cletad_").to_numpy()
young13=(oldage==0).any(axis=1)
ages = r.filter(regex="^eta_mfr").to_numpy()
schools = r.filter(regex="^istr_mfr").to_numpy()
active = np.arange(5)[None,:] < r.ncomp_mfr.to_numpy()[:,None]
young = ((ages<=2)&active).any(axis=1)
ok = (r.ncomp_mfr<=5) & (~(active & (~np.isin(schools,[1,3,4,6]) | ~np.isin(ages,range(1,12))))).all(axis=1)
assert not young[ok].any()
use = r.loc[ok].reset_index(drop=True)
usages = ages[ok]; uschools = schools[ok]
nums = use.ncomp_mfr.to_numpy()
w = use.PESO.to_numpy()
mask = active[ok]
mids = [np.nan,2.5,8,12.5,17,24.5,34.5,44.5,54.5,64.5,74.5,90]
age_mid = np.array(mids)[np.nan_to_num(usages).astype(int)]
school = np.select([uschools==x for x in (1,3,4,6)],[12,8,12,18],default=np.nan)
def school_index(years, ages):
    denom = np.select([ages<15,ages<25,ages<35], [0,8,18], default=21)
    frac = np.minimum(1,np.divide(years,denom,out=np.ones_like(years),where=denom!=0))
    return np.where(mask,frac,0).sum(1)/nums
def coord(ages, index, exact=False):
    lo, hi=np.nanmin(ages,axis=1), np.nanmax(ages,axis=1)
    L=lo*.25+hi*.75-10*nums; G=10*nums+(lo+2*hi)*(1/3 if exact else .33)
    return np.stack([(L-a.loneliness.min())/np.ptp(a.loneliness),(G-a.generation.min())/np.ptp(a.generation),index],axis=1)
edu=school_index(school,age_mid)
X=coord(age_mid,edu)
lab, distances=nn(X,c)
support=((X<0)|(X>1)).any(axis=1)
d=read("assigned_2021_proxy.csv")
check_est(use.ID_RECORD, d.id); check_est(w,d.weight); check_est(lab,d.archetype); check_est(edu,d.studio_ind)
assert np.array_equal(support,d.out_of_support)
summary=read("2021_fixed_proxy_summary.csv")
check_est(shares(lab,np.ones(len(w))),summary.share_pct);check_est(shares(lab,w),summary.weighted_share_pct)
check_est(shares(lab,nums),summary.people_share_pct)
newregions = (use.REGIO//10).to_numpy()
assert set(newregions)==set(m.reg)==set(range(1,21))
reg=read("regional_common_cohort_changes.csv")
p13=[]; p21=[]; n13=[]; n21=[]
for region in range(1,21):
    t=m[m.reg==region]; u=newregions==region
    o_uw=shares(t.segment+1,np.ones(len(t))); o_w=shares(t.segment+1,t.coef_red)
    n_uw=shares(lab[u],np.ones(sum(u))); n_w=shares(lab[u],w[u])
    rr=reg[reg.region==region].sort_values("archetype")
    for col,v in {"share2013":o_uw,"share2021":n_uw,"weighted2013":o_w,"weighted2021":n_w,"change_pp":n_uw-o_uw,"weighted_change_pp":n_w-o_w}.items():check_est(rr[col],v)
    p13.append([o_uw,o_w]); p21.append([n_uw,n_w]); n13.append([len(t),t.coef_red.sum()]); n21.append([u.sum(),w[u].sum()])
mix=lambda counts: np.asarray(counts)/np.sum(counts,axis=0,keepdims=True)
sumover=lambda P,q: np.einsum("rma,rm->ma",P,q)
std = sumover(p21,mix(n13)); old = sumover(p13,mix(n13)); new=sumover(p21,mix(n21))
tt=read("regional_composition_standardization.csv")
for col, v in {"archive2013_unweighted":old[0],"archive2013_surveyweighted":old[1],"proxy2021_at2013_sample_region_mix":std[0],"proxy2021_at2013_surveyweighted_region_mix":std[1],"region_standardized_change_pp":std[0]-old[0]}.items():check_est(tt[col],v)
sens=read("transfer_sensitivities.csv")
for name,x in {"education_code1_5years":coord(age_mid,school_index(np.where(uschools==1,5,school),age_mid)),"clip_to_archive_range":np.clip(X,0,1)}.items():
    alt=nn(x,c)[0]; check_est(shares(alt,np.ones(len(w))),sens[sens.scenario==name].share_pct)
rc=read("2021_refit_centroids.csv")
cost=np.linalg.norm(rc[acols].to_numpy()[:,None]-c,axis=2)**2
match=tuple(1+i for i in min(itertools.permutations(range(4)),key=lambda perm:sum(cost[j,p] for j,p in enumerate(perm))))
assert match==tuple(rc.matched_archetype)
newlab=nn(X,rc.sort_values("matched_archetype")[acols].to_numpy())[0]
ari,nmi=agreement(lab,newlab)
refit=read("2021_refit_k4.csv").query("seed==42")
check_est(ari,refit.ARI_vs_fixed);check_est(nmi,refit.NMI_vs_fixed);check_est(shares(newlab,np.ones(len(w))),refit.share_pct)
cm=pd.crosstab(lab,newlab).to_numpy()
check_est(cm,read("2021_fixed_vs_refit_confusion.csv").iloc[:,1:])
# Verify K4 sample silhouette using its saved centers, summing per point.
idx=np.random.default_rng(42).choice(len(w),1500,replace=False)
ls=newlab[idx]; z=X[idx]; ss=[]
for i,xi in enumerate(z):
    di=np.linalg.norm(z-xi,axis=1); gr=ls==ls[i]; own=di[gr].sum()/(gr.sum()-1)
    other=min(di[ls==j].mean() for j in set(ls) if j!=ls[i]); ss.append((other-own)/max(own,other))
ks=read("2021_k_search.csv")
check_est(np.mean(ss),ks[ks.k==4].silhouette)
# New independent sensitivities. Examine archive's unusual within-age-bin placement.
# Uniform integer ages within each released bin; then age-class expectation under historical code.
bins=[(0,5),(6,10),(11,14),(15,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,79),(80,100)]
midlegacy=[np.nan]
for lo,hi in bins:
    aa=np.arange(lo,hi+1); code=np.where(aa<15,0,np.minimum(75,15+10*((aa-15)//10)))
    midlegacy.append(np.where(code==0,12.5,code+7).mean())
legacyage=np.array(midlegacy)[np.nan_to_num(usages).astype(int)]
legacylab=nn(coord(legacyage,edu),c)[0]
caplab=nn(coord(np.where(age_mid==90,82,age_mid),edu),c)[0]
aged21=np.nanmin(usages,axis=1)>=4
rcount=np.c_[np.bincount(newregions)[1:],np.bincount(r.REGIO.to_numpy()//10)[1:]]
audit={"verified_protected_files":len(b["sources"]),"archived_replay":len(a),"regions":20,"new_n":len(w),"new_raw":len(r),"old_matched_size_1_to_5":len(m),"old_raw":len(o),"size_eligible_2021_with_child_0_to_10":int((young & (r.ncomp_mfr<=5)).sum()),"excluded_size_eligible_2021_without_child_0_to_10":int((~young & ~ok & (r.ncomp_mfr<=5)).sum()),"under_11_2021_accepted":int(young[ok].sum()),"old_domain_includes_any_under_15":int(young13.sum()),"eligible_weight_pct":100*w.sum()/r.PESO.sum(),"effective_n":w.sum()**2/(w@w),"national":[shares(lab,np.ones(len(w))),shares(lab,w)],"region_standardized":[std[0],std[1]],"std_minus_actual_2021":std-new,"raw_change":new-old,"standardized_change":std-old,"region_eligible_n_min_max":[rcount[:,0].min(),rcount[:,0].max()],"region_retention_extrema_pct":100*np.array([(rcount[:,0]/rcount[:,1]).min(),(rcount[:,0]/rcount[:,1]).max()]),"box_outside":int(support.sum()),"box_cross_by_type":np.bincount(lab[support],minlength=5)[1:],"nearest_ties_n":int((np.diff(distances[:,:2],axis=1)==0).sum()),"confusion":cm,"seed42_diag_agreement":np.trace(cm)/cm.sum(),"ARI_NMI":[ari,nmi],"K4_silhouette":np.mean(ss),"Ksearch_table_best":int(ks.loc[ks.silhouette.idxmax(),"k"]),"2013_age15plus_size15_archive_share":shares(m.segment[~young13]+1,np.ones(sum(~young13))),"2021_age15plus_size15_eligible_share":shares(lab[aged21],np.ones(aged21.sum())),"age15plus_size15_n":[sum(~young13),aged21.sum()],"age_expected_archive_style_conditional_share":shares(legacylab,np.ones(len(w))),"age_expected_archive_style_reassigned_pct":100*(legacylab!=lab).mean(),"age80plus_represented_by82_share":shares(caplab,np.ones(len(w))),"age80plus_represented_by82_reassigned_pct":100*(caplab!=lab).mean()}
(HERE/"independent_results.json").write_text(dump(audit),encoding="utf-8")
print(dump(audit))
