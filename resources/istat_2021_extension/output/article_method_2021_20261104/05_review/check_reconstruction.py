"""Read-only independent checks of native 2021 records, new labels and probability fits."""
from pathlib import Path
import hashlib,itertools,json,sys
here=Path(__file__).resolve().parent; pack=here.parent; root=pack.parents[1]; p=pack/"02_probabilities"; sys.path.insert(0,str(pack/".scientific"))
import pandas as pd, numpy as np, statsmodels.api as sm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
def read(f):return pd.read_csv(p/f)
def J(x):return json.loads(x.read_text(encoding="utf-8-sig"))
def eq(a,b):np.testing.assert_allclose(a,b,atol=2e-9,rtol=1e-10,equal_nan=True)
def matrix(a,s):
    return np.stack((np.ones_like(a),a,s,a*s,a*a,s*s),axis=1)
def fit(a,y):
    cols=list(range(6))
    while True:
        m=sm.OLS(y,a[:,cols]).fit()
        worst=1+int(np.argmax(m.pvalues[1:])) if len(cols)>1 else 0
        if worst==0 or m.pvalues[worst]<=.05:break
        del cols[worst]
    out=np.zeros(6);out[cols]=m.params
    return out
native=next((root/"istat/Microdati21").rglob("*Microdati*DELIMITED.txt"))
fields={"ID_RECORD","PESO","REGIO","RIP","ncomp_mfr","v1_1","v1_2_new","v1_7_mfr","v3_7","v3_8a","v3_8b","v3_8c"}
d=pd.read_csv(native,sep="^",dtype=str,usecols=lambda s:s in fields or s.startswith(("eta_mfr","istr_mfr"))).apply(lambda col:pd.to_numeric(col.str.strip(),errors="coerce")).set_index("ID_RECORD",verify_integrity=True)
inp=pd.read_csv(pack/"01_inputs/analysis_input_2021.csv",usecols=["id","weight","dwelling_type_code_2021","macroregion_code","heating_daily_hours","electricity_annual_kwh","dwelling_area_lower","dwelling_area_upper","building_age_lower","building_age_upper"]).set_index("id",verify_integrity=True).loc[d.index]
alllab=read("household_archetypes_2021.csv").set_index("id",verify_integrity=True).loc[d.index]
eq(inp.weight,d.PESO);eq(inp.dwelling_type_code_2021,d.v1_1);eq(inp.macroregion_code,np.minimum(d.RIP,4))
blocks=d[["v3_8a","v3_8b","v3_8c"]]; target=blocks.sum(1,min_count=3).where(d.v3_7.isin([1,2])&(blocks>=0).all(axis=1)&(blocks<=8).all(axis=1));eq(inp.heating_daily_hours,target)
eng=pd.read_csv(root/"revised_paper_outputs/appliance_audit/electricity_2021.csv").set_index("id",verify_integrity=True).loc[d.index];eq(inp.electricity_annual_kwh,eng.appliances_el_kWh)
age=d.filter(regex="^eta_").to_numpy();school=d.filter(regex="^istr_").to_numpy();n=d.ncomp_mfr.to_numpy()
active=np.arange(5)[None,:]<n[:,None]
observed=active&(age>=3)
ok=(n<=5)&(~active|np.isin(age,range(1,12))).all(1)&(~observed|np.isin(school,[1,3,4,6])).all(1)&observed.any(1)
assert np.array_equal(ok,alllab.eligible)
young=(active&(age<=2)).any(1)
v=alllab[ok]; bounds=[(0,5),(6,10),(11,14),(15,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,79),(80,100)]; rng=np.random.RandomState(42)
assert not np.isfinite(school[(age<=2)&active]).any()
recon=[]
for i in np.flatnonzero(n<=5):
    ages=[]; ratios=[]
    for j in range(n[i]):
        code=int(age[i,j]);lo,hi=bounds[code-1]
        ar=rng.randint(lo,hi+1);ages.append(ar)
        if code<3:continue
        if not np.isfinite(school[i,j]):break
        e={1:5,3:8,4:12,6:18}[school[i,j]]
        denom=np.array([5,8,12,16,18,21])[np.searchsorted([15,19,23,25,29],ar,side="right")]
        ratios.append(min(1,e/denom))
    if ok[i]:recon.append([min(ages),max(ages),np.mean(ratios)])
eq(recon,v[["min_age","max_age","studio_ind"]])
lo,hi,ed=np.array(recon).T
ag=np.stack([.25*lo+.75*hi-10*n[ok],.33*lo+.66*hi+10*n[ok]],axis=1)
x=np.c_[(ag-ag.min(0))/np.ptp(ag,axis=0),ed]
eq(x,v[["normal_l","normal_g","normal_s"]]);labels=v.archetype.astype(int).to_numpy()
cent=np.stack([x[labels==l].mean(0) for l in range(1,5)]);assert np.array_equal(cdist(x,cent).argmin(1)+1,labels)
old=pd.read_csv(root/"clusters.csv");cc=old.groupby("segment")[["normal_l","normal_g","normal_s"]].mean().to_numpy()
M=np.c_[(ag-old[["loneliness","generation"]].min().to_numpy())/np.ptp(old[["loneliness","generation"]].to_numpy(),axis=0),ed]
aa=np.stack([M[labels==l].mean(0) for l in range(1,5)])
assert np.array_equal(linear_sum_assignment(cdist(aa,cc,"sqeuclidean"))[1],range(4))
pcvar=np.linalg.eigvalsh(np.cov(x.T))[::-1];eq(pcvar/pcvar.sum(),J(p/"archetype_fit_parameters.json")["pca_explained_ratio"])
s=read("archetype_summary_2021.csv")
for r in s.itertuples():
    z=v[v.archetype==r.archetype]
    eq([len(z),len(z)/len(v)*100,z.weight.sum()/v.weight.sum()*100,z.number.sum()/v.number.sum()*100],[r.n,r.household_share,r.weighted_household_share,r.person_share])
samples=read("dwelling_samples_2021.csv").set_index("id",verify_integrity=True)
dsub=d[ok & d.v1_2_new.isin(range(1,9)) & d.v1_7_mfr.isin(range(1,17))]
assert samples.index.equals(dsub.index)
eq(samples.archetype,v.loc[dsub.index].archetype)
# Native-code intervals at the 2021 reference date. Bounds on open categories are assumptions.
ac=[(0,11),(12,21),(22,31),(32,41),(42,51),(52,61),(62,71),(72,150)]
sc=[(10,39),(40,49),(50,59),(60,69),(70,79),(80,89),(90,99),(100,109),(110,119),(120,129),(130,149),(150,169),(170,199),(200,249),(250,299),(300,400)]
rng=np.random.RandomState(42)
for col,arr,res in [("v1_2_new",ac,"building_age_years"),("v1_7_mfr",sc,"dwelling_area_m2")]:
    draws=[rng.randint(arr[k-1][0],arr[k-1][1]+1) for k in dsub[col].astype(int)];eq(draws,samples[res])
grid=read("probability_grid_empirical.csv");wg=read("probability_grid_surveyweighted_sensitivity.csv");sa=samples.building_age_years.to_numpy();ss=samples.dwelling_area_m2.to_numpy();l=samples.archetype.to_numpy();w=dsub.PESO.to_numpy()
def grid_sample(ids):
    counts=[];ratios=[];wr=[]
    for t in grid.itertuples():
        mask=ids & (abs(sa-t.building_age_years)<=6)&(abs(ss-t.dwelling_area_m2)<=8)
        freq=np.bincount(l[mask],minlength=5)[1:]
        counts.append(freq);ratios.append(freq/max(1,freq.sum()))
        wr.append(np.bincount(l[mask],weights=w[mask],minlength=5)[1:]/max(1,w[mask].sum()))
    return np.array(counts),np.array(ratios),np.array(wr)
freq,rat,wr=grid_sample(np.ones(len(sa),bool));eq(freq,grid[[f"archetype_{k}_count" for k in range(1,5)]])
pcols=[f"prob_archetype_{k}" for k in range(1,5)];eq(rat,grid[pcols]);eq(wr,wg[pcols])
X=matrix(grid.building_age_years.to_numpy()/100,grid.dwelling_area_m2.to_numpy()/100)
infit=(grid.dwelling_area_m2<150).to_numpy()&(freq.sum(1)>=5); B=np.stack([fit(X[infit],col[infit]) for col in rat.T]);WB=np.stack([fit(X[infit],col[infit]) for col in wr.T])
eq(B,read("polynomial_coefficient_matrix.csv").iloc[:,1:]);eq(WB,read("polynomial_matrix_surveyweighted_sensitivity.csv").iloc[:,1:]);P=X@B.T; eq(P,read("polynomial_full_grid_predictions_flags.csv")[pcols])
pr=np.random.RandomState(2021).permutation(len(sa));train=np.zeros(len(sa),bool);train[pr[:int(.7*len(sa))]]=True
f1,r1,_=grid_sample(train);f2,r2,_=grid_sample(~train);u=(f1.sum(1)>=5)&(grid.dwelling_area_m2<150);t=(f2.sum(1)>=5)&(grid.dwelling_area_m2<150);C=np.stack([fit(X[u],col[u]) for col in r1.T]);diff=r2[t]-X[t]@C.T;hs=read("household_holdout_probability_scores.csv");eq(np.sqrt(np.mean(diff**2,axis=0)),hs.RMSE)
ref=J(root/"output/temporal_extension_2013_2021_20261104/00_baseline/baseline_freeze.json")
for row in ref["sources"]:assert hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()==row["sha256"]
audit={"raw_n":len(d),"eligible_n":len(v),"retained_with_under11":int((young&ok).sum()),"eligible_weight_fraction":v.weight.sum()/d.PESO.sum(),"pfe_n":len(samples),"pooled_window_n":int(freq.sum()),"fit_windows":int(infit.sum()),"fit_range":list(map(float,[P[infit].sum(1).min(),P[infit].sum(1).max()])),"fullgrid_outside01":int(((P<0)|(P>1)).any(1).sum()),"infit_outside01":int(((P[infit]<0)|(P[infit]>1)).any(1).sum()),"B":B.tolist(),"holdout_rmse":np.sqrt(np.mean(diff**2,axis=0)).tolist(),"protected_sources":len(ref["sources"])}
(here/"reconstruction_checks.json").write_text(json.dumps(audit,indent=2));print(audit)
