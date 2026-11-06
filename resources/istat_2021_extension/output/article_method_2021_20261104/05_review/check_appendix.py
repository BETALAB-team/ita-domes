"""Verify 2021 Appendix A, plot strata and estimable ANOVA table independently."""
from pathlib import Path
import json,sys
here=Path(__file__).resolve().parent;pack=here.parent;sys.path.insert(0,str(pack/".scientific"))
import pandas as pd,numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
p=pack/"03_energy"
keys=["location","dwelling_type","archetype"]
ap=pd.read_csv(p/"appendix_A_assignment_2021.csv").set_index(keys,verify_integrity=True); assert len(ap)==4*5*4
wide=pd.read_csv(p/"appendix_A_assignment_2021_wide.csv").set_index(["Location","Dwelling Type"],verify_integrity=True);assert len(wide)==20
raw=pd.read_csv(pack/"01_inputs/analysis_input_2021.csv",usecols=["id","location","dwelling_type","cooling_frequency","cooling_inventory_status","heating_daily_hours","electricity_annual_kwh"]).set_index("id",verify_integrity=True)
labels=pd.read_csv(pack/"02_probabilities/household_archetypes_2021.csv",usecols=["id","archetype"]).set_index("id",verify_integrity=True).dropna();d=raw.loc[labels.index].join(labels).copy(); d["archetype"]=d.archetype.astype(int)
d["cooling_system_use"]=np.select([d.cooling_frequency.isin(range(1,6)),(d.cooling_frequency==6)|(d.cooling_inventory_status=="absent")],["present_or_used","absent_or_unused"],default="unknown")
d=d.join(pd.read_csv(pack/"02_probabilities/dwelling_samples_2021.csv",usecols=["id","building_age_years","dwelling_area_m2"]).set_index("id",verify_integrity=True))
aa=pd.read_csv(p/"anova_results_2021.csv");saved=json.loads((p/"anova_models_2021.json").read_text()); rec={}
for pref,c,m in [("H","heating_daily_hours","heating_group"),("E","electricity_annual_kwh","electricity_group")]:
 look=pd.read_csv(p/f"{pref}_group_lookup_2021.csv").set_index(keys,verify_integrity=True).group
 for ix,r in ap.iterrows():
    assert r[m]==look.loc[ix]
    assert wide.loc[(r.location_label,r.dwelling_label),f"{pref} User Cluster {ix[2]}"]==r[m]
 dd=d.dropna(subset=[c]).join(look,on=keys)
 if pref=="E":
    counts=dd.groupby(["group","cooling_system_use"]).size()
    out=pd.read_csv(p/"electricity_plot_counts_2021.csv").set_index(["group","cooling_system_use"]).n.sort_index()
    assert np.array_equal(counts.loc[out.index].values,out.values)
 q=dd.dropna(subset=["building_age_years","dwelling_area_m2"]);q=q[q.cooling_system_use!="unknown"]
 info=next(s for s in saved if s["response"]==c)
 # Independent canonical formula, with all available two-way categorical terms and age × area.
 model=ols(f"{c} ~ (C(location)+C(dwelling_type)+C(archetype))**2+building_age_years*dwelling_area_m2+C(cooling_system_use)",data=q).fit()
 n=model.nobs; assert n==info["n"] and model.model.rank==info["rank"]
 np.testing.assert_allclose(model.rsquared,info["R2"],atol=1e-9)
 a=anova_lm(model,typ=2); src=aa[aa.response==c]
 norm=lambda k:":".join(sorted(k.split(":")))
 for row in src.itertuples():
    sr=a.loc[next(k for k in a.index if norm(k)==norm(row.term))]
    np.testing.assert_allclose([sr["sum_sq"],sr.F,sr.df],[row.sum_squares,row.F,row.df],atol=1e-6,rtol=1e-8,equal_nan=True)
 rec[pref]={"anova_n":int(n),"anova_rank":int(model.model.rank),"R2":model.rsquared}
rec["wide_group_cells"]=160;rec["cooling_split"]=d.cooling_system_use.value_counts().to_dict()
(here/"appendix_checks.json").write_text(json.dumps(rec,indent=2));print(rec)
