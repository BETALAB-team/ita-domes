"""Check new direct survey and archetype-conditional results from native raw files."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
here=Path(__file__).resolve().parent
root=here.parents[2]
out=here.parent/"02_analysis"
raw13=next((root/"istat/ECH_2013_IT/MICRODATI").glob("*.txt"))
raw21=next((root/"istat/Microdati21").rglob("*Microdati*DELIMITED.txt"))
summary=pd.read_csv(out/"full_size1_5_direct_survey.csv")
regional=pd.read_csv(out/"full_size1_5_direct_regional.csv")
conditional=pd.read_csv(out/"conditional_heating_cooling_new_diagnostic.csv")
new=pd.read_csv(out/"assigned_2021_proxy.csv").set_index("id").archetype
arc=pd.read_csv(root/"clusters.csv").set_index("identi").segment+1
domains={}; metrics={}; labels={}
for y,p,sep,ageprefix,size,idkey,w,geo,typ,fields in [
    (2013,raw13,"\t","cletad_","q_1_1_sq1","nprog","coef_red","reg",arc,[["q_3_10","q_3_11A","q_3_11B","q_3_11C"],["q_5_7","q_5_8A","q_5_8B","q_5_8C"]]),
    (2021,raw21,"^","eta_mfr","ncomp_mfr","ID_RECORD","PESO","REGIO",new,[["v3_7","v3_8a","v3_8b","v3_8c"],["v5_6","v5_7a","v5_7b","v5_7c"]])]:
    cols={size,idkey,w,geo}|set(sum(fields,[]))
    d=pd.read_csv(p,sep=sep,dtype=str,usecols=lambda col:col in cols or col.startswith(ageprefix)).apply(lambda col:pd.to_numeric(col.str.strip().str.replace(",","."),errors="coerce"))
    d=d[d[size].between(1,5)].copy()
    ages=d.filter(regex="^"+ageprefix).to_numpy()
    mem=np.arange(ages.shape[1])[None,:]<d[size].to_numpy()[:,None]
    incomplete=mem & ~np.isin(ages,([0,15,25,35,45,55,65,75] if y==2013 else range(1,12)))
    child=((ages==0 if y==2013 else ages<=3)&mem).any(1)
    ws=d[w].to_numpy()
    a=d[idkey].map(typ).to_numpy()
    df=summary[summary.year==y].set_index("metric")
    tests={"household_size_mean":d[size],"has_under15_pct":100*child}
    tests.update({f"household_size{i}_pct":100*(d[size]==i) for i in range(1,6)})
    for key,v in tests.items():
        np.testing.assert_allclose([np.mean(v),np.average(v,weights=ws)],[df.loc[key,"unweighted"],df.loc[key,"surveyweighted"]])
    domains[y]={"n":len(d),"age_incomplete_profiles":int(incomplete.any(1).sum()),"weight_sum":ws.sum(),"under15_pct_w":np.average(100*child,weights=ws),"mean_size_w":np.average(d[size],weights=ws)}
    for (kind,cc) in zip(["heating","cooling"],fields):
        arr=d[cc[1:]].to_numpy()
        vv=(d[cc[0]].isin([1,2]) & np.isfinite(arr).all(1) & (arr>=0).all(1) & (arr<=8).all(1)).to_numpy()
        hours=arr.sum(1)
        use=d[geo].to_numpy()
        R=regional[(regional.year==y)&(regional.metric==kind+"_conditional_hours")]
        # File can be regenerated with canonical geography; the national total is unchanged.
        if y==2021 and R.region.max()<=20: use=use//10
        for rr in R.itertuples():
            mm=vv if rr.region==0 else vv&(use==rr.region)
            np.testing.assert_allclose([mm.sum(),np.mean(hours[mm]),np.average(hours[mm],weights=ws[mm])],[rr.valid_n,rr.unweighted,rr.surveyweighted])
        selected=conditional[(conditional.year==y)&(conditional.outcome==kind)]
        for rr in selected.itertuples():
            mm=vv&(a==rr.archetype)
            np.testing.assert_allclose([mm.sum(),np.mean(hours[mm]),np.std(hours[mm],ddof=1),np.average(hours[mm],weights=ws[mm]),ws[mm].sum()**2/(ws[mm]@ws[mm])],[rr.conditional_valid_n,rr.mean_hours_unweighted,rr.sd_hours_unweighted,rr.mean_hours_surveyweighted,rr.effective_n])
        metrics[f"{y}_{kind}"]={"denominator":int(vv.sum()),"zero_sums":int((hours[vv]==0).sum()),"unweighted":np.mean(hours[vv]),"weighted":np.average(hours[vv],weights=ws[vv])}
result={"raw_1_to_5":domains,"full_domain_hours":metrics}
(here/"direct_results.json").write_text(json.dumps(result,indent=2,default=lambda x:x.item()))
print(result)

