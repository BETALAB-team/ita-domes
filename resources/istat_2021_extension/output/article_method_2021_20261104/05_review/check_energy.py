"""Independent sufficient-statistic replay of the two new 2021 grouping histories."""
from pathlib import Path
import sys,json
here=Path(__file__).resolve().parent; pack=here.parent; root=pack.parents[1]; p=pack/"03_energy"; sys.path.insert(0,str(pack/".scientific"))
import numpy as np,pandas as pd
from scipy.stats import studentized_range
import statsmodels.api as sm
def eq(a,b):np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-7)
native=next((root/"istat/Microdati21").rglob("*Microdati*DELIMITED.txt"))
d=pd.read_csv(native,sep="^",dtype=str,usecols=["ID_RECORD","PESO","RIP","v1_1","v3_7","v3_8a","v3_8b","v3_8c","v5_6","v1_2_new"]+[f"v2_11{c}" for c in "abcde"]).apply(lambda s:pd.to_numeric(s.str.strip(),errors="coerce")).set_index("ID_RECORD",verify_integrity=True)
arc=pd.read_csv(pack/"02_probabilities/household_archetypes_2021.csv",usecols=["id","eligible","archetype"]).set_index("id",verify_integrity=True)
d=d.loc[arc[arc.eligible].index].copy();d["archetype"]=arc.loc[d.index].archetype.astype(int)
d["dwelling_type"]=d.v1_1.map(dict(enumerate(["single_family_house","multi_family_house","apartment_ground_basement_raised","apartment_intermediate_floor","apartment_top_floor"],1)))
d["location"]=d.RIP.map({1:"north_west",2:"north_east",3:"central",4:"south_and_islands",5:"south_and_islands"})
h=d[["v3_8a","v3_8b","v3_8c"]];d["H"]=h.sum(1,min_count=3).where(d.v3_7.isin([1,2])&(h<=8).all(axis=1)&(h>=0).all(axis=1))
e=pd.read_csv(root/"revised_paper_outputs/appliance_audit/electricity_2021.csv",usecols=["id","appliances_el_kWh"]).set_index("id",verify_integrity=True);d["E"]=e.loc[d.index].appliances_el_kWh
factors=["location","dwelling_type","archetype"]
audit={}
for prefix in ("H","E"):
    z=d.dropna(subset=[prefix]).copy()
    vals=z[prefix]; cell=z[factors].astype(str).agg(" | ".join,axis=1)
    groups={k:(int(len(xx)),float(vals.loc[xx].sum()),float((vals.loc[xx]**2).sum()),{k}) for k,xx in cell.groupby(cell).groups.items()}
    hist=pd.read_csv(p/f"{prefix}_merge_history_2021.csv")
    lookup=pd.read_csv(p/f"{prefix}_group_lookup_2021.csv"); assert len(groups)==len(lookup)==80
    def pair(u):
        keys=sorted(u); nn=np.array([u[k][0] for k in keys]); sums=np.array([u[k][1] for k in keys]); ss=np.array([u[k][2] for k in keys]);means=sums/nn
        df=int(sum(nn)-len(nn));mse=(ss-sums*sums/nn).sum()/df; a,b=np.triu_indices(len(nn),1)
        q=np.abs(means[a]-means[b])/np.sqrt(mse/2*(1/nn[a]+1/nn[b]))
        return keys,a,b,q,df
    for row in hist.itertuples():
        keys,i,j,q,df=pair(groups); a=keys.index(row.left_group);b=keys.index(row.right_group); selected=q[(i==min(a,b))&(j==max(a,b))][0]
        eq(selected,q.min()); assert row.adjusted_p_value>=.05
        g1=groups.pop(row.left_group);g2=groups.pop(row.right_group);groups[row.merged_group]=(g1[0]+g2[0],g1[1]+g2[1],g1[2]+g2[2],g1[3]|g2[3])
    names=sorted(groups,key=lambda k:groups[k][1]/groups[k][0])
    group_for_cell={cell:f"{prefix}{j}" for j,k in enumerate(names,1) for cell in groups[k][3]}
    assert lookup.set_index("original_combination").group.to_dict()==group_for_cell
    assigned=z.merge(lookup[factors+["group"]],on=factors,validate="many_to_one")
    sums=pd.read_csv(p/f"{prefix}_group_summary_2021.csv").set_index("group"); wsum=pd.read_csv(p/f"{prefix}_surveyweighted_summaries_2021.csv").set_index("group")
    for k,xx in assigned.groupby("group"):
        a=xx[prefix];w=xx.PESO
        eq([len(a),a.mean(),a.std(ddof=1),np.average(a,weights=w)],[sums.loc[k,"sample_count"],sums.loc[k,"mean"],sums.loc[k,"standard_deviation"],wsum.loc[k,"weighted_mean"]])
    keys,i,j,q,df=pair(groups); mostsimilarq=q.min()
    terminal_p=float(studentized_range.sf(mostsimilarq,len(groups),df))
    t=pd.read_csv(p/f"{prefix}_final_pairwise_tests_2021.csv");eq(terminal_p,t.p_adjusted.max());assert terminal_p<.05
    audit[prefix]={"n":len(z),"n_start":80,"n_groups":len(groups),"merges":len(hist),"mean":z[prefix].mean(),"weighted_mean":np.average(z[prefix],weights=z.PESO),"terminal_max_p":terminal_p,"cell_min_n":int(cell.value_counts().min())}
(here/"energy_checks.json").write_text(json.dumps(audit,indent=2));print(audit)
