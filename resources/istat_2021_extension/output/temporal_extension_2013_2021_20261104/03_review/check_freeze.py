"""Frozen-file and source-inventory audit."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
here=Path(__file__).resolve().parent; root=here.parents[2]; out=here.parent/"02_analysis"
meta=json.loads((out/"artifact_hashes.json").read_text(encoding="utf-8-sig"))
for name,h in meta.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h,name
expected=pd.read_csv(out/"full_size1_5_cooling_inventory.csv")
e=[]
for y,path,sep,size,w,items in [(2013,next((root/"istat/ECH_2013_IT/MICRODATI").glob("*.txt")),"\t","q_1_1_sq1","coef_red",["q_2_37"+s for s in "ABCDEF"]),(2021,next((root/"istat/Microdati21").rglob("*Microdati*DELIMITED.txt")),"^","ncomp_mfr","PESO",["v2_11"+s for s in "abcde"])]:
    raw=pd.read_csv(path,sep=sep,dtype=str,usecols=[size,w]+items).apply(lambda s:pd.to_numeric(s.str.strip().str.replace(",","."),errors="coerce")); d=raw[raw[size].between(1,5)]
    obs=d[items].to_numpy()
    yes=(obs==1).any(1);no=(obs==2).all(1)
    for name,mask in [("affirmative_inventory",yes),("all_items_explicit_no",no),("unknown_no_affirmative",~yes & ~no)]:
        z=expected.query("year==@y and status==@name").iloc[0]
        np.testing.assert_allclose([mask.sum(),100*np.average(mask,weights=d[w])],[z.n,z.surveyweighted_pct])
    e.append([y,int(yes.sum()),int(no.sum()),int((~yes & ~no).sum()),100*np.average(yes,weights=d[w])])
audit={"verified_analysis_files":len(meta),"analysis_manifest_sha256":hashlib.sha256((out/"artifact_hashes.json").read_bytes()).hexdigest(),"cooling_inventory_year_affirmative_explicit_negative_unknown_weighted_affirmative":e}
(here/"freeze_audit.json").write_text(json.dumps(audit,indent=2));print(audit)
