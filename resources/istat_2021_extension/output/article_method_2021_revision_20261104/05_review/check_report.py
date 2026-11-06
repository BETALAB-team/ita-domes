"""Pointwise fidelity of report graphics, examples, editable equation and appendix."""
from pathlib import Path
import json,sys,hashlib,zipfile,xml.etree.ElementTree as ET
here=Path(__file__).resolve().parent;d=here.parent;sys.path.insert(0,str(d.parent/"article_method_2021_20261104/.scientific"))
import pandas as pd,numpy as np
R=d/"06_report";P=d/"02_probability_candidates";E=d/"03_energy_candidates";F=d/"04_figures";J=lambda x:json.loads(x.read_text(encoding="utf-8-sig"))
report=J(R/"report_content.json"); tabs=[x for x in report if x["kind"]=="table"]; eq=lambda x,y:np.testing.assert_allclose(x,y,rtol=1e-9,atol=1e-9)
manifest=J(P/"artifact_hashes.json");assert all(hashlib.sha256((P/k).read_bytes()).hexdigest()==h for k,h in manifest.items())
B=pd.read_csv(P/"recommended_full_quadratic_matrix.csv",index_col=0).to_numpy()
names={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main","m":"http://schemas.openxmlformats.org/officeDocument/2006/math"}
x=ET.fromstring(zipfile.ZipFile(next(R.glob("*.docx"))).read("word/document.xml"));cms=x.findall(".//m:m",names)
cm=next(v for v in cms if len(v.findall("m:mr",names))==4 and len(v.findall("m:mr",names)[0].findall("m:e",names))==6)
fmt=lambda v:f"{v:.5f}"
for r,row in zip(B,cm.findall("m:mr",names)):assert [fmt(y) for y in r]==["".join(v.itertext()) for v in row.findall("m:e",names)]
reg=next(t for t in tabs if t["headers"][0]=="Term");assert [rr[1:] for rr in reg["rows"][:6]]==[[fmt(v) for v in rr] for rr in B.T]
node=pd.read_csv(F/"figure_6_7_node_data.csv");o=pd.read_csv(P/"observed_supported_windows.csv");select=["building_age_years","dwelling_area_m2"];oo=o.set_index(select).loc[pd.MultiIndex.from_frame(node[select])];assert len(node)==125 and not node.duplicated(select).any()
a=node.building_age_years/100;s=node.dwelling_area_m2/100;X=np.stack([np.ones(len(a)),a,s,a*s,a*a,s*s],axis=1);eq(node[[f"p{i}_fit" for i in range(1,5)]],X@B.T);eq(node[[f"p{i}_obs" for i in range(1,5)]],oo[[f"p{i}" for i in range(1,5)]]);eq(node.n,oo.n)
flat=pd.read_csv(E/"appendix_A_assignment_2021_wide_flat.csv");ap=pd.read_csv(E/"appendix_A_compact_long.csv").set_index(["location","dwelling_type","archetype"],verify_integrity=True)
apt={"ground/semi-basement/raised-floor apartment":"apartment_ground_basement_raised","Apartment on ground, semi-basement or raised floor":"apartment_ground_basement_raised","Apartment on intermediate floor":"apartment_intermediate_floor","Apartment on top floor":"apartment_top_floor","Single Family House":"single_family_house","Multi Family House":"multi_family_house","single-family house":"single_family_house","multi-family house":"multi_family_house"}
loc={"North East":"north_east","North West":"north_west","Central":"central","South and Islands":"south_and_islands"}
at=next(t for t in tabs if t["headers"][0]=="Location");assert flat.astype(str).values.tolist()==at["rows"]
for rr in at["rows"]:
    for j in range(4):
        al=ap.loc[(loc[rr[0]],apt[rr[1]],j+1)];assert al.H_group==rr[2+j] and al.E_group==rr[6+j]
st=pd.read_csv(E/"selected_group_summaries.csv").set_index("group")
for tb in (t for t in tabs if t["headers"][0]=="Group"):
    for group,stats,n,ncells,example in tb["rows"]:
        r=st.loc[group];assert stats==f'{r["mean"]:.2f} ({r.sd:.2f})' and int(n.replace(",",""))==r.n and int(ncells)==r.cell_count
        l,t,a=example.split(", ");assert ap.loc[(loc[l],apt[t],int(a[1:]))][group[0]+"_group"]==group
print({"candidate_manifest_entries":len(manifest),"blocks":len(report),"tables":len(tabs),"nodes":len(node),"means_examples":len(st),"matrix_shape":B.shape,"appendix_entries":len(ap)*2})
