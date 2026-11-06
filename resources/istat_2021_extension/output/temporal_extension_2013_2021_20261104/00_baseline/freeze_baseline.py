"""Read-only source audit: no clustering/PCA fit, no baseline rewrite."""
import csv, json, hashlib, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
CTX=ROOT/'output/claim_checked_article/00_context/archetype_extension'
PDF=Path('C:/Users/khajmoh18975/Downloads/1-s2.0-S0378778825006541-main.pdf')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
sources=[PDF,ROOT/'clusters.csv',ROOT/'Clustered.csv',ROOT/'family_trees.csv',ROOT/'istat_microdata_2013.csv',ROOT/'01_PCA_Exploratory.py',ROOT/'02_Families_Analysis.py',ROOT/'03_PCA_Clustering.py',ROOT/'archetype_pipeline.py',ROOT/'baseline_outputs/household_archetypes.csv',ROOT/'revised_paper_outputs/2013/households.csv',CTX/'local_assignment_audit.json',CTX/'published_appendix_A_group_lookup.csv',CTX/'primary_complete_text.txt']
manifest=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sources if p.exists()]
audit=json.loads((CTX/'local_assignment_audit.json').read_text())
rows=list(csv.DictReader((ROOT/'clusters.csv').open(newline='')))
cols=['normal_l','normal_g','normal_s']
centers={str(k+1):[sum(float(r[c]) for r in rows if int(r['segment'])==k)/sum(int(r['segment'])==k for r in rows) for c in cols] for k in range(4)}
dis=[]
for r in rows:
    ds=[sum((float(r[c])-centers[str(k+1)][j])**2 for j,c in enumerate(cols)) for k in range(4)]
    pred=min(range(4),key=lambda k:ds[k])
    if pred!=int(r['segment']): dis.append(dict(id=r['identi'],saved=int(r['segment'])+1,nearest=pred+1))
scalers={c:dict(min=min(float(r[c]) for r in rows),max=max(float(r[c]) for r in rows)) for c in ['loneliness','generation','studio_ind']}
residuals={}
for raw,norm in [('loneliness','normal_l'),('generation','normal_g')]:
    lo,hi=scalers[raw]['min'],scalers[raw]['max']
    residuals[raw]=max(abs((float(r[raw])-lo)/(hi-lo)-float(r[norm])) for r in rows)
params=dict(provenance='Means of saved normalized features grouped by unchanged archived labels; NOT recovered original fitted estimator and NOT a refit.',features=cols,formula={'loneliness':'0.25*min_age+0.75*max_age-10*number','generation':'0.33*min_age+0.66*max_age+10*number','studio':'studio_ind'},scalers=scalers,centers=centers,normalization_max_abs_residual=residuals,saved_label_nearest_center_disagreement_n=len(dis),saved_label_nearest_center_disagreements=dis)
(OUT/'archive_transfer_parameters.json').write_text(json.dumps(params,indent=2))
bench=[]
for r in audit['published_rounded_comparison']:
    bench.append(dict(kind='demographic',group='A'+str(r['archetype_id']),metric=r['metric'],value=r['published'],sd='',unit='percent' if 'pct' in r['metric'] else '',locator='p7 Table3 / Fig5',note='Printed benchmark; no replacement by archive-derived value'))
heating=[(4.67,3.54),(5.25,3.77),(6.91,4.43),(7.53,4.41),(8.53,4),(9.14,4.17),(9.67,4.28)]
electric=[(1801,1502),(2023,1631),(2320,1502),(2646,1844),(2895,1853)]
for kind,prefix,vals,unit in [('heating','H',heating,'h/day'),('electricity','E',electric,'kWh/year')]:
    for i,(v,sd) in enumerate(vals,1):bench.append(dict(kind=kind,group=prefix+str(i),metric='mean',value=v,sd=sd,unit=unit,locator='p10 Table6',note='Declared on-hours' if prefix=='H' else 'Survey-based appliance model, not metered electricity'))
terms=['intercept','age/100','area/100','age*area/10000','age^2/10000','area^2/10000']
eq=[[.289,.692,0,0,-.372,-.084],[.323,-.523,.104,0,.269,0],[.335,-.230,-.028,0,.127,0],[.033,.039,0,.014,.152,0]]
tab=[[.29,.69,0,0,-.37,-.08],[.32,-.52,.10,0,.27,0],[.34,-.23,-.03,0,.13,0],[.03,.04,0,.01,.02,0]]
for name,mat,loc in [('polynomial_Eq3',eq,'p7 Eq3'),('polynomial_Table4',tab,'p8 Table4')]:
    for k,co in enumerate(mat,1):
        for term,v in zip(terms,co):bench.append(dict(kind=name,group='A'+str(k),metric=term,value=v,sd='',unit='',locator=loc,note='Printed absence encoded0. A4 age^2 differs materially between Eq3 and Table4; retained separately.'))
for k,vals in enumerate(zip([648.9,571,282.8,219.7],[.94,.93,.87,.84],[.020,.016,.008,.011]),1):
    for metric,v in zip(['F_statistic','adjusted_R2','RMSE'],vals):bench.append(dict(kind='polynomial_fit',group='A'+str(k),metric=metric,value=v,sd='',unit='',locator='p8 Table4',note='Original model fit; no new validation'))
for group,vals in [('Wilks_lambda',[.8105,.2226,1]),('Roy_greatest_root',[.1687,.5329,.9357])]:
    for metric,v in zip(['value','F_value','p_value'],vals):bench.append(dict(kind='regional_MANOVA',group=group,metric=metric,value=v,sd='',unit='',locator='p6 Table2',note='Printed inference concerns regional centroid effects, not equality of energy or shares'))
(OUT/'published_benchmarks.json').write_text(json.dumps(bench,indent=2))
with (OUT/'published_benchmarks.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(bench[0]));w.writeheader();w.writerows(bench)
freeze=dict(policy='Published paper numbers frozen exactly as printed. Archived membership is supplementary article-consistent evidence, not certified identical published membership.',sources=manifest,archive_n=len(rows),label_map={'0':1,'1':2,'2':3,'3':4},published_cell_matches=39,published_cell_checks=40,unresolved=['A4 max-age72.2 paper versus72.1 archive rounded','Eq3 A4 age-squared0.152 versus Table4 0.02','Six electricity groups in prose versus E1-E5 tables/appendix','Duplicated H6/H7 examples','p9 prose conflicts with Table5'],integrity_after=[dict(path=m['path'],sha256=sha(Path(m['path'])),unchanged=sha(Path(m['path']))==m['sha256']) for m in manifest])
(OUT/'baseline_freeze.json').write_text(json.dumps(freeze,indent=2))
print(json.dumps(dict(n=len(rows),scalers=scalers,centers=centers,disagreement_n=len(dis),integrity_all_unchanged=all(r['unchanged'] for r in freeze['integrity_after'])),indent=2))
