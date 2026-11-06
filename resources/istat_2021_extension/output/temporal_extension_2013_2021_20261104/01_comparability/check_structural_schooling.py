from pathlib import Path
import pandas as pd,json
r=Path.cwd();o=r/'output/temporal_extension_2013_2021_20261104/01_comparability'
p=next((r/'istat/Microdati21').glob('*/MICRODATI/*2021*DELIMITED.txt'));d=pd.read_csv(p,sep='^',dtype=str,keep_default_na=False);n=pd.to_numeric(d.ncomp_mfr);w=pd.to_numeric(d.PESO)
u=pd.Series(False,index=d.index);blank=True;members=0
for k in range(1,6):
 age=pd.to_numeric(d[f'eta_mfr{k}'].str.strip(),errors='coerce');m=(n>=k)&age.isin([1,2]);u|=m;members+=int(m.sum());blank=blank and bool(d.loc[m,f'istr_mfr{k}'].str.strip().eq('').all())
a=(n<=5)&u
out={'year':2021,'under11_member_records':members,'all_under11_schooling_blank':blank,'households_under11_all_release':int(u.sum()),'households_under11_size1to5':int(a.sum()),'size1to5_total':int(n.le(5).sum()),'under11_household_weight_share_size1to5':float(w[a].sum()/w[n.le(5)].sum()),'implication':'Strict all-member schooling completeness necessarily excludes every household containing an under11 member. Regional weighting cannot recover structural zero inclusion.'}
(o/'structural_schooling_coverage.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(out)
for name in ['comparability_matrix','verified_analysis_input_spec','regional_factor_inventory']:
 q=o/(name+'.md');q.write_text(q.read_text(encoding='utf8')+'\n\n**Critical coverage restriction (fresh raw audit):**2021education is structurally blank for all released members aged0–10. A strict all-member complete-schooling profile excludes every household containing an under11child. This is selection,not population prevalence;regional weights cannot repair zero inclusion. See structural_schooling_coverage.json and02_analysis/coverage_under15_2021.csv.\n',encoding='utf8')
 q=o/(name+'.json');j=json.loads(q.read_text(encoding='utf8'));j['critical_schooling_coverage']=out;q.write_text(json.dumps(j,indent=2,ensure_ascii=False),encoding='utf8')
