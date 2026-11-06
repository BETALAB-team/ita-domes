from pathlib import Path
import json,hashlib,itertools,math,time
import numpy as np,pandas as pd
from PIL import Image,ImageDraw
ROOT=Path.cwd(); OUT=ROOT/'output/temporal_extension_2013_2021_20261104/02_analysis'; BASE=OUT.parent/'00_baseline'
P21=ROOT/'istat/Microdati21/500Tj00000M4IEcIAN-1_00250525-1/MICRODATI/ISTAT_MFR_ECH_Microdati_2021_DELIMITED.txt'
P13=ROOT/'istat/ECH_2013_IT/MICRODATI/ECH_Microdati_Anno_2013.txt'
PARAM=BASE/'archive_transfer_parameters.json'
paths=[P21,P13,ROOT/'clusters.csv',PARAM,OUT/'analysis_spec.md']
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
raw=pd.read_csv(P21,sep='^',low_memory=False); old=pd.read_csv(P13,sep='\t',low_memory=False); arc=pd.read_csv(ROOT/'clusters.csv')
par=json.loads(PARAM.read_text(encoding='utf-8-sig')); centers=np.array(list(par['centers'].values()))
regmap={10*i:i for i in range(1,21)};regmap.update({41:4,42:4})
ranges={1:(0,5),2:(6,10),3:(11,14),4:(15,19),5:(20,29),6:(30,39),7:(40,49),8:(50,59),9:(60,69),10:(70,79),11:(80,100)}
def num(s):return pd.to_numeric(s.astype(str).str.replace(',','.',regex=False),errors='coerce')
raw['PESO']=num(raw.PESO);old['coef_red']=num(old.coef_red);old['q_1_1_sq1']=num(old.q_1_1_sq1)
def build(mode='midpoint',seed=42,school='legacy'):
 rng=np.random.default_rng(seed);records=[];ex=[]
 for _,r in raw.iterrows():
  n=int(r.ncomp_mfr)
  if not 1<=n<=5:ex.append((r.ID_RECORD,'size_outside_1_5'));continue
  ac=[int(r[f'eta_mfr{i}']) for i in range(1,n+1)];ec=[int(str(r[f'istr_mfr{i}']).strip() or 0) for i in range(1,n+1)]
  if any(a not in ranges for a in ac) or any(e not in [1,3,4,6] for e in ec):ex.append((r.ID_RECORD,'invalid_age_or_education'));continue
  if not r.PESO>0:ex.append((r.ID_RECORD,'invalid_weight'));continue
  ages=np.array([(sum(ranges[a])/2 if mode=='midpoint' else rng.integers(ranges[a][0],ranges[a][1]+1)) for a in ac])
  # Coarse source categories cannot recover archived within-bin ages or education.
  # Legacy primary-school cascading recode 3->5->12 is preserved as explicit proxy.
  years=np.array([{1:(12 if school=='legacy' else 5),3:8,4:12,6:18}[e] for e in ec])
  cls=np.where(ages<15,0,(np.floor((ages-15)/10)*10+15))
  potential=np.select([cls<10,cls<15,cls<19,cls<23,cls<25,cls<29],[0,5,8,12,16,18],default=21)
  ratios=np.divide(years,potential,out=np.ones(n,dtype=float),where=potential>0);edu=np.minimum(ratios,1).mean()
  records.append((r.ID_RECORD,n,ages.min(),ages.max(),edu,regmap[int(r.REGIO)],int(r.RIP),r.PESO))
 d=pd.DataFrame(records,columns=['id','number','min_age','max_age','studio_ind','region','macroregion','weight'])
 X=transform(d);d['archetype']=assign(X);d['out_of_support']=((X<0)|(X>1)).any(axis=1)
 return d,X,pd.DataFrame(ex,columns=['id','reason'])
def transform(d):return np.column_stack(((.25*d.min_age+.75*d.max_age-10*d.number+40.5)/114.5,(.33*d.min_age+.66*d.max_age+10*d.number-30.79)/91.05,d.studio_ind))
def assign(X):return ((X[:,None,:]-centers[None,:,:])**2).sum(2).argmin(1)+1
def km(X,k,seed):
 rng=np.random.default_rng(seed);best=None
 for run in range(10):
  c=X[rng.choice(len(X),k,replace=False)].copy()
  for j in range(150):
   lab=((X[:,None,:]-c[None,:,:])**2).sum(2).argmin(1)
   nc=np.array([X[lab==i].mean(0) if (lab==i).any() else X[rng.integers(len(X))] for i in range(k)])
   if np.max(abs(nc-c))<1e-8:break
   c=nc
  inertia=float(((X-c[lab])**2).sum())
  if best is None or inertia<best[0]:best=(inertia,lab,c)
 return best
def agreement(a,b):
 tab=pd.crosstab(a,b).to_numpy();n=tab.sum();comb=lambda x:(x*(x-1)/2).sum();s=comb(tab);aa=comb(tab.sum(1));bb=comb(tab.sum(0));exp=aa*bb/(n*(n-1)/2);ari=(s-exp)/((aa+bb)/2-exp)
 p=tab/n;pa=p.sum(1);pb=p.sum(0);nz=p>0;mi=(p[nz]*np.log((p/(pa[:,None]*pb[None,:]))[nz])).sum();ha=-(pa[pa>0]*np.log(pa[pa>0])).sum();hb=-(pb[pb>0]*np.log(pb[pb>0])).sum()
 return float(ari),float(2*mi/(ha+hb))
def summaries(d):
 rows=[]
 for a in range(1,5):
  z=d[d.archetype==a];rows.append(dict(archetype=a,n=len(z),denominator=len(d),share_pct=100*len(z)/len(d),weighted_share_pct=100*z.weight.sum()/d.weight.sum(),people_share_pct=100*z.number.sum()/d.number.sum(),weight_sum=z.weight.sum(),effective_n=z.weight.sum()**2/(z.weight**2).sum(),min_age=z.min_age.mean(),max_age=z.max_age.mean(),number=z.number.mean(),studio_ind=z.studio_ind.mean()))
 return pd.DataFrame(rows)
d,X,ex=build();d.to_csv(OUT/'assigned_2021_proxy.csv',index=False);ex.to_csv(OUT/'exclusions_2021.csv',index=False)
summ=summaries(d);summ.to_csv(OUT/'2021_fixed_proxy_summary.csv',index=False)
# New archive-domain diagnostic; never substitutes for printed table.
a13=arc.merge(old[['nprog','reg','coef_red','q_1_1_sq1']],left_on='identi',right_on='nprog',validate='one_to_one');a13=a13[(a13.q_1_1_sq1.between(1,5))&(a13.coef_red>0)].copy();a13['archetype']=a13.segment+1;a13['region']=a13.reg;a13['weight']=a13.coef_red
regionrows=[]
for r in sorted(set(a13.region)&set(d.region)):
 for a in range(1,5):
  z0=a13[a13.region==r];z1=d[d.region==r];p0=(z0.archetype==a).mean();p1=(z1.archetype==a).mean();w0=z0.loc[z0.archetype==a,'weight'].sum()/z0.weight.sum();w1=z1.loc[z1.archetype==a,'weight'].sum()/z1.weight.sum()
  regionrows.append(dict(region=r,archetype=a,n2013=len(z0),n2021=len(z1),count2013=int((z0.archetype==a).sum()),count2021=int((z1.archetype==a).sum()),share2013=100*p0,share2021=100*p1,change_pp=100*(p1-p0),weighted2013=100*w0,weighted2021=100*w1,weighted_change_pp=100*(w1-w0)))
reg=pd.DataFrame(regionrows);reg.to_csv(OUT/'regional_common_cohort_changes.csv',index=False)
comp=[]
for a in range(1,5):
 rr=reg[reg.archetype==a];q=rr.n2013/rr.n2013.sum();qw=np.array([a13.loc[a13.region==r,'weight'].sum() for r in rr.region],dtype=float);qw/=qw.sum()
 comp.append(dict(archetype=a,archive2013_common_n=len(a13),new2021_n=len(d),archive2013_unweighted=rr.share2013@q,proxy2021_unweighted=rr.share2021@(rr.n2021/rr.n2021.sum()),proxy2021_at2013_sample_region_mix=rr.share2021@q,region_standardized_change_pp=(rr.share2021-rr.share2013)@q,archive2013_surveyweighted=rr.weighted2013@qw,proxy2021_at2013_surveyweighted_region_mix=rr.weighted2021@qw))
pd.DataFrame(comp).to_csv(OUT/'regional_composition_standardization.csv',index=False)
sens=[]
for name,mode,seed,school in [('primary','midpoint',42,'legacy'),('age_seed0','random',0,'legacy'),('age_seed42','random',42,'legacy'),('age_seed2021','random',2021,'legacy'),('education_code1_5years','midpoint',42,'conventional')]:
 dd,xx,_=build(mode,seed,school); ss=summaries(dd)
 ari,nmi=agreement(d.archetype,dd.archetype)
 for _,r in ss.iterrows():sens.append(dict(scenario=name,archetype=int(r.archetype),share_pct=r.share_pct,n=len(dd),ARI_vs_primary=ari,NMI_vs_primary=nmi,changed_pct=100*(dd.archetype.to_numpy()!=d.archetype.to_numpy()).mean()))
for name,xx in [('clip_to_archive_range',np.clip(X,0,1)),('paper_exact_thirds',np.column_stack((X[:,0],((d.min_age/3+2*d.max_age/3+10*d.number)-30.79)/91.05,X[:,2])))]:
 lab=assign(xx);ari,nmi=agreement(d.archetype,lab)
 for a in range(1,5):sens.append(dict(scenario=name,archetype=a,share_pct=100*(lab==a).mean(),n=len(d),ARI_vs_primary=ari,NMI_vs_primary=nmi,changed_pct=100*(lab!=d.archetype).mean()))
inside=d[~d.out_of_support]; ss=summaries(inside)
for _,r in ss.iterrows():sens.append(dict(scenario='in_support_only',archetype=int(r.archetype),share_pct=r.share_pct,n=len(inside)))
pd.DataFrame(sens).to_csv(OUT/'transfer_sensitivities.csv',index=False)
refits=[]; aligned=None
for seed in [0,42,2021]:
 inertia,lab,c=km(X,4,seed);cost=((c[:,None,:]-centers[None,:,:])**2).sum(2);perm=min(itertools.permutations(range(4)),key=lambda p:sum(cost[i,p[i]] for i in range(4)));ali=np.array(perm)[lab]+1;ari,nmi=agreement(d.archetype,ali)
 for a in range(1,5):refits.append(dict(seed=seed,archetype=a,share_pct=100*(ali==a).mean(),ARI_vs_fixed=ari,NMI_vs_fixed=nmi,inertia=inertia,matching='exhaustive 4! optimal sum squared centroid distance (Hungarian-equivalent)',matched_refit_cluster=int(perm.index(a-1))))
 if seed==42:aligned=ali;pd.DataFrame(c,columns=['normal_l','normal_g','normal_s']).assign(matched_archetype=np.array(perm)+1).to_csv(OUT/'2021_refit_centroids.csv',index=False)
pd.DataFrame(refits).to_csv(OUT/'2021_refit_k4.csv',index=False);pd.crosstab(d.archetype,pd.Series(aligned,name='aligned_refit')).to_csv(OUT/'2021_fixed_vs_refit_confusion.csv')
# deterministic 1500-point silhouette to bound memory/time; all rows fit.
idx=np.random.default_rng(42).choice(len(X),min(1500,len(X)),replace=False);Z=X[idx];dist=np.sqrt(((Z[:,None,:]-Z[None,:,:])**2).sum(2));search=[]
for k in range(2,9):
 inertia,lab,c=km(X,k,42);ls=lab[idx];a=np.zeros(len(Z));b=np.full(len(Z),np.inf)
 for j in range(k):
  mask=ls==j;nn=mask.sum();mean=dist[:,mask].mean(1);b[~mask]=np.minimum(b[~mask],mean[~mask]);a[mask]=dist[np.ix_(mask,mask)].sum(1)/max(nn-1,1)
 sil=np.divide(b-a,np.maximum(a,b),out=np.zeros_like(a),where=np.maximum(a,b)>0).mean();search.append(dict(k=k,inertia=inertia,silhouette=float(sil),silhouette_n=len(Z)))
pd.DataFrame(search).to_csv(OUT/'2021_k_search.csv',index=False)
res=dict(raw2021_n=len(raw),eligible2021_n=len(d),excluded2021_n=len(ex),raw2013_n=len(old),archived2013_n=len(arc),common_archived2013_n=len(a13),out_of_support_n=int(d.out_of_support.sum()),weight2021_effective_n=float(d.weight.sum()**2/(d.weight**2).sum()),transfer_status='archive-calibrated fixed-centroid proxy; not exact published estimator',published_benchmarks_unchanged=True,best_silhouette_k=max(search,key=lambda v:v['silhouette'])['k'],source_hashes=hashes,source_hashes_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()),software=dict(python='bundled 3.12',numpy=np.__version__,pandas=pd.__version__,kmeans='bounded numpy Lloyd, 10 random starts, 150 iterations, tolerance 1e-8',matching='optimal 4! enumeration'),limitations=['2021 age11 upper limit100 is assumption for open-ended80+','Education code1 aggregates no/primary schooling; archive legacy collision cannot be recovered exactly','2013 paper whole archive and 2021 size<=5 cohort differ; descriptive benchmark contrast is not a clean temporal trend','No confidence intervals: harmonization uncertainty and incompatible survey design not represented by binomial precision','Heating/equipment and modeled energy deferred pending verified comparability'])
(OUT/'results.json').write_text(json.dumps(res,indent=2))
# Scientific grouped bar plots, direct SVG and raster with identical geometry.
def bars(filename,series,title):
 colors=['#3c6382','#e58e26','#6a89cc'];W,H=1100,620;im=Image.new('RGB',(W,H),'white');draw=ImageDraw.Draw(im);svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/>']
 def txt(x,y,s,size=18):
  draw.text((x,y),s,fill='black',font_size=size);svg.append(f'<text x="{x}" y="{y+size}" font-family="Arial" font-size="{size}">{s}</text>')
 txt(55,20,title,24)
 for t in range(0,61,10):
  y=510-t*6;draw.line((70,y,1050,y),fill='#dddddd');svg.append(f'<path d="M70 {y} H1050" stroke="#ddd"/>');txt(30,y-10,str(t),15)
 for j,(label,values) in enumerate(series):
  txt(80+j*330,65,label,16)
  for i,v in enumerate(values):
   x=120+i*235+j*55;y=510-float(v)*6;color=colors[j];draw.rectangle((x,y,x+48,510),fill=color);svg.append(f'<rect x="{x}" y="{y}" width="48" height="{510-y}" fill="{color}"/>');txt(x,y-23,f'{v:.1f}',15)
 for i in range(4):txt(130+i*235,530,f'A{i+1}',20)
 txt(70,580,'Household share (%) — proxy labels; cross-sectional samples',17)
 svg.append('</svg>');(OUT/f'{filename}.svg').write_text('\n'.join(svg),encoding='utf-8');im.save(OUT/f'{filename}.png');assert Image.open(OUT/f'{filename}.png').size==(W,H)
bars('published_vs_2021_proxy',[('Published 2013 (all archived)',[40.1,26.9,22.9,10.1]),('Eligible 2021 proxy',summ.share_pct.tolist())],'Published benchmark and 2021 transfer (different cohort coverage)')
bars('fixed_vs_refit_2021',[('Frozen transfer proxy',summ.share_pct.tolist()),('2021 K4 aligned refit',[v['share_pct'] for v in refits if v['seed']==42])],'2021 fixed transfer versus exploratory aligned refit')
print(json.dumps(res,indent=2))



