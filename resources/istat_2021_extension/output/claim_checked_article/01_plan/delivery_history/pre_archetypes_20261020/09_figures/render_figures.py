"""Reproduce the two approved scientific figures; read-only v5 source verification.
Run: C:/Works/EUReCA/EUReCA/.venv/Scripts/python.exe -B 09_figures/render_figures.py
"""
import os, csv, json, hashlib, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / '09_figures'
os.environ['MPLCONFIGDIR'] = str(OUT / '.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def close(a,b): assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-8), (a,b)
def export(name, rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
brief = ROOT/'09_figures/figure_briefs.md'
assert sha(brief)=='3a3f9f3bd803c75e97e3369524d7e7dcee4792550c15a00cfb3d6f412c42bd4d'
paired=ROOT/'04_analysis/paired_schedule_contrasts_v5.csv'
partition=ROOT/'04_analysis/posthoc_setpoint_partition_v5.csv'
rows=read(paired); assert len(rows)==12
sites=['Padova','Bolzano','Roma','Palermo']; schedules=['seeded','early','late']
assert {(r['site'],r['schedule']) for r in rows}=={(s,t) for s in sites for t in schedules}
sourcepaths={brief,paired,partition,ROOT/'06_claim_drafts/article_claims_final.md',ROOT/'06_claim_drafts/final_acceptance.md',ROOT/'07_reviews/final_closure.md',ROOT/'01_plan/prose_gate.json'}
checks=[]; raw={}
for r in rows:
    s,t=r['site'],r['schedule']; base=next(q for q in rows if q['site']==s and q['schedule']=='seeded')
    p=ROOT/'05_simulations'/r['run_id'].rsplit('/',1)[0]/(t+'_hourly.csv')
    assert p.resolve()==Path(r['hourly_path']).resolve()
    h=read(p); raw[(s,t)]=h; assert len(h)==8760
    positive=[max(float(x['sensible_load_W']),0) for x in h]
    close(math.fsum(positive)/1000,r['annual_sensible_heating_kWh']); close(max(positive)/1000,r['peak_sensible_heating_kW'])
    for field,delta,percent in [('annual_sensible_heating_kWh','delta_annual_kWh','delta_annual_percent'),('peak_sensible_heating_kW','delta_peak_kW','delta_peak_percent')]:
        d=float(r[field])-float(base[field]); close(d,r[delta]); close(100*d/float(base[field]),r[percent])
    sourcepaths.add(p)
    for name in ['summary.json','completion_manifest.json','design.json']:
        q=p.parent/name; assert q.exists(); sourcepaths.add(q)
    checks.append({'run_id':r['run_id'],'hours':len(h),'annual_and_peak_raw_check':'pass','stored_same_site_percentages':'pass'})
for t in schedules:
    h=raw[('Bolzano',t)]; pk=max(h,key=lambda x:float(x['sensible_load_W'])); assert int(pk['hour'])==6912
    close(pk['heating_setpoint_C'],19); close(h[6911]['heating_setpoint_C'],0)
    r=next(r for r in rows if r['site']=='Bolzano' and r['schedule']==t); assert float(r['delta_peak_percent'])==0
selected=[r for r in read(partition) if r['site']=='Padova' and r['schedule']=='seeded' and float(r['setpoint_C'])==19]
assert len(selected)==1
r=selected[0]; subset=[h for h in raw[('Padova','seeded')] if float(h['heating_setpoint_C'])==19]
tol=float(r['positive_load_tolerance_W']); assert tol==1e-6
pos=[h for h in subset if float(h['sensible_load_W'])>tol]
assert len(subset)==int(r['calendar_hours_at_setpoint'])==2730
assert len(pos)==int(r['positive_heating_hours_gt_1e_minus6W'])==1323
close(math.fsum(float(h['sensible_load_W']) for h in pos)/1000,r['positive_sensible_heating_kWh'])
plotrows=[r for r in rows if r['schedule']!='seeded']
export('FIG-01_plotting_data.csv',plotrows); export('FIG-02_plotting_data.csv',selected)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#777777','xtick.color':'#333333','ytick.color':'#333333','savefig.facecolor':'white'})
colors={'early':'#0072B2','late':'#D55E00'}; markers={'early':'o','late':'s'}
fig,axes=plt.subplots(1,2,figsize=(7.2,4.1),sharey=True)
fig.subplots_adjust(left=.105,right=.98,top=.77,bottom=.34,wspace=.26)
for ax,field,title,limits,ticks in zip(axes,['delta_annual_percent','delta_peak_percent'],['A  Annual useful sensible\n    heating','B  Annual maximum hourly\n    heating load'],[(-.7,4.65),(-3.6,10.4)],[[0,1,2,3,4],[-2,0,2,4,6,8,10]]):
    ax.axvline(0,color='#777777',lw=.7,zorder=0)
    ax.set_axisbelow(True); ax.grid(axis='y',color='#eeeeee',lw=.6)
    for i,s in enumerate(sites):
        for t,offset in [('early',-.13),('late',.13)]:
            q=next(q for q in plotrows if q['site']==s and q['schedule']==t); x=float(q[field]); y=i+offset
            ax.scatter(x,y,s=32,marker=markers[t],color=colors[t],zorder=3)
            label='0.000' if x==0 else f'{x:+.3f}'
            ax.annotate(label,(x,y),xytext=(6,0),textcoords='offset points',va='center',fontsize=8,color=colors[t])
    ax.set(yticks=range(4),yticklabels=sites,xlim=limits,xticks=ticks,ylim=(3.45,-.45))
    ax.tick_params(axis='y',length=0,pad=7)
    ax.set_title(title,loc='left',pad=12,fontweight='bold'); ax.set_xlabel('Change from same-site\nseeded placement (%)',labelpad=9)
handles=[Line2D([0],[0],marker=markers[t],color='none',markerfacecolor=colors[t],markeredgecolor=colors[t],label=t.capitalize()+' placement',markersize=6) for t in ['early','late']]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.98),ncol=2,frameon=False)
fig.text(.105,.105,'Bolzano: all three profile maxima occur at the shared imposed season restart.',fontsize=8)
fig.text(.105,.04,'One assumed dwelling; one response; common season; deterministic alternatives',fontsize=8,color='#444444')
for ext in ['svg','png']: fig.savefig(OUT/f'FIG-01_paired_contrasts.{ext}',dpi=400)
plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,3.2)); fig.subplots_adjust(left=.10,right=.93,top=.71,bottom=.43)
ax.barh([0],[2730],height=.44,color='#e3e6e9',edgecolor='#777777',linewidth=.7)
ax.barh([0],[1323],height=.44,color='#0072B2')
ax.text(661.5,0,'1323 positive-load hours',ha='center',va='center',color='white',fontsize=10,fontweight='bold')
ax.text(2730,.38,'2730 calendar hours at 19°C',ha='right',va='center',fontsize=9)
ax.set(xlim=(0,2730),ylim=(-.5,.65),yticks=[],xticks=[0,500,1000,1500,2000,2500,2730],xlabel='Calendar time at modeled 19°C setback (h)')
ax.spines['bottom'].set_color('#777777'); ax.tick_params(axis='x',labelsize=8)
fig.text(.10,.88,'Post-hoc; Padova seeded model output',fontsize=11,fontweight='bold')
fig.text(.10,.16,'Associated useful sensible heat: 2375.62 kWh',fontsize=10)
fig.text(.10,.055,'Positive-load threshold: >10⁻⁶ W. One modeled case; no measured-operation comparator.',fontsize=8,color='#444444')
for ext in ['svg','png']: fig.savefig(OUT/f'FIG-02_setback_diagnostic.{ext}',dpi=400)
plt.close(fig)
# Check vector labels and image dimensions independently of the plotting calls.
for name,required in [('FIG-01_paired_contrasts.svg',['Padova','Bolzano','Roma','Palermo','0.000','+8.326','-2.103','common season']),('FIG-02_setback_diagnostic.svg',['1323 positive-load hours','2730 calendar hours','2375.62 kWh','Post-hoc; Padova'])]:
    svg=(OUT/name).read_text(encoding='utf-8')
    for label in required: assert label in svg,label
from PIL import Image
images={}
for p in [OUT/'FIG-01_paired_contrasts.png', OUT/'FIG-02_setback_diagnostic.png']:
    im=Image.open(p); assert min(im.info['dpi'])>=399; images[p.name]={'pixels':im.size,'dpi':im.info['dpi']}
provenance={'status':'all numerical/source selection checks passed; visual inspection recorded separately','python':sys.version,'matplotlib':matplotlib.__version__,'brief_sha256':sha(brief),'sources_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(sourcepaths)},'checks':checks,'FIG-02':{'selection':'Padova / seeded / 19 C','calendar_hours':2730,'positive_load_hours':1323,'tolerance_W':tol,'heat_kWh_label':'2375.62'},'transforms':'Stored percent values plotted after raw positive-W hourly integration, maximum and same-site seeded denominator checks; labels three decimals. FIG-02 stored selected counts and heat only; no new comparative result.','images':images,'outputs_sha256':{p.name:sha(p) for p in sorted(OUT.glob('FIG-*')) if p.suffix in ['.svg','.png','.csv'] and '_QA_' not in p.name},'script_sha256':sha(Path(__file__)),'FIG-03':'blocked; no empirical data or image produced'}
(OUT/'figure_provenance_qa.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'checks':'passed','images':images,'outputs':provenance['outputs_sha256']},indent=2))




