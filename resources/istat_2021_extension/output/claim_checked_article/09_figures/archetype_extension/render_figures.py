"""Render frozen archetype-extension figures; no model execution or new result estimation.
Run read-only existing EUReCA Python with -B. Outputs and cache confined beside script.
"""
from pathlib import Path
import os,csv,json,hashlib,math,sys
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
PRE=ROOT/'05_simulations/archetype_campaign_v2/pre_run'; RES=ROOT/'04_analysis/archetype_extension/results_v2'
SITES=['Padova','Bolzano','Roma','Palermo']
sourcehash={}; checks=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p,expected=None):
    h=sha(p); assert expected is None or h==expected,(p,h,expected)
    sourcehash[str(p.relative_to(ROOT)).replace('\\','/')]=h
    return p
def read(p,expected=None):
    bind(p,expected)
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def js(p,expected=None):return json.loads(bind(p,expected).read_text(encoding='utf-8-sig'))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-9),(a,b)
def writecsv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
bind(OUT/'figure_briefs.md','9eaad3a1892bf650600857986b350b1abb234c598f26bff9627c7e5b269bf993')
bind(ROOT/'06_claim_drafts/article_claims_final.md','9aa90444db044ab1480872fbf496db171f359e39e10cf1207f76a4c586757df9')
for p in ['06_claim_drafts/final_acceptance.md','07_reviews/archetype_extension/final_closure.md','01_plan/archetype_extension/prose_gate.json']:bind(ROOT/p)
components=read(PRE/'component_manifest.csv','4a6f07041552f312e292da1b24f50e36b36cd31f72df43438bb89a748ed90e00')
profiles=js(PRE/'profile_manifest.json','663a54e956020c606b8189305093dd78fe2456724d9135c46ec827ba6c384dcc')
prob=js(PRE/'mixture_probabilities.json','f97a68f29aca5fac7e0c6fa3f8284c82e8960133afd4c5eee032df4f3dda8a43')
assert prob['training_n']==13592 and prob['free_probability_parameters']=={'pooled':3,'conditional':12}
profiledata=[]; curves=[]
assert [r['household_id'] for r in components]==['2941','981','5303','9574']
for c in components:
    p=profiles[c['profile_id']]; annual=read(PRE/(c['profile_id']+'.csv'))
    vals=np.array([float(r['lower_C']) for r in annual],dtype='<f8')
    assert len(vals)==8760 and hashlib.sha256(vals.tobytes()).hexdigest()==c['profile_sha256']==p['profile_sha256']
    daily=[p['on_C'] if x else p['off_C'] for x in p['daily_high_setpoint']]
    assert len(daily)==24 and list(vals[:24])==daily
    curves.append(daily)
    for h,v in enumerate(daily):profiledata.append({'role_id':c['role_id'],'household_id':c['household_id'],'control_branch':c['control_branch'],'profile_sha256':c['profile_sha256'],'hour_fixed_UTC_plus_1':h,'modeled_lower_setpoint_C':v})
branches=prob['branch_order']; matrix=[]; probs=[]
for group in ['Pooled','1','2','3','4']:
    counts={b:sum(prob['group_branch_training_counts'][g][b] for g in ['1','2','3','4']) for b in branches} if group=='Pooled' else prob['group_branch_training_counts'][group]
    total=sum(counts.values()); stored=prob['pooled_P_branch'] if group=='Pooled' else prob['training_P_branch_given_group'][group]
    if group=='Pooled':assert total==13592
    matrix.append([stored[b] for b in branches]);close(sum(stored.values()),1)
    for b in branches:
        close(stored[b],counts[b]/total);probs.append({'row':group,'branch':b,'training_count':counts[b],'training_row_denominator':total,'stored_probability':stored[b]})
writecsv('FIG-04_profile_plotting_data.csv',profiledata);writecsv('FIG-04_probability_plotting_data.csv',probs)
(OUT/'FIG-04_plotting_data.json').write_text(json.dumps({'profiles':profiledata,'probabilities':probs},indent=2)+'\n')
metrics=read(RES/'weighted_metrics.csv','c2b0b514d7d3e6f2a489cec6ce1535d803ea8355fd921e28289f24e70bd143a6')
js(RES/'analysis_manifest.json','f8e5803ede9299103d36a3ccfd2432d517d4220a5227b83362858d90ea94aa9b')
endpoints=read(RES/'independent_endpoints.csv'); deviations=read(RES/'individual_deviations.csv'); inventory=read(RES/'execution_inventory.csv')
test=read(PRE/'test_respondents.csv'); roles=read(PRE/'role_manifest.csv'); strata=read(PRE/'stratum_counts.csv','d23a457ff25a03405fdd1d6eb6c7af6aecdb9a43f3e22d20814dfea94a8eb085');bind(PRE/'selection_manifest.json')
assert len(test)==32 and len({r['household_id'] for r in test})==32
for r in test:
    s=next(s for s in strata if s['archetype_id']==r['archetype_id'] and s['control_branch']==r['control_branch'])
    close(r['overall_empirical_weight'],int(s['eligible_n'])/(2*13624));close(r['within_group_weight'],int(s['eligible_n'])/(2*int(s['group_n'])))
assert len(inventory)==152 and sum(r['role_category']=='scientific' for r in inventory)==144
# Inventory paths are admitted source execution IDs, not guessed household identifiers.
for r in inventory:
    if r['role_category']!='scientific':continue
    path=ROOT/'05_simulations/archetype_campaign_v2'/r['site']/(r['case']+'_hourly.csv')
    bind(path,r['hourly_sha256'])
    e=next(e for e in endpoints if e['site']==r['site'] and e['profile_id']==r['profile_id']);assert e['hourly_sha256']==r['hourly_sha256']
# Recheck the stored displayed RMSE against saved individual deviations and domain weights.
selected=[r for r in metrics if r['domain']=='all_eligible_empirical_groups' or (r['site']=='Bolzano' and r['endpoint']=='peak_sensible_heating_kW')]
assert len(selected)==12
for r in selected:
    group=None if r['domain']=='all_eligible_empirical_groups' else r['domain'].split('_')[-1]
    d=[d for d in deviations if d['site']==r['site'] and d['endpoint']==r['endpoint'] and (group is None or d['archetype_id']==group)]
    assert len(d)==int(r['n_test'])==(32 if group is None else 8)
    w='overall_empirical_weight' if group is None else 'within_group_weight';close(math.fsum(float(x[w]) for x in d),1)
    for method in ['conditional','pooled']:
        mse=math.fsum(float(x[w])*(float(x[method+'_prediction'])-float(x['detailed_reference']))**2 for x in d)
        close(mse,r[method+'_MSE']);close(math.sqrt(mse),r[method+'_RMSE'])
    close(float(r['conditional_MSE'])-float(r['pooled_MSE']),r['paired_MSE_difference_conditional_minus_pooled'])
    checks.append({'site':r['site'],'endpoint':r['endpoint'],'domain':r['domain'],'n':len(d),'weight_sum_and_stored_RMSE':'pass'})
for group in ['2','4']:
    r=next(r for r in selected if r['domain']=='archetype_'+group);assert float(r['conditional_RMSE'])>float(r['pooled_RMSE'])
ctx=read(ROOT/'04_analysis/archetype_extension/posthoc_peak_context_v1/cohort_counts.csv','ce81c5fb21338365fdb084b1c846f9829e4e20d669993ccd259c946170af45f2')
for role,n,k in [('test',32,27),('component',4,2)]:
    r=next(r for r in ctx if r['site']=='Bolzano' and r['role']==role);assert int(r['denominator'])==n and int(r['n_first_max_on_restart_day'])==k
for f in ['scientific_peak_context.csv','diagnostic_manifest.json']:bind(ROOT/'04_analysis/archetype_extension/posthoc_peak_context_v1'/f)
bind(ROOT/'03_methods/archetype_extension/posthoc_peak_context_report.md')
writecsv('FIG-05_plotting_data.csv',selected)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'svg.fonttype':'none','svg.hashsalt':'archetype-extension-v07','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#777777','xtick.color':'#333333','ytick.color':'#333333'})
COLOR={'conditional':'#0072B2','pooled':'#D55E00'};MARKER={'conditional':'o','pooled':'s'}
metadata={'Date':None,'Creator':'Matplotlib; frozen v07 programmatic figures'}
def save(fig,stem):
    for ext in ['svg','png']:fig.savefig(OUT/(stem+'.'+ext),dpi=400,metadata=metadata if ext=='svg' else None)
    plt.close(fig)
    im=Image.open(OUT/(stem+'.png'));im.resize((1000,round(im.height*1000/im.width))).save(OUT/(stem+'_QA_1000px.png'))
fig=plt.figure(figsize=(7.6,5.4));gs=fig.add_gridspec(4,2,width_ratios=[1,1.18],left=.09,right=.96,top=.82,bottom=.20,wspace=.50,hspace=.58)
fig.text(.09,.94,'A  Shared executable component profiles',weight='bold',fontsize=10)
fig.text(.57,.94,'B  Training branch probabilities',weight='bold',fontsize=10)
for i,(c,daily) in enumerate(zip(components,curves)):
    ax=fig.add_subplot(gs[i,0]);ax.step(range(25),daily+[daily[-1]],where='post',lw=1.4,color='#0072B2')
    for b in [5,13,21]:ax.axvline(b,color='#d1d1d1',lw=.6,ls='--',zorder=0)
    ax.set(xlim=(0,24),ylim=(9,26),yticks=[11,19,24],xticks=[0,5,13,21,24]);ax.tick_params(labelsize=7,length=2)
    ax.text(.02,1.05,f"{c['control_branch']}°C · respondent {c['household_id']}",transform=ax.transAxes,fontsize=8,backgroundcolor='white')
    if i<3:ax.tick_params(labelbottom=False)
    else:ax.set_xlabel('Hour of day (fixed UTC+1)',labelpad=6)
fig.text(.018,.53,'Modeled lower heating setpoint (°C)',rotation=90,va='center',fontsize=9)
ax=fig.add_subplot(gs[:,1]);im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
ax.set(xticks=range(4),xticklabels=branches,yticks=range(5),yticklabels=['Pooled','Group 1','Group 2','Group 3','Group 4'])
ax.tick_params(length=0,labelsize=8);ax.set_xlabel('Control branch (high/setback °C)',labelpad=8)
for y in range(5):
    for x in range(4):ax.text(x,y,f'{matrix[y][x]:.3f}',ha='center',va='center',fontsize=9,color='#222222')
cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.055,pad=.17,aspect=20);cb.set_ticks([0,.5,1]);cb.ax.tick_params(labelsize=7);cb.set_label('Training probability',fontsize=8)
fig.text(.09,.085,'13,592 training records · conditional: 12 free probabilities · pooled: 3',fontsize=9)
fig.text(.09,.035,'Modeled setpoints; shared components simulated separately; dwelling, gains and season fixed.',fontsize=8,color='#444444')
save(fig,'FIG-04_components_probabilities')
fig=plt.figure(figsize=(7.6,7.1));gs=fig.add_gridspec(2,2,left=.17,right=.98,top=.78,bottom=.20,hspace=.58,wspace=.48)
fig.text(.17,.975,'Same-model detailed-input reference; one held-out split',fontsize=11,weight='bold')
handles=[Line2D([],[],marker=MARKER[m],color='none',markerfacecolor=COLOR[m],markeredgecolor=COLOR[m],label=m.capitalize(),markersize=6) for m in ['conditional','pooled']]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.60,.95),ncol=2,frameon=False,fontsize=9)
def dots(ax,rr,labels,title,limit,decimals,xlabel):
    for i,r in enumerate(rr):
        a=float(r['conditional_RMSE']);b=float(r['pooled_RMSE'])
        ax.plot([a,b],[i-.14,i+.14],color='#bbbbbb',lw=.8,zorder=1)
        for m,dy in [('conditional',-.14),('pooled',.14)]:
            x=float(r[m+'_RMSE']);ax.scatter(x,i+dy,s=27,marker=MARKER[m],color=COLOR[m],zorder=3)
            ax.annotate(f'{x:.{decimals}f}',(x,i+dy),xytext=(6,0),textcoords='offset points',va='center',fontsize=8,color=COLOR[m])
    ax.set(yticks=range(4),yticklabels=labels,ylim=(3.50,-.50),xlim=(0,limit));ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=6,labelsize=8)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#eeeeee',lw=.6);ax.set_title(title,loc='left',pad=12,weight='bold',fontsize=10);ax.set_xlabel(xlabel,labelpad=7)
for j,(ep,title,lim,dec,unit) in enumerate([('annual_sensible_heating_kWh','A  Annual useful heating\n    Overall (n = 32)',3500,2,'Model-reference RMSE (kWh)'),('peak_sensible_heating_kW','B  Expected individual\n    annual maximum\n    Overall (n = 32)',6.6,4,'Model-reference RMSE (kW)')]):
    rr=[next(r for r in selected if r['site']==s and r['endpoint']==ep and r['domain']=='all_eligible_empirical_groups') for s in SITES]
    ax=fig.add_subplot(gs[0,j]);dots(ax,rr,SITES,title,lim,dec,unit)
ax=fig.add_subplot(gs[1,:]);rr=[next(r for r in selected if r['domain']=='archetype_'+str(i)) for i in range(1,5)]
dots(ax,rr,['Group 1','Group 2\nconditional worse','Group 3','Group 4\nconditional worse'],'C  Bolzano expected individual annual maximum · all groups (n = 8 each)',3.8,6,'Model-reference RMSE (kW)')
fig.text(.17,.10,'Post hoc — Bolzano: 27/32 test and 2/4 component maxima on imposed restart day',fontsize=8)
fig.text(.17,.055,'One dwelling; fixed gains/season; conditional 12 versus pooled 3 free probabilities.',fontsize=8,color='#444444')
fig.text(.17,.02,'Expected individual maxima; no coincident-peak, measured-accuracy or sizing inference.',fontsize=8,color='#444444')
save(fig,'FIG-05_fidelity_metrics')
# SVG text checks bind visible precision and caveats to admitted stored values.
for stem,labels in [('FIG-04_components_probabilities',['2941','981','5303','9574','12 free probabilities','0.630']),('FIG-05_fidelity_metrics',['1787.94','2.3571','2.008767','1.965770','2.941238','2.929172','conditional worse','27/32','2/4'])]:
    text=(OUT/(stem+'.svg')).read_text(encoding='utf-8')
    for label in labels:assert label in text,(stem,label)
imageinfo={}
for stem in ['FIG-04_components_probabilities','FIG-05_fidelity_metrics']:
    im=Image.open(OUT/(stem+'.png'));assert min(im.info['dpi'])>399;imageinfo[stem]={'pixels':im.size,'dpi':im.info['dpi'],'width_inches':7.6}
qa={'status':'numerical checks passed; actual visual QA recorded in figure_manifest.md','sources_sha256':sourcehash,'checks':checks,'profile_arrays':'all 4 full8760 little-endian float64 hashes verified; first active-day curves equal frozen daily mappings','probabilities':'20 count/denominator probabilities and row sums verified','test_weights':'all32 overall and within-group weights verified against16 eligible strata','raw_sources':'144 scientific hourly hashes match inventory and independent endpoints;8technical executions excluded','Bolzano_restart_context':'stored cohort counts27/32test and2/4component verified; posthoc only','images':imageinfo,'python':sys.version,'matplotlib':matplotlib.__version__,'outputs_sha256':{p.name:sha(p) for p in sorted(OUT.glob('FIG-*')) if p.suffix in ['.svg','.png','.csv','.json']},'script_sha256':sha(Path(__file__)),'FIG-06':'blocked; no rendering or data produced'}
(OUT/'figure_provenance_qa.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'numerical_checks':'pass','outputs':qa['outputs_sha256']},indent=2))



