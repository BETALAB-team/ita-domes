"""Run the article's cell grouping on 2021; all outputs remain in this folder."""
from pathlib import Path
import sys,json,hashlib,warnings
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'.scientific'));sys.path.insert(0,str(REPO))
import numpy as np,pandas as pd
from energy_analysis import iterative_tukey_groups
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import studentized_range
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch,Rectangle
sys.stdout.reconfigure(encoding='utf-8');HERE.mkdir(exist_ok=True)
F=ROOT/'04_figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'axes.linewidth':.6,'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10,'svg.fonttype':'none','savefig.facecolor':'white'})
raw=pd.read_csv(ROOT/'01_inputs/analysis_input_2021.csv',low_memory=False)
labels=pd.read_csv(ROOT/'02_probabilities/household_archetypes_2021.csv')
assert raw.id.is_unique and labels.id.is_unique
labelcols=labels.loc[labels.eligible,['id','archetype']].copy();labelcols.archetype=labelcols.archetype.astype(int)
d=raw.merge(labelcols,on='id',validate='one_to_one');assert len(d)==len(labelcols)
samples=pd.read_csv(ROOT/'02_probabilities/dwelling_samples_2021.csv')
d=d.drop(columns=['building_age_years','dwelling_area_m2']).merge(samples[['id','building_age_years','dwelling_area_m2']],on='id',how='left',validate='one_to_one')
# Use reported frequencies to distinguish rare actual cooling use from never.
freq=d.cooling_frequency
d['cooling_system_use']=np.select([freq.isin([1,2,3,4,5]),freq.eq(6)|d.cooling_inventory_status.eq('absent')],['present_or_used','absent_or_unused'],default='unknown')
d['cooling_use_conflict']=freq.isin([1,2,3,4,5])&d.cooling_inventory_status.eq('absent')
assert not d.cooling_use_conflict.any()
factors=['location','dwelling_type','archetype']
domains={'heating_daily_hours':d[d.heating_daily_hours.notna()&d.heating_daily_hours.between(0,24)].copy(),
         'electricity_annual_kwh':d[d.electricity_annual_kwh.notna()&d.electricity_annual_kwh.gt(0)].copy()}
results={};maps={};stats={};histories=[];terminal=[]
for response,prefix in [('heating_daily_hours','H'),('electricity_annual_kwh','E')]:
    z=domains[response]
    print(f'Grouping {prefix}: {len(z)} households, {len(z[factors].drop_duplicates())} initial cells',flush=True)
    mapping,summary,history=iterative_tukey_groups(z,response,factors,.05,prefix)
    mapping['archetype']=pd.to_numeric(mapping['archetype']).astype(int)
    summary=summary.drop(columns=[c for c in summary if c.startswith('paper_') or 'delta_from_paper' in c])
    mapping=mapping.sort_values(['group',*factors]);maps[prefix]=mapping;stats[prefix]=summary
    mapping.to_csv(HERE/f'{prefix}_group_lookup_2021.csv',index=False)
    summary.to_csv(HERE/f'{prefix}_group_summary_2021.csv',index=False)
    history.to_csv(HERE/f'{prefix}_merge_history_2021.csv',index=False);histories.append(history)
    z=z.merge(mapping[factors+['group']],on=factors,validate='many_to_one');assert z.group.notna().all()
    z.to_csv(HERE/f'{prefix}_household_assignments_2021.csv',index=False)
    domains[response]=z
    d=d.merge(z[['id','group']].rename(columns={'group':prefix+'_group'}),on='id',how='left',validate='one_to_one')
    # Terminal pairwise table is an audit of the final partition, not a new search.
    from energy_analysis import _tukey_pairwise_pvalues
    labs,left,right,q,means,df=_tukey_pairwise_pvalues(z[response].to_numpy(),z.group.to_numpy())
    pp=studentized_range.sf(q,len(labs),df)
    pair=pd.DataFrame({'left':[labs[i] for i in left],'right':[labs[i] for i in right],'q':q,'p_adjusted':pp,'reject_at_0_05':pp<.05})
    pair.to_csv(HERE/f'{prefix}_final_pairwise_tests_2021.csv',index=False)
    weighted=[]
    for g,gg in z.groupby('group'):
        w=gg.weight.to_numpy();v=gg[response].to_numpy()
        weighted.append({'group':g,'n':len(gg),'weight_sum':float(w.sum()),'weighted_mean':float(np.average(v,weights=w)),'kish_effective_n':float(w.sum()**2/(w@w))})
    pd.DataFrame(weighted).to_csv(HERE/f'{prefix}_surveyweighted_summaries_2021.csv',index=False)
    results[prefix]={'response':response,'n':len(z),'initial_cells':len(mapping),'groups':len(summary),'merges':len(history),'mean':float(z[response].mean()),'weighted_mean':float(np.average(z[response],weights=z.weight)),'terminal_max_p':float(pp.max()) if len(pp) else None,'final_pairs_all_reject':bool((pp<.05).all())}
    print(json.dumps(results[prefix]),flush=True)
d.to_csv(HERE/'analysis_households_2021.csv',index=False)

# ANOVA recovers the estimable terms; 2021 has no subjective insulation rating.
terms=['C(archetype)','C(dwelling_type)','C(location)','building_age_years','dwelling_area_m2','C(cooling_system_use)','C(archetype):C(dwelling_type)','C(archetype):C(location)','C(dwelling_type):C(location)','building_age_years:dwelling_area_m2']
anova=[];modelinfo=[]
for response,prefix in [('heating_daily_hours','H'),('electricity_annual_kwh','E')]:
    z=domains[response];z=z[z.building_age_years.notna()&z.dwelling_area_m2.notna()&z.cooling_system_use.ne('unknown')].copy()
    formula=response+' ~ '+' + '.join(terms);model=ols(formula,z).fit()
    with warnings.catch_warnings(record=True) as ws:
        aa=anova_lm(model,typ=2).reset_index().rename(columns={'index':'term','sum_sq':'sum_squares','df':'df','F':'F','PR(>F)':'p'})
    aa.insert(0,'response',response);aa['sample_n']=len(z);aa['significant_0_05']=aa.p<.05;anova.append(aa)
    modelinfo.append({'response':response,'formula':formula,'n':len(z),'R2':float(model.rsquared),'rank':int(model.model.rank),'design_columns':int(model.model.exog.shape[1]),'warnings':[str(w.message) for w in ws]})
pd.concat(anova,ignore_index=True).to_csv(HERE/'anova_results_2021.csv',index=False)
(HERE/'anova_models_2021.json').write_text(json.dumps(modelinfo,indent=2),encoding='utf-8')

# Appendix A: the published table's location×dwelling×four-user layout.
locations=['north_east','north_west','central','south_and_islands']
dwellings=['apartment_ground_basement_raised','apartment_intermediate_floor','apartment_top_floor','single_family_house','multi_family_house']
location_names={'north_east':'North East','north_west':'North West','central':'Central','south_and_islands':'South and Islands'}
dwelling_names={'apartment_ground_basement_raised':'Apartment on ground, semi-basement or raised floor','apartment_intermediate_floor':'Apartment on intermediate floor','apartment_top_floor':'Apartment on top floor','single_family_house':'Single Family House','multi_family_house':'Multi Family House'}
look={p:df.set_index(factors).group.to_dict() for p,df in maps.items()}
cells=[]
for loc in locations:
    for dw in dwellings:
        for a in range(1,5):
            key=(loc,dw,a);cells.append({'location':loc,'location_label':location_names[loc],'dwelling_type':dw,'dwelling_label':dwelling_names[dw],'archetype':a,'heating_group':look['H'].get(key,''),'electricity_group':look['E'].get(key,'')})
appendix=pd.DataFrame(cells);assert not appendix[['heating_group','electricity_group']].eq('').any().any()
appendix.to_csv(HERE/'appendix_A_assignment_2021.csv',index=False)
wide=[]
for (loc,dw),z in appendix.groupby(['location','dwelling_type'],sort=False):
    rr={'Location':location_names[loc],'Dwelling Type':dwelling_names[dw]}
    for p,field in [('H','heating_group'),('E','electricity_group')]:
        for a in range(1,5):rr[f'{p} User Cluster {a}']=z.loc[z.archetype==a,field].iloc[0]
    wide.append(rr)
pd.DataFrame(wide).to_csv(HERE/'appendix_A_assignment_2021_wide.csv',index=False)

def save(fig,name):
    fig.savefig(F/(name+'.svg'),bbox_inches='tight');fig.savefig(F/(name+'.png'),dpi=240,bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def sortkey(v):return int(v[1:])

# Same outcomes and group axes as original Figure 8. H is pooled across insulation
# because the 2021 survey does not release that subjective rating.
fig,axes=plt.subplots(2,1,figsize=(8.2,8.7))
for ax,prefix,response in [(axes[0],'H','heating_daily_hours'),(axes[1],'E','electricity_annual_kwh')]:
    z=domains[response];groups=sorted(z.group.unique(),key=sortkey)
    if prefix=='H':
        sns.boxplot(data=z,x='group',y=response,order=groups,color='#E9AC72',showfliers=False,ax=ax,linewidth=.65,width=.65)
        ax.set_ylabel('Daily On-Time of Heating System (hours)')
    else:
        pz=z[z.cooling_system_use.ne('unknown')]
        sns.boxplot(data=pz,x='group',y=response,hue='cooling_system_use',order=groups,hue_order=['absent_or_unused','present_or_used'],palette=['#A3C1E3','#1B365D'],showfliers=False,ax=ax,linewidth=.65,width=.8)
        ax.set_ylabel('Annual Appliance Consumption (kWh)')
        handles,_=ax.get_legend_handles_labels();ax.legend(handles,['Cooling System Not Used','Cooling System Used'],title='Cooling System',loc='upper left',fontsize=9,title_fontsize=10)
    ax.set_xlabel('');ax.set_xticks(range(len(groups)),['Group\n'+g for g in groups]);ax.grid(axis='y',ls='--',alpha=.5);ax.set_axisbelow(True)
    ax.text(.5,-.19,'(a)' if prefix=='H' else '(b)',transform=ax.transAxes,ha='center',va='top',fontsize=12)
fig.subplots_adjust(hspace=.42,left=.12,right=.98,top=.98,bottom=.1);save(fig,'energy_group_boxplots_2021')
plotcounts=domains['electricity_annual_kwh'].groupby(['group','cooling_system_use']).size().rename('n').reset_index();plotcounts.to_csv(HERE/'electricity_plot_counts_2021.csv',index=False)

# Appendix visual with the same merged headers and orange/blue group shading.
heats=sorted(stats['H'].group,key=sortkey);elecs=sorted(stats['E'].group,key=sortkey)
hc={g:plt.cm.Oranges(v) for g,v in zip(heats,np.linspace(.18,.9,len(heats)))};ec={g:plt.cm.Blues(v) for g,v in zip(elecs,np.linspace(.18,.9,len(elecs)))}
widths=[.6,3.25]+[.72]*8;x=np.cumsum([0]+widths);fig,ax=plt.subplots(figsize=(11,9.8));ax.set_xlim(0,x[-1]);ax.set_ylim(0,22);ax.axis('off')
def cell(col,y,width,height,label,face='white',rotation=0,size=9):
    xx=x[col];ax.add_patch(Rectangle((xx,y),width,height,facecolor=face,edgecolor='#A8A8A8',lw=.5))
    rgb=matplotlib.colors.to_rgb(face);tc='white' if np.dot(rgb,[.2126,.7152,.0722])<.43 else 'black'
    ax.text(xx+width/2,y+height/2,label,ha='center',va='center',fontsize=size,rotation=rotation,color=tc)
cell(0,20,widths[0],2,'Location','#FAFAFA',90);cell(1,20,widths[1],2,'Dwelling Type','#FAFAFA')
cell(2,21,sum(widths[2:6]),1,'Heating Group','#FAFAFA');cell(6,21,sum(widths[6:]),1,'Electricity Group','#FAFAFA')
for j in range(8):cell(j+2,20,widths[j+2],1,f'User\nCluster\n{j%4+1}',size=8.5)
for i,rr in enumerate(wide):
    y=19-i
    if i%5==0:cell(0,y-4,widths[0],5,rr['Location'],'#FAFAFA',90)
    label=rr['Dwelling Type'].replace('Apartment on ground, semi-basement or raised floor','Apartment on ground,\nsemi-basement or raised floor').replace('Apartment on intermediate floor','Apartment on\nintermediate floor')
    cell(1,y,widths[1],1,label,'#FAFAFA',size=9)
    for j,p in enumerate(['H','E']):
        for a in range(1,5):
            g=rr[f'{p} User Cluster {a}'];face=(hc if p=='H' else ec)[g]
            cell(2+j*4+a-1,y,widths[2+j*4+a-1],1,g,face,size=10)
fig.tight_layout(pad=.12);save(fig,'appendix_A_group_assignment_matrix_2021')

metadata={'year':2021,'archetype_n':len(d),'groups':results,'group_factors':factors,'primary_grouping':'unweighted original-style iterative Tukey-Kramer reduction at alpha0.05, starting from location×dwelling×archetype cells; final groups ordered by mean; counts not forced','weights':'PESO used for separately reported group means, not as physical energy multipliers','insulation':'subjective2013rating unavailable; H2021boxplots pooled across insulation; no improvement variable substituted','electricity_model':'audited2021 ownership/use model using fixed2013reference appliance coefficients; measured spending not used as kWh','cooling_plot_rule':'used if2021frequency1–5; notused iffrequency6 or allcoolinginventoryitems explicitlyno; unknown excluded from colour split only','cooling_plot_counts':d.cooling_system_use.value_counts().to_dict(),'construction':'unknownconstruction retained for H/E factor grouping; excluded from age/area ANOVA and PFE','appendix_rows':len(appendix),'appendix_cells':len(appendix)*2,'code_lineage':{'group_algorithm':'energy_analysis.iterative_tukey_groups, recovered from07_ANOVA_analysis.py','factor_resolution':'Final new_elet.csv carries locationAbition with archetype, and published Appendix A varies E by user archetype; both H and E use the complete3factorcellstructure'},'sources':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'01_inputs/analysis_input_2021.csv',ROOT/'02_probabilities/household_archetypes_2021.csv',REPO/'energy_analysis.py',REPO/'07_ANOVA_analysis.py']},'versions':{'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__}}
(HERE/'results.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');print(json.dumps(metadata,indent=2),flush=True)
