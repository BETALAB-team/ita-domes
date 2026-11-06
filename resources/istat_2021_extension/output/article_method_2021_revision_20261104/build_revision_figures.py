from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'article_method_2021_20261104'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0,str(OLD/'.scientific'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
P=ROOT/'02_probability_candidates';E=ROOT/'03_energy_candidates';F=ROOT/'04_figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'axes.linewidth':.6,'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10,'svg.fonttype':'none','savefig.facecolor':'white'})
colours=['#4169E1','#50C878','#FFBF00','#D50032']
def save(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(F/f'{name}.{ext}',dpi=220,bbox_inches='tight')
 plt.close(fig)

grid=pd.read_csv(P/'candidate_grid_predictions.csv')
fitted=grid[(grid.candidate=='full_quadratic')&(grid.fit_domain=='primary_area_lt150')&grid.dwelling_area_m2.lt(150)].copy()
assert len(fitted)==125
observed=pd.read_csv(P/'observed_supported_windows.csv').query('dwelling_area_m2 < 150')
fitted=fitted.merge(observed,on=['building_age_years','dwelling_area_m2'],suffixes=('_fit','_obs'),validate='one_to_one')
fitted.to_csv(F/'figure_6_7_node_data.csv',index=False)
curves=pd.read_csv(P/'matched_vintage_area_curves.csv')
eras=['Before 1950','1950–1969','1970–1989','1990 onward']
print('ERAS',curves.vintage.unique().tolist(),flush=True)
# Keep the implementation's area axis and fitted-node crosses; empirical fit is Fig.7.
fig,axes=plt.subplots(2,2,figsize=(8,7.2),sharex=True,sharey=True)
styles=['--','-.',':',(0,(5,1,1,1))]
for a,ax in enumerate(axes.flat,1):
 ax.scatter(fitted.dwelling_area_m2,fitted[f'p{a}_fit'],c=colours[a-1],marker='x',s=14,alpha=.55,lw=.7)
 for era,style in zip(eras,styles):
  v=curves[curves.vintage.str.lower().str.replace(' ','')==era.lower().replace(' ','')]
  assert len(v),era
  ax.plot(v.area,v[f'p{a}'],color='#656565',linestyle=style,lw=1.25,label=era)
 ax.set_title(f'Cluster {a}',fontsize=12);ax.set_xlim(36,140);ax.set_ylim(.08,.49)
 ax.set_xticks([40,64,88,112,136]);ax.grid(color='#D2D2D2',ls='--',lw=.5,alpha=.65);ax.set_axisbelow(True)
fig.supxlabel('Dwelling Area (m²)',y=.08);fig.supylabel('Predicted Probability',x=.025)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,fontsize=9,bbox_to_anchor=(.51,.005))
fig.subplots_adjust(hspace=.22,wspace=.12,left=.1,right=.99,bottom=.18,top=.95)
save(fig,'dwelling_probability_profiles_2021')

fig,ax=plt.subplots(figsize=(6.5,5.6))
for a,c in enumerate(colours,1):ax.scatter(fitted[f'p{a}_obs'],fitted[f'p{a}_fit'],color=c,marker='x',s=26,alpha=.65,label=f'Cluster {a}',linewidth=.9)
ax.plot([0,.6],[0,.6],'--',color='gray',lw=1,label='y = x');ax.set(xlabel='Actual Proportion',ylabel='Predicted Probability',xlim=(0,.6),ylim=(0,.6));ax.legend(loc='upper left',frameon=False);fig.tight_layout();save(fig,'probability_model_validation_2021')

# Paired household validation, evaluating both geometries against identical targets.
fair=pd.read_csv(P/'fair_old21_vs_new125_same_holdout.csv');names=['old21_backward','new125_full_quadratic']
fig,ax=plt.subplots(figsize=(7,3.5))
for seed,z in fair[fair.candidate.isin(names)].groupby('seed'):
 v=z.set_index('candidate').loc[names,'RMSE'];ax.plot([0,1],v,color='#AAAAAA',marker='o',ms=3,lw=.7)
means=fair[fair.candidate.isin(names)].groupby('candidate').RMSE.mean().reindex(names)
ax.plot([0,1],means,c='#1B365D',lw=2,marker='o',ms=6,label='Mean of five household splits')
ax.set_xticks([0,1],['Previous reconstruction\n21 training windows','Revised reconstruction\n125 training windows']);ax.set_ylabel('Held-out probability RMSE');ax.grid(axis='y',ls='--',alpha=.4);ax.legend(frameon=False,fontsize=9);fig.tight_layout();save(fig,'probability_holdout_comparison_2021')

house=pd.read_csv(E/'household_compact_assignments.csv')
source=pd.read_csv(OLD/'03_energy/analysis_households_2021.csv',usecols=['id','heating_daily_hours','electricity_annual_kwh','cooling_system_use'])
house=house.merge(source,on='id',validate='one_to_one')
house.to_csv(E/'compact_assignments_with_outcomes.csv',index=False)
fig,axes=plt.subplots(2,1,figsize=(8,8.1))
for ax,p,response in [(axes[0],'H','heating_daily_hours'),(axes[1],'E','electricity_annual_kwh')]:
 field=f'{p}_compact_group';z=house[house[field].notna()];order=sorted(z[field].unique(),key=lambda v:int(v[1:]))
 if p=='H':
  sns.boxplot(data=z,x=field,y=response,order=order,color='#E9AC72',showfliers=False,ax=ax,linewidth=.65,width=.6)
  ax.set_ylabel('Daily On-Time of Heating System (hours)')
 else:
  sns.boxplot(data=z,x=field,y=response,order=order,hue='cooling_system_use',hue_order=['absent_or_unused','present_or_used'],palette=['#A3C1E3','#1B365D'],showfliers=False,ax=ax,linewidth=.65,width=.75)
  handles,_=ax.get_legend_handles_labels();ax.legend(handles,['Cooling System Not Used','Cooling System Used'],title='Cooling System',loc='upper left',fontsize=9,title_fontsize=10)
  ax.set_ylabel('Annual Appliance Consumption (kWh)')
 ax.set_xlabel('');ax.set_xticks(range(len(order)),['Group '+g for g in order]);ax.grid(axis='y',ls='--',alpha=.5);ax.set_axisbelow(True);ax.text(.5,-.15,'(a)' if p=='H' else '(b)',transform=ax.transAxes,ha='center',va='top',fontsize=12)
fig.subplots_adjust(hspace=.38,left=.13,right=.98,top=.98,bottom=.07);save(fig,'energy_group_boxplots_2021')

locs=['north_east','north_west','central','south_and_islands'];dws=['apartment_ground_basement_raised','apartment_intermediate_floor','apartment_top_floor','single_family_house','multi_family_house']
ln=dict(zip(locs,['North East','North West','Central','South and Islands']))
dn=dict(zip(dws,['Apartment on ground, semi-basement or raised floor','Apartment on intermediate floor','Apartment on top floor','Single Family House','Multi Family House']))
app=pd.read_csv(E/'appendix_A_compact_long.csv');lookup=app.set_index(['location','dwelling_type','archetype'])
wide=[]
for loc in locs:
 for dw in dws:
  rr={'Location':ln[loc],'Dwelling Type':dn[dw]}
  for p in ['H','E']:
   for a in range(1,5):rr[f'{p} User Cluster {a}']=lookup.loc[(loc,dw,a),f'{p}_group']
  wide.append(rr)
wide=pd.DataFrame(wide);wide.to_csv(E/'appendix_A_assignment_2021_wide_flat.csv',index=False)
heats=['H'+str(i) for i in range(1,5)];elecs=['E'+str(i) for i in range(1,6)]
hc={g:plt.cm.Oranges(v) for g,v in zip(heats,np.linspace(.18,.9,len(heats)))};ec={g:plt.cm.Blues(v) for g,v in zip(elecs,np.linspace(.18,.9,len(elecs)))}
widths=[.6,3.25]+[.72]*8;x=np.cumsum([0]+widths);fig,ax=plt.subplots(figsize=(11,9.8));ax.set_xlim(0,x[-1]);ax.set_ylim(0,22);ax.axis('off')
def cell(col,y,width,height,label,face='white',rotation=0,size=9):
 xx=x[col];ax.add_patch(Rectangle((xx,y),width,height,facecolor=face,edgecolor='#A8A8A8',lw=.5));rgb=matplotlib.colors.to_rgb(face);tc='white' if np.dot(rgb,[.2126,.7152,.0722])<.43 else 'black';ax.text(xx+width/2,y+height/2,label,ha='center',va='center',fontsize=size,rotation=rotation,color=tc)
cell(0,20,widths[0],2,'Location','#FAFAFA',90);cell(1,20,widths[1],2,'Dwelling Type','#FAFAFA');cell(2,21,sum(widths[2:6]),1,'Heating Group','#FAFAFA');cell(6,21,sum(widths[6:]),1,'Electricity Group','#FAFAFA')
for j in range(8):cell(j+2,20,widths[j+2],1,f'User\nCluster\n{j%4+1}',size=8.5)
for i,rr in wide.iterrows():
 y=19-i
 if i%5==0:cell(0,y-4,widths[0],5,rr.Location,'#FAFAFA',90)
 label=rr['Dwelling Type'].replace('Apartment on ground, semi-basement or raised floor','Apartment on ground,\nsemi-basement or raised floor').replace('Apartment on intermediate floor','Apartment on\nintermediate floor')
 cell(1,y,widths[1],1,label,'#FAFAFA',size=9)
 for j,p in enumerate(['H','E']):
  for a in range(1,5):
   g=rr[f'{p} User Cluster {a}'];cell(2+j*4+a-1,y,widths[2+j*4+a-1],1,g,(hc if p=='H' else ec)[g],size=10)
fig.tight_layout(pad=.12);save(fig,'appendix_A_group_assignment_matrix_2021')

# A matched pair of decision curves; retain the 95% design floor explicitly.
selection=pd.read_csv(E/'candidate_curves.csv');fig,axes=plt.subplots(1,2,figsize=(8,3.45))
for ax,p in zip(axes,['H','E']):
 z=selection[selection.response==p];chosen=z[z.selected].iloc[0]
 ax.plot(z.K,100*z.between_cell_variance_retained,color='#1B365D',marker='o',ms=3)
 ax.axhline(95,color='gray',ls='--',lw=.8);ax.axvline(chosen.K,color='#E9AC72',lw=1.2)
 ax.scatter([chosen.K],[100*chosen.between_cell_variance_retained],color='#D50032',s=28,zorder=3)
 ax.set(title='Heating' if p=='H' else 'Electricity',xlabel='Number of Groups',ylabel='Between-Cell Mean Variance Retained (%)',xticks=[1,3,5,7,9,11,13,15],ylim=(0,102));ax.grid(axis='y',ls='--',alpha=.35)
fig.tight_layout();save(fig,'compact_group_fidelity_2021')
print(json.dumps({'figure_sets':6,'appendix_assignments':len(app)*2,'new_households':len(house)}),flush=True)
