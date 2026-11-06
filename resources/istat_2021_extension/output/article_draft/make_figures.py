from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'revised_paper_outputs'/'2021'
OUT=Path(__file__).parent/'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
BLUE='#285E8E'; ORANGE='#D07C2C'; TEAL='#187F78'; PURPLE='#8662A6'
def save(fig,name):
    fig.tight_layout(pad=1.7)
    for ext in ['png','svg']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
a=pd.read_csv(SRC/'archetypes.csv'); x=np.arange(4)
fig,ax=plt.subplots(figsize=(8,3.7))
ax.bar(x-.18,a.household_share,.34,label='Households',color=BLUE)
ax.bar(x+.18,a.people_share,.34,label='People',color=ORANGE)
ax.set(xticks=x,xticklabels=['A1','A2','A3','A4'],ylabel='Weighted share (%)',ylim=(0,62))
ax.legend(frameon=False,ncol=2,loc='upper left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
save(fig,'fig01_archetype_shares')
e=pd.read_csv(SRC/'equipment_summary.csv');fig,axs=plt.subplots(1,3,figsize=(9,3.5))
for ax,measure,title,unit,col in zip(axs,['AC ownership','Insulation/window improvements','Cooling hours (frequency 1-3)'],['Air conditioning','Reported improvement','Cooling duration'],['Ownership (%)','Known responses (%)','Hours per day'],[BLUE,TEAL,ORANGE]):
    sub=e[(e.measure==measure)&(e.archetype.astype(str)!='All')]
    ax.plot(sub.estimate,x,'o',ms=7,color=col);ax.set(yticks=x,yticklabels=['A1','A2','A3','A4'],xlabel=unit,title=title);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
save(fig,'fig02_equipment')
h=pd.read_csv(SRC/'heating_test.csv');p=pd.read_csv(SRC/'heating_insulation_pairs.csv')
def stats(g):
    v=g.heating.to_numpy();w=g.weight.to_numpy();o=np.argsort(v);v=v[o];w=w[o]
    q=np.interp([.25,.5,.75],(np.cumsum(w)-.5*w)/w.sum(),v)
    low=q[0]-1.5*(q[2]-q[0]);high=q[2]+1.5*(q[2]-q[0]);inside=v[(v>=low)&(v<=high)]
    return dict(q1=q[0],med=q[1],q3=q[2],whislo=inside.min(),whishi=inside.max(),fliers=[])
fig,ax=plt.subplots(figsize=(9,4.5))
for j,(val,color,label) in enumerate([('Reported',TEAL,'Improvement yes'),('Not reported',PURPLE,'Improvement no')]):
    boxes=[]
    for group in p.group: boxes.append(stats(h[(h.group==group)&(h.insulation_improvement==val)]))
    artists=ax.bxp(boxes,positions=np.arange(8)+(j-.5)*.32,widths=.28,patch_artist=True,showfliers=False,manage_ticks=False)
    for box in artists['boxes']:box.set(facecolor=color,alpha=.65)
    for med in artists['medians']:med.set(color='#202020')
    ax.plot([],[],color=color,lw=8,label=label)
for i,row in p.iterrows():ax.text(i,26,f'p={row.p_holm:.3f}',ha='center',fontsize=8)
ax.set(xticks=np.arange(8),xticklabels=p.group,ylabel='Reported heating hours per day',ylim=(0,28));ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.2));ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
save(fig,'fig03_heating_improvements')
g=pd.read_csv(SRC/'electricity_groups.csv');fig,axs=plt.subplots(1,2,figsize=(9,3.8))
axs[0].bar(['Lower','Reference','Upper'],[1527.1184153223548,1949.1860095094185,2705.67106271211],color=['#9BACBA',BLUE,'#E7AE75'])
axs[0].set(title='Assumption scenarios',ylabel='kWh per household per year',ylim=(0,3200))
axs[1].plot(g['mean'],np.arange(len(g)),'o',color=BLUE);axs[1].set(yticks=np.arange(8),yticklabels=g.group,xlabel='kWh per household per year',title='Eligible domain energy groups');axs[1].invert_yaxis()
for ax in axs:ax.grid(alpha=.15);ax.set_axisbelow(True)
save(fig,'fig04_appliance_sensitivity')
s=pd.read_csv(SRC/'probability_scores.csv');s=s.set_index('model').loc[['training_marginal','quadratic','quadratic_plus_location_type']]
fig,axs=plt.subplots(1,2,figsize=(8.5,3.3))
for ax,key,title in zip(axs,['log_loss','brier'],['Weighted log loss','Multiclass Brier score']):
    ax.plot(s[key],np.arange(3),'o',color=BLUE,ms=7);ax.set(yticks=np.arange(3),yticklabels=['Marginal','Quadratic','Extended'],title=title);ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.margins(y=.3)
save(fig,'fig05_probability_scores')
print('Created five figures as PNG and SVG')
