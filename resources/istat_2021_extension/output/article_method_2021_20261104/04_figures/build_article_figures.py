from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'.scientific'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'axes.linewidth':.6,'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10,'svg.fonttype':'none','savefig.facecolor':'white'})
P=ROOT/'02_probabilities';COL=['#4169E1','#50C878','#FFBF00','#D50032']
def save(fig,name):
    for suffix in ['svg','pdf']:fig.savefig(HERE/f'{name}.{suffix}',bbox_inches='tight')
    fig.savefig(HERE/f'{name}.png',dpi=240,bbox_inches='tight');plt.close(fig)
def grid(ax):ax.grid(color='#CCCCCC',linewidth=.5);ax.set_axisbelow(True)

# Figure 5, same stacked household/person proportions and cluster colours.
s=pd.read_csv(P/'archetype_summary_2021.csv').sort_values('archetype')
fig,axes=plt.subplots(1,2,figsize=(9,4.8),sharey=True)
for ax,year,values in [(axes[0],2013,[[40.1,29.9],[26.9,40.2],[22.9,21.3],[10.1,8.6]]),(axes[1],2021,s[['household_share','person_share']].to_numpy())]:
    bottom=np.zeros(2)
    for i,v in enumerate(values):
        v=np.array(v);ax.bar([0,1],v/100,bottom=bottom,color=COL[i],width=.55,label=f'Cluster {i+1}')
        for x,(bb,vv) in enumerate(zip(bottom,v)):ax.text(x,bb+vv/200,f'{vv:.1f}%',ha='center',va='center',color='white',fontsize=10)
        bottom+=v/100
    ax.set_ylim(0,1);ax.set_xticks([0,1],['households','people']);ax.set_title(str(year),fontsize=13);ax.grid(False)
axes[0].set_ylabel('Proportion of Total')
handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles[::-1],labels[::-1],loc='upper center',ncol=4,bbox_to_anchor=(.5,1.03),frameon=False)
fig.tight_layout(rect=[0,0,1,.94]);save(fig,'archetype_population_shares_2021')

# Article Figure 3, freshly fitted 2021 WCSS rather than an old curve.
d=pd.read_csv(P/'eligible_household_features_2021.csv');X=d[['normal_l','normal_g','normal_s']].to_numpy();krows=[]
with threadpool_limits(limits=1):
    for k in range(1,10):
        m=KMeans(n_clusters=k,n_init=20,random_state=42,max_iter=500,tol=1e-8).fit(X);krows.append({'k':k,'within_cluster_sum_squares':m.inertia_})
kdf=pd.DataFrame(krows);kdf.to_csv(HERE/'within_cluster_sum_squares_2021.csv',index=False)
fig,ax=plt.subplots(figsize=(6.6,4.6));ax.plot(kdf.k,kdf.within_cluster_sum_squares,color='black',marker='o',markersize=4,linewidth=1);ax.set_xticks(range(1,10));ax.set_xlabel('Number of Clusters');ax.set_ylabel('Within Cluster Sum of Squares');grid(ax);fig.tight_layout();save(fig,'clustering_elbow_2021')

# Article Figure 6: four cluster panels, area x, empirical crosses and fitted age lines.
B=pd.read_csv(P/'polynomial_coefficient_matrix.csv',index_col=0).to_numpy();emp=pd.read_csv(P/'probability_grid_empirical.csv');training=emp[emp.meets_minimum_support&emp.dwelling_area_m2.lt(150)]
def design(age,area):
    aa=np.asarray(age)/100;ss=np.asarray(area)/100;return np.column_stack([np.ones(len(aa)),aa,ss,aa*ss,aa*aa,ss*ss])
fig,axes=plt.subplots(2,2,figsize=(8.2,7.2),sharex=True)
ages=[20,60,100,140];styles=['-', '--', ':', '-.']
for i,ax in enumerate(axes.flat):
    ax.scatter(training.dwelling_area_m2,training[f'prob_archetype_{i+1}'],marker='x',s=22,linewidths=.7,color=COL[i],alpha=.8)
    area=np.linspace(40,140,150)
    for age,style in zip(ages,styles):
        ax.plot(area,design(np.full(len(area),age),area)@B[i],color='#888888',linestyle=style,linewidth=.8,label=f'Age {age} years')
    ax.set_title(f'Cluster {i+1}',fontsize=12);ax.set_xlim(35,145);ax.set_xticks([40,60,80,100,120,140]);ax.set_ylabel('Predicted Probability');grid(ax)
axes[1,0].set_xlabel('Dwelling Area (m²)');axes[1,1].set_xlabel('Dwelling Area (m²)')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.02),fontsize=10)
fig.tight_layout(rect=[0,0,1,.96]);save(fig,'dwelling_probability_profiles_2021')

# Article Figure 7, one coloured actual/predicted plot with the y=x guide.
t=pd.read_csv(P/'empirical_vs_fitted_training.csv');fig,ax=plt.subplots(figsize=(6,6))
for a in range(1,5):ax.scatter(t[f'prob_archetype_{a}'],t[f'fitted_archetype_{a}'],marker='x',s=24,linewidths=.8,color=COL[a-1],label=f'Cluster {a}')
ax.plot([0,1],[0,1],color='#999999',linestyle='--',linewidth=.8,label='y = x');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('Actual Proportion');ax.set_ylabel('Predicted Probability');ax.legend(loc='upper left',fontsize=10);grid(ax);fig.tight_layout();save(fig,'probability_model_validation_2021')

# A matrix image is provided as a standalone vector/raster asset; Word also gets native OMML.
fig,ax=plt.subplots(figsize=(10,2.65));ax.axis('off')
left=r'$\left[p_1, p_2, p_3, p_4\right]^T =$'
ax.text(.01,.55,left,fontsize=13,ha='left',va='center',transform=ax.transAxes)
table=ax.table(cellText=[[f'{v:.5f}' for v in row] for row in B],cellLoc='center',bbox=[.27,.18,.58,.7])
for (i,j),c in table.get_celld().items():c.set_edgecolor('white');c.set_fontsize(11)
for x in [.265,.855]:ax.plot([x,x],[.16,.9],color='black',linewidth=.8,transform=ax.transAxes)
ax.text(.88,.53,'×',ha='center',va='center',fontsize=15,transform=ax.transAxes)
ax.text(.92,.53,'1\na / 100\ns / 100\na s / 10000\na² / 10000\ns² / 10000',ha='left',va='center',fontsize=10,transform=ax.transAxes)
fig.tight_layout();save(fig,'matrix_equation_2021')
(HERE/'article_plot_specification.json').write_text(json.dumps({'source':'Original paper Figures3,5,6,7,8 and AppendixA','new_plots':'All2021statisticalpanelsderivefromnew2021fit;2013sharepanelusesunchangedprintedpercentages','font':'Times New Roman','cluster_colours':COL,'probability_basis':['1','age/100','area/100','age*area/10000','age^2/10000','area^2/10000'],'primary_probability_fit_centres':'age20–140 years;area40,80,120 m²','figure6_lines':'Four illustrative building ages20,60,100,140years;crossesare21empiricalfittingwindows;graylinesarethefittedmatrix','equation_coefficients_rounded_digits':5},indent=2),encoding='utf-8')
print('Generated six new article-style figure assets',flush=True)
