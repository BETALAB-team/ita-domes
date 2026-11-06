from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path('output/article_method_2021_revision_20261104/02_probability_candidates');c=pd.read_csv(O/'matched_vintage_area_curves.csv');p=pd.read_csv(O/'matched_vintage_observed_fitted_nodes.csv');plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','font.size':10});fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True);order=['Before1950','1950–1969','1970–1989','1990onward'];colors=['#225ea8','#41b6c4','#fd8d3c','#bd0026']
for a,ax in enumerate(axs.ravel(),1):
 for vintage,col in zip(order,colors):
  z=c[c.vintage==vintage];q=p[p.vintage==vintage];ax.plot(z.area,z[f'p{a}'],color=col,label={'Before1950':'Before 1950','1950–1969':'1950–1969','1970–1989':'1970–1989','1990onward':'1990 onward'}[vintage]);ax.scatter(q.area,q[f'observed_p{a}'],color=col,marker='x',s=26)
 ax.set(title=f'Archetype{a}',ylabel='Mean window probability',xlabel='Dwelling area (m2)',xlim=(40,136));ax.grid(alpha=.2)
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,title='Vintage of age-window centers',loc='lower center',ncol=4,fontsize=9,title_fontsize=9);fig.suptitle('2021 probability model using recovered 125-node geometry\nLines: full quadratic; points: observed windows with identical age-node weights',fontsize=12);fig.tight_layout(rect=(0,.10,1,.95));fig.savefig(O/'recommended_vintage_area_profiles.png',dpi=180);fig.savefig(O/'recommended_vintage_area_profiles.svg');plt.close(fig)
m=pd.read_csv(O/'fair_old21_vs_new125_same_holdout.csv');fig,ax=plt.subplots(figsize=(7,4));names=['old21_backward','new125_backward','new125_full_quadratic'];labels=['Old21 backward','New125 backward','New125 full quadratic']
for seed,z in m.groupby('seed'):
 vals=[float(z.loc[z.candidate==name,'RMSE'].iloc[0]) for name in names];ax.plot(range(3),vals,color='gray',alpha=.45,marker='o')
ax.plot(range(3),[m.loc[m.candidate.eq(n),'RMSE'].mean() for n in names],color='#225ea8',linewidth=3,marker='o',label='Mean across5 splits');ax.set_xticks(range(3),labels);ax.set_ylabel('RMSE on identical heldout125-node proportions');ax.set_title('Fair regression comparison: refit each model on training households');ax.legend();fig.tight_layout();fig.savefig(O/'fair_holdout_rmse_comparison.png',dpi=180);fig.savefig(O/'fair_holdout_rmse_comparison.svg');print('Two diagnostic SVG/PNG pairs rendered')

