from pathlib import Path
import json,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('output/article_method_2021_20261104/02_probabilities');plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
def save(fig,name):
 fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(fig)
s=pd.read_csv(OUT/'archetype_summary_2021.csv');fig,ax=plt.subplots(figsize=(8,4));x=np.arange(4);ax.bar(x-.18,s.household_share,.36,label='Unweighted (fit estimand)');ax.bar(x+.18,s.weighted_household_share,.36,label='Official survey weights');ax.set_xticks(x,[f'A{i}' for i in range(1,5)]);ax.set_ylabel('Households (%)');ax.set_title('New 2021 archetypes, aligned to historical interpretations');ax.legend();save(fig,'archetype_shares_2021')
d=pd.read_csv(OUT/'eligible_household_features_2021.csv');fig,axs=plt.subplots(1,3,figsize=(13,4));cols=['min_age','max_age','studio_ind'];labels=['Minimum age (imputed years)','Maximum age (imputed years)','Education proxy'];
for ax,col,label in zip(axs,cols,labels):
 for a in range(1,5):
  z=d[d.archetype==a];ax.hist(z[col],bins=25,density=True,histtype='step',label=f'A{a}')
 ax.set_xlabel(label);ax.set_ylabel('Density')
axs[0].legend();fig.suptitle('New 2021 fitted archetypes: within-group feature distributions');fig.tight_layout();save(fig,'archetype_feature_distributions_2021')
p=pd.read_csv(OUT/'probability_grid_empirical.csv');f=pd.read_csv(OUT/'polynomial_full_grid_predictions_flags.csv');fig,axs=plt.subplots(2,4,figsize=(14,7),sharex=True,sharey=True)
for j in range(4):
 for i,df in enumerate([p,f]):
  ax=axs[i,j];g=df.pivot(index='building_age_years',columns='dwelling_area_m2',values=f'prob_archetype_{j+1}');im=ax.imshow(g.to_numpy(),origin='lower',aspect='auto',extent=(20,380,10,150),vmin=0,vmax=1,cmap='viridis');ax.set_title(f'A{j+1}: '+('empirical windows' if i==0 else 'new quadratic fit'));ax.set_xlabel('Area (m2)');ax.set_ylabel('Building age (years)');ax.axvline(150,color='red',linestyle='--',linewidth=1);bad=(g.to_numpy()<0)|(g.to_numpy()>1);yy,xx=np.where(bad);ax.scatter(40+40*xx,20+20*yy,marker='x',color='red',s=24,linewidths=1)
fig.colorbar(im,ax=axs.ravel().tolist(),label='Probability (colors capped0–1; raw flags in CSV)',shrink=.8,extend='both');fig.suptitle('2021 dwelling probabilities: red line = fit boundary; crosses = invalid raw probabilities');save(fig,'probability_empirical_fitted_heatmaps_2021')
e=pd.read_csv(OUT/'empirical_vs_fitted_training.csv');fig,axs=plt.subplots(1,4,figsize=(13,3.6))
for a,ax in enumerate(axs,1):
 ax.scatter(e[f'prob_archetype_{a}'],e[f'fitted_archetype_{a}'],s=np.sqrt(e.sample_count)*3,alpha=.7);ax.plot([0,1],[0,1],color='gray',linestyle='--');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_title(f'A{a}');ax.set_xlabel('Empirical window probability');ax.set_ylabel('Fitted probability')
fig.suptitle('2021 apparent polynomial fit (point area reflects support)');fig.tight_layout();save(fig,'probability_empirical_vs_fitted_2021')
params=json.loads((OUT/'archetype_fit_parameters.json').read_text());fig,ax=plt.subplots(figsize=(6,3.5));ratio=np.array(params['pca_explained_ratio']);ax.bar([1,2,3],ratio*100);ax.plot([1,2,3],np.cumsum(ratio)*100,marker='o',color='black',label='Cumulative');ax.set_xticks([1,2,3]);ax.set_xlabel('Full PCA component');ax.set_ylabel('Variance explained (%)');ax.set_title('2021 normalized demographic features');ax.legend();save(fig,'pca_explained_variance_2021')
print('Saved five scientific figures as SVG and PNG')

