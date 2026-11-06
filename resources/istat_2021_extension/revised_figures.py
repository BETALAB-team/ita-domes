"""Vector scientific figures corresponding to Figures 1-10 and A.1 of the article."""
from __future__ import annotations
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
from matplotlib.colors import Normalize, ListedColormap
from scipy.special import softmax

from revised_analysis import ROOT, LOCATIONS, TYPES, AREA_CENTERS, probability_design, weighted_mean_sd

COLORS = ['#2964a8','#28966b','#df9638','#ab4c83']
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.titlesize':10,
                     'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                     'svg.fonttype':'none','svg.hashsalt':'istat-corrected-v1','pdf.fonttype':42,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16})


def save(fig, folder, name):
    folder.mkdir(parents=True,exist_ok=True)
    for extension in ['svg','pdf','png']:
        metadata = {'Date':None} if extension=='svg' else {'CreationDate':None,'ModDate':None} if extension=='pdf' else None
        fig.savefig(folder/f'{name}.{extension}',bbox_inches='tight',dpi=150,metadata=metadata)
    plt.close(fig)


def unavailable(ax, message):
    ax.set_facecolor('#f4f4f4'); ax.grid(False)
    ax.text(.5,.5,message,transform=ax.transAxes,ha='center',va='center',fontsize=10,
            linespacing=1.6,color='#555555')
    ax.set_xticks([]); ax.set_yticks([])


def weighted_box(g, outcome, label):
    x,w = g[outcome].to_numpy(float),g.weight.to_numpy(float)
    order = np.argsort(x); x,w = x[order],w[order]
    cdf = (np.cumsum(w)-.5*w)/w.sum()
    q1,med,q3 = np.interp([.25,.5,.75],cdf,x)
    inside = x[(x>=q1-1.5*(q3-q1))&(x<=q3+1.5*(q3-q1))]
    return {'label':label,'q1':q1,'med':med,'q3':q3,'whislo':inside.min(),
            'whishi':inside.max(),'fliers':[]}


def draw_flow(folder,year):
    fig,ax = plt.subplots(figsize=(9.5,4.6)); ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
    boxes = [(0,2.1,2.2,1.6,f'ISTAT {year}\nCodebook checks\nSurvey weights'),
             (3,3.9,2.5,1.6,'Demographic features\nFitted PCA\nFour fixed archetypes'),
             (3,.3,2.5,1.6,'Dwelling categories\nAge / area proxies\nMissingness audit'),
             (6.5,3.9,2.5,1.6,'Train / tune / test\nJoint softmax model\nCalibration + scores'),
             (6.5,.3,2.5,1.6,'Heating + modeled kWh\nTrain groups; tune size\nTest-only inference'),
             (9.7,2.1,2.3,1.6,'Simulation inputs\nProbabilities + groups\nLimits + sensitivity')]
    for x,y,w,h,label in boxes:
        ax.add_patch(Rectangle((x,y),w,h,facecolor='#f6f8fa',edgecolor='#34465b',lw=1))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8.5,linespacing=1.5)
    for start,end in [((2.2,3.2),(3,4.5)),((2.2,2.6),(3,1.1)),((5.5,4.7),(6.5,4.7)),
                      ((5.5,1.1),(6.5,1.1)),((4.3,3.9),(6.5,1.8)),((4.9,1.9),(6.5,4)),
                      ((9,4.5),(9.7,3.2)),((9,1.2),(9.7,2.7))]:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,color='#34465b',lw=.8))
    save(fig,folder,'fig01_process')


def read_region_polygons(path):
    """Read this supplied ESRI polygon shapefile and DBF, without external GIS services."""
    dbf = path.with_suffix('.dbf').read_bytes()
    n = struct.unpack_from('<I',dbf,4)[0]
    header, size = struct.unpack_from('<HH',dbf,8)
    fields=[];offset=32
    while dbf[offset]!=13:
        block=dbf[offset:offset+32]
        fields.append((block[:11].split(b'\0')[0].decode('ascii'),block[16]));offset+=32
    records=[]
    for i in range(n):
        block=dbf[header+i*size:header+(i+1)*size]; offset=1; record={}
        for name,length in fields:
            record[name]=block[offset:offset+length].decode('utf-8',errors='replace').strip();offset+=length
        records.append(record)
    data=path.read_bytes();offset=100; polygons=[]
    while offset<len(data):
        number,length=struct.unpack_from('>ii',data,offset);offset+=8
        block=data[offset:offset+2*length];offset+=2*length
        shape_type=struct.unpack_from('<i',block,0)[0]
        if shape_type not in [5,15,25]:raise ValueError(f'Expected polygon, got {shape_type}')
        nparts,npoints=struct.unpack_from('<ii',block,36)
        parts=list(struct.unpack_from('<'+'i'*nparts,block,44))+[npoints]
        points=np.frombuffer(block,dtype='<f8',offset=44+4*nparts,count=npoints*2).reshape(-1,2)
        record=records[number-1]; code=int(record['COD_REG'])
        polygons.append((code,[points[parts[j]:parts[j+1]] for j in range(nparts)]))
    return polygons


def make_all_figures(out, model, cfg, years=(2013,2021)):
    for year in years:
        source=out/str(year);folder=source/'figures';folder.mkdir(exist_ok=True)
        households=pd.read_csv(source/'households.csv'); summaries=pd.read_csv(source/'archetypes.csv')
        results=json.loads((source/'results.json').read_text())
        draw_flow(folder,year)
        if year == 2021:
            equipment = pd.read_csv(source/'equipment_summary.csv')
            fig, axes = plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
            for ax, measure, title, unit in zip(axes,
                    ['AC ownership','Insulation/window improvements','Cooling hours (frequency 1-3)'],
                    ['(a) AC ownership','(b) Improvements in last 5 years','(c) Cooling use, summer 2020'],
                    ['Weighted households (%)','Weighted responding households (%)','Hours/day, frequency codes 1-3']):
                g=equipment[equipment.measure.eq(measure)&equipment.archetype.ne('All')]
                ax.bar(['A'+str(a) for a in g.archetype],g.estimate,color=COLORS)
                ax.set(title=title,ylabel=unit)
            save(fig,folder,'fig11_equipment')
        fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
        for ax,pca,title in zip(axes,[model['all_variable_pca'],model['pca']],['(a) Four variables','(b) Demographics only']):
            cumulative=100*np.cumsum(pca['explained_ratio']);ax.plot(range(1,len(cumulative)+1),cumulative,'o-',color=COLORS[0])
            ax.set(xlabel='Number of principal components',ylabel='Cumulative explained variance (%)',ylim=(0,105),xticks=range(1,len(cumulative)+1),title=title)
        save(fig,folder,'fig02_pca')
        elbow=pd.read_csv(source/'elbow.csv')
        fig,ax=plt.subplots(figsize=(7.3,3.5),layout='constrained')
        ax.plot(elbow.k,elbow.weighted_mean_squared_distance,'o-',color=COLORS[0])
        ax.axvline(4,color=COLORS[3],ls='--',label='Four archetypes retained by design')
        ax.set(xlabel='Number of clusters',ylabel='Weighted mean squared distance',xticks=range(1,11));ax.legend(fontsize=8)
        save(fig,folder,'fig03_elbow')
        centroids=pd.read_csv(source/'regional_centroids.csv')
        fig,ax=plt.subplots(figsize=(7.8,4.9),layout='constrained')
        for region,g in centroids.groupby('region'):
            g=g.sort_values('archetype');xx=g[['coordinate_3','coordinate_1']].to_numpy()
            ax.plot(*xx.T,color='#b6b6b6',alpha=.6,lw=.7)
            ax.scatter(*xx.T,s=10+g.n.iloc[0]/250,color='#b6b6b6',alpha=.6)
        center=np.array(model['centers'])[:,[2,0]];ax.plot(*center.T,color='#b92f3d',lw=1.8,label='Frozen 2013 reference')
        ax.scatter(*center.T,c=COLORS,s=75,edgecolors='black',linewidths=.4)
        for a,point in enumerate(center,1):ax.text(*point,f' A{a}',fontsize=9)
        ax.set(xlabel='Normalized schooling proxy',ylabel='Normalized PC1')
        ax.plot([],[],color='#b6b6b6',marker='o',lw=.7,label=f'{year} regional refits');ax.legend(fontsize=8)
        save(fig,folder,'fig04_regional_centroids')
        fig,ax=plt.subplots(figsize=(6.8,4.3),layout='constrained')
        bottom=np.zeros(2)
        for i,row in enumerate(summaries.itertuples()):
            values=np.array([row.household_share,row.people_share])/100
            ax.bar([0,1],values,bottom=bottom,width=.5,color=COLORS[i],label=f'Archetype {i+1}')
            for x,value,start in zip([0,1],values,bottom):
                ax.text(x,start+value/2,f'{value*100:.1f}%',ha='center',va='center',color='white',fontsize=10)
            bottom+=values
        ax.set(xticks=[0,1],xticklabels=['Households','People'],ylabel='Proportion of total',ylim=(0,1),xlim=(-.6,1.9))
        handles,labels=ax.get_legend_handles_labels();ax.legend(handles[::-1],labels[::-1],loc='lower right',fontsize=8)
        save(fig,folder,'fig05_shares')
        cells=pd.read_csv(source/'probability_cells.csv'); cells=cells[cells.split.eq('test')]
        beta=np.array(results['probability']['models']['quadratic']['beta'])
        fig,axes=plt.subplots(2,2,figsize=(9,6.2),sharex=True,sharey=True,layout='constrained')
        for a,ax in enumerate(axes.flat,1):
            for linestyle,(age,g) in zip(['-', '--', ':', '-.'],sorted(cells.groupby('age_proxy'),key=lambda pair:pair[0])):
                area=np.linspace(20,200,120);fake=pd.DataFrame({'age_proxy':age,'area_proxy':area})
                design,_=probability_design(fake);pr=softmax(design@beta,axis=1)
                ax.plot(area,pr[:,a-1],color='#777777',linestyle=linestyle,label=f'{age:g} years')
                strong=g[g.supported];weak=g[~g.supported]
                ax.scatter(strong.area_proxy,strong[f'observed_{a}'],s=20,color=COLORS[a-1])
                ax.scatter(weak.area_proxy,weak[f'observed_{a}'],s=25,marker='x',color=COLORS[a-1])
            ax.set(title=f'Archetype {a}',ylim=(0,1),xlim=(10,210))
        for ax in axes[-1]:ax.set_xlabel('Dwelling floor-area proxy (m²)')
        for ax in axes[:,0]:ax.set_ylabel('Archetype probability')
        axes[0,0].legend(title='Building-age proxy',fontsize=7,title_fontsize=8,ncol=2)
        save(fig,folder,'fig06_probability_profiles')
        fig,ax=plt.subplots(figsize=(7.4,5.4),layout='constrained')
        ax.plot([0,1],[0,1],'--',color='#888888',lw=.8,label='Calibration reference')
        for a in range(1,5):
            for supported,marker in [(True,'o'),(False,'x')]:
                g=cells[cells.supported.eq(supported)]
                ax.scatter(g[f'observed_{a}'],g[f'predicted_{a}'],s=np.clip(g.effective_n/8,15,80),c=COLORS[a-1],marker=marker,alpha=.8,label=f'Archetype {a}' if supported else None)
        ax.set(xlabel='Held-out weighted cell proportion',ylabel='Predicted probability',xlim=(0,1),ylim=(0,1))
        ax.legend(fontsize=8)
        save(fig,folder,'fig07_calibration')
        fig,axes=plt.subplots(2,1,figsize=(9,7.0),layout='constrained')
        for ax,outcome in zip(axes,['heating','electricity']):
            if outcome=='electricity' and 'electricity' not in results['energy']:
                unavailable(ax,'(b) Appliance electricity - unavailable\nNo matched 2021 modeled or metered appliance-kWh outcome\nNo 2013 values transferred into this panel.');continue
            dat=pd.read_csv(source/f'{outcome}_test.csv');dat=dat[dat.group.ne('Unsupported')]
            groups=sorted(dat.group.unique(),key=lambda v:int(v[1:]))
            hue=('insulation' if year==2013 else 'insulation_improvement') if outcome=='heating' else 'cooling'
            levels=['High/sufficient','Low/draughty'] if outcome=='heating' and year==2013 else ['Present','Absent'] if outcome=='electricity' else ['Reported','Not reported']
            box_colors=['#efd8b6','#c98c51'] if outcome=='heating' else ['#263e5b','#acc9e3']
            for j,level in enumerate(levels):
                bxp=[];positions=[]
                for i,group in enumerate(groups):
                    g=dat[dat.group.eq(group)&dat[hue].eq(level)]
                    if len(g)>=5:
                        bxp.append(weighted_box(g,outcome,group));positions.append(i+(j-(len(levels)-1)/2)*.3)
                plotted=ax.bxp(bxp,positions=positions,widths=.25,showfliers=False,patch_artist=True,manage_ticks=False)
                for box in plotted['boxes']:box.set_facecolor(box_colors[j]);box.set_alpha(.85)
                legend_label = ('Insulation improvements: '+('yes' if level=='Reported' else 'no')) if year==2021 and outcome=='heating' else level
                ax.plot([],[],color=box_colors[j],lw=5,label=legend_label)
            if year==2021 and outcome=='heating':
                pairs=pd.read_csv(source/'heating_insulation_pairs.csv').set_index('group')
                for i, group in enumerate(groups):
                    p=pairs.loc[group,'p_holm']
                    ax.text(i,25.5,('p<0.001' if p<.001 else f'p={p:.3f}'),ha='center',fontsize=7)
                ax.set_ylim(0,29)
            ax.set(xticks=range(len(groups)),xticklabels=groups,ylabel='Daily heating hours' if outcome=='heating' else 'Modeled appliance kWh/year',title='(a) Heating' if outcome=='heating' else '(b) Electricity')
            if year==2021 and outcome=='heating':
                ax.legend(fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
                ax.set_title('(a) Heating',pad=30)
            else:
                ax.legend(fontsize=8)
            ax.set_xlabel('Training-defined group; held-out distributions')
        save(fig,folder,'fig08_energy_groups')
        fig,ax=plt.subplots(figsize=(8.6,4.2),layout='constrained')
        if 'electricity' in results['energy']:
            dat=pd.read_csv(source/'electricity_test.csv')
            groups=sorted(set(dat.group)-{'Unsupported'},key=lambda value:int(value[1:]))
            boxes=[];areas=[]
            for group in groups:
                g=dat[dat.group.eq(group)]
                boxes.append(weighted_box(g,'electricity',group))
                known=g.dropna(subset=['area_proxy']);areas.append(np.average(known.area_proxy,weights=known.weight))
            positions=np.arange(1,len(groups)+1)
            plotted=ax.bxp(boxes,positions=positions,widths=.48,vert=False,showfliers=False,manage_ticks=False,patch_artist=True)
            for i,box in enumerate(plotted['boxes']):box.set_facecolor(plt.get_cmap('Blues')(.3+.55*i/max(1,len(groups)-1)))
            ax.axvline(1717,color='#c33a48',ls='--',label='ISO example in source paper: 1717 kWh')
            ax.plot(1879/120*np.array(areas),positions,'o--',color='#279657',label='EN example scaled to each group mean area')
            ax.set(yticks=positions,yticklabels=groups,ylabel='Electricity group (ordered; not an area scale)',ylim=(.4,len(groups)+.6))
            secondary=ax.twinx();secondary.set(ylim=ax.get_ylim(),yticks=positions,yticklabels=[f'{area:.1f}' for area in areas],ylabel='Group mean area proxy (m²)');secondary.grid(False)
            ax.set(xlabel='Modeled appliance electricity (kWh/year)',xlim=(0,8000))
            ax.legend(fontsize=7.5,loc='lower right')
        else:
            unavailable(ax,'Horizontal group distributions and benchmark comparison unavailable\nNo matched 2021 appliance-kWh outcome\nNo electricity groups or area-based benchmarks imputed.')
        save(fig,folder,'fig09_benchmarks')
        fig,ax=plt.subplots(figsize=(7,5.7),layout='constrained');ax.axis('off');ax.set_aspect('equal')
        regions=pd.read_csv(source/'regional_heating.csv').set_index('region')
        cmap=plt.get_cmap('YlOrRd');normal=Normalize(4,12)
        all_points=[]
        for code,parts in read_region_polygons(ROOT/'Reg01012024_g/Reg01012024_g_WGS84.shp'):
            color=cmap(normal(regions.loc[code,'heating_hours'])) if code in regions.index else '#cccccc'
            for part in parts:
                ax.add_patch(Polygon(part,closed=True,facecolor=color,edgecolor='white',lw=.25));all_points.append(part)
        points=np.concatenate(all_points);ax.set_xlim(points[:,0].min(),points[:,0].max());ax.set_ylim(points[:,1].min(),points[:,1].max())
        fig.colorbar(plt.cm.ScalarMappable(norm=normal,cmap=cmap),ax=ax,shrink=.75,label='Mean daily heating hours')
        save(fig,folder,'fig10_regional_map')
        fig,axes=plt.subplots(1,2,figsize=(10.6,7),layout='constrained')
        for ax,outcome in zip(axes,['heating','electricity']):
            if outcome=='electricity' and 'electricity' not in results['energy']:
                unavailable(ax,'Electricity assignments unavailable\nNo 2021 appliance-kWh outcome');continue
            mapping=pd.read_csv(source/f'{outcome}_assignment.csv')
            lookup=dict(zip(mapping.cell,mapping.group))
            mat=np.full((12,4),np.nan);labels=[]
            for r,(location,typ) in enumerate([(l,t) for l in LOCATIONS for t in TYPES]):
                labels.append(location+' / '+typ)
                for a in range(1,5):
                    label=lookup.get(f'{location}|{typ}|{a}')
                    if label:mat[r,a-1]=int(label[1:])
            ax.imshow(mat,cmap='Oranges' if outcome=='heating' else 'Blues',vmin=-2,vmax=12,aspect='auto')
            for r in range(12):
                for c in range(4):
                    ax.text(c,r,('-' if np.isnan(mat[r,c]) else ('H' if outcome=='heating' else 'E')+str(int(mat[r,c]))),ha='center',va='center',fontsize=9)
            ax.set(xticks=range(4),xticklabels=['A1','A2','A3','A4'],yticks=range(12),yticklabels=labels,title=outcome.capitalize());ax.grid(False)
            ax.tick_params(axis='y',labelsize=7)
        save(fig,folder,'figA1_assignment')
        # Equations are also vector assets; their numeric loadings match the saved model.
        for number,loading in enumerate(model['pca']['loadings'][:2],1):
            text=rf'$PC_{{{number}}} = {loading[0]:+.6f}z_{{\min}} {loading[1]:+.6f}z_{{\max}} {loading[2]:+.6f}z_{{n}}$'
            fig,ax=plt.subplots(figsize=(8,.65));ax.axis('off');ax.text(.5,.5,text,ha='center',va='center',fontsize=15)
            save(fig,folder,f'eq{number:02d}_pca')
    # Optional sensitivity figure is included in each manuscript appendix.
    sensitivity=pd.read_csv(out/'sensitivity.csv')
    pivot=sensitivity.pivot(index=['scenario','archetype'],columns='year',values='share');pivot['change']=pivot[2021]-pivot[2013]
    for year in years:
        fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
        for a,color in enumerate(COLORS,1):
            g=sensitivity[(sensitivity.year==year)&(sensitivity.archetype==a)]
            v=g[g.scenario.eq('reference')].share.iloc[0]
            axes[0].plot([a,a],[g.share.min(),g.share.max()],color=color,lw=4);axes[0].scatter(a,v,color=color,s=50)
            changes=pivot.xs(a,level='archetype')['change'];v=changes.loc['reference']
            axes[1].plot([a,a],[changes.min(),changes.max()],color=color,lw=4);axes[1].scatter(a,v,color=color,s=50)
        axes[0].set(ylabel=f'{year} household share (%)',title='(a) Input-assumption range')
        axes[1].set(ylabel='2021 minus 2013 (percentage points)',title='(b) Cross-wave sensitivity');axes[1].axhline(0,color='#888888',ls='--')
        for ax in axes:ax.set(xticks=range(1,5),xticklabels=['A1','A2','A3','A4'],xlim=(.5,4.5))
        save(fig,out/str(year)/'figures','figB1_sensitivity')
