"""Training-only feature fitting and exact weighted 1-D energy clustering."""
import numpy as np
import pandas as pd


def weighted_mean(data, col):
    v=data.dropna(subset=[col,'weight'])
    return float(np.average(v[col],weights=v.weight)) if len(v) else np.nan


def grouping_factors(cfg,outcome):
    factors=list(cfg['analysis']['group_factors'])
    if outcome=='electricity' and cfg['analysis'].get('electricity_include_cooling',True) and 'cooling' not in factors:
        factors.append('cooling')
    return factors


def exact_partitions(x,w,kmax):
    x,w=np.asarray(x,float),np.asarray(w,float)
    if len(x)<kmax or kmax<1 or np.any(np.diff(x)<0) or np.any(w<=0):
        raise ValueError('Invalid sorted inputs / group count.')
    n=len(x);a,b,c=[np.r_[0.,np.cumsum(v)] for v in [w,w*x,w*x*x]]
    costs=np.full((n,n+1),np.inf)
    for i in range(n):
        for j in range(i+1,n+1): costs[i,j]=max(0.,c[j]-c[i]-(b[j]-b[i])**2/(a[j]-a[i]))
    dp=np.full((kmax+1,n+1),np.inf);prev=np.zeros_like(dp,dtype=int);dp[0,0]=0;parts={}
    for k in range(1,kmax+1):
        for j in range(k,n+1):
            candidates=dp[k-1,k-1:j]+costs[k-1:j,j];i=k-1+int(np.argmin(candidates))
            dp[k,j]=candidates.min();prev[k,j]=i
        groups=[];j=n
        for m in range(k,0,-1):i=prev[m,j];groups.append((i,j));j=i
        parts[k]=groups[::-1]
    return parts,dp[1:,n]


def kmeans(x,w,k,seed,starts):
    best=None
    for start in range(starts):
        rng=np.random.default_rng(seed+start);centers=[x[rng.choice(len(x),p=w/w.sum())]]
        while len(centers)<k:
            dist=((x[:,None]-np.array(centers)[None])**2).sum(2).min(1);prob=w*dist
            if prob.sum()<=0: raise ValueError('Not enough distinct feature combinations.')
            centers.append(x[rng.choice(len(x),p=prob/prob.sum())])
        centers=np.array(centers)
        for iteration in range(300):
            dist=((x[:,None]-centers[None])**2).sum(2);labels=dist.argmin(1)
            new=np.array([np.average(x[labels==i],weights=w[labels==i],axis=0) if np.any(labels==i) else x[np.argmax(w*dist.min(1))] for i in range(k)])
            if np.max(abs(new-centers))<1e-7: centers=new;break
            centers=new
        loss=float(np.average(((x[:,None]-centers[None])**2).sum(2).min(1),weights=w))
        if best is None or loss<best[0]:best=(loss,centers)
    return best


def prepare(cfg,out,log):
    a=cfg['analysis'];frame=pd.read_csv(cfg['paths']['households'])
    needed=['id','weight','split','archetype','heating','electricity','nmembers','location','dwelling_type','cooling_hours',*a['features'],*grouping_factors(cfg,'electricity')]
    missing=sorted(set(needed)-set(frame))
    if missing:raise ValueError(f'Missing CSV columns: {missing}')
    if not set(frame.split.unique())<={'train','tune','test'}:raise ValueError('CSV must contain train/tune/test splits.')
    if not frame.id.is_unique or not np.isfinite(frame.weight).all() or (frame.weight<=0).any():raise ValueError('IDs must be unique and weights positive and finite.')
    features=a['features']; model={'mode':a['archetype_mode'],'features':features}
    if a['archetype_mode']=='refit':
        n0=len(frame);frame=frame.dropna(subset=features).copy();log(f'Refit: excluded {n0-len(frame)} households with missing selected features.')
        train=frame[frame.split.eq('train')];x=train[features].to_numpy(float);w=train.weight.to_numpy()
        mean=np.average(x,weights=w,axis=0);scale=np.sqrt(np.average((x-mean)**2,weights=w,axis=0))
        if np.any(scale<=0):raise ValueError('Remove constant clustering features.')
        x=(x-mean)/scale;allx=(frame[features].to_numpy(float)-mean)/scale
        components=a['pca_components'];rotation=np.eye(len(features))
        if components:
            if not 1<=components<=len(features):raise ValueError('PCA components exceed selected features.')
            val,vec=np.linalg.eigh((x*w[:,None]).T@x/w.sum());rotation=vec[:,np.argsort(val)[::-1]][:,:components]
        x=x@rotation;allx=allx@rotation
        loss,centers=kmeans(x,w,a['n_archetypes'],a['seed'],a['kmeans_starts'])
        # Stable within this fit, not a claim of equivalence to published labels.
        centers=centers[np.lexsort(centers.T[::-1])]
        frame['archetype']=((allx[:,None]-centers[None])**2).sum(2).argmin(1)+1
        model.update(mean=mean.tolist(),scale=scale.tolist(),rotation=rotation.tolist(),centers=centers.tolist(),inertia=loss)
    elif frame.archetype.nunique()!=a['n_archetypes']:
        raise ValueError('Existing labels have a fixed count; choose Refit to change archetype count.')
    unknown={'Unknown','Unavailable','Not reported','nan'}
    before=len(frame)
    for f in a['group_factors']:
        frame=frame[frame[f].notna() & ~frame[f].astype(str).isin(unknown)]
    log(f'Grouping: excluded {before-len(frame)} households missing selected factors. Insulation only matters if selected.')
    mappings={};curves={}
    for outcome in ['heating','electricity']:
        factors=grouping_factors(cfg,outcome)
        data=frame.dropna(subset=[outcome,*factors]).copy()
        for f in factors:data=data[~data[f].astype(str).isin(unknown)]
        data['cell']=data[factors].astype(str).agg('|'.join,axis=1)
        train=data[data.split.eq('train')]
        cells=pd.DataFrame([{'cell':key,'n':len(g),'weight':g.weight.sum(),'mean':weighted_mean(g,outcome)} for key,g in train.groupby('cell') if len(g)>=a['minimum_train_cell']])
        if cells.empty:raise ValueError('No supported training cells. Reduce the minimum cell count or remove factors.')
        cells=cells.sort_values(['mean','cell']).reset_index(drop=True)
        requested=a[f'{outcome}_groups'];maxk=min(max(a['elbow_max_k'],requested),len(cells),len(cells['mean'].unique()))
        if requested>maxk and a['group_selection']=='fixed':raise ValueError(f'{outcome}: {requested} groups exceed {maxk} distinct supported cells.')
        parts,losses=exact_partitions(cells['mean'],cells.weight,maxk)
        curve=pd.DataFrame({'k':range(1,maxk+1),'wcss':losses})
        curve['between_cell_explained_pct']=100*(1-curve.wcss/curve.wcss.iloc[0]) if losses[0]>0 else 0.
        elbow=min(a['elbow_max_k'],maxk)
        if elbow==1 or losses[0]==losses[elbow-1]:chosen=1
        else:
            ys=(losses[:elbow]-losses[elbow-1])/(losses[0]-losses[elbow-1]);xs=np.linspace(0,1,elbow)
            chosen=int(np.argmax(1-xs-ys)+1)
        if a['group_selection']=='fixed':chosen=requested
        prefix='H' if outcome=='heating' else 'E'
        for i,(l,r) in enumerate(parts[chosen],1):
            g=cells.iloc[l:r];cells.loc[l:r-1,'group']=f'{prefix}{i}'
            cells.loc[l:r-1,'prediction']=np.average(g['mean'],weights=g.weight)
        cells.to_csv(out/f'{outcome}_assignment.csv',index=False);curve.to_csv(out/f'{outcome}_elbow.csv',index=False)
        mappings[outcome]=cells;curves[outcome]=curve
        data['group']=data.cell.map(cells.set_index('cell').group).fillna('Unsupported')
        data[data.split.eq('test')].to_csv(out/f'{outcome}_test.csv',index=False)
        log(f'{outcome}: {chosen} groups from {len(cells)} supported cells.')
    frame.to_csv(out/'households_assigned.csv',index=False)
    import json
    (out/'archetype_model.json').write_text(json.dumps(model,indent=2))
    return frame,mappings,curves


def scenario_profile(frame,mappings,cfg,location,dwelling,archetype):
    train=frame[frame.split.eq('train')]
    cohort=train[(train.location==location)&(train.dwelling_type==dwelling)&(train.archetype==archetype)]
    # Transfer indices retain the exact joint cohort and never use schedule defaults.
    indices={}
    members_valid=cohort.nmembers.notna() & np.isfinite(cohort.nmembers)
    indices['members_index_n']=int(members_valid.sum())
    indices['members_index']=weighted_mean(cohort.loc[members_valid],'nmembers') if members_valid.sum()>=cfg['analysis']['minimum_train_cell'] else np.nan
    for outcome,index in [('heating','heating'),('electricity','appliance'),('cooling_hours','cooling')]:
        mode=cfg['hybrid']['index_source'] if index!='cooling' else 'reported_cohort_mean'
        if mode=='group_means':
            keys=cohort[grouping_factors(cfg,outcome)].astype(str).agg('|'.join,axis=1) if len(cohort) else pd.Series(dtype=str)
            values=keys.map(mappings[outcome].set_index('cell').prediction)
            # Exclude missing outcome records when pooling supported predictions.
            valid=values.notna() & cohort[outcome].notna()
        else:
            values=cohort[outcome];valid=values.notna() & np.isfinite(values) & values.ge(0)
        count=int(valid.sum())
        coverage=float(cohort.loc[valid,'weight'].sum()/cohort.weight.sum()) if len(cohort) else 0.
        value=float(np.average(values[valid],weights=cohort.loc[valid,'weight'])) if count>=cfg['analysis']['minimum_train_cell'] else np.nan
        indices.update({index+'_index':value,index+'_index_n':count,index+'_index_coverage':coverage,index+'_index_source':mode})
    source='location × dwelling × archetype'
    if len(cohort)<cfg['analysis']['minimum_train_cell']:
        cohort=train[train.archetype.eq(archetype)];source='archetype fallback'
    profile={'members':weighted_mean(cohort,'nmembers'),'heating_hours':weighted_mean(cohort,'heating'),
             'appliance_kwh':weighted_mean(cohort,'electricity'),
             'cooling_hours':weighted_mean(cohort,'cooling_hours'),
             'support_n':len(cohort),'profile_source':source}
    # Pool fitted cell predictions over the scenario cohort for any additional factors.
    for outcome,target in [('heating','heating_hours'),('electricity','appliance_kwh')]:
        keys=cohort[grouping_factors(cfg,outcome)].astype(str).agg('|'.join,axis=1)
        pred=keys.map(mappings[outcome].set_index('cell').prediction)
        valid=pred.notna()
        if valid.any():profile[target]=float(np.average(pred[valid],weights=cohort.loc[valid,'weight']))
    if not np.isfinite(profile['cooling_hours']):profile['cooling_hours']=cfg['simulation']['default_cooling_hours']
    for key in ['members','heating_hours','appliance_kwh']:
        if not np.isfinite(profile[key]):raise ValueError(f'Cannot estimate {key} for archetype {archetype}.')
    profile.update(cfg['simulation']['archetype_overrides'].get(str(archetype),{}))
    profile.update(indices)
    return profile
