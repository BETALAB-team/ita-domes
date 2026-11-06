"""Independent grouped-coordinate verification of the 2021-only candidate-K grid."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
out=HERE.parent/"02_analysis"
d=pd.read_csv(out/"assigned_2021_proxy.csv")
X=np.column_stack(((d.min_age+3*d.max_age-40*d.number+162)/458, (.33*d.min_age+.66*d.max_age+10*d.number-30.79)/91.05, d.studio_ind))
z,idx,freq=np.unique(X,axis=0,return_inverse=True,return_counts=True)
sample=np.random.default_rng(42).choice(len(X),1500,replace=False)
sub=X[sample]
distance=np.sqrt(np.maximum(0,((sub[:,None,:]-sub[None,:,:])**2).sum(2)))
def fit(k,seed):
    rng=np.random.default_rng(seed)
    best=np.inf
    for j in range(10):
        means=X[rng.choice(len(X),k,replace=False)]
        for _ in range(150):
            ds=np.sum((means-z[:,None])**2,axis=2)
            pred=np.argmin(ds,axis=1)
            count=np.bincount(pred,weights=freq,minlength=k)
            means2=np.stack([np.bincount(pred,weights=col*freq,minlength=k) for col in z.T],axis=1)/np.maximum(1,count[:,None])
            for empty in np.flatnonzero(count==0): means2[empty]=X[rng.integers(len(X))]
            if np.max(np.abs(means2-means))<1e-8: break
            means=means2
        inertia=freq@np.sum((z-means[pred])**2,axis=1)
        if inertia<best: best, bestlab=inertia,pred[idx[sample]]
    within=np.zeros(len(sample)); between=np.ones(len(sample))*np.inf
    for cluster in range(k):
        mask=bestlab==cluster
        if not mask.any():continue
        taus=distance[:,mask].sum(1)
        within[mask]=taus[mask]/max(1,mask.sum()-1)
        between[~mask]=np.minimum(between[~mask],taus[~mask]/mask.sum())
    return best, float(np.mean((between-within)/np.maximum(between,within)))
ks=pd.read_csv(out/"2021_k_search.csv")
results=[]
for row in ks.itertuples():
    inertia,sil=fit(row.k,42);results.append(dict(k=row.k,inertia=inertia,silhouette=sil,table_inertia=row.inertia,table_silhouette=row.silhouette))
(HERE/"kgrid_audit.json").write_text(json.dumps(results,indent=2))
print(results)
