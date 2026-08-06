"""Local conditional surprise on California Housing with injected label errors."""
from __future__ import annotations
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def main(seed=0):
    rng=np.random.default_rng(seed); ds=fetch_california_housing(); X=StandardScaler().fit_transform(ds.data); y=(ds.target-ds.target.mean())/ds.target.std()
    idx=rng.choice(len(y),12000,replace=False); X=X[idx]; y=y[idx]; bad=rng.choice(len(y),int(.05*len(y)),replace=False); labels=np.zeros(len(y)); labels[bad]=1; yc=y.copy(); yc[bad]+=rng.choice([-1,1],len(bad))*1.5
    model=Ridge(alpha=1.0).fit(X,yc); residual=yc-model.predict(X)
    nbrs=NearestNeighbors(n_neighbors=31).fit(X); neigh=nbrs.kneighbors(return_distance=False)[:,1:]
    local=np.empty(len(y))
    for i,n in enumerate(neigh):
        med=np.median(residual[n]); mad=1.4826*np.median(np.abs(residual[n]-med))+1e-9; local[i]=abs(residual[i]-med)/mad
    print({"global_auc":roc_auc_score(labels,abs(residual)),"local_auc":roc_auc_score(labels,local)})


if __name__=='__main__': main()
