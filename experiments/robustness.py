"""Compare ridge and bounded-geometric ridge under gross contamination."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from geometric_register.robust import fit_bgr, ridge_fit


def problem(seed: int, n=1200, p=12):
    rng=np.random.default_rng(seed); X=rng.normal(size=(n,p)); beta=rng.normal(size=p); y=X@beta+rng.normal(scale=.5,size=n)
    return X,y,beta,rng


def contaminate(X,y,rng,kind,frac=.05,amp_x=50.,amp_y=8.):
    X=X.copy(); y=y.copy(); idx=rng.choice(len(y),int(frac*len(y)),replace=False)
    if kind in {'x','mixed'}: X[idx]+=amp_x*rng.normal(size=(len(idx),X.shape[1]))
    if kind in {'y','mixed'}: y[idx]+=amp_y*rng.choice([-1.,1.],size=len(idx))
    return X,y


def main():
    rows=[]
    for seed in range(10):
        X,y,_,rng=problem(seed); cut=800; scaler=StandardScaler().fit(X[:cut]); Xtr=scaler.transform(X[:cut]); Xte=scaler.transform(X[cut:]); ytr=y[:cut]; yte=y[cut:]
        for kind in ['clean','y','x','mixed']:
            Xa,ya=(Xtr,ytr) if kind=='clean' else contaminate(Xtr,ytr,rng,kind)
            br=ridge_fit(Xa,ya,1.0); bg,_=fit_bgr(Xa,ya,1.0)
            rows.append((kind,mean_squared_error(yte,Xte@br),mean_squared_error(yte,Xte@bg)))
    for kind in ['clean','y','x','mixed']:
        a=np.array([(r,b) for k,r,b in rows if k==kind]); print(kind,{"ridge":a[:,0].mean(),"bgr":a[:,1].mean()})


if __name__ == '__main__': main()
