"""Private SeLoger stress test. The source CSV is intentionally not distributed."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from geometric_register.robust import fit_bgr, ridge_fit

NUM=['surface','nb_pieces','nb_chambres','nb_photos','si_balcon']
CAT=['codepostal','typedebien','idtypecuisine']

def main():
    p=argparse.ArgumentParser(); p.add_argument('csv',type=Path); a=p.parse_args(); d=pd.read_csv(a.csv).drop_duplicates('idannonce'); d=d[(d.prix>0)].copy(); y=np.log(d.prix.to_numpy())
    tr,te=train_test_split(np.arange(len(d)),test_size=.25,random_state=0); pre=ColumnTransformer([('n',StandardScaler(),NUM),('c',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)]); Xtr=pre.fit_transform(d.iloc[tr]); Xte=pre.transform(d.iloc[te]); ytr=y[tr]; yte=y[te]
    rng=np.random.default_rng(0); bad=rng.choice(len(ytr),int(.05*len(ytr)),replace=False); Xc=Xtr.copy(); yc=ytr.copy(); Xc[bad]+=50*rng.normal(size=(len(bad),Xtr.shape[1])); yc[bad]+=rng.choice([-1,1],len(bad))*4
    br=ridge_fit(Xc,yc,10.0); bg,_=fit_bgr(Xc,yc,10.0)
    print({'ridge_rmse':mean_squared_error(yte,Xte@br)**.5,'bgr_rmse':mean_squared_error(yte,Xte@bg)**.5})

if __name__=='__main__': main()
