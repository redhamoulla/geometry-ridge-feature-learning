#!/usr/bin/env python3
"""Public robustness benchmark on UCI Concrete Compressive Strength.

The protocol mirrors the SeLoger stress test on a redistributable numeric table.
Exact feature duplicates are grouped before splitting so an identical concrete
recipe cannot leak from training to validation/test. A global group-disjoint
pool is reserved for development; thresholds are calibrated there and frozen
before ten repeated evaluation splits.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUT = Path(os.environ.get("CONCRETE_CSV", str(ROOT / "data" / "public" / "Concrete_Data_Yeh.csv")))
OUTDIR = Path(os.environ.get("EXPERIMENT_OUTPUT_DIR", str(HERE)))
OUTDIR.mkdir(parents=True, exist_ok=True)

DETAIL_PATH = OUTDIR / "concrete_bgr_public_results.csv"
SUMMARY_PATH = OUTDIR / "concrete_bgr_public_summary.csv"
JSON_PATH = OUTDIR / "concrete_bgr_public_summary.json"
TUNING_PATH = OUTDIR / "concrete_bgr_public_tuning.csv"
HYPER_PATH = OUTDIR / "concrete_bgr_public_hyperparameters.json"
FIGURE_PATH = OUTDIR / "concrete_bgr_public_stress.pdf"

FEATURES = ["cement", "slag", "flyash", "water", "superplasticizer", "coarseaggregate", "fineaggregate", "age"]
TARGET = "csMPa"
ALPHAS = np.logspace(-3, 4, 15)
POOL_SEED = 20260804
DEV_FRACTION = 0.25
DEV_SEEDS = [7101, 7202, 7303, 7404, 7505]
EVAL_SEEDS = [8101, 8202, 8303, 8404, 8505, 8606, 8707, 8808, 8909, 9010]
NOMINAL_RATIO_LIMIT = 1.005
NOMINAL_MAX_LIMIT = 1.01
MAX_IRLS = 30
TOL = 2e-7
BGR_LADDER = [(0.90,1.5),(0.90,2.0),(0.95,2.0),(0.95,3.0),(0.975,3.0),(0.99,4.0),(1.0,6.0)]
HUBER_LADDER = [1.5,2.0,3.0,4.0,6.0]
GEOM_LADDER = [0.90,0.95,0.975,0.99,1.0]
METHODS = ["ridge","huber","geometry","schweppe_type","bgr"]

@dataclass
class Fit:
    model: Ridge
    geom_weights: np.ndarray
    resid_weights: np.ndarray
    combined_weights: np.ndarray
    n_iter: int


def load_data():
    df = pd.read_csv(INPUT)
    x = df[FEATURES].to_numpy(float)
    y = df[TARGET].to_numpy(float)
    # Exact feature tuple identifies a recipe group.
    hashes = pd.util.hash_pandas_object(df[FEATURES], index=False).to_numpy(np.uint64)
    _, groups = np.unique(hashes, return_inverse=True)
    return x, y, groups, df


def global_group_pools(groups: np.ndarray):
    unique = np.unique(groups)
    rng = np.random.default_rng(POOL_SEED)
    perm = rng.permutation(unique)
    n_dev = max(1, int(round(DEV_FRACTION*len(unique))))
    dev_g = np.sort(perm[:n_dev]); eval_g = np.sort(perm[n_dev:])
    dev_mask = np.isin(groups, dev_g); eval_mask = np.isin(groups, eval_g)
    payload = {
        "pool_seed": POOL_SEED,
        "total_groups": int(len(unique)),
        "development_groups": int(len(dev_g)),
        "evaluation_groups": int(len(eval_g)),
        "intersection_size": int(np.intersect1d(dev_g, eval_g).size),
    }
    return np.flatnonzero(dev_mask), np.flatnonzero(eval_mask), payload


def split_indices(indices: np.ndarray, groups: np.ndarray, seed: int):
    g = groups[indices]
    s1 = GroupShuffleSplit(n_splits=1, train_size=0.60, random_state=seed)
    tr_rel, temp_rel = next(s1.split(indices, groups=g))
    temp = indices[temp_rel]
    s2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed+10000)
    va_rel, te_rel = next(s2.split(temp, groups=groups[temp]))
    return indices[tr_rel], temp[va_rel], temp[te_rel]


def transform_fit(xtr,ytr):
    sc=RobustScaler(quantile_range=(25,75)); sc.fit(xtr)
    ym=float(np.mean(ytr)); ys=max(float(np.std(ytr)),1e-8)
    return sc,ym,ys

def tx(sc,x): return np.column_stack([np.ones(len(x)),sc.transform(x)])
def ty(ym,ys,y): return (y-ym)/ys

def ridge(alpha): return Ridge(alpha=float(alpha),fit_intercept=False,solver="cholesky",tol=1e-10)

def tune_alpha(x,y,xv,yv):
    best=(math.inf,None)
    for a in ALPHAS:
        m=ridge(a);m.fit(x,y);r=float(np.sqrt(mean_squared_error(yv,m.predict(xv))))
        if r<best[0]: best=(r,float(a))
    return best[1]

def scale_ridge(x,y,a):
    m=ridge(a);m.fit(x,y);e=y-m.predict(x);med=np.median(e);return max(1.4826*np.median(np.abs(e-med)),1e-4)

def geom_weights(x,kx):
    n=np.linalg.norm(x,axis=1)
    if not np.isfinite(kx): return np.ones(len(x)),n
    w=np.ones(len(x));mask=n>kx;w[mask]=(kx/np.maximum(n[mask],1e-12))**2;return w,n

def fit_method(method,x,y,a,kx,ky,scale):
    if method=="ridge":
        m=ridge(a);m.fit(x,y);o=np.ones(len(y));return Fit(m,o,o,o,1)
    if method=="schweppe_type":
        norms=np.linalg.norm(x,axis=1);v=np.minimum(1.0,kx/np.maximum(norms,1e-12))
        m=ridge(a);m.fit(x,y);prev=m.coef_.copy();w=np.ones(len(y))
        for it in range(1,MAX_IRLS+1):
            u=(y-m.predict(x))/scale;w=np.minimum(1.0,ky*v/np.maximum(np.abs(u),1e-12))
            new=ridge(a);new.fit(x,y,sample_weight=w);rel=np.linalg.norm(new.coef_-prev)/(np.linalg.norm(prev)+1e-12);m,prev=new,new.coef_.copy()
            if rel<TOL: break
        z=np.abs((y-m.predict(x))/scale);wy=np.minimum(1.0,ky/np.maximum(z,1e-12))
        return Fit(m,v**2,wy,w,it)
    useg=method in {"geometry","bgr"};useh=method in {"huber","bgr"}
    wx,_=geom_weights(x,kx if useg else math.inf);m=ridge(a);m.fit(x,y,sample_weight=wx)
    if not useh:
        o=np.ones(len(y));return Fit(m,wx,o,wx,1)
    prev=m.coef_.copy();wy=np.ones(len(y))
    for it in range(1,MAX_IRLS+1):
        z=np.abs(y-m.predict(x))/scale;wy=np.minimum(1.0,ky/np.maximum(z,1e-12));w=wx*wy
        new=ridge(a);new.fit(x,y,sample_weight=w);rel=np.linalg.norm(new.coef_-prev)/(np.linalg.norm(prev)+1e-12);m,prev=new,new.coef_.copy()
        if rel<TOL: break
    return Fit(m,wx,wy,wx*wy,it)

def rmse(m,x,y): return float(np.sqrt(mean_squared_error(y,m.predict(x))))

def select(table,fam):
    sub=table[table.family==fam]
    agg=(sub.groupby(["rank","q","ky"],as_index=False).nominal_ratio.agg(["mean","max"]).reset_index())
    eligible=agg[(agg["mean"]<=NOMINAL_RATIO_LIMIT)&(agg["max"]<=NOMINAL_MAX_LIMIT)]
    if eligible.empty:
        best=agg.sort_values(["mean","max"]).iloc[0];status="fallback"
    else:
        best=eligible.sort_values("rank").iloc[0];status="most_aggressive_under_mean_and_max_constraints"
    return {"q_geometry":float(best.q),"kappa_y":float(best.ky),"nominal_ratio_mean":float(best["mean"]),"nominal_ratio_max":float(best["max"]),"status":status}

def calibrate(x,y,groups,dev_idx):
    rows=[]
    for seed in DEV_SEEDS:
        tr,va,_=split_indices(dev_idx,groups,seed);sc,ym,ys=transform_fit(x[tr],y[tr]);xt,xv=tx(sc,x[tr]),tx(sc,x[va]);yt,yv=ty(ym,ys,y[tr]),ty(ym,ys,y[va]);a=tune_alpha(xt,yt,xv,yv);base=ridge(a);base.fit(xt,yt);br=rmse(base,xv,yv);scale=scale_ridge(xt,yt,a);norms=np.linalg.norm(xt,axis=1)
        candidates=[]
        for rank,ky in enumerate(HUBER_LADDER):candidates.append(("huber",rank,1.0,ky))
        for rank,q in enumerate(GEOM_LADDER):candidates.append(("geometry",rank,q,1e9))
        for fam in ["bgr","schweppe_type"]:
            for rank,(q,ky) in enumerate(BGR_LADDER):candidates.append((fam,rank,q,ky))
        for fam,rank,q,ky in candidates:
            kx=math.inf if fam=="huber" else float(np.quantile(norms,q));f=fit_method(fam,xt,yt,a,kx,ky,scale);rr=rmse(f.model,xv,yv)
            rows.append({"seed":seed,"family":fam,"rank":rank,"q":q,"ky":ky,"nominal_ratio":rr/br,"ridge_rmse":br,"rmse":rr,"alpha":a,"kx":kx,"scale":scale})
    tab=pd.DataFrame(rows);tab.to_csv(TUNING_PATH,index=False)
    hyp={f:select(tab,f) for f in ["huber","geometry","bgr","schweppe_type"]}
    payload={"protocol":{"development_seeds":DEV_SEEDS,"nominal_ratio_mean_limit":NOMINAL_RATIO_LIMIT,"nominal_ratio_max_limit":NOMINAL_MAX_LIMIT,"scale":"ordinary-Ridge residual MAD"},**hyp};HYPER_PATH.write_text(json.dumps(payload,indent=2));return payload

def params(hyp,m,norms):
    if m=="ridge":return math.inf,1e9
    q=hyp[m]["q_geometry"];ky=hyp[m]["kappa_y"];return (math.inf if m=="huber" else float(np.quantile(norms,q))),ky

def evaluate(x,y,groups,eval_idx,hyp):
    rows=[]
    for seed in EVAL_SEEDS:
        tr,va,te=split_indices(eval_idx,groups,seed);sc,ym,ys=transform_fit(x[tr],y[tr]);xt,xv,xte=tx(sc,x[tr]),tx(sc,x[va]),tx(sc,x[te]);yt,yv,yte=ty(ym,ys,y[tr]),ty(ym,ys,y[va]),ty(ym,ys,y[te]);a=tune_alpha(xt,yt,xv,yv);scale=scale_ridge(xt,yt,a);norms=np.linalg.norm(xt,axis=1);pr={m:params(hyp,m,norms) for m in METHODS};nom={m:fit_method(m,xt,yt,a,*pr[m],scale) for m in METHODS}
        rng=np.random.default_rng(seed+50000);n=max(1,int(round(.05*len(tr))));bad=rng.choice(len(tr),n,replace=False);cols=rng.integers(1,xt.shape[1],n);xs=rng.choice([-1.,1.],n);ysign=rng.choice([-1.,1.],n)
        scenarios={"nominal":(0.,xt,yt,np.array([],int))};yc=yt.copy();yc[bad]+=4*ysign;scenarios["vertical"]=(4.,xt,yc,bad);xl=xt.copy();xl[bad,cols]+=50*xs;scenarios["leverage"]=(50.,xl,yt,bad);xm=xt.copy();yy=yt.copy();xm[bad,cols]+=50*xs;yy[bad]+=4*ysign;scenarios["mixed"]=(50.,xm,yy,bad);noise=np.random.default_rng(seed+60000).normal(size=len(yt));scenarios["gaussian"]=(.4,xt,yt+.4*noise,np.arange(len(yt)))
        for m in METHODS:
            nr=rmse(nom[m].model,xte,yte);kx,ky=pr[m]
            for scen,(lev,xx,yyy,idx) in scenarios.items():
                f=nom[m] if scen=="nominal" else fit_method(m,xx,yyy,a,kx,ky,scale);rr=rmse(f.model,xte,yte)
                rows.append({"seed":seed,"scenario":scen,"level":lev,"method":m,"rmse":rr,"delta_rmse":rr-nr,"mean_corrupt_geom_weight":float(np.mean(f.geom_weights[idx])) if len(idx) else math.nan,"mean_corrupt_resid_weight":float(np.mean(f.resid_weights[idx])) if len(idx) else math.nan,"mean_corrupt_combined_weight":float(np.mean(f.combined_weights[idx])) if len(idx) else math.nan})
    return pd.DataFrame(rows)

def main():
    x,y,groups,df=load_data();dev,ev,pools=global_group_pools(groups);hyp=calibrate(x,y,groups,dev);detail=evaluate(x,y,groups,ev,hyp);detail.to_csv(DETAIL_PATH,index=False);summ=(detail.groupby(["scenario","level","method"],as_index=False).agg(n=("seed","size"),rmse_mean=("rmse","mean"),rmse_std=("rmse","std"),delta_rmse_mean=("delta_rmse","mean"),delta_rmse_std=("delta_rmse","std"),corrupt_geom_weight=("mean_corrupt_geom_weight","mean"),corrupt_resid_weight=("mean_corrupt_resid_weight","mean"),corrupt_combined_weight=("mean_corrupt_combined_weight","mean")));summ.to_csv(SUMMARY_PATH,index=False)
    payload={"dataset":{"rows":len(df),"feature_groups":int(len(np.unique(groups))),"sha256":hashlib.sha256(INPUT.read_bytes()).hexdigest()},"pools":pools,"hyperparameters":hyp,"summary":summ.to_dict(orient="records"),"notes":["All residual scales use ordinary Ridge; exact feature duplicates are group-split.","Repeated evaluation splits overlap; dispersion is descriptive."]};JSON_PATH.write_text(json.dumps(payload,indent=2))
    specs=[("vertical",4.,"Labels"),("leverage",50.,"Leverage"),("mixed",50.,"Mixte"),("gaussian",.4,"Gaussien")];lab={"ridge":"Ridge","huber":"Huber","geometry":"Géométrie","schweppe_type":"Schweppe","bgr":"BGR"};xx=np.arange(4);w=.155;fig,ax=plt.subplots(figsize=(10.4,5.4))
    for j,m in enumerate(METHODS):
        vals=[float(summ[(summ.scenario==s)&np.isclose(summ.level,l)&(summ.method==m)].iloc[0].delta_rmse_mean) for s,l,_ in specs];ax.bar(xx+(j-2)*w,vals,width=w,label=lab[m])
    ax.axhline(0,lw=.8);ax.set_xticks(xx,[z for _,_,z in specs]);ax.set_ylabel("Augmentation de RMSE standardisée");ax.set_title("Concrete public : dégradation sous contamination");ax.legend(ncol=3,loc="upper left");fig.tight_layout();fig.savefig(FIGURE_PATH);fig.savefig(OUTDIR/"concrete_bgr_public_stress.png",dpi=220);plt.close(fig)
    print(json.dumps(hyp,indent=2));print(summ.to_string(index=False))
if __name__=="__main__":main()
