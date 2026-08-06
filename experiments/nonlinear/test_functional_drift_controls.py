#!/usr/bin/env python3
"""Frozen-feature and linearized controls for the functional tangent drift."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
from sklearn.datasets import make_friedman1
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
@dataclass
class Config:
    n_train:int=500; n_test:int=1000; n_probe:int=140; d:int=10; h:int=8; epochs:int=300; lr:float=.02; weight_decay:float=1e-4; replicates:int=10
class MLP(torch.nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.fc1=torch.nn.Linear(d,h); self.fc2=torch.nn.Linear(h,1)
        torch.nn.init.normal_(self.fc1.weight,0,.18); torch.nn.init.zeros_(self.fc1.bias); torch.nn.init.normal_(self.fc2.weight,0,.18); torch.nn.init.zeros_(self.fc2.bias)
    def forward(self,x): return self.fc2(torch.relu(self.fc1(x))).squeeze(-1)
def flat(m):
    with torch.no_grad(): return np.concatenate([m.fc1.weight.detach().numpy().ravel(),m.fc1.bias.detach().numpy(),m.fc2.weight.detach().numpy().ravel(),m.fc2.bias.detach().numpy()]).astype(float)
def unpack(t,d,h):
    k=0; W=t[k:k+h*d].reshape(h,d); k+=h*d; b=t[k:k+h]; k+=h; a=t[k:k+h]; k+=h; c=float(t[k]); return W,b,a,c
def f_j(t,X,d,h):
    W,b,a,c=unpack(t,d,h); Z=X@W.T+b; mask=(Z>0).astype(float); H=np.maximum(Z,0); pred=H@a+c; coeff=mask*a[None,:]; JW=np.einsum("nh,nd->nhd",coeff,X).reshape(len(X),h*d); return pred,np.concatenate([JW,coeff,H,np.ones((len(X),1))],axis=1)
def head_j(t,X,d,h):
    W,b,_,_=unpack(t,d,h); H=np.maximum(X@W.T+b,0); return np.concatenate([H,np.ones((len(X),1))],axis=1)
def align(A,B):
    den=np.linalg.norm(A,"fro")*np.linalg.norm(B,"fro"); return float(np.sum(A*B)/den)
def drift(A,B): return float(np.linalg.norm(A/np.linalg.norm(A,"fro")-B/np.linalg.norm(B,"fro"),"fro"))
def ker(J): return (J@J.T)/max(J.shape[1],1)
def prepare(seed,c):
    X,y=make_friedman1(n_samples=c.n_train+c.n_test,n_features=c.d,noise=1,random_state=seed); Xtr,Xte=X[:c.n_train],X[c.n_train:]; ytr,yte=y[:c.n_train],y[c.n_train:]; sx=StandardScaler().fit(Xtr); Xtr=sx.transform(Xtr); Xte=sx.transform(Xte); ym,ys=float(ytr.mean()),float(ytr.std()); ytr=(ytr-ym)/ys; yte=(yte-ym)/ys; rng=np.random.default_rng(seed+991); Xp=Xte[rng.choice(len(Xte),c.n_probe,replace=False)]; return Xtr,ytr,Xte,yte,Xp
def init(seed,c): torch.manual_seed(seed); return MLP(c.d,c.h).double()
def full(seed,c,Xtr,ytr,Xte,yte,Xp):
    m=init(seed,c); _,J0=f_j(flat(m),Xp,c.d,c.h); K0=ker(J0); xt=torch.from_numpy(Xtr).double(); yt=torch.from_numpy(ytr).double(); opt=torch.optim.Adam(m.parameters(),lr=c.lr,weight_decay=c.weight_decay)
    for _ in range(c.epochs): opt.zero_grad(set_to_none=True); loss=.5*torch.mean((m(xt)-yt)**2); loss.backward(); opt.step()
    _,Jf=f_j(flat(m),Xp,c.d,c.h); Kf=ker(Jf); mse=float(np.mean((f_j(flat(m),Xte,c.d,c.h)[0]-yte)**2)); return align(K0,Kf),drift(K0,Kf),mse
def frozen(seed,c,Xtr,ytr,Xte,yte,Xp):
    m=init(seed,c); [p.requires_grad_(False) for p in m.fc1.parameters()]; K0=ker(head_j(flat(m),Xp,c.d,c.h)); xt=torch.from_numpy(Xtr).double(); yt=torch.from_numpy(ytr).double(); opt=torch.optim.Adam(m.fc2.parameters(),lr=c.lr,weight_decay=c.weight_decay)
    for _ in range(c.epochs): opt.zero_grad(set_to_none=True); loss=.5*torch.mean((m(xt)-yt)**2); loss.backward(); opt.step()
    Kf=ker(head_j(flat(m),Xp,c.d,c.h)); mse=float(np.mean((f_j(flat(m),Xte,c.d,c.h)[0]-yte)**2)); return align(K0,Kf),drift(K0,Kf),mse
def linearized(seed,c,Xtr,ytr,Xte,yte,Xp):
    m=init(seed,c); t0=flat(m); ftr,Jtr=f_j(t0,Xtr,c.d,c.h); fte,Jte=f_j(t0,Xte,c.d,c.h); _,Jp=f_j(t0,Xp,c.d,c.h); K0=ker(Jp); delta=torch.zeros(Jtr.shape[1],dtype=torch.float64,requires_grad=True); jt=torch.from_numpy(Jtr).double(); ft=torch.from_numpy(ftr).double(); yt=torch.from_numpy(ytr).double(); opt=torch.optim.Adam([delta],lr=c.lr,weight_decay=c.weight_decay)
    for _ in range(c.epochs): opt.zero_grad(set_to_none=True); loss=.5*torch.mean((ft+jt@delta-yt)**2); loss.backward(); opt.step()
    pred=fte+Jte@delta.detach().numpy(); Kf=ker(Jp); return align(K0,Kf),drift(K0,Kf),float(np.mean((pred-yte)**2))
def main():
    c=Config(); rows=[]
    for r in range(c.replicates):
        seed=9300+r; data=prepare(seed,c)
        for name,fn in [("réseau complet",full),("extracteur gelé",frozen),("modèle linéarisé",linearized)]:
            a,d,m=fn(seed,c,*data); rows.append({"seed":seed,"regime":name,"alignment":a,"normalized_drift":d,"test_mse":m})
    df=pd.DataFrame(rows); df.to_csv(OUT/"functional_drift_controls_replicates.csv",index=False); agg=df.groupby("regime").agg(alignment_mean=("alignment","mean"),alignment_std=("alignment","std"),drift_mean=("normalized_drift","mean"),drift_std=("normalized_drift","std"),test_mse_mean=("test_mse","mean"),test_mse_std=("test_mse","std")).reset_index(); summary={"config":c.__dict__,"regimes":agg.to_dict(orient="records")}; (OUT/"functional_drift_controls_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    order=["réseau complet","extracteur gelé","modèle linéarisé"]; means=[float(agg.loc[agg.regime==x,"alignment_mean"].iloc[0]) for x in order]; stds=[float(agg.loc[agg.regime==x,"alignment_std"].iloc[0]) for x in order]; fig,ax=plt.subplots(figsize=(7,4)); ax.bar(np.arange(3),means,yerr=stds,capsize=4); ax.set_xticks(np.arange(3)); ax.set_xticklabels(order); ax.set_ylim(0,1.04); ax.set_ylabel("Alignement $K_0^{(0)}$ / $K_T^{(0)}$"); ax.set_title("Contrôles à géométrie gelée"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(OUT/"functional_drift_controls.pdf"); plt.close(fig); print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
