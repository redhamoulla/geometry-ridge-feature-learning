#!/usr/bin/env python3
"""Stagewise experiment: random points receive early or late training exposure.

Group membership is assigned independently of x and y. After a final joint phase,
static final diagnostics should not recover the random schedule, whereas trajectory
features can. This tests the claim that the dynamic register contains temporal role
information erased by final scalar/static attribution.
"""
from __future__ import annotations
import json, math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np, pandas as pd, torch
from sklearn.datasets import make_friedman1
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent

@dataclass
class Cfg:
    n_train:int=400; n_test:int=700; d:int=10; h:int=10; noise:float=1.0
    phase1:int=60; phase2:int=60; phase3:int=120
    checkpoints:Tuple[int,...]=(0,20,60,80,120,160,240)
    lr:float=.018; wd:float=1e-4; ridge:float=2e-2; reps:int=8

class MLP(torch.nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.fc1=torch.nn.Linear(d,h);self.fc2=torch.nn.Linear(h,1)
        torch.nn.init.normal_(self.fc1.weight,std=.16);torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.normal_(self.fc2.weight,std=.16);torch.nn.init.zeros_(self.fc2.bias)
    def forward(self,x):return self.fc2(torch.tanh(self.fc1(x))).squeeze(-1)

def flat(m):
    with torch.no_grad():
        return np.r_[m.fc1.weight.detach().numpy().ravel(),m.fc1.bias.detach().numpy(),m.fc2.weight.detach().numpy().ravel(),m.fc2.bias.detach().numpy()]

def unpack(t,d,h):
    k=0;W=t[k:k+h*d].reshape(h,d);k+=h*d;b=t[k:k+h];k+=h;a=t[k:k+h];k+=h;c=float(t[k]);return W,b,a,c

def fj(t,X,d,h):
    W,b,a,c=unpack(t,d,h);Z=X@W.T+b;H=np.tanh(Z);s=1-H*H;pred=H@a+c;coef=s*a[None,:]
    J=np.c_[np.einsum('nh,nd->nhd',coef,X).reshape(len(X),h*d),coef,H,np.ones(len(X))]
    return pred,J

def rsig(r):
    med=np.median(r);return float(max(1.4826*np.median(np.abs(r-med)),.10))

def registers(theta,X,y,Xt,yt,cfg):
    pred,J=fj(theta,X,cfg.d,cfg.h);pt,Jt=fj(theta,Xt,cfg.d,cfg.h);sig=rsig(pred-y)
    G=cfg.ridge*np.eye(J.shape[1])+(J.T@J)/(sig*sig);P=np.linalg.inv(G)
    mu=np.einsum('ij,jk,ik->i',J,P,J)/(sig*sig);mu=np.maximum(mu,0);c=mu/(1+mu)
    z=(pred-y)/sig;eff=np.abs(z)*np.sqrt(mu)
    # exact local rank-one displacement and test-loss influence for every point
    PJ=P@J.T; den=sig*sig+np.einsum('ij,ji->i',J,PJ)
    D=-(PJ*((pred-y)/den)[None,:]).T
    gtest=(Jt.T@(pt-yt))/(sig*sig*len(Xt)); infl=-(D@gtest)
    return c,eff,np.abs(z),infl

def one(seed,cfg):
    rng=np.random.default_rng(seed);X,y=make_friedman1(n_samples=cfg.n_train+cfg.n_test,n_features=cfg.d,noise=cfg.noise,random_state=seed)
    Xtr,Xt=X[:cfg.n_train],X[cfg.n_train:];ytr,yt=y[:cfg.n_train],y[cfg.n_train:]
    xs=StandardScaler().fit(Xtr);Xtr=xs.transform(Xtr);Xt=xs.transform(Xt);ym,ys=ytr.mean(),ytr.std();ytr=(ytr-ym)/ys;yt=(yt-ym)/ys
    perm=rng.permutation(cfg.n_train);early=perm[:cfg.n_train//2];late=perm[cfg.n_train//2:]
    group=np.empty(cfg.n_train,dtype='U5');group[early]='early';group[late]='late'
    torch.manual_seed(seed);m=MLP(cfg.d,cfg.h).double();opt=torch.optim.Adam(m.parameters(),lr=cfg.lr,weight_decay=cfg.wd)
    Xtt=torch.from_numpy(Xtr).double();ytt=torch.from_numpy(ytr).double()
    rows=[]; total=cfg.phase1+cfg.phase2+cfg.phase3
    for e in range(total+1):
        if e in cfg.checkpoints:
            c,ef,z,inf=registers(flat(m),Xtr,ytr,Xt,yt,cfg)
            for i in range(cfg.n_train):rows.append({'seed':seed,'point':i,'schedule':group[i],'epoch':e,'contraction':c[i],'effort':ef[i],'residual':z[i],'pred_influence':inf[i]})
        if e==total:break
        ids=early if e<cfg.phase1 else late if e<cfg.phase1+cfg.phase2 else np.arange(cfg.n_train)
        opt.zero_grad(set_to_none=True);loss=.5*torch.mean((m(Xtt[ids])-ytt[ids])**2);loss.backward();opt.step()
    return pd.DataFrame(rows)

def classify(df,cfg):
    parts=[]
    for metric in ['contraction','effort','residual','pred_influence']:
        q=df.pivot_table(index=['seed','point','schedule'],columns='epoch',values=metric);q.columns=[f'{metric}_e{int(x)}' for x in q.columns];parts.append(q)
    w=pd.concat(parts,axis=1).reset_index();y=(w.schedule=='late').astype(int).to_numpy();groups=w.seed.to_numpy();cv=GroupKFold(n_splits=len(np.unique(groups)));last=cfg.checkpoints[-1]
    sets={
      'final_predictive_attribution_scalar':[f'pred_influence_e{last}'],
      'final_register_2d':[f'contraction_e{last}',f'effort_e{last}'],
      'dynamic_predictive_attribution':[f'pred_influence_e{e}' for e in cfg.checkpoints],
      'dynamic_residual':[f'residual_e{e}' for e in cfg.checkpoints],
      'dynamic_register':[f'{m}_e{e}' for e in cfg.checkpoints for m in ['contraction','effort']],
    }
    rows=[];preds={}
    for name,cols in sets.items():
        X=w[cols].to_numpy(float);med=np.median(X,0);sc=1.4826*np.median(np.abs(X-med),0)+1e-8;X=(X-med)/sc
        pr=cross_val_predict(LogisticRegression(max_iter=500,class_weight='balanced'),X,y,groups=groups,cv=cv,method='predict_proba')[:,1];preds[name]=pr
        for seed in sorted(np.unique(groups)):
            m=groups==seed;rows.append({'method':name,'seed':int(seed),'auc':roc_auc_score(y[m],pr[m])})
    return w,pd.DataFrame(rows)

def main():
    cfg=Cfg(); dfs=[]
    for r in range(cfg.reps):print('stagewise',r+1,'/',cfg.reps,flush=True);dfs.append(one(9700+r,cfg))
    df=pd.concat(dfs,ignore_index=True);w,auc=classify(df,cfg);df.to_csv(OUT/'stagewise_dynamic_register_points.csv',index=False);auc.to_csv(OUT/'stagewise_dynamic_register_auc.csv',index=False)
    tab=auc.groupby('method').auc.agg(['mean','std','min','max'])
    # median trajectories for figure
    med=df.groupby(['schedule','epoch'])[['contraction','effort','pred_influence']].median().reset_index()
    fig,ax=plt.subplots(figsize=(6.6,4.5))
    for g,ls in [('early','-'),('late','--')]:
        s=med[med.schedule==g];ax.plot(s.epoch,s.effort,marker='o',ls=ls,label=f'{g} - effort')
    ax.axvline(cfg.phase1,lw=1,ls=':');ax.axvline(cfg.phase1+cfg.phase2,lw=1,ls=':');ax.set_xlabel('Époque');ax.set_ylabel('Effort naturel médian');ax.set_title('Même distribution, exposition précoce ou tardive');ax.legend(frameon=False);ax.grid(alpha=.22);fig.tight_layout();fig.savefig(OUT/'stagewise_dynamic_register.pdf');plt.close(fig)
    summary={'config':cfg.__dict__,'auc':{k:{kk:float(vv) for kk,vv in row.items()} for k,row in tab.to_dict('index').items()}}
    with open(OUT/'stagewise_dynamic_register_summary.json','w') as f:json.dump(summary,f,indent=2)
    print(tab);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
