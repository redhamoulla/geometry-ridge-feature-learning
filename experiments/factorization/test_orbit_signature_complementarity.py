#!/usr/bin/env python3
"""Complementarity experiment for the orbital signature.

Four classes cross two contraction spectra with two effort-energy profiles.
The binary target is their XOR. Hence neither half of the signature carries
information about that target, whereas the full signature determines it.
Random orthogonal rotations remove coordinate artefacts.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parent
SPECTRA = {0: np.array([0.78, 0.43, 0.12]), 1: np.array([0.66, 0.54, 0.08])}
ENERGIES = {0: np.array([1.70, 0.25, 0.05]), 1: np.array([0.05, 0.25, 1.70])}

def haar(rng, p=3):
    q, r = np.linalg.qr(rng.normal(size=(p,p)))
    s=np.sign(np.diag(r)); s[s==0]=1
    q=q@np.diag(s)
    if np.linalg.det(q)<0: q[:,0]*=-1
    return q

def recover(B,b):
    vals, vecs=np.linalg.eigh((B+B.T)/2)
    order=np.argsort(vals)[::-1]; vals=vals[order]; vecs=vecs[:,order]
    return vals,(vecs.T@b)**2

def make_data(n_groups=400,seed=20260803):
    rng=np.random.default_rng(seed); rows=[]
    for gid in range(n_groups):
        q=haar(rng); signs=rng.choice([-1.,1.],3)
        for sid in (0,1):
            for eid in (0,1):
                B=q@np.diag(SPECTRA[sid])@q.T
                b=q@(signs*np.sqrt(ENERGIES[eid]))
                vals,en=recover(B,b)
                row={"group_id":gid,"spectrum_id":sid,"energy_id":eid,"xor_label":sid^eid,"four_class":2*sid+eid}
                for j in range(3):
                    row[f"eigenvalue_{j+1}"]=float(vals[j]); row[f"effort_energy_{j+1}"]=float(en[j])
                rows.append(row)
    return pd.DataFrame(rows)

def preds(df,cols,target,proba):
    X=df[cols].to_numpy(float); y=df[target].to_numpy(int); groups=df.group_id.to_numpy(int)
    model=RandomForestClassifier(n_estimators=250,max_depth=5,min_samples_leaf=2,random_state=20260803,n_jobs=1,class_weight="balanced")
    p=cross_val_predict(model,X,y,groups=groups,cv=GroupKFold(10),method="predict_proba" if proba else "predict",n_jobs=1)
    return p[:,1] if proba else p

def main():
    df=make_data(); spectral=[f"eigenvalue_{j}" for j in range(1,4)]; effort=[f"effort_energy_{j}" for j in range(1,4)]; full=spectral+effort
    auc={}; acc={}
    for name,cols in [("spectrum_only",spectral),("effort_only",effort),("full_signature",full)]:
        auc[name]=float(roc_auc_score(df.xor_label,preds(df,cols,"xor_label",True)))
        acc[name]=float(accuracy_score(df.four_class,preds(df,cols,"four_class",False)))
    serr=eerr=0.0
    for _,r in df.iterrows():
        serr=max(serr,float(np.max(np.abs(r[spectral].to_numpy(float)-SPECTRA[int(r.spectrum_id)]))))
        eerr=max(eerr,float(np.max(np.abs(r[effort].to_numpy(float)-ENERGIES[int(r.energy_id)]))))
    summary={"n_rotation_groups":int(df.group_id.nunique()),"n_rows":int(len(df)),"spectra":{str(k):v.tolist() for k,v in SPECTRA.items()},"effort_energies":{str(k):v.tolist() for k,v in ENERGIES.items()},"binary_xor_auc":auc,"four_class_accuracy":acc,"max_signature_recovery_error":{"spectrum":serr,"effort_energy":eerr}}
    df.to_csv(OUT/"orbit_signature_complementarity_points.csv",index=False)
    (OUT/"orbit_signature_complementarity_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    labels=["Spectre seul","Énergies seules","Signature complète"]; x=np.arange(3); w=.34
    fig,ax=plt.subplots(figsize=(7.2,4.1)); ax.bar(x-w/2,[auc["spectrum_only"],auc["effort_only"],auc["full_signature"]],w,label="AUC XOR"); ax.bar(x+w/2,[acc["spectrum_only"],acc["effort_only"],acc["full_signature"]],w,label="Exactitude 4 classes"); ax.axhline(.5,ls="--",lw=1); ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0,1.05); ax.set_ylabel("Performance en validation croisée"); ax.set_title("Complémentarité des composantes de la signature"); ax.legend(frameon=False); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(OUT/"orbit_signature_complementarity.pdf"); plt.close(fig)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
