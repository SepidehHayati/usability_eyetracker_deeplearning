# 11_eval_protocols_loso_louo_baselines.py
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

DATA_DIR = os.path.join("..","data")
Xtab      = np.load(os.path.join(DATA_DIR,"X_tabular.npy"))
Xtab_user = np.load(os.path.join(DATA_DIR,"X_tabular_usernorm.npy"))
Y         = np.load(os.path.join(DATA_DIR,"Y8.npy")).astype(int)
G         = np.load(os.path.join(DATA_DIR,"G8.npy")).astype(int)
meta      = pd.read_csv(os.path.join(DATA_DIR,"meta_features_8cols.csv"))

def onehot_site_task(df):
    enc = OneHotEncoder(sparse_output=False, drop=None)
    Z = enc.fit_transform(df[["website","task"]])
    return Z, enc

def eval_protocol(groups, group_names, Xbase, add_site_task=False, model="logreg"):
    accs,f1s,aucs=[],[],[]
    # one-hot site/task
    if add_site_task:
        Z, enc = onehot_site_task(meta)
        Xuse = np.hstack([Xbase, Z])
    else:
        Xuse = Xbase

    for g in group_names:
        te_idx = np.where(groups==g)[0]
        tr_idx = np.where(groups!=g)[0]
        Xtr, Xte = Xuse[tr_idx], Xuse[te_idx]
        ytr, yte = Y[tr_idx], Y[te_idx]
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr); Xte = scaler.transform(Xte)
        if model=="logreg":
            clf = LogisticRegression(max_iter=500, class_weight="balanced")
        else:
            clf = SVC(kernel="rbf", probability=True, class_weight="balanced")
        clf.fit(Xtr,ytr)
        pr  = clf.predict(Xte)
        prp = clf.predict_proba(Xte)[:,1] if hasattr(clf,"predict_proba") else None
        accs.append(accuracy_score(yte,pr))
        f1s.append(f1_score(yte,pr,zero_division=0))
        aucs.append(roc_auc_score(yte,prp) if prp is not None else np.nan)
    return {"acc_mean":float(np.nanmean(accs)),"f1_mean":float(np.nanmean(f1s)),
            "auc_mean":float(np.nanmean(aucs))}

# 1) LOSO: گروه = website
sites = meta["website"].values
res_loso_raw   = eval_protocol(sites, np.unique(sites), Xtab, add_site_task=False, model="logreg")
res_loso_user  = eval_protocol(sites, np.unique(sites), Xtab_user, add_site_task=False, model="logreg")
res_loso_raw_st= eval_protocol(sites, np.unique(sites), Xtab, add_site_task=True,  model="logreg")
res_loso_user_st=eval_protocol(sites, np.unique(sites), Xtab_user, add_site_task=True, model="logreg")

# 2) LOUO: گروه = user
users = meta["user"].values
res_louo_raw   = eval_protocol(users, np.unique(users), Xtab, add_site_task=False, model="logreg")
res_louo_user  = eval_protocol(users, np.unique(users), Xtab_user, add_site_task=False, model="logreg")
res_louo_raw_st= eval_protocol(users, np.unique(users), Xtab, add_site_task=True,  model="logreg")
res_louo_user_st=eval_protocol(users, np.unique(users), Xtab_user, add_site_task=True, model="logreg")

print("[LOSO] tab:", res_loso_raw,
      "| tab_usernorm:", res_loso_user,
      "| +site/task:", res_loso_raw_st, res_loso_user_st)
print("[LOUO] tab:", res_louo_raw,
      "| tab_usernorm:", res_louo_user,
      "| +site/task:", res_louo_raw_st, res_louo_user_st)
