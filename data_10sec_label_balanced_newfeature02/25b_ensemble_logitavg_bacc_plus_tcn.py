# 25b_ensemble_logitavg_bacc_plus_tcn.py
# Logit-averaging ensemble + جستجوی آستانه با اولویت Balanced Accuracy
# مدل‌های قبلی + TCN جدید

import os, json, itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results", "25b_ensemble_logitavg_bacc_plus_tcn")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ← این چهار تا را داریم
MODELS = [
    "11_resnet1d_runner06.py",
    "10_cnn_bn_runner06.py",
    "18_resnet1d_accfirst_runner08.py",
    "28_tcn_runner06.py",
]

WEIGHTS = [0, 1, 2, 3, 4]   # 0 یعنی حذف آن مدل، دامنه کوچک = اجرا سریع‌تر
THRS    = np.linspace(0.0, 1.0, 501)

def load_probs(fold_dir):
    df = pd.read_csv(os.path.join(fold_dir, "predictions.csv"))
    return df["y_true"].values.astype(int), df["y_prob"].values.astype(float)

def metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    tpr = tp / max(tp+fn, 1)
    tnr = tn / max(tn+fp, 1)
    bacc = 0.5 * (tpr + tnr)
    return acc, prec, rec, f1, bacc

def to_logit(p):
    eps = 1e-6
    p = np.clip(p, eps, 1.0-eps)
    return np.log(p/(1-p))

def from_logit(z):
    return 1.0/(1.0+np.exp(-z))

fold_rows = []
best_weights_per_fold = {}
for k in range(1, 6):
    # خواندن y_true و y_prob هر مدل
    probs = []
    y_true_ref = None
    for m in MODELS:
        fold_dir = os.path.join(ROOT, "results", m, f"fold{k}")
        y_true, y_prob = load_probs(fold_dir)
        if y_true_ref is None: y_true_ref = y_true
        probs.append(y_prob)
    probs = np.stack(probs, axis=1)  # (N, M)

    # logit-averaging
    logits = to_logit(probs)  # (N,M)

    best_key = None
    best_row = None

    for w in itertools.product(WEIGHTS, repeat=len(MODELS)):
        if sum(w) == 0:
            continue
        ww = np.array(w, dtype=float)
        ww = ww / ww.sum()

        z = (logits * ww).sum(axis=1)
        p = from_logit(z)

        # اولویت: بیشینه کردن Balanced Accuracy؛ سپس Acc؛ بعد F1، Rec، Prec
        best_bacc = -1.0
        best_tuple = None
        best_thr = 0.5

        for thr in THRS:
            acc, prec, rec, f1, bacc = metrics(y_true_ref, p, thr)
            key = (bacc, acc, f1, rec, prec)
            if key > best_tuple if best_tuple is not None else True:
                best_tuple = key
                best_thr = thr

        bacc, acc, f1, rec, prec = best_tuple
        row = dict(fold=k, acc=acc, prec=prec, rec=rec, f1=f1, bacc=bacc, thr=float(best_thr), weights=list(ww), weight_raw=list(w))
        if (best_key is None) or ((bacc, acc, f1, rec, prec) > best_key):
            best_key = (bacc, acc, f1, rec, prec)
            best_row = row

    fold_rows.append(best_row)
    best_weights_per_fold[str(k)] = best_row["weight_raw"]

df = pd.DataFrame(fold_rows)
df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)

avg = df[["acc","prec","rec","f1","bacc"]].mean().to_dict()
print(f"[DONE] results → {RESULTS_DIR}")
print(df)
print("AVG:", {k: round(v,4) for k,v in avg.items()})

with open(os.path.join(RESULTS_DIR, "meta.json"), "w") as f:
    json.dump({
        "avg": avg,
        "note": "Logit-avg ensemble with TCN added; selection by (BAcc,Acc,F1,Rec,Prec)",
        "models": MODELS,
        "weight_range": WEIGHTS,
        "thresholds": {"count": len(THRS), "min": float(THRS.min()), "max": float(THRS.max())},
        "best_weights_per_fold": best_weights_per_fold
    }, f, indent=2)
