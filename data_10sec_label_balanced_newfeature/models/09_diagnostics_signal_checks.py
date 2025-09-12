# 09_diagnostics_signal_checks.py
# هدف: بررسی وجود سیگنال، اندازه‌گیری AUC/F1 با مدل‌های ساده، پرمیوتیشن تست،
#      بیس‌لاین‌های مبتنی بر website/task، و تاثیر Downsample.
# ورودی‌ها: X8.npy (N,1500,8), Y8.npy (N,), G8.npy (N,), meta_features_8cols.csv (اختیاری برای filename/website/task/user)
# خروجی: results/09_diagnostics/... شامل گزارش‌ها و نتایج

import os, json, warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "09_diagnostics_signal_checks")
os.makedirs(RESULT_DIR, exist_ok=True)

# ---- Load ----
X = np.load(os.path.join(DATA_DIR, "X8.npy"))         # (N,1500,8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy")).astype(int)
G = np.load(os.path.join(DATA_DIR, "G8.npy")).astype(int)

meta_path = os.path.join(DATA_DIR, "meta_features_8cols.csv")
meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None

N, T, C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}")

# ---- Utility: feature summarization over time ----
def summarize_time_series(arr):  # arr: (T,C)
    df = pd.DataFrame(arr, columns=[f"ch{c}" for c in range(arr.shape[1])])
    feats = {}
    for col in df.columns:
        s = df[col]
        feats[f"{col}_mean"]  = float(s.mean())
        feats[f"{col}_std"]   = float(s.std(ddof=1) if s.shape[0]>1 else 0.0)
        feats[f"{col}_min"]   = float(s.min())
        feats[f"{col}_max"]   = float(s.max())
        feats[f"{col}_median"]= float(s.median())
        feats[f"{col}_p25"]   = float(np.percentile(s,25))
        feats[f"{col}_p75"]   = float(np.percentile(s,75))
        feats[f"{col}_iqr"]   = feats[f"{col}_p75"] - feats[f"{col}_p25"]
        # rough dynamics:
        diff = np.diff(s.values, prepend=s.values[0])
        feats[f"{col}_rmsdiff"] = float(np.sqrt(np.mean(diff**2)))
        feats[f"{col}_zerox"]   = int(np.sum(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0))
    return feats

# Build per-sample summary features
feat_rows = [summarize_time_series(X[i]) for i in range(N)]
Xsum = pd.DataFrame(feat_rows)
if meta is not None:
    Xsum["website"] = meta["website"].values
    Xsum["task"]    = meta["task"].values
    Xsum["user"]    = meta["user"].values

# Numeric features only
num_cols = [c for c in Xsum.columns if c.startswith("ch")]
Xnum = Xsum[num_cols].values

# ---- Group CV setup ----
outer = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)

def eval_simple_model(model_name="logreg", downsample=False):
    accs, f1s, aucs = [], [], []
    for tr_idx, te_idx in outer.split(Xnum, Y, groups=G):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = Y[tr_idx], Y[te_idx]
        # Optionally downsample time (every 5th)
        if downsample:
            Xtr_ds = Xtr[:, ::5, :]  # ( ,300,8)
            Xte_ds = Xte[:, ::5, :]
        else:
            Xtr_ds, Xte_ds = Xtr, Xte

        # summarize again for consistency (works both ds/full)
        Xtr_sum = np.vstack([list(summarize_time_series(x).values()) for x in Xtr_ds])
        Xte_sum = np.vstack([list(summarize_time_series(x).values()) for x in Xte_ds])

        scaler = StandardScaler().fit(Xtr_sum)
        Xtr_s = scaler.transform(Xtr_sum)
        Xte_s = scaler.transform(Xte_sum)

        if model_name == "logreg":
            clf = LogisticRegression(max_iter=500, class_weight="balanced")
        elif model_name == "svm_rbf":
            clf = SVC(kernel="rbf", probability=True, class_weight="balanced")
        else:
            raise ValueError

        clf.fit(Xtr_s, ytr)
        pr  = clf.predict(Xte_s)
        prp = clf.predict_proba(Xte_s)[:,1]

        accs.append(accuracy_score(yte, pr))
        f1s.append(f1_score(yte, pr, zero_division=0))
        try:
            aucs.append(roc_auc_score(yte, prp))
        except:
            aucs.append(np.nan)

    return {
        "acc_mean": float(np.nanmean(accs)), "acc_std": float(np.nanstd(accs)),
        "f1_mean": float(np.nanmean(f1s)),   "f1_std": float(np.nanstd(f1s)),
        "auc_mean": float(np.nanmean(aucs)), "auc_std": float(np.nanstd(aucs)),
    }

# ---- 1) Simple baselines on summary features ----
res_logreg    = eval_simple_model("logreg", downsample=False)
res_svm       = eval_simple_model("svm_rbf", downsample=False)
res_logreg_ds = eval_simple_model("logreg", downsample=True)
res_svm_ds    = eval_simple_model("svm_rbf", downsample=True)

with open(os.path.join(RESULT_DIR, "01_simple_models.json"), "w") as f:
    json.dump({
        "logreg": res_logreg,
        "svm_rbf": res_svm,
        "logreg_downsample": res_logreg_ds,
        "svm_rbf_downsample": res_svm_ds
    }, f, indent=2)

print("[01] simple models:", res_logreg, res_svm, res_logreg_ds, res_svm_ds)

# ---- 2) Permutation test (are we above chance?) ----
def permutation_test(model_name="logreg", n_perm=200):
    rng = np.random.default_rng(123)
    # true score
    real = eval_simple_model(model_name, downsample=False)["f1_mean"]
    # shuffled scores
    sh = []
    for _ in range(n_perm):
        y_shuf = Y.copy()
        rng.shuffle(y_shuf)
        accs, f1s = [], []
        for tr_idx, te_idx in outer.split(Xnum, y_shuf, groups=G):
            Xtr_sum = Xnum[tr_idx]; Xte_sum = Xnum[te_idx]
            ytr = y_shuf[tr_idx]; yte = y_shuf[te_idx]
            scaler = StandardScaler().fit(Xtr_sum)
            Xtr_s = scaler.transform(Xtr_sum); Xte_s = scaler.transform(Xte_sum)
            if model_name=="logreg":
                clf = LogisticRegression(max_iter=500, class_weight="balanced")
            else:
                clf = SVC(kernel="rbf", probability=False, class_weight="balanced")
            clf.fit(Xtr_s, ytr)
            pr = clf.predict(Xte_s)
            f1s.append(f1_score(yte, pr, zero_division=0))
        sh.append(np.mean(f1s))
    pval = (np.sum(np.array(sh) >= real) + 1) / (n_perm + 1)
    return {"real_f1": float(real), "perm_mean": float(np.mean(sh)), "pval": float(pval)}

perm_logreg = permutation_test("logreg", n_perm=200)
with open(os.path.join(RESULT_DIR, "02_permutation_logreg.json"), "w") as f:
    json.dump(perm_logreg, f, indent=2)

print("[02] permutation (logreg):", perm_logreg)

# ---- 3) Website/task baselines ----
site_task_res = {}
if meta is not None:
    dfm = meta.copy()
    dfm["y"] = Y
    # majority by website
    site_major = dfm.groupby("website")["y"].agg(lambda s: int(np.mean(s)>=0.5)).to_dict()
    # majority by task
    task_major = dfm.groupby("task")["y"].agg(lambda s: int(np.mean(s)>=0.5)).to_dict()

    y_pred_site = dfm["website"].map(site_major).values
    y_pred_task = dfm["task"].map(task_major).values

    site_task_res = {
        "site_acc": float(accuracy_score(Y, y_pred_site)),
        "site_f1":  float(f1_score(Y, y_pred_site, zero_division=0)),
        "task_acc": float(accuracy_score(Y, y_pred_task)),
        "task_f1":  float(f1_score(Y, y_pred_task, zero_division=0)),
    }

    with open(os.path.join(RESULT_DIR, "03_site_task_baselines.json"), "w") as f:
        json.dump(site_task_res, f, indent=2)
    print("[03] site/task baselines:", site_task_res)
else:
    print("[03] meta file not found; skipping site/task baselines.")

# ---- 4) Quick LOUO (Leave-One-User-Out) on summary features ----
def eval_louo_logreg():
    users = np.unique(G)
    f1s = []
    for u in users:
        te_idx = np.where(G==u)[0]
        tr_idx = np.where(G!=u)[0]
        Xtr = Xnum[tr_idx]; Xte = Xnum[te_idx]
        ytr = Y[tr_idx];     yte = Y[te_idx]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(Xtr_s, ytr)
        pr = clf.predict(Xte_s)
        f1s.append(f1_score(yte, pr, zero_division=0))
    return {"louo_f1_mean": float(np.mean(f1s)), "louo_f1_std": float(np.std(f1s))}

louo_res = eval_louo_logreg()
with open(os.path.join(RESULT_DIR, "04_louo_logreg.json"), "w") as f:
    json.dump(louo_res, f, indent=2)
print("[04] LOUO:", louo_res)

# ---- 5) Save a short summary ----
summary = {
    "simple_models": {
        "logreg": res_logreg, "svm_rbf": res_svm,
        "logreg_downsample": res_logreg_ds, "svm_rbf_downsample": res_svm_ds
    },
    "permutation_logreg": perm_logreg,
    "site_task_baselines": site_task_res,
    "louo_logreg": louo_res
}
with open(os.path.join(RESULT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("[DONE] Results saved to:", RESULT_DIR)
