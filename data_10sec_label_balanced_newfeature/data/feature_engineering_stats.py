# 10_feature_engineering_stats.py
# خروجی: data/X_tabular.npy, data/X_tabular_cols.txt, data/X_tabular_usernorm.npy

import os, numpy as np, pandas as pd

DATA_DIR = os.path.join("..","data")
meta_csv = os.path.join(DATA_DIR,"meta_features_8cols.csv")

X = np.load(os.path.join(DATA_DIR,"X8.npy"))      # (N,1500,8) deltas
Y = np.load(os.path.join(DATA_DIR,"Y8.npy"))
G = np.load(os.path.join(DATA_DIR,"G8.npy"))
meta = pd.read_csv(meta_csv) if os.path.exists(meta_csv) else None
N,T,C = X.shape

def summarize(arr):  # (T,C)
    out = {}
    for c in range(arr.shape[1]):
        s = arr[:,c]
        diff = np.diff(s, prepend=s[0])
        out.update({
            f"ch{c}_mean": float(s.mean()),
            f"ch{c}_std":  float(s.std(ddof=1) if len(s)>1 else 0.0),
            f"ch{c}_min":  float(s.min()),
            f"ch{c}_max":  float(s.max()),
            f"ch{c}_med":  float(np.median(s)),
            f"ch{c}_p25":  float(np.percentile(s,25)),
            f"ch{c}_p75":  float(np.percentile(s,75)),
            f"ch{c}_iqr":  float(np.percentile(s,75)-np.percentile(s,25)),
            f"ch{c}_rmsd": float(np.sqrt(np.mean(diff**2))),
            f"ch{c}_zxc":  int(np.sum(np.sign(diff[:-1])*np.sign(diff[1:])<0)),
            f"ch{c}_mabs": float(np.mean(np.abs(s))),
            f"ch{c}_mabsdiff": float(np.mean(np.abs(diff))),
        })
    return out

# 1) Tabular روی دیتای فعلی
rows = [summarize(X[i]) for i in range(N)]
Xtab = pd.DataFrame(rows)
cols = list(Xtab.columns)
np.save(os.path.join(DATA_DIR,"X_tabular.npy"), Xtab.values.astype(np.float32))
with open(os.path.join(DATA_DIR,"X_tabular_cols.txt"),"w",encoding="utf-8") as f: f.write("\n".join(cols))

# 2) نسخه‌ی نرمال‌سازی درون-کاربری (per-user z-score در زمان، سپس خلاصه)
X_usernorm = np.empty_like(X)
for u in np.unique(G):
    idx = np.where(G==u)[0]
    for i in idx:
        # z-score هر کانال روی همان 10s
        x = X[i]
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True) + 1e-8
        X_usernorm[i] = (x - mu)/sd

rowsN = [summarize(X_usernorm[i]) for i in range(N)]
XtabN = pd.DataFrame(rowsN)[cols]  # همان ترتیب ستون‌ها
np.save(os.path.join(DATA_DIR,"X_tabular_usernorm.npy"), XtabN.values.astype(np.float32))

print("[OK] Saved:",
      "X_tabular.npy, X_tabular_usernorm.npy, X_tabular_cols.txt",
      f"shape={Xtab.shape}")
