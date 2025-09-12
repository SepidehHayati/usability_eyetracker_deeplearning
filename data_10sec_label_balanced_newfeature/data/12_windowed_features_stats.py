# 12_windowed_features_stats.py
# از X8.npy (N,1500,8) پنجره می‌سازد و از هر پنجره ویژگی آماری می‌گیرد.
# خروجی:
#   X_win_tab.npy (Σ_i W_i, F)
#   y_win.npy     (Σ_i W_i,)
#   g_win.npy     (Σ_i W_i,)  گروه کاربر
#   file_idx.npy  (Σ_i W_i,)  ایندکس فایل مادر برای انسامبل سطح فایل
#   cols_win.txt  لیست نام ستون‌ها

import os, numpy as np, pandas as pd

DATA_DIR = os.path.join("..","data")
WIN_DIR  = os.path.join("..","data","windowed")
os.makedirs(WIN_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR,"X8.npy"))      # (N,1500,8)
Y = np.load(os.path.join(DATA_DIR,"Y8.npy")).astype(int)
G = np.load(os.path.join(DATA_DIR,"G8.npy")).astype(int)
N,T,C = X.shape

WIN = 300
STR = 150

def summarize(arr):  # (t,c)
    out = {}
    for c in range(arr.shape[1]):
        s = arr[:,c]
        d = np.diff(s, prepend=s[0])
        out.update({
            f"ch{c}_mean": float(s.mean()),
            f"ch{c}_std":  float(s.std(ddof=1) if len(s)>1 else 0.0),
            f"ch{c}_min":  float(s.min()),
            f"ch{c}_max":  float(s.max()),
            f"ch{c}_med":  float(np.median(s)),
            f"ch{c}_p25":  float(np.percentile(s,25)),
            f"ch{c}_p75":  float(np.percentile(s,75)),
            f"ch{c}_iqr":  float(np.percentile(s,75)-np.percentile(s,25)),
            f"ch{c}_rmsd": float(np.sqrt(np.mean(d**2))),
            f"ch{c}_zxc":  int(np.sum(np.sign(d[:-1])*np.sign(d[1:])<0)),
            f"ch{c}_mabs": float(np.mean(np.abs(s))),
            f"ch{c}_mabsdiff": float(np.mean(np.abs(d))),
        })
    return out

feat_rows, y_rows, g_rows, file_rows = [], [], [], []

for i in range(N):
    x = X[i]
    for start in range(0, T - WIN + 1, STR):
        sl = x[start:start+WIN, :]
        feats = summarize(sl)
        feat_rows.append(feats)
        y_rows.append(Y[i])
        g_rows.append(G[i])
        file_rows.append(i)

Xwin = pd.DataFrame(feat_rows)
cols = list(Xwin.columns)

np.save(os.path.join(WIN_DIR,"X_win_tab.npy"), Xwin.values.astype(np.float32))
np.save(os.path.join(WIN_DIR,"y_win.npy"),      np.array(y_rows, dtype=np.int64))
np.save(os.path.join(WIN_DIR,"g_win.npy"),      np.array(g_rows, dtype=np.int64))
np.save(os.path.join(WIN_DIR,"file_idx.npy"),   np.array(file_rows, dtype=np.int64))
with open(os.path.join(WIN_DIR,"cols_win.txt"),"w",encoding="utf-8") as f: f.write("\n".join(cols))

print("[OK] Saved windowed features:",
      Xwin.shape, "to", WIN_DIR)
