# -*- coding: utf-8 -*-
# Step 2: Build GroupKFold splits (by user) and save indices

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DATA_DIR   = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data"
MANIFEST   = os.path.join(DATA_DIR, "manifest_clean.csv")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)

# ---- load manifest (we need 'user' and 'label' aligned with X/Y order)
mf = pd.read_csv(MANIFEST)
if not {"user","label"}.issubset(mf.columns):
    raise ValueError("manifest_clean.csv must contain 'user' and 'label' columns.")

n = len(mf)
groups = mf["user"].astype(int).to_numpy()
y      = mf["label"].astype(int).to_numpy()

# (اختیاری) هم‌ترازی با X/Y را چک کن
x_path = os.path.join(DATA_DIR, "X_clean.npy")
y_path = os.path.join(DATA_DIR, "Y_clean.npy")
if os.path.exists(x_path):
    X = np.load(x_path, mmap_mode="r")
    assert X.shape[0] == n, f"X N={X.shape[0]} != manifest N={n}"
if os.path.exists(y_path):
    Y = np.load(y_path, mmap_mode="r")
    assert Y.shape[0] == n, f"Y N={Y.shape[0]} != manifest N={n}"

# ---- build 5-fold GroupKFold (each user appears in exactly one fold's val)
n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

summary_rows = []
for k, (tr_idx, va_idx) in enumerate(gkf.split(np.zeros((n,1)), y, groups), start=1):
    # save indices
    np.save(os.path.join(SPLITS_DIR, f"fold{k}_train_idx.npy"), tr_idx)
    np.save(os.path.join(SPLITS_DIR, f"fold{k}_val_idx.npy"), va_idx)

    # quick label balance
    tr_y = y[tr_idx]
    va_y = y[va_idx]
    tr_pos, tr_neg = int((tr_y==1).sum()), int((tr_y==0).sum())
    va_pos, va_neg = int((va_y==1).sum()), int((va_y==0).sum())

    print(f"[fold {k}] train={len(tr_idx)} (pos={tr_pos}, neg={tr_neg}) | "
          f"val={len(va_idx)} (pos={va_pos}, neg={va_neg}) | "
          f"val_users={sorted(pd.unique(groups[va_idx]).tolist())}")

    summary_rows.append({
        "fold": k,
        "n_train": len(tr_idx), "train_pos": tr_pos, "train_neg": tr_neg,
        "n_val": len(va_idx),   "val_pos": va_pos,   "val_neg": va_neg,
        "n_val_users": len(pd.unique(groups[va_idx]))
    })

# save summary
pd.DataFrame(summary_rows).to_csv(os.path.join(SPLITS_DIR, "splits_summary.csv"), index=False)
print(f"[OK] saved indices in {SPLITS_DIR} and summary.csv")
