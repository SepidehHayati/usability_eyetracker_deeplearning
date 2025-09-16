# -*- coding: utf-8 -*-
# Step 1: Build G_clean.npy (users) from manifest_clean.csv

import os, numpy as np, pandas as pd

DATA_DIR = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data"
MANIFEST = os.path.join(DATA_DIR, "manifest_clean.csv")
G_PATH   = os.path.join(DATA_DIR, "G_clean.npy")
SPLITS   = os.path.join(DATA_DIR, "splits")
os.makedirs(SPLITS, exist_ok=True)

if not os.path.exists(MANIFEST):
    raise FileNotFoundError(f"Not found: {MANIFEST}")

mf = pd.read_csv(MANIFEST)
if "user" not in mf.columns:
    raise ValueError("manifest_clean.csv must contain a 'user' column.")

G = mf["user"].astype(np.int64).to_numpy()
np.save(G_PATH, G)

# sanity checks (اختیاری)
x_path = os.path.join(DATA_DIR, "X_clean.npy")
y_path = os.path.join(DATA_DIR, "Y_clean.npy")
if os.path.exists(x_path):
    X = np.load(x_path, mmap_mode="r")
    assert X.shape[0] == len(G), f"X N={X.shape[0]} != len(G)={len(G)}"
if os.path.exists(y_path):
    Y = np.load(y_path, mmap_mode="r")
    assert Y.shape[0] == len(G), f"Y N={Y.shape[0]} != len(G)={len(G)}"

print(f"[OK] Saved G_clean.npy → {G_PATH}  shape={G.shape}, unique_users={mf['user'].nunique()}")
print(f"[INFO] splits dir: {SPLITS}")
