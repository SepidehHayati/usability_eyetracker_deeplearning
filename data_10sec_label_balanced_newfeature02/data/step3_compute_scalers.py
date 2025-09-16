# -*- coding: utf-8 -*-
# Step 3: Compute per-fold Z-Score scalers (train-only) for each of 16 channels

import os
import numpy as np
import pandas as pd

DATA_DIR   = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data"
X_PATH     = os.path.join(DATA_DIR, "X_clean.npy")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)

# ترتیب کانال‌ها باید دقیقاً با فایل‌های cleaned یکی باشد:
BASE = ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR']
ALL  = BASE + [f"delta_{c}" for c in BASE]  # 16 کانال
CHANNELS_TXT = os.path.join(DATA_DIR, "channels_names.txt")
with open(CHANNELS_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(ALL))
print(f"[INFO] channels saved → {CHANNELS_TXT}")

# بارگذاری داده (mmap برای مصرف رم کمتر)
if not os.path.exists(X_PATH):
    raise FileNotFoundError(f"Not found: {X_PATH}")
X = np.load(X_PATH, mmap_mode="r")  # shape (N, 1500, 16)
N, T, C = X.shape
assert (T, C) == (1500, 16), f"Unexpected shape: {X.shape}"

# پیدا کردن فایل‌های ایندکس فولدها
def load_idx(path):
    return np.load(path).astype(np.int64)

fold_files = []
for k in range(1, 6):
    tr_p = os.path.join(SPLITS_DIR, f"fold{k}_train_idx.npy")
    va_p = os.path.join(SPLITS_DIR, f"fold{k}_val_idx.npy")
    if os.path.exists(tr_p) and os.path.exists(va_p):
        fold_files.append((k, tr_p, va_p))
if not fold_files:
    raise RuntimeError(f"No folds found in {SPLITS_DIR}. Run step2_make_splits.py first.")

EPS = 1e-6  # برای جلوگیری از تقسیم بر صفر

for k, tr_p, va_p in fold_files:
    train_idx = load_idx(tr_p)
    # val_idx = load_idx(va_p)  # نیازی بهش نیست برای محاسبه‌ی اسکیل

    # داده‌ی Train این فولد
    Xt = X[train_idx]   # shape (~85, 1500, 16) ≈ چند مگابایت

    # میانگین و انحراف معیار برای هر کانال، روی محور (نمونه × زمان)
    mean = Xt.mean(axis=(0, 1))              # shape (16,)
    std  = Xt.std(axis=(0, 1))               # shape (16,)
    std  = np.where(std < EPS, EPS, std)     # جلوگیری از صفر

    # چک‌های پایه
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError(f"Non-finite scaler stats in fold {k}")

    # ذخیره‌ی اسکیلر این فولد
    np.save(os.path.join(SPLITS_DIR, f"fold{k}_mean.npy"), mean.astype(np.float32))
    np.save(os.path.join(SPLITS_DIR, f"fold{k}_std.npy"),  std.astype(np.float32))
    print(f"[fold {k}] saved mean/std → splits/fold{k}_mean.npy, fold{k}_std.npy")

    # --- sanity check: آیا روی Train همین فولد، Z-Score باعث میانگین≈0 و std≈1 می‌شود؟
    # برای سرعت، به جای ساخت آرایه‌ی کامل، آمار را با فرمول محاسبه می‌کنیم:
    # mean((X - m)/s) = 0  و  std((X - m)/s) = 1  در حالت ایده‌آل؛
    # به خاطر ممیز شناور، کمی خطا داریم.
    Xt_reshaped = Xt.reshape(-1, C)              # ((N_train*T), 16)
    # میانگین و std پس از نرمال‌سازی (کانال‌به‌کانال)
    norm_mean = (Xt_reshaped - mean) / std
    m2 = norm_mean.mean(axis=0)
    s2 = norm_mean.std(axis=0)

    # گزارش CSV کوچک برای هر فولد
    rep = pd.DataFrame({
        "channel": ALL,
        "train_mean": mean,
        "train_std": std,
        "post_norm_train_mean": m2,
        "post_norm_train_std": s2
    })
    rep_path = os.path.join(SPLITS_DIR, f"fold{k}_scaler_summary.csv")
    rep.to_csv(rep_path, index=False)
    print(f"[fold {k}] summary → {rep_path}  (should be near mean≈0, std≈1 on train)")

print("[OK] scaler stats computed and saved for all folds.")
