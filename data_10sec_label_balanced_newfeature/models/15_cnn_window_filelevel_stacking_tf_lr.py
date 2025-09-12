# 15_cnn_window_filelevel_stacking_tf_lr.py
# ایده: CNN روی پنجره‌ها (مثل 14d) برای تولید احتمالِ پنجره → استخراج ویژگی‌های فایل‌سطح
# (آمارهای توزیعِ احتمال + site/task + (اختیاری) X_tabular) → LogisticRegression برای تصمیم فایل‌سطح.
#
# ورودی‌ها:
#   ../data/X8.npy, Y8.npy, G8.npy
#   ../data/meta_features_8cols.csv   (حاوی website, task برای هر فایل)
#   (اختیاری) ../data/X_tabular.npy  و X_tabular_cols.txt
#
# خروجی‌ها:
#   ../results/15_cnn_window_filelevel_stacking_tf_lr/metrics_file_level.csv, summary.txt

import os, math, random, json
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, BatchNormalization, Activation,
                                     MaxPooling1D, GlobalAveragePooling1D,
                                     Dense, Dropout)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ============ Paths ============
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "15_cnn_window_filelevel_stacking_tf_lr")
os.makedirs(RESULT_DIR, exist_ok=True)

# core arrays
X = np.load(os.path.join(DATA_DIR, "X8.npy"))   # (N,1500,8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy")).astype(int)
G = np.load(os.path.join(DATA_DIR, "G8.npy")).astype(int)
N, T, C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}", flush=True)

# meta (site/task) — ساخته‌شده در مرحله‌ی مپینگ
meta_path = os.path.join(DATA_DIR, "meta_features_8cols.csv")
meta_df = pd.read_csv(meta_path)  # columns: filename, website, task, user, label, ...
# نگاشت file-index → (site, task)
# فرض: ترتیب فایل‌ها در X/Y/G همان ترتیب meta_df است.
assert len(meta_df) == N, "meta_features_8cols.csv rows must match N"
sites = meta_df["website"].astype(str).str.lower().values
tasks = meta_df["task"].astype(int).values

# (اختیاری) tabular features (96d)
tab_path = os.path.join(DATA_DIR, "X_tabular.npy")
USE_TAB = os.path.exists(tab_path)
if USE_TAB:
    X_tab = np.load(tab_path)  # shape (N, 96)
    print("[INFO] Loaded X_tabular:", X_tab.shape, flush=True)

# ============ Config ============
WIN, STR        = 400, 200   # توصیه: طول بلندتر برای افزایش Acc
N_SPLITS        = 6
EPOCHS          = 80
BATCH_SIZE      = 16
LR_INIT         = 1e-3
L2_REG          = 1e-5
DROPOUT_BLOCK   = 0.30
DROPOUT_HEAD    = 0.35
RANDOM_STATE    = 42
SEEDS           = [7, 17, 23]
TOPK_K          = 5

np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE); random.seed(RANDOM_STATE)

# ============ Windowing ============
def make_windows(X, Y, G, win=WIN, stride=STR):
    Xw, Yw, Gw, Fw = [], [], [], []
    for i in range(X.shape[0]):
        xi = X[i]
        for s in range(0, T - win + 1, stride):
            Xw.append(xi[s:s+win, :]); Yw.append(Y[i]); Gw.append(G[i]); Fw.append(i)
    return (np.asarray(Xw, dtype=np.float32),
            np.asarray(Yw, dtype=np.int64),
            np.asarray(Gw, dtype=np.int64),
            np.asarray(Fw, dtype=np.int64))

Xw, Yw, Gw, Fw = make_windows(X, Y, G, WIN, STR)
print(f"[WINDOWS] Xw={Xw.shape}, Yw={Yw.shape}, Gw={Gw.shape}, Fw={Fw.shape}", flush=True)

# ============ Helpers ============
def standardize_per_fold(train_arr, *others):
    t_len, c = train_arr.shape[1], train_arr.shape[2]
    sc = StandardScaler(); sc.fit(train_arr.reshape(-1, c))
    out = []
    for arr in (train_arr,) + others:
        A2 = arr.reshape(-1, c); A2 = sc.transform(A2)
        out.append(A2.reshape(arr.shape[0], t_len, c))
    return out

def compute_class_weights(y):
    classes = np.unique(y)
    cnts = np.array([(y==c).sum() for c in classes], dtype=np.float32)
    n, k = len(y), len(classes)
    return {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, cnts)}

def build_cnn(input_shape):
    m = Sequential([
        tf.keras.Input(shape=input_shape),
        Conv1D(48, 7, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(2), Dropout(DROPOUT_BLOCK),

        Conv1D(64, 5, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(2), Dropout(DROPOUT_BLOCK),

        Conv1D(64, 3, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(2), Dropout(DROPOUT_BLOCK),

        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(DROPOUT_HEAD),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return m

def fileprob_stats(win_probs_for_files: dict):
    """از دیکشنری file_id->list(prob) چند ویژگی آماری می‌سازد."""
    rows = []
    for f, lst in win_probs_for_files.items():
        arr = np.array(lst, dtype=float)
        feats = {
            "file": int(f),
            "n_win": int(len(arr)),
            "p_mean": float(np.mean(arr)),
            "p_std":  float(np.std(arr)),
            "p_min":  float(np.min(arr)),
            "p_q25":  float(np.percentile(arr, 25)),
            "p_med":  float(np.median(arr)),
            "p_q75":  float(np.percentile(arr, 75)),
            "p_max":  float(np.max(arr)),
            "p_prop_gt_0p6": float(np.mean(arr > 0.6)),
            "p_prop_gt_0p7": float(np.mean(arr > 0.7)),
            "p_prop_gt_0p8": float(np.mean(arr > 0.8)),
        }
        rows.append(feats)
    return pd.DataFrame(rows).set_index("file").sort_index()

def collect_win_probs_by_file(probs, file_ids):
    from collections import defaultdict
    d = defaultdict(list)
    for p, f in zip(probs, file_ids):
        d[int(f)].append(float(p))
    return d

# ============ CV (GroupKFold by user, تصمیم فایل‌سطح با LR) ============
gkf = GroupKFold(n_splits=N_SPLITS)

rows = []
fold = 1

for tr_idx, te_idx in gkf.split(Xw, Yw, groups=Gw):
    print(f"\n===== Fold {fold} =====", flush=True)

    Xtr, Xte = Xw[tr_idx], Xw[te_idx]
    Ytr, Yte = Yw[tr_idx], Yw[te_idx]
    Ftr, Fte = Fw[tr_idx], Fw[te_idx]

    # Standardize per fold
    Xtr, Xte = standardize_per_fold(Xtr, Xte)

    # Class weights
    cw = compute_class_weights(Ytr)
    print("[CLASS WEIGHTS]", cw, flush=True)

    # Seed ensemble → window probs (برای تر/تست)
    te_win_probs_list = []
    tr_win_probs_list = []

    for seed in SEEDS:
        tf.keras.backend.clear_session()
        np.random.seed(seed); tf.random.set_seed(seed)

        model = build_cnn((Xtr.shape[1], Xtr.shape[2]))
        es  = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)

        # برای سادگی: 10% از تر برای ولیدیشن داخلی Keras (به‌خاطر EarlyStopping)
        model.fit(Xtr, Ytr,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  validation_split=0.1,
                  class_weight=cw,
                  verbose=0,
                  callbacks=[es, rlr])

        tr_win_probs_list.append(model.predict(Xtr, verbose=0).ravel())
        te_win_probs_list.append(model.predict(Xte, verbose=0).ravel())

    tr_win_probs = np.mean(tr_win_probs_list, axis=0)
    te_win_probs = np.mean(te_win_probs_list, axis=0)

    # جمع‌آوری به ازای فایل
    tr_dict = collect_win_probs_by_file(tr_win_probs, Ftr)
    te_dict = collect_win_probs_by_file(te_win_probs, Fte)

    # ساخت Featureهای فایل‌سطح
    tr_feats = fileprob_stats(tr_dict)
    te_feats = fileprob_stats(te_dict)

    # اضافه‌کردن site/task (one-hot)
    tr_sites = pd.Series({i: sites[i] for i in tr_feats.index}, name="site")
    te_sites = pd.Series({i: sites[i] for i in te_feats.index}, name="site")
    tr_tasks = pd.Series({i: tasks[i] for i in tr_feats.index}, name="task").astype(int)
    te_tasks = pd.Series({i: tasks[i] for i in te_feats.index}, name="task").astype(int)

    # OneHotEncoder روی تر
    enc_site = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc_task = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    site_tr_oh = enc_site.fit_transform(tr_sites.values.reshape(-1,1))
    task_tr_oh = enc_task.fit_transform(tr_tasks.values.reshape(-1,1))
    site_te_oh = enc_site.transform(te_sites.values.reshape(-1,1))
    task_te_oh = enc_task.transform(te_tasks.values.reshape(-1,1))

    # (اختیاری) افزودن X_tabular
    if USE_TAB:
        tr_tab = X_tab[tr_feats.index.values]  # انتخاب همان فایل‌ها
        te_tab = X_tab[te_feats.index.values]
    else:
        tr_tab = np.empty((len(tr_feats),0), dtype=float)
        te_tab = np.empty((len(te_feats),0), dtype=float)

    # ماتریس نهایی ویژگی‌ها
    tr_X = np.hstack([tr_feats.values, site_tr_oh, task_tr_oh, tr_tab])
    te_X = np.hstack([te_feats.values, site_te_oh, task_te_oh, te_tab])

    # استانداردسازی فقط روی ستون‌های پیوسته (اولین ستون‌های tr_feats و tabular)
    # site/task one-hot را دست‌نخورده می‌گذاریم.
    n_cont = tr_feats.shape[1] + tr_tab.shape[1]
    sc = StandardScaler().fit(tr_X[:, :n_cont])
    tr_X[:, :n_cont] = sc.transform(tr_X[:, :n_cont])
    te_X[:, :n_cont] = sc.transform(te_X[:, :n_cont])

    # برچسب فایل‌سطح
    tr_y = Y[tr_feats.index.values]
    te_y = Y[te_feats.index.values]

    # مدل فایل‌سطح: LogisticRegression
    clf = LogisticRegression(max_iter=2000)
    clf.fit(tr_X, tr_y)
    te_p = clf.predict_proba(te_X)[:,1]
    te_pred = (te_p > 0.5).astype(int)

    acc  = accuracy_score(te_y, te_pred)
    prec = precision_score(te_y, te_pred, zero_division=0)
    rec  = recall_score(te_y, te_pred, zero_division=0)
    f1   = f1_score(te_y, te_pred, zero_division=0)

    rows.append({"fold":fold, "acc":acc, "precision":prec, "recall":rec, "f1":f1,
                 "n_test_files": len(te_y)})

    print(f"[FOLD {fold}] File-level LR → Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")
    fold += 1

# ============ Summary ============
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_file_level.csv"), index=False)
mean = df.mean(numeric_only=True); std = df.std(numeric_only=True)

summary = [
    f"WIN={WIN}, STR={STR}, Seeds={SEEDS}, TopK={TOPK_K}, USE_TAB={USE_TAB}",
    "Mean ± STD (file-level, LR stacking)",
    f"Accuracy: {mean['acc']:.4f} ± {std['acc']:.4f}",
    f"Precision: {mean['precision']:.4f} ± {std['precision']:.4f}",
    f"Recall: {mean['recall']:.4f} ± {std['recall']:.4f}",
    f"F1: {mean['f1']:.4f} ± {std['f1']:.4f}",
]
with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary))

print("\n".join(summary), flush=True)
print("Saved to:", RESULT_DIR, flush=True)
