# 14f_cnn_window_ensemble_hybridrule_accopt_tf_v2.py
# اصلاح شده:
# - سه حالت هیبرید: fraction / count / quantile
# - بازه‌های نرم‌تر برای T_WIN و ALPHA/K/q
# - انتخاب بهترین حالت بر اساس Accuracy ولیدیشن
# - فول‌بک: اگر هیبرید < قاعده ساده بود، از همان قاعده ساده استفاده کن.

import os, math, random
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, BatchNormalization, Activation,
                                     MaxPooling1D, GlobalAveragePooling1D,
                                     Dense, Dropout)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ======================= Paths & I/O =======================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "14f_cnn_window_ensemble_hybridrule_accopt_tf_v2")
os.makedirs(RESULT_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X8.npy"))   # (N,1500,8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy")).astype(int)
G = np.load(os.path.join(DATA_DIR, "G8.npy")).astype(int)
N, T, C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}", flush=True)

# ======================= Config =======================
WIN, STR        = 500, 250           # همان بهترین‌های قبلی
N_SPLITS        = 6
EPOCHS          = 80
BATCH_SIZE      = 16
LR_INIT         = 1e-3
L2_REG          = 1e-5
DROPOUT_BLOCK   = 0.30
DROPOUT_HEAD    = 0.35
RANDOM_STATE    = 42
VAL_FILE_RATIO  = 0.40
THR_MIN, THR_MAX = 0.45, 0.85
THR_STEPS       = 41
SEEDS           = [7, 17, 23]
TOPK_K          = 5
NEG_BOOST       = 1.6
AGG_MODE        = "median"           # 'mean'/'median'/'trimmed'

# --- نرم‌تر: بازه‌های هیبرید ---
TWIN_GRID    = np.linspace(0.45, 0.65, 5)   # قبلاً 0.55..0.75
ALPHA_GRID   = np.array([0.15, 0.20, 0.25, 0.30, 0.35])  # سهم برای fraction
K_GRID       = np.array([1, 2])             # حداقل پنجره برای count-rule (با 5 پنجره منطقی)
Q_GRID       = np.array([0.70, 0.80, 0.90]) # صدک برای quantile-rule

np.random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE); random.seed(RANDOM_STATE)

# ======================= Windowing =======================
def make_windows(X, Y, G, win=WIN, stride=STR):
    Xw, Yw, Gw, Fw = [], [], [], []
    for i in range(X.shape[0]):
        xi = X[i]
        for s in range(0, T - win + 1, stride):
            Xw.append(xi[s:s+win, :])
            Yw.append(Y[i]); Gw.append(G[i]); Fw.append(i)
    return (np.asarray(Xw, dtype=np.float32),
            np.asarray(Yw, dtype=np.int64),
            np.asarray(Gw, dtype=np.int64),
            np.asarray(Fw, dtype=np.int64))

Xw, Yw, Gw, Fw = make_windows(X, Y, G, WIN, STR)
print(f"[WINDOWS] Xw={Xw.shape}, Yw={Yw.shape}, Gw={Gw.shape}, Fw={Fw.shape}", flush=True)

# ======================= Utils =======================
def standardize_per_fold(train_arr, *others):
    t_len, c = train_arr.shape[1], train_arr.shape[2]
    sc = StandardScaler()
    sc.fit(train_arr.reshape(-1, c))
    out = []
    for arr in (train_arr,) + others:
        A2 = arr.reshape(-1, c)
        A2 = sc.transform(A2)
        out.append(A2.reshape(arr.shape[0], t_len, c))
    return out

def compute_class_weights(y, neg_boost=1.0):
    classes = np.unique(y)
    cnts = np.array([(y==c).sum() for c in classes], dtype=np.float32)
    n, k = len(y), len(classes)
    w = {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, cnts)}
    if 0 in w:
        w[0] = w[0] * neg_boost
    return w

def agg_probs_file_level(probs, file_ids, mode=AGG_MODE):
    from collections import defaultdict
    d = defaultdict(list)
    for p, f in zip(probs, file_ids):
        d[int(f)].append(float(p))
    out = {}
    for f, ps in d.items():
        arr = np.array(ps, dtype=float)
        if mode == "median":
            out[f] = float(np.median(arr))
        elif mode == "trimmed":
            lo, hi = np.percentile(arr, [10, 90])
            trimmed = arr[(arr >= lo) & (arr <= hi)]
            out[f] = float(np.mean(trimmed)) if len(trimmed) else float(np.mean(arr))
        else:
            out[f] = float(np.mean(arr))
    return out

def choose_files_for_validation_stratified(train_file_ids, y_file_dict, ratio, seed=0):
    rng = np.random.default_rng(seed)
    files0 = [f for f in train_file_ids if y_file_dict[f]==0]
    files1 = [f for f in train_file_ids if y_file_dict[f]==1]
    n_val = max(1, int(math.ceil(len(train_file_ids)*ratio)))
    n1 = max(1, int(round(n_val * (len(files1)/(len(files0)+len(files1)+1e-9)))))
    n0 = max(1, n_val - n1)
    val0 = rng.choice(files0, size=min(n0,len(files0)), replace=False).tolist() if files0 else []
    val1 = rng.choice(files1, size=min(n1,len(files1)), replace=False).tolist() if files1 else []
    val = set(val0 + val1)
    remaining = [f for f in train_file_ids if f not in val]
    while len(val) < n_val and remaining:
        val.add(remaining.pop())
    return val

def choose_threshold_for_accuracy(val_file_probs, val_file_true, tmin=THR_MIN, tmax=THR_MAX, steps=THR_STEPS):
    grid = np.linspace(tmin, tmax, steps)
    files = list(val_file_probs.keys())
    y_true = np.array([val_file_true[f] for f in files], dtype=int)
    p_vec  = np.array([val_file_probs[f] for f in files], dtype=float)
    best_t, best_acc = 0.5, -1.0
    for t in grid:
        y_hat = (p_vec > t).astype(int)
        acc = accuracy_score(y_true, y_hat)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc

# --------- Hybrid rules ----------
def decide_fraction(win_probs, thr_file, t_win, alpha_min, agg_mode=AGG_MODE):
    arr = np.array(win_probs, dtype=float)
    agg = np.median(arr) if agg_mode=="median" else (np.mean(arr[(arr>=np.percentile(arr,10))&(arr<=np.percentile(arr,90))]) if agg_mode=="trimmed" and len(arr)>0 else np.mean(arr))
    frac = np.mean(arr > t_win) if len(arr) else 0.0
    return int((agg > thr_file) and (frac >= alpha_min))

def decide_count(win_probs, thr_file, t_win, k_min, agg_mode=AGG_MODE):
    arr = np.array(win_probs, dtype=float)
    agg = np.median(arr) if agg_mode=="median" else (np.mean(arr[(arr>=np.percentile(arr,10))&(arr<=np.percentile(arr,90))]) if agg_mode=="trimmed" and len(arr)>0 else np.mean(arr))
    cnt = int(np.sum(arr > t_win))
    return int((agg > thr_file) and (cnt >= k_min))

def decide_quantile(win_probs, thr_file, t_win, q, agg_mode=AGG_MODE):
    arr = np.array(win_probs, dtype=float)
    agg = np.median(arr) if agg_mode=="median" else (np.mean(arr[(arr>=np.percentile(arr,10))&(arr<=np.percentile(arr,90))]) if agg_mode=="trimmed" and len(arr)>0 else np.mean(arr))
    qv = float(np.quantile(arr, q)) if len(arr) else 0.0
    return int((agg > thr_file) and (qv > t_win))

def grid_search_hybrid_on_val(win_probs_val, files_val, y_file, thr_file):
    from collections import defaultdict
    per_file = defaultdict(list)
    for p, f in zip(win_probs_val, files_val):
        per_file[int(f)].append(float(p))
    files = sorted(per_file.keys())
    y_true = np.array([y_file[f] for f in files], dtype=int)

    best = {"acc": -1.0, "mode": "fraction", "param1": None, "param2": None}
    # fraction
    for tw in TWIN_GRID:
        for a in ALPHA_GRID:
            preds = [decide_fraction(per_file[f], thr_file, tw, a) for f in files]
            acc = accuracy_score(y_true, preds)
            if acc > best["acc"]:
                best = {"acc": acc, "mode": "fraction", "param1": float(tw), "param2": float(a)}
    # count
    for tw in TWIN_GRID:
        for k in K_GRID:
            preds = [decide_count(per_file[f], thr_file, tw, int(k)) for f in files]
            acc = accuracy_score(y_true, preds)
            if acc > best["acc"]:
                best = {"acc": acc, "mode": "count", "param1": float(tw), "param2": float(k)}
    # quantile
    for tw in TWIN_GRID:
        for q in Q_GRID:
            preds = [decide_quantile(per_file[f], thr_file, tw, float(q)) for f in files]
            acc = accuracy_score(y_true, preds)
            if acc > best["acc"]:
                best = {"acc": acc, "mode": "quantile", "param1": float(tw), "param2": float(q)}
    return best

class TopKSaver(tf.keras.callbacks.Callback):
    def __init__(self, model, k=TOPK_K):
        super().__init__()
        self.model_ref = model
        self.k = k
        self.snaps = []
    def on_epoch_end(self, epoch, logs=None):
        vloss = float(logs.get('val_loss', np.inf))
        self.snaps.append((vloss, [w.copy() for w in self.model_ref.get_weights()]))
    def set_topk_weights(self):
        if not self.snaps: return
        snaps_sorted = sorted(self.snaps, key=lambda x: x[0])
        top = snaps_sorted[:min(self.k, len(self.snaps))]
        avg = [np.mean([w[i] for (_vl, w) in top], axis=0) for i in range(len(top[0][1]))]
        self.model_ref.set_weights(avg)

def build_cnn(input_shape):
    model = Sequential([
        tf.keras.Input(shape=input_shape),
        Conv1D(48, kernel_size=7, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(pool_size=2), Dropout(DROPOUT_BLOCK),
        Conv1D(64, kernel_size=5, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(pool_size=2), Dropout(DROPOUT_BLOCK),
        Conv1D(64, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(L2_REG)),
        BatchNormalization(), Activation('relu'),
        MaxPooling1D(pool_size=2), Dropout(DROPOUT_BLOCK),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(DROPOUT_HEAD),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(LR_INIT), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ======================= GroupKFold =======================
gkf = GroupKFold(n_splits=N_SPLITS)
rows = []
fold = 1

for tr_idx, te_idx in gkf.split(Xw, Yw, groups=Gw):
    print(f"\n===== Fold {fold} =====", flush=True)

    Xtr, Xte = Xw[tr_idx], Xw[te_idx]
    Ytr, Yte = Yw[tr_idx], Yw[te_idx]
    Ftr, Fte = Fw[tr_idx], Fw[te_idx]

    train_files = np.unique(Ftr)
    y_file = {int(fi): int(Y[fi]) for fi in train_files}

    # stratified validation at file-level
    val_files = choose_files_for_validation_stratified(train_files, y_file, VAL_FILE_RATIO,
                                                       seed=RANDOM_STATE + fold)
    tr_mask = np.array([f not in val_files for f in Ftr])
    va_mask = np.array([f in  val_files for f in Ftr])

    Xtr_in, Ytr_in, Ftr_in = Xtr[tr_mask], Ytr[tr_mask], Ftr[tr_mask]
    Xva_in, Yva_in, Fva_in = Xtr[va_mask], Ytr[va_mask], Ftr[va_mask]

    # standardize
    Xtr_in, Xva_in, Xte = standardize_per_fold(Xtr_in, Xva_in, Xte)

    # class weights (+NEG_BOOST)
    cw = compute_class_weights(Ytr_in, neg_boost=NEG_BOOST)
    print("[CLASS WEIGHTS]", cw, flush=True)

    # seed-ensemble + Top-K
    val_win_probs_list, te_win_probs_list = [], []
    for seed in SEEDS:
        tf.keras.backend.clear_session()
        np.random.seed(seed); tf.random.set_seed(seed)

        model = build_cnn((Xtr_in.shape[1], Xtr_in.shape[2]))
        es  = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=0)
        topk = TopKSaver(model, k=TOPK_K)

        model.fit(Xtr_in, Ytr_in,
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(Xva_in, Yva_in),
                  class_weight=cw, verbose=0,
                  callbacks=[es, rlr, topk])

        topk.set_topk_weights()
        val_win_probs_list.append(model.predict(Xva_in, verbose=0).ravel())
        te_win_probs_list.append(model.predict(Xte,   verbose=0).ravel())

    # average probs at window-level
    val_win_probs = np.mean(val_win_probs_list, axis=0)
    te_win_probs  = np.mean(te_win_probs_list,  axis=0)

    # classic file-level threshold (baseline)
    val_file_probs = agg_probs_file_level(val_win_probs, Fva_in, mode=AGG_MODE)
    te_file_probs  = agg_probs_file_level(te_win_probs,  Fte,   mode=AGG_MODE)
    val_file_true  = {int(fi): int(Y[fi]) for fi in val_file_probs.keys()}

    thr_file, acc_val = choose_threshold_for_accuracy(val_file_probs, val_file_true,
                                                      tmin=THR_MIN, tmax=THR_MAX, steps=THR_STEPS)
    print(f"[FOLD {fold}] thr_file={thr_file:.3f} | Acc(val-files) baseline={acc_val:.3f}")

    # grid-search hybrid (pick best mode & params on validation)
    best_h = grid_search_hybrid_on_val(val_win_probs, Fva_in, val_file_true, thr_file)
    print(f"[FOLD {fold}] hybrid best: mode={best_h['mode']} acc={best_h['acc']:.3f} "
          f"param1={best_h['param1']:.2f} param2={best_h['param2']:.2f}")

    # prepare per-file windows for test
    from collections import defaultdict
    per_file_te = defaultdict(list)
    for p, f in zip(te_win_probs, Fte):
        per_file_te[int(f)].append(float(p))
    te_files_sorted = sorted(per_file_te.keys())
    y_true = np.array([int(Y[f]) for f in te_files_sorted], dtype=int)

    # decide function chosen
    def predict_file(win_probs):
        if best_h["acc"] >= acc_val:  # only use hybrid if it beats baseline on val
            if best_h["mode"] == "fraction":
                return decide_fraction(win_probs, thr_file, best_h['param1'], best_h['param2'])
            elif best_h["mode"] == "count":
                return decide_count(win_probs, thr_file, best_h['param1'], int(best_h['param2']))
            else:
                return decide_quantile(win_probs, thr_file, best_h['param1'], best_h['param2'])
        else:
            # fallback to simple file threshold
            arr = np.array(win_probs, dtype=float)
            agg = np.median(arr) if AGG_MODE=="median" else (np.mean(arr[(arr>=np.percentile(arr,10))&(arr<=np.percentile(arr,90))]) if AGG_MODE=="trimmed" and len(arr)>0 else np.mean(arr))
            return int(agg > thr_file)

    y_pred = [predict_file(per_file_te[f]) for f in te_files_sorted]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    rows.append({"fold":fold,
                 "thr_file": float(thr_file),
                 "hybrid_mode": best_h["mode"],
                 "param1": float(best_h["param1"]),
                 "param2": float(best_h["param2"]),
                 "acc_val_baseline": float(acc_val),
                 "acc_val_hybrid": float(best_h["acc"]),
                 "acc":acc,"precision":prec,"recall":rec,"f1":f1})

    fold += 1

# ======================= Summary & Save =======================
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_file_level.csv"), index=False)

mean = df.mean(numeric_only=True); std = df.std(numeric_only=True)
summary = [
    f"WIN={WIN}, STR={STR}, Seeds={SEEDS}, TopK={TOPK_K}, NEG_BOOST={NEG_BOOST}, AGG_MODE={AGG_MODE}",
    "Mean ± STD (file-level, hybrid with fallback)",
    f"Accuracy: {mean['acc']:.4f} ± {std['acc']:.4f}",
    f"Precision: {mean['precision']:.4f} ± {std['precision']:.4f}",
    f"Recall: {mean['recall']:.4f} ± {std['recall']:.4f}",
    f"F1: {mean['f1']:.4f} ± {std['f1']:.4f}",
]
with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary))

print("\n".join(summary), flush=True)
print("Saved to:", RESULT_DIR, flush=True)
