# 14b_cnn_window_ensemble_topk_thrconstraint_tf.py
# بهبودهای نسبت به v14:
# - Top-K snapshot averaging
# - انتخاب آستانه با قید حداقل Precision فایل‌سطح (قابل تنظیم)

import os, json, math, random
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, BatchNormalization, Activation,
                                     MaxPooling1D, GlobalAveragePooling1D,
                                     Dense, Dropout)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ============== Paths & I/O ==============
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "14b_cnn_window_ensemble_topk_thrconstraint_tf")
os.makedirs(RESULT_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X8.npy"))   # (N, 1500, 8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy")).astype(int)
G = np.load(os.path.join(DATA_DIR, "G8.npy")).astype(int)
N, T, C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}", flush=True)

# ============== Config ==============
WIN, STR        = 300, 150
N_SPLITS        = 6
EPOCHS          = 80
BATCH_SIZE      = 16
LR_INIT         = 1e-3
L2_REG          = 1e-5
DROPOUT_BLOCK   = 0.30
DROPOUT_HEAD    = 0.35
RANDOM_STATE    = 42
VAL_FILE_RATIO  = 0.2
THR_MIN, THR_MAX = 0.30, 0.70
THR_STEPS       = 41
SEEDS           = [7, 17, 23]
TOPK_K          = 5         # اسنپ‌شات‌های برتر
PREC_MIN        = 0.70      # حداقل Precision فایل‌سطح؛ می‌تونی مثلاً 0.65/0.75 تست کنی

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ============== Windowing ==============
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

# ============== Utils ==============
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

def compute_class_weights(y):
    classes = np.unique(y)
    cnts = np.array([(y==c).sum() for c in classes], dtype=np.float32)
    n, k = len(y), len(classes)
    return {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, cnts)}

def agg_probs_file_level(probs, file_ids):
    from collections import defaultdict
    d = defaultdict(list)
    for p, f in zip(probs, file_ids):
        d[int(f)].append(float(p))
    return {f: float(np.mean(ps)) for f, ps in d.items()}

def choose_threshold_with_precision(val_file_probs, val_file_true,
                                    tmin=THR_MIN, tmax=THR_MAX, steps=THR_STEPS,
                                    prec_min=PREC_MIN):
    grid = np.linspace(tmin, tmax, steps)
    y_true = np.array([val_file_true[f] for f in val_file_probs.keys()], dtype=int)
    p_vec  = np.array([val_file_probs[f] for f in val_file_probs.keys()], dtype=float)

    best_t, best_f1 = 0.5, -1.0
    fallback_t, fallback_f1 = 0.5, -1.0
    for t in grid:
        y_hat = (p_vec > t).astype(int)
        f1  = f1_score(y_true, y_hat, zero_division=0)
        prec= precision_score(y_true, y_hat, zero_division=0)
        if f1 > fallback_f1:
            fallback_f1, fallback_t = f1, t
        if prec >= prec_min and f1 > best_f1:
            best_f1, best_t = f1, t
    if best_f1 < 0:
        return fallback_t, fallback_f1, False
    return best_t, best_f1, True

class TopKSaver(tf.keras.callbacks.Callback):
    def __init__(self, model, k=TOPK_K):
        super().__init__()
        self.model_ref = model
        self.k = k
        self.snaps = []  # (val_loss, weights)

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

# ============== GroupKFold (user) ==============
gkf = GroupKFold(n_splits=N_SPLITS)

rows = []
fold = 1

for tr_idx, te_idx in gkf.split(Xw, Yw, groups=Gw):
    print(f"\n===== Fold {fold} =====", flush=True)

    Xtr, Xte = Xw[tr_idx], Xw[te_idx]
    Ytr, Yte = Yw[tr_idx], Yw[te_idx]
    Ftr, Fte = Fw[tr_idx], Fw[te_idx]

    # فایل‌های ولیدیشن داخل ترین برای آستانه
    uniq_tr_files = np.unique(Ftr)
    rng = np.random.default_rng(RANDOM_STATE + fold)
    n_val_files = max(1, int(math.ceil(len(uniq_tr_files) * VAL_FILE_RATIO)))
    val_files = set(rng.choice(uniq_tr_files, size=n_val_files, replace=False).tolist())

    tr_mask = np.array([f not in val_files for f in Ftr])
    va_mask = np.array([f in  val_files for f in Ftr])

    Xtr_in, Ytr_in, Ftr_in = Xtr[tr_mask], Ytr[tr_mask], Ftr[tr_mask]
    Xva_in, Yva_in, Fva_in = Xtr[va_mask], Ytr[va_mask], Ftr[va_mask]

    # نرمال‌سازی per-fold
    Xtr_in, Xva_in, Xte = standardize_per_fold(Xtr_in, Xva_in, Xte)

    # class weights
    cw = compute_class_weights(Ytr_in)
    print("[CLASS WEIGHTS]", cw, flush=True)

    # Seed-ensemble + Top-K
    val_win_probs_list, te_win_probs_list = [], []

    for seed in SEEDS:
        tf.keras.backend.clear_session()
        np.random.seed(seed); tf.random.set_seed(seed)

        model = build_cnn((Xtr_in.shape[1], Xtr_in.shape[2]))
        es  = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=0)
        topk = TopKSaver(model, k=TOPK_K)

        model.fit(Xtr_in, Ytr_in,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  validation_data=(Xva_in, Yva_in),
                  class_weight=cw,
                  verbose=0,
                  callbacks=[es, rlr, topk])

        topk.set_topk_weights()

        val_win_probs_list.append(model.predict(Xva_in, verbose=0).ravel())
        te_win_probs_list.append(model.predict(Xte,   verbose=0).ravel())

    val_win_probs = np.mean(val_win_probs_list, axis=0)
    te_win_probs  = np.mean(te_win_probs_list,  axis=0)

    # تجمیع به سطح فایل
    val_file_probs = agg_probs_file_level(val_win_probs, Fva_in)
    te_file_probs  = agg_probs_file_level(te_win_probs,  Fte)

    val_file_true = {int(fi): int(Y[fi]) for fi in val_file_probs.keys()}
    te_file_true  = {int(fi): int(Y[fi]) for fi in te_file_probs.keys()}

    # انتخاب آستانه با قید precision
    thr, f1_val, met_prec = choose_threshold_with_precision(val_file_probs, val_file_true,
                                                            prec_min=PREC_MIN)
    tag = "with_prec" if met_prec else "fallback_f1max"
    print(f"[FOLD {fold}] thr={thr:.3f} ({tag}) | F1(val-files)={f1_val:.4f}")

    # ارزیابی فایل‌سطح
    te_files_sorted = sorted(te_file_probs.keys())
    y_true = np.array([te_file_true[f]  for f in te_files_sorted], dtype=int)
    p_mean = np.array([te_file_probs[f] for f in te_files_sorted], dtype=float)
    y_pred = (p_mean > thr).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    rows.append({"fold":fold,"thr":thr,"tag":tag,
                 "acc":acc,"precision":prec,"recall":rec,"f1":f1,
                 "n_test_files":len(te_files_sorted)})
    fold += 1

# Summary
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_file_level.csv"), index=False)

mean = df.mean(numeric_only=True); std = df.std(numeric_only=True)
summary = [
    f"WIN={WIN}, STR={STR}, Seeds={SEEDS}, TopK={TOPK_K}, PREC_MIN={PREC_MIN}",
    "File-level thresholds: " + ", ".join([f"{t:.3f}" for t in df['thr'].tolist()]),
    "Mean ± STD (file-level)",
    f"Accuracy: {mean['acc']:.4f} ± {std['acc']:.4f}",
    f"Precision: {mean['precision']:.4f} ± {std['precision']:.4f}",
    f"Recall: {mean['recall']:.4f} ± {std['recall']:.4f}",
    f"F1: {mean['f1']:.4f} ± {std['f1']:.4f}",
]
with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary))

print("\n".join(summary), flush=True)
print("Saved to:", RESULT_DIR, flush=True)
