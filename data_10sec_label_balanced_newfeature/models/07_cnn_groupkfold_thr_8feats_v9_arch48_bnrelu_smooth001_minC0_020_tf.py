# 07_cnn_groupkfold_thr_8feats_v9_arch48_bnrelu_smooth001_minC0_020_tf.py
# CNN + StratifiedGroupKFold + per-fold thresholding (constrained) + Seed-ensemble + Top-K snapshot averaging
# Changes vs v8:
# - SEEDS=[7,17,23,31,57], TOPK_CHECKPOINTS=7
# - LABEL_SMOOTH=0.01
# - MIN_C0_RATIO=0.20
# - Threshold grid clip [0.30, 0.70]
# - Conv -> BN -> ReLU order (no activation in Conv layers)

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, GlobalAveragePooling1D,
                                     Dense, Dropout, BatchNormalization, Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# Paths & I/O
# ===========================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "07_cnn_groupkfold_thr_8feats_v9_arch48_bnrelu_smooth001_minC0_020_tf")
os.makedirs(RESULT_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X8.npy"))   # (N, 1500, 8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy")).astype(int)   # (N,)
G = np.load(os.path.join(DATA_DIR, "G8.npy")).astype(int)   # (N,)
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}", flush=True)

# ===========================
# Config
# ===========================
N_SPLITS          = 6
SEEDS             = [7, 17, 23, 31, 57]
RANDOM_STATE      = 42
EPOCHS            = 80
BATCH_SIZE        = 16
LR_INIT           = 1e-3
MIN_C0_RATIO      = 0.20          # سخت‌تر از قبل برای وادارکردن به پیش‌بینیِ کلاس 0
LABEL_SMOOTH      = 0.01
TOPK_CHECKPOINTS  = 7
L2_REG            = 1e-5

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ===========================
# Utilities
# ===========================
def standardize_per_fold(X_tr, X_va, X_te):
    """Channel-wise standardization fit on TRAIN only."""
    T, C = X_tr.shape[1], X_tr.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_tr.reshape(-1, C))
    def trf(A):
        A2 = A.reshape(-1, C)
        A2 = scaler.transform(A2)
        return A2.reshape(A.shape[0], T, C)
    return trf(X_tr), trf(X_va), trf(X_te)

def compute_class_weights(y):
    classes = np.unique(y)
    counts  = np.array([(y == c).sum() for c in classes], dtype=np.float32)
    n, k = len(y), len(classes)
    return {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, counts)}

def best_threshold_with_constraint(val_probs, val_true, min_c0_ratio=MIN_C0_RATIO):
    """Grid search threshold maximizing macro-F1 with an optional min share of class 0 predictions."""
    p_lo, p_hi = np.percentile(val_probs, 10), np.percentile(val_probs, 90)
    grid = np.linspace(p_lo, p_hi, 41)
    grid = np.clip(grid, 0.30, 0.70)  # widened range

    best_t, best_macro = 0.5, -1.0
    for t in grid:
        yhat = (val_probs > t).astype(int)
        if min_c0_ratio is not None and np.mean(yhat == 0) < min_c0_ratio:
            continue
        macro = f1_score(val_true, yhat, average='macro', zero_division=0)
        if macro > best_macro:
            best_macro, best_t = macro, t

    if best_macro < 0:
        for t in grid:
            yhat = (val_probs > t).astype(int)
            macro = f1_score(val_true, yhat, average='macro', zero_division=0)
            if macro > best_macro:
                best_macro, best_t = macro, t
    return best_t, best_macro

def average_weights(weight_list):
    """Average a list of model.get_weights() (same shapes)."""
    return [np.mean([w[i] for w in weight_list], axis=0) for i in range(len(weight_list[0]))]

class TopKSaver(tf.keras.callbacks.Callback):
    """Keep (weights, val_loss) each epoch; at end, average top-K by lowest val_loss."""
    def __init__(self, model, k=TOPK_CHECKPOINTS):
        super().__init__()
        self.model_ref = model
        self.k = k
        self.snaps = []  # list of (val_loss, weights)

    def on_epoch_end(self, epoch, logs=None):
        vloss = float(logs.get('val_loss', np.inf))
        self.snaps.append((vloss, [w.copy() for w in self.model_ref.get_weights()]))

    def set_topk_weights(self):
        if not self.snaps:
            return
        snaps_sorted = sorted(self.snaps, key=lambda x: x[0])
        top = snaps_sorted[:min(self.k, len(self.snaps))]
        avg_w = average_weights([w for (_vl, w) in top])
        self.model_ref.set_weights(avg_w)

# ===========================
# Model (BN→ReLU variant)
# ===========================
def build_model(input_shape, lr=LR_INIT, label_smooth=LABEL_SMOOTH, l2_reg=L2_REG):
    model = Sequential([
        tf.keras.Input(shape=input_shape),

        # Block 1
        Conv1D(48, kernel_size=7, padding='same',
               kernel_regularizer=regularizers.l2(l2_reg)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.30),

        # Block 2
        Conv1D(64, kernel_size=5, padding='same',
               kernel_regularizer=regularizers.l2(l2_reg)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.30),

        # Block 3
        Conv1D(64, kernel_size=3, padding='same',
               kernel_regularizer=regularizers.l2(l2_reg)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.30),

        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.35),
        Dense(1, activation='sigmoid')
    ])
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smooth)
    model.compile(optimizer=Adam(lr), loss=loss_fn, metrics=['accuracy'])
    return model

# ===========================
# Training per fold (+seed-ensemble)
# ===========================
outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

rows, all_reports = [], []
fold = 1

for tr_idx, te_idx in outer.split(X, Y, groups=G):
    print(f"\n===== Fold {fold} =====", flush=True)
    fold_dir = os.path.join(RESULT_DIR, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    X_tr_full, X_te = X[tr_idx], X[te_idx]
    Y_tr_full, Y_te = Y[tr_idx], Y[te_idx]
    G_tr_full       = G[tr_idx]

    # inner split (group-aware) for validation
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_tr, inner_va = next(inner.split(X_tr_full, Y_tr_full, groups=G_tr_full))
    X_tr, Y_tr = X_tr_full[inner_tr], Y_tr_full[inner_tr]
    X_va, Y_va = X_tr_full[inner_va], Y_tr_full[inner_va]

    # standardize
    X_tr, X_va, X_te = standardize_per_fold(X_tr, X_va, X_te)

    # class weights
    class_w = compute_class_weights(Y_tr)
    print("[CLASS WEIGHTS]", class_w, flush=True)

    val_probs_seeds, te_probs_seeds = [], []

    for seed in SEEDS:
        tf.keras.backend.clear_session()
        np.random.seed(seed); tf.random.set_seed(seed)

        model = build_model((X_tr.shape[1], X_tr.shape[2]),
                            lr=LR_INIT, label_smooth=LABEL_SMOOTH, l2_reg=L2_REG)

        es  = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=0)
        topk = TopKSaver(model, k=TOPK_CHECKPOINTS)

        model.fit(X_tr, Y_tr,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  validation_data=(X_va, Y_va),
                  class_weight=class_w,
                  verbose=0,
                  callbacks=[es, rlr, topk])

        # average top-K snapshots (by lowest val_loss)
        topk.set_topk_weights()

        val_probs_seeds.append(model.predict(X_va, verbose=0).ravel())
        te_probs_seeds.append(model.predict(X_te, verbose=0).ravel())

    # average seeds
    val_probs_mean = np.mean(val_probs_seeds, axis=0)
    te_probs_mean  = np.mean(te_probs_seeds,  axis=0)

    # per-fold threshold
    t_fold, macro_val = best_threshold_with_constraint(val_probs_mean, Y_va, MIN_C0_RATIO)
    print(f"[FOLD {fold}] best_thr={t_fold:.3f} | macroF1(val)={macro_val:.4f}")

    Y_pred = (te_probs_mean > t_fold).astype(int)

    acc  = accuracy_score(Y_te, Y_pred)
    prec = precision_score(Y_te, Y_pred, zero_division=0)
    rec  = recall_score(Y_te, Y_pred, zero_division=0)
    f1   = f1_score(Y_te, Y_pred, zero_division=0)
    f1m  = f1_score(Y_te, Y_pred, average='macro', zero_division=0)

    rep = classification_report(Y_te, Y_pred, digits=4)
    all_reports.append(f"Fold {fold}\n{rep}\n")

    # save confusion matrix per fold
    cm = confusion_matrix(Y_te, Y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["non-easy", "easy"], yticklabels=["non-easy", "easy"])
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Fold {fold} | thr={t_fold:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, "confusion_matrix.png"))
    plt.close()

    with open(os.path.join(fold_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"best_thr={t_fold:.3f} | macroF1(val)={macro_val:.4f}\n\n")
        f.write(rep)

    rows.append({
        "fold": fold, "thr": t_fold,
        "acc": acc, "precision": prec, "recall": rec, "f1": f1, "f1_macro": f1m
    })
    fold += 1

# ===========================
# Summaries & Save
# ===========================
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_per_fold.csv"), index=False)

mean = df.mean(numeric_only=True); std = df.std(numeric_only=True)
summary = [
    f"Runs: {len(df)}",
    f"SEEDS: {SEEDS}",
    f"MIN_C0_RATIO: {MIN_C0_RATIO}",
    f"LABEL_SMOOTH: {LABEL_SMOOTH}",
    f"TOPK_CHECKPOINTS: {TOPK_CHECKPOINTS}",
    "Per-fold thresholds: " + ", ".join([f"{t:.3f}" for t in df['thr'].tolist()]),
    "Mean ± STD",
    f"Accuracy: {mean['acc']:.4f} ± {std['acc']:.4f}",
    f"Precision: {mean['precision']:.4f} ± {std['precision']:.4f}",
    f"Recall: {mean['recall']:.4f} ± {std['recall']:.4f}",
    f"F1: {mean['f1']:.4f} ± {std['f1']:.4f}",
    f"F1 (macro): {mean['f1_macro']:.4f} ± {std['f1_macro']:.4f}",
]
with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary))
with open(os.path.join(RESULT_DIR, "reports.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(all_reports))

print("\n".join(summary), flush=True)
print("Saved to:", RESULT_DIR, flush=True)
