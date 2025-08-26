# 31_cnn_groupkfold_dualratio_globalthr_tf.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, balanced_accuracy_score
)
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

# =========================
# Paths
# =========================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "28_cnn_groupkfold_dualratio_globalthr_tf")
os.makedirs(RESULT_DIR, exist_ok=True)

X_NAME, Y_NAME, G_NAME = "X2.npy", "Y2.npy", "G1.npy"   # خروجی مرحله‌ی XYG

# =========================
# Params
# =========================
N_SPLITS      = 6
RANDOM_STATE  = 42
EPOCHS        = 50
BATCH_SIZE    = 16
LR            = 1e-3

# --- جست‌وجوی آستانه با قید سهم دو کلاس (روی ولیدیشن) ---
MIN_RATIO_0   = 0.25     # حداقل سهم پیش‌بینی کلاس 0 روی ولیدیشن
MIN_RATIO_1   = 0.25     # حداقل سهم پیش‌بینی کلاس 1 روی ولیدیشن
THRESH_GRID   = np.linspace(0.01, 0.99, 99)

# --- وزن‌های Loss (ملایم/بی‌طرف) ---
W_CLASS0      = 1.00     # گزینه‌ها: 1.00 (بی‌وزن) یا 1.15 برای ملایم
W_CLASS1      = 1.00

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# =========================
# Load
# =========================
X = np.load(os.path.join(DATA_DIR, X_NAME))
Y = np.load(os.path.join(DATA_DIR, Y_NAME)).astype(int)
G = np.load(os.path.join(DATA_DIR, G_NAME)).astype(int)
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}", flush=True)

# =========================
# Loss: weighted BCE from logits
# =========================
def weighted_bce_from_logits(w0=1.0, w1=1.0):
    def loss(y_true, logits):
        y_true = tf.cast(y_true, tf.float32)
        weights = y_true * w1 + (1.0 - y_true) * w0
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits)
        return tf.reduce_mean(weights * ce)
    return loss

# =========================
# Model
# =========================
def create_model(input_shape, w0=1.0, w1=1.0):
    model = Sequential([
        tf.keras.Input(shape=input_shape),

        Conv1D(16, 5, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.4),

        Conv1D(32, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.4),

        GlobalAveragePooling1D(),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(1)  # logits
    ])
    model.compile(optimizer=Adam(LR),
                  loss=weighted_bce_from_logits(w0, w1),
                  metrics=['accuracy'])
    return model

# =========================
# Standardize per-fold (z-score روی کانال‌ها)
# =========================
def standardize_fold(X_tr, X_va, X_te):
    T, C = X_tr.shape[1], X_tr.shape[2]
    scaler = StandardScaler()
    Xtr2 = X_tr.reshape(-1, C); scaler.fit(Xtr2)
    Xva2 = X_va.reshape(-1, C); Xva2 = scaler.transform(Xva2)
    Xte2 = X_te.reshape(-1, C); Xte2 = scaler.transform(Xte2)
    Xtr2 = scaler.transform(Xtr2)
    return (Xtr2.reshape(X_tr.shape[0], T, C),
            Xva2.reshape(X_va.shape[0], T, C),
            Xte2.reshape(X_te.shape[0], T, C))

# =========================
# Simple oversample (train-only) تا 1:1
# =========================
def oversample_to_balance(X_tr, Y_tr, minority_label=0):
    idx0 = np.where(Y_tr == minority_label)[0]
    idx1 = np.where(Y_tr != minority_label)[0]
    n0, n1 = len(idx0), len(idx1)
    if n0 == 0 or n1 == 0:
        return X_tr, Y_tr
    if n0 < n1:
        need = n1 - n0
        reps = int(np.ceil(need / n0))
        extra = np.tile(idx0, reps)[:need]
    else:
        need = n0 - n1
        reps = int(np.ceil(need / n1))
        extra = np.tile(idx1, reps)[:need]
    Xb = np.concatenate([X_tr, X_tr[extra]], axis=0)
    Yb = np.concatenate([Y_tr, Y_tr[extra]], axis=0)
    p = np.random.permutation(len(Yb))
    return Xb[p], Yb[p]

# =========================
# Threshold search with dual ratio constraints
# =========================
def best_threshold_with_constraint(val_probs, val_true,
                                   grid=THRESH_GRID,
                                   min_ratio_0=MIN_RATIO_0,
                                   min_ratio_1=MIN_RATIO_1):
    best_t, best_macro = 0.5, -1.0
    for t in grid:
        yhat = (val_probs > t).astype(int)
        if len(np.unique(yhat)) < 2:
            continue
        # enforce min ratio for both classes
        ratio0 = np.mean(yhat == 0)
        ratio1 = np.mean(yhat == 1)
        if ratio0 < min_ratio_0 or ratio1 < min_ratio_1:
            continue
        macro = f1_score(val_true, yhat, average='macro', zero_division=0)
        if macro > best_macro:
            best_macro, best_t = macro, t

    # fallback: both classes predicted, no ratio constraint
    if best_macro < 0:
        for t in grid:
            yhat = (val_probs > t).astype(int)
            if len(np.unique(yhat)) < 2:
                continue
            macro = f1_score(val_true, yhat, average='macro', zero_division=0)
            if macro > best_macro:
                best_macro, best_t = macro, t

    # fallback: pure macro-F1
    if best_macro < 0:
        for t in grid:
            macro = f1_score(val_true, (val_probs > t).astype(int),
                             average='macro', zero_division=0)
            if macro > best_macro:
                best_macro, best_t = macro, t

    return best_t, best_macro

# =========================
# CV (Group-aware by user)
# =========================
outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

all_val_probs, all_val_true = [], []
fold_artifacts = []

fold = 1
for tr_idx, te_idx in outer.split(X, Y, groups=G):
    print(f"\n===== Fold {fold} =====", flush=True)

    X_tr_full, X_te = X[tr_idx], X[te_idx]
    Y_tr_full, Y_te = Y[tr_idx], Y[te_idx]
    G_tr_full       = G[tr_idx]

    # inner split (group-aware) برای ولیدیشن
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_tr, inner_va = next(inner.split(X_tr_full, Y_tr_full, groups=G_tr_full))
    X_tr, Y_tr = X_tr_full[inner_tr], Y_tr_full[inner_tr]
    X_va, Y_va = X_tr_full[inner_va], Y_tr_full[inner_va]

    # oversample روی Train
    X_tr, Y_tr = oversample_to_balance(X_tr, Y_tr, minority_label=0)
    print(f"[Fold {fold}] Train balanced: n0={np.sum(Y_tr==0)}, n1={np.sum(Y_tr==1)}", flush=True)

    # استانداردسازی
    X_tr, X_va, X_te = standardize_fold(X_tr, X_va, X_te)

    # مدل (گزینه‌ها: W_CLASS0=1.00 یا 1.15)
    model = create_model((X.shape[1], X.shape[2]), w0=W_CLASS0, w1=W_CLASS1)
    cbs = [
        EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, monitor='val_loss', verbose=0)
    ]
    model.fit(X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_va, Y_va), verbose=0, callbacks=cbs)

    # ولیدیشن/تست logits -> probs
    val_logits = model.predict(X_va, verbose=0).ravel()
    val_probs  = 1.0 / (1.0 + np.exp(-val_logits))
    all_val_probs.append(val_probs); all_val_true.append(Y_va)

    te_logits = model.predict(X_te, verbose=0).ravel()
    te_probs  = 1.0 / (1.0 + np.exp(-te_logits))
    fold_artifacts.append((te_probs, Y_te))

    fold += 1

# =========================
# Global threshold
# =========================
all_val_probs = np.concatenate(all_val_probs)
all_val_true  = np.concatenate(all_val_true)
t_global, macro_val = best_threshold_with_constraint(
    all_val_probs, all_val_true,
    grid=THRESH_GRID, min_ratio_0=MIN_RATIO_0, min_ratio_1=MIN_RATIO_1
)
print(f"\n[GLOBAL] best_threshold={t_global:.3f} | MacroF1(val_all)={macro_val:.4f}", flush=True)

# =========================
# Evaluate per fold (global threshold)
# =========================
rows, reports = [], []
for i, (te_probs, Y_te) in enumerate(fold_artifacts, start=1):
    Y_pred = (te_probs > t_global).astype(int)

    acc  = accuracy_score(Y_te, Y_pred)
    bal  = balanced_accuracy_score(Y_te, Y_pred)
    prec = precision_score(Y_te, Y_pred, zero_division=0)
    rec  = recall_score(Y_te, Y_pred, zero_division=0)
    f1   = f1_score(Y_te, Y_pred, zero_division=0)

    rep  = classification_report(Y_te, Y_pred, digits=4)
    reports.append(f"Fold {i}\n{rep}\n")
    print(f"\nFold {i}\n{rep}")

    # کلاس‌محور
    prec_macro = precision_score(Y_te, Y_pred, average='macro', zero_division=0)
    rec_macro  = recall_score(Y_te, Y_pred, average='macro', zero_division=0)
    f1_macro   = f1_score(Y_te, Y_pred, average='macro', zero_division=0)

    prec_c0 = precision_score((Y_te==0).astype(int), (Y_pred==0).astype(int), zero_division=0)
    rec_c0  = recall_score((Y_te==0).astype(int),    (Y_pred==0).astype(int), zero_division=0)
    f1_c0   = f1_score((Y_te==0).astype(int),        (Y_pred==0).astype(int), zero_division=0)

    prec_c1 = precision_score((Y_te==1).astype(int), (Y_pred==1).astype(int), zero_division=0)
    rec_c1  = recall_score((Y_te==1).astype(int),    (Y_pred==1).astype(int), zero_division=0)
    f1_c1   = f1_score((Y_te==1).astype(int),        (Y_pred==1).astype(int), zero_division=0)

    rows.append({
        "fold": i,
        "acc": acc, "balanced_acc": bal,
        "precision": prec, "recall": rec, "f1": f1,
        "precision_macro": prec_macro, "recall_macro": rec_macro, "f1_macro": f1_macro,
        "precision_c0": prec_c0, "recall_c0": rec_c0, "f1_c0": f1_c0,
        "precision_c1": prec_c1, "recall_c1": rec_c1, "f1_c1": f1_c1
    })

    # Confusion matrix
    cm = confusion_matrix(Y_te, Y_pred)
    plt.figure(figsize=(5, 4.4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['non-easy(0)', 'easy(1)'],
                yticklabels=['non-easy(0)', 'easy(1)'])
    plt.title(f"Confusion Matrix - Fold {i} (thr={t_global:.3f})")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"cm_fold_{i:02d}.png"))
    plt.close()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_per_fold.csv"), index=False)

mean = df.mean(numeric_only=True)
std  = df.std(numeric_only=True)

summary = [
    f"Global Threshold: {t_global:.3f}",
    f"Runs: {len(rows)}",
    "Mean ± STD",
    f"Accuracy: {mean['acc']:.4f} ± {std['acc']:.4f}",
    f"Balanced Acc: {mean['balanced_acc']:.4f} ± {std['balanced_acc']:.4f}",
    f"Precision (bin): {mean['precision']:.4f} ± {std['precision']:.4f}",
    f"Recall (bin): {mean['recall']:.4f} ± {std['recall']:.4f}",
    f"F1 (bin): {mean['f1']:.4f} ± {std['f1']:.4f}",
    f"Precision (macro): {mean['precision_macro']:.4f} ± {std['precision_macro']:.4f}",
    f"Recall (macro): {mean['recall_macro']:.4f} ± {std['recall_macro']:.4f}",
    f"F1 (macro): {mean['f1_macro']:.4f} ± {std['f1_macro']:.4f}",
    (
        "Class 0 — Precision: "
        f"{mean['precision_c0']:.4f} ± {std['precision_c0']:.4f}, "
        f"Recall: {mean['recall_c0']:.4f} ± {std['recall_c0']:.4f}, "
        f"F1: {mean['f1_c0']:.4f} ± {std['f1_c0']:.4f}"
    ),
    (
        "Class 1 — Precision: "
        f"{mean['precision_c1']:.4f} ± {std['precision_c1']:.4f}, "
        f"Recall: {mean['recall_c1']:.4f} ± {std['recall_c1']:.4f}, "
        f"F1: {mean['f1_c1']:.4f} ± {std['f1_c1']:.4f}"
    ),
]

with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary))
with open(os.path.join(RESULT_DIR, "reports.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(reports))

print("\n".join(summary), flush=True)
print("Saved to:", RESULT_DIR, flush=True)
