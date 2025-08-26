# 31_cnn_groupkfold_perfoldthr_quantmatch_tf.py
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

# ===== مسیرها =====
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "29_cnn_groupkfold_perfoldthr_quantmatch_tf")
os.makedirs(RESULT_DIR, exist_ok=True)

X_NAME, Y_NAME, G_NAME = "X2.npy", "Y2.npy", "G1.npy"

# ===== پارامترها =====
N_SPLITS      = 6
RANDOM_STATE  = 42
EPOCHS        = 50
BATCH_SIZE    = 16
LR            = 1e-3

# آستانه‌جویی
THRESH_GRID   = np.linspace(0.05, 0.95, 37)  # شبکه جستجو برای آستانه
MIN_NEG_RATIO = 0.20  # حداقل سهم پیش‌بینی‌های کلاس 0 در VAL (برای جلوگیری از تک‌کلاسه شدن)
MIN_POS_RATIO = 0.20  # حداقل سهم پیش‌بینی‌های کلاس 1 در VAL

# روی تست، اگر احتمال‌ها خیلی بایاس باشند با کوانتیل، نرخ مثبت/منفی را تضمین می‌کنیم
TEST_MIN_NEG  = 0.20  # حداقل سهم 0 در TEST پیش‌بینی
TEST_MIN_POS  = 0.20  # حداقل سهم 1 در TEST پیش‌بینی

# Oversample TRAIN؟ (فعلا خاموش برای جلوگیری از بایاس)
USE_OVERSAMPLE = False
MINORITY_LABEL = 0  # اگر روشن شد، کدام را اقلیت فرض کنیم

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ===== بارگذاری =====
X = np.load(os.path.join(DATA_DIR, X_NAME))
Y = np.load(os.path.join(DATA_DIR, Y_NAME)).astype(int)
G = np.load(os.path.join(DATA_DIR, G_NAME)).astype(int)
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}", flush=True)

# ===== مدل =====
def create_model(input_shape):
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

        Dense(1, activation='sigmoid')  # خروجی probability مستقیم
    ])
    model.compile(optimizer=Adam(LR), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ===== استانداردسازی per-fold =====
def standardize_fold(X_tr, X_va, X_te):
    T, C = X_tr.shape[1], X_tr.shape[2]
    scaler = StandardScaler()
    Xtr2 = X_tr.reshape(-1, C)
    scaler.fit(Xtr2)
    Xtr2 = scaler.transform(Xtr2)
    Xva2 = scaler.transform(X_va.reshape(-1, C))
    Xte2 = scaler.transform(X_te.reshape(-1, C))
    return (Xtr2.reshape(X_tr.shape[0], T, C),
            Xva2.reshape(X_va.shape[0], T, C),
            Xte2.reshape(X_te.shape[0], T, C))

# ===== oversample ساده (در صورت نیاز) =====
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

# ===== انتخاب آستانه با قیود نسبت کلاس‌ها روی VAL (per-fold) =====
def best_threshold_with_ratio(val_probs, val_true,
                              grid=THRESH_GRID,
                              min_neg=MIN_NEG_RATIO,
                              min_pos=MIN_POS_RATIO):
    best_t, best_macro = 0.5, -1.0
    for t in grid:
        yhat = (val_probs > t).astype(int)
        if len(np.unique(yhat)) < 2:
            continue
        neg_ratio = np.mean(yhat == 0)
        pos_ratio = 1.0 - neg_ratio
        if (neg_ratio < min_neg) or (pos_ratio < min_pos):
            continue
        macro = f1_score(val_true, yhat, average='macro', zero_division=0)
        if macro > best_macro:
            best_macro, best_t = macro, t
    # اگر چیزی پیدا نشد، فقط بهترین macro F1 را برگردان (بدون قید)
    if best_macro < 0:
        for t in grid:
            yhat = (val_probs > t).astype(int)
            if len(np.unique(yhat)) < 2:
                continue
            macro = f1_score(val_true, yhat, average='macro', zero_division=0)
            if macro > best_macro:
                best_macro, best_t = macro, t
    return best_t, best_macro

# ===== تنظیم آستانه روی TEST با Quantile Matching =====
def quantile_match_threshold(test_probs, base_threshold,
                             min_neg=TEST_MIN_NEG, min_pos=TEST_MIN_POS):
    # اگر base_threshold باعث تک‌کلاسه شد، با کوانتیل نرخ را تضمین کن
    yhat = (test_probs > base_threshold).astype(int)
    neg_ratio = np.mean(yhat == 0)
    pos_ratio = 1.0 - neg_ratio

    t = base_threshold
    if neg_ratio < min_neg:
        # منفی‌ها خیلی کم‌اند => آستانه را بالا ببر تا منفی‌ها بیشتر شوند
        target_neg = max(min_neg, neg_ratio)
        q = 1.0 - target_neg  # چون threshold بالاتر => منفی بیشتر
        q = min(max(q, 0.0), 1.0)
        t = np.quantile(test_probs, q)
    elif pos_ratio < min_pos:
        # مثبت‌ها خیلی کم‌اند => آستانه را پایین بیار تا مثبت‌ها بیشتر شوند
        target_pos = max(min_pos, pos_ratio)
        q = 1.0 - target_pos
        q = min(max(q, 0.0), 1.0)
        t = np.quantile(test_probs, q)

    return float(t)

# ===== Group K-Fold خارجی =====
outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

rows, reports = [], []
fold = 1
for tr_idx, te_idx in outer.split(X, Y, groups=G):
    print(f"\n===== Fold {fold} =====", flush=True)

    X_tr_full, X_te = X[tr_idx], X[te_idx]
    Y_tr_full, Y_te = Y[tr_idx], Y[te_idx]
    G_tr_full       = G[tr_idx]

    # inner split گروه‌محور برای VAL همان فولد
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_tr, inner_va = next(inner.split(X_tr_full, Y_tr_full, groups=G_tr_full))
    X_tr, Y_tr = X_tr_full[inner_tr], Y_tr_full[inner_tr]
    X_va, Y_va = X_tr_full[inner_va], Y_tr_full[inner_va]

    if USE_OVERSAMPLE:
        X_tr, Y_tr = oversample_to_balance(X_tr, Y_tr, minority_label=MINORITY_LABEL)
        print(f"[Fold {fold}] Oversampled Train: n0={np.sum(Y_tr==0)}, n1={np.sum(Y_tr==1)}", flush=True)

    # استانداردسازی
    X_tr, X_va, X_te = standardize_fold(X_tr, X_va, X_te)

    # مدل
    model = create_model((X.shape[1], X.shape[2]))
    cbs = [
        EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-6, monitor='val_loss', verbose=0),
    ]
    model.fit(X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_va, Y_va), verbose=0, callbacks=cbs)

    # probs
    val_probs = model.predict(X_va, verbose=0).ravel()
    te_probs  = model.predict(X_te, verbose=0).ravel()

    # آستانه per-fold از VAL
    t_val, macro_val = best_threshold_with_ratio(
        val_probs, Y_va,
        grid=THRESH_GRID,
        min_neg=MIN_NEG_RATIO, min_pos=MIN_POS_RATIO
    )

    # کوانتیل‌مچ روی TEST
    t_test = quantile_match_threshold(
        te_probs, t_val, min_neg=TEST_MIN_NEG, min_pos=TEST_MIN_POS
    )
    print(f"[Fold {fold}] thr_val={t_val:.3f} | MacroF1(val)={macro_val:.4f} | thr_test={t_test:.3f}", flush=True)

    # ارزیابی
    Y_pred = (te_probs > t_test).astype(int)
    acc  = accuracy_score(Y_te, Y_pred)
    bal  = balanced_accuracy_score(Y_te, Y_pred)
    prec = precision_score(Y_te, Y_pred, zero_division=0)
    rec  = recall_score(Y_te, Y_pred, zero_division=0)
    f1   = f1_score(Y_te, Y_pred, zero_division=0)
    rep  = classification_report(Y_te, Y_pred, digits=4)
    reports.append(f"Fold {fold}\n{rep}\n")

    # کلاس‌محور
    prec_c0 = precision_score((Y_te==0).astype(int), (Y_pred==0).astype(int), zero_division=0)
    rec_c0  = recall_score((Y_te==0).astype(int),    (Y_pred==0).astype(int), zero_division=0)
    f1_c0   = f1_score((Y_te==0).astype(int),        (Y_pred==0).astype(int), zero_division=0)
    prec_c1 = precision_score((Y_te==1).astype(int), (Y_pred==1).astype(int), zero_division=0)
    rec_c1  = recall_score((Y_te==1).astype(int),    (Y_pred==1).astype(int), zero_division=0)
    f1_c1   = f1_score((Y_te==1).astype(int),        (Y_pred==1).astype(int), zero_division=0)

    rows.append({
        "fold": fold, "acc": acc, "balanced_acc": bal,
        "precision": prec, "recall": rec, "f1": f1,
        "precision_macro": precision_score(Y_te, Y_pred, average='macro', zero_division=0),
        "recall_macro":    recall_score(Y_te, Y_pred,    average='macro', zero_division=0),
        "f1_macro":        f1_score(Y_te, Y_pred,        average='macro', zero_division=0),
        "precision_c0": prec_c0, "recall_c0": rec_c0, "f1_c0": f1_c0,
        "precision_c1": prec_c1, "recall_c1": rec_c1, "f1_c1": f1_c1,
        "thr_val": t_val, "thr_test": t_test, "macroF1_val": macro_val
    })

    cm = confusion_matrix(Y_te, Y_pred)
    plt.figure(figsize=(5, 4.4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['non-easy(0)','easy(1)'],
                yticklabels=['non-easy(0)','easy(1)'])
    plt.title(f"Confusion Matrix - Fold {fold} (thr_test={t_test:.3f})")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"cm_fold_{fold:02d}.png")); plt.close()

    fold += 1

# ===== خروجی نهایی =====
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_per_fold.csv"), index=False)

mean = df.mean(numeric_only=True)
std  = df.std(numeric_only=True)

summary = [
    f"Runs: {len(df)}",
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
    f.write("\n".join([r for r in df.apply(lambda r: f"Fold {int(r['fold'])}\n", axis=1)]))  # فقط تیترها
with open(os.path.join(RESULT_DIR, "fold_thresholds.csv"), "w", encoding="utf-8") as f:
    df[["fold","thr_val","thr_test","macroF1_val"]].to_csv(f, index=False)

print("\n".join(summary), flush=True)
print("Saved to:", RESULT_DIR, flush=True)
