import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, balanced_accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =====================
# تنظیمات
# =====================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "27_cnn_cv_std_overs14_focala005_thresh_constrained_tf")

N_SPLITS     = 10
N_REPEATS    = 4
RANDOM_STATE = 42
EPOCHS       = 50
BATCH_SIZE   = 16
LR           = 1e-3

VAL_SIZE      = 0.2
THRESH_GRID   = np.linspace(0.1, 0.9, 33)

# Oversampling: نسبت کلاس 0 به کلاس 1 در Train هر فولد
OVERSAMPLE_RATIO = 1.4   # یعنی n_min_target ≈ 1.4 * n_maj

# Focal Loss: alpha کوچک → تاکید بیشتر روی کلاس 0 (label=0)
FOCAL_ALPHA = 0.05
FOCAL_GAMMA = 2.0

# قید تصمیم‌گیری روی ولیدیشن: حداقل سهم کلاس 0 در پیش‌بینی‌ها
MIN_C0_PRED_RATIO = 0.30

os.makedirs(RESULT_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# =====================
# بارگذاری داده
# =====================
X = np.load(os.path.join(DATA_DIR, "X.npy"))         # (N, T, C)
Y = np.load(os.path.join(DATA_DIR, "Y.npy")).astype(int)  # (N,)

# =====================
# Focal Loss (from logits)
# =====================
def focal_loss_from_logits(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    """
    Binary focal loss with logits.
    alpha وزن کلاس مثبت (label=1) است؛ پس وزن کلاس 0 برابر (1-alpha) می‌شود.
    با کوچک‌کردن alpha، کلاس 0 وزن بیشتری می‌گیرد.
    """
    def loss(y_true, logits):
        y_true = tf.cast(y_true, tf.float32)
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits)
        p = tf.sigmoid(logits)
        pt = y_true * p + (1.0 - y_true) * (1.0 - p)
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        focal = alpha_factor * tf.pow(1.0 - pt, gamma) * ce
        return tf.reduce_mean(focal)
    return loss

# =====================
# مدل (خروجی: logits)
# =====================
def create_model(input_shape):
    model = Sequential([
        tf.keras.Input(shape=input_shape),
        Conv1D(32, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(1)  # logits (بدون sigmoid)
    ])
    model.compile(optimizer=Adam(LR), loss=focal_loss_from_logits(), metrics=['accuracy'])
    return model

# =====================
# استانداردسازی per-fold
# =====================
def standardize_fold(X_tr, X_va, X_te):
    Ntr, T, C = X_tr.shape
    scaler = StandardScaler()
    Xtr2D = X_tr.reshape(-1, C)
    Xva2D = X_va.reshape(-1, C)
    Xte2D = X_te.reshape(-1, C)
    scaler.fit(Xtr2D)
    Xtr2D = scaler.transform(Xtr2D)
    Xva2D = scaler.transform(Xva2D)
    Xte2D = scaler.transform(Xte2D)
    return (Xtr2D.reshape(Ntr, T, C),
            Xva2D.reshape(X_va.shape[0], T, C),
            Xte2D.reshape(X_te.shape[0], T, C))

# =====================
# Oversampling کلاس 0 تا نسبت دلخواه
# =====================
def oversample_minority_ratio(X_tr, Y_tr, minority_label=0, ratio=1.4):
    idx_min = np.where(Y_tr == minority_label)[0]
    idx_maj = np.where(Y_tr != minority_label)[0]
    n_min, n_maj = len(idx_min), len(idx_maj)
    if n_min == 0:
        return X_tr, Y_tr
    target_min = int(np.ceil(ratio * n_maj))
    if target_min <= n_min:
        return X_tr, Y_tr  # به اندازه کافی هست
    n_need = target_min - n_min
    reps = int(np.ceil(n_need / n_min))
    extra_idx = np.tile(idx_min, reps)[:n_need]
    X_aug = np.concatenate([X_tr, X_tr[extra_idx]], axis=0)
    Y_aug = np.concatenate([Y_tr, Y_tr[extra_idx]], axis=0)
    # شافل
    perm = np.random.permutation(len(Y_aug))
    return X_aug[perm], Y_aug[perm]

# =====================
# انتخاب آستانه: Macro-F1 + قیود دوکلاسه و حداقل سهم کلاس 0
# =====================
def choose_threshold_macro_f1_safe(val_probs, val_y, grid=THRESH_GRID):
    best_t, best_macro = 0.5, -1.0
    candidate_found = False
    for t in grid:
        yhat = (val_probs > t).astype(int)
        if len(np.unique(yhat)) < 2:
            continue
        pred_ratio_c0 = np.mean(yhat == 0)
        if pred_ratio_c0 < MIN_C0_PRED_RATIO:
            continue
        macro_f1 = f1_score(val_y, yhat, average='macro', zero_division=0)
        if (not candidate_found) or (macro_f1 > best_macro):
            candidate_found = True
            best_macro = macro_f1
            best_t = t
    # اگر هیچ آستانه‌ای قید را برآورده نکرد، fallback: دوکلاسه بدون نسبت
    if not candidate_found:
        best_t2, best_macro2 = 0.5, -1.0
        for t in grid:
            yhat = (val_probs > t).astype(int)
            if len(np.unique(yhat)) < 2:
                continue
            macro_f1 = f1_score(val_y, yhat, average='macro', zero_division=0)
            if macro_f1 > best_macro2:
                best_macro2, best_t2 = macro_f1, t
        if best_macro2 > -1:
            return best_t2, best_macro2
        # آخرین fallback: بهترین macro بدون هیچ قیدی
        best_t3, best_macro3 = 0.5, -1.0
        for t in grid:
            yhat = (val_probs > t).astype(int)
            macro_f1 = f1_score(val_y, yhat, average='macro', zero_division=0)
            if macro_f1 > best_macro3:
                best_macro3, best_t3 = macro_f1, t
        return best_t3, best_macro3
    return best_t, best_macro

# =====================
# Cross Validation
# =====================
rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

all_rows = []
all_reports = []
fold_idx = 1

for tr_idx, te_idx in rskf.split(X, Y):
    print(f"\n===== Fold {fold_idx} =====")
    X_tr_full, X_te = X[tr_idx], X[te_idx]
    Y_tr_full, Y_te = Y[tr_idx], Y[te_idx]

    # ساخت Val از Train فولد
    X_tr, X_va, Y_tr, Y_va = train_test_split(
        X_tr_full, Y_tr_full, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=Y_tr_full
    )

    # Oversample روی Train (به نفع کلاس 0 با نسبت 1.4×)
    X_tr, Y_tr = oversample_minority_ratio(X_tr, Y_tr, minority_label=0, ratio=OVERSAMPLE_RATIO)

    # استانداردسازی per-fold
    X_tr, X_va, X_te = standardize_fold(X_tr, X_va, X_te)

    # class_weight per-fold (با وجود oversample همچنان مفید)
    classes = np.unique(Y_tr)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=Y_tr)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    # مدل
    model = create_model((X.shape[1], X.shape[2]))

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor='val_loss', verbose=0),
    ]

    # آموزش
    model.fit(
        X_tr, Y_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        validation_data=(X_va, Y_va),
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    # انتخاب آستانه با قیود
    val_logits = model.predict(X_va, verbose=0).ravel()
    val_probs  = 1.0 / (1.0 + np.exp(-val_logits))  # sigmoid
    best_t, best_macro = choose_threshold_macro_f1_safe(val_probs, Y_va, THRESH_GRID)

    # ارزیابی روی Test
    te_logits = model.predict(X_te, verbose=0).ravel()
    te_probs  = 1.0 / (1.0 + np.exp(-te_logits))
    Y_pred = (te_probs > best_t).astype(int)

    # متریک‌ها
    acc  = accuracy_score(Y_te, Y_pred)
    prec = precision_score(Y_te, Y_pred, zero_division=0)
    rec  = recall_score(Y_te, Y_pred, zero_division=0)
    f1   = f1_score(Y_te, Y_pred, zero_division=0)
    balacc = balanced_accuracy_score(Y_te, Y_pred)

    # کلاس‌محور
    prec_c0 = precision_score((Y_te==0).astype(int), (Y_pred==0).astype(int), zero_division=0)
    rec_c0  = recall_score((Y_te==0).astype(int),  (Y_pred==0).astype(int), zero_division=0)
    f1_c0   = f1_score((Y_te==0).astype(int),      (Y_pred==0).astype(int), zero_division=0)

    prec_c1 = precision_score((Y_te==1).astype(int), (Y_pred==1).astype(int), zero_division=0)
    rec_c1  = recall_score((Y_te==1).astype(int),    (Y_pred==1).astype(int), zero_division=0)
    f1_c1   = f1_score((Y_te==1).astype(int),        (Y_pred==1).astype(int), zero_division=0)

    report = classification_report(Y_te, Y_pred, digits=4)
    all_reports.append(f"Fold {fold_idx} | best_threshold={best_t:.3f} | MacroF1(val)={best_macro:.4f}\n{report}\n")

    # Confusion Matrix
    cm = confusion_matrix(Y_te, Y_pred)
    plt.figure(figsize=(5.2, 4.6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['non-easy(0)','easy(1)'],
                yticklabels=['non-easy(0)','easy(1)'])
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion Matrix - Fold {fold_idx}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"cm_fold_{fold_idx:02d}.png"))
    plt.close()

    # ردیف CSV
    all_rows.append({
        "fold": fold_idx,
        "best_threshold": best_t,
        "acc": acc,
        "balanced_acc": balacc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "precision_macro": precision_score(Y_te, Y_pred, average='macro', zero_division=0),
        "recall_macro":    recall_score(Y_te, Y_pred,    average='macro', zero_division=0),
        "f1_macro":        f1_score(Y_te, Y_pred,        average='macro', zero_division=0),
        "precision_c0": prec_c0, "recall_c0": rec_c0, "f1_c0": f1_c0,
        "precision_c1": prec_c1, "recall_c1": rec_c1, "f1_c1": f1_c1
    })

    fold_idx += 1

# ذخیره نتایج
df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_per_fold.csv"), index=False)

mean = df[['acc','balanced_acc','precision','recall','f1',
           'precision_macro','recall_macro','f1_macro',
           'precision_c0','recall_c0','f1_c0',
           'precision_c1','recall_c1','f1_c1']].mean()
std  = df[['acc','balanced_acc','precision','recall','f1',
           'precision_macro','recall_macro','f1_macro',
           'precision_c0','recall_c0','f1_c0',
           'precision_c1','recall_c1','f1_c1']].std()

summary_lines = [
    f"Runs: {N_SPLITS * N_REPEATS}",
    f"Mean ± STD",
    f"Accuracy: {mean['acc']:.4f} ± {std['acc']:.4f}",
    f"Balanced Acc: {mean['balanced_acc']:.4f} ± {std['balanced_acc']:.4f}",
    f"Precision (bin): {mean['precision']:.4f} ± {std['precision']:.4f}",
    f"Recall (bin): {mean['recall']:.4f} ± {std['recall']:.4f}",
    f"F1 (bin): {mean['f1']:.4f} ± {std['f1']:.4f}",
    f"Precision (macro): {mean['precision_macro']:.4f} ± {std['precision_macro']:.4f}",
    f"Recall (macro): {mean['recall_macro']:.4f} ± {std['recall_macro']:.4f}",
    f"F1 (macro): {mean['f1_macro']:.4f} ± {std['f1_macro']:.4f}",
    f"Class 0 — Precision: {mean['precision_c0']:.4f} ± {std['precision_c0']:.4f}, "
    f"Recall: {mean['recall_c0']:.4f} ± {std['recall_c0']:.4f}, F1: {mean['f1_c0']:.4f} ± {std['f1_c0']:.4f}",
    f"Class 1 — Precision: {mean['precision_c1']:.4f} ± {std['precision_c1']:.4f}, "
    f"Recall: {mean['recall_c1']:.4f} ± {std['recall_c1']:.4f}, F1: {mean['f1_c1']:.4f} ± {std['f1_c1']:.4f}",
]

with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

with open(os.path.join(RESULT_DIR, "reports.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(all_reports))

print("\n".join(summary_lines))
print("Saved to:", RESULT_DIR)
