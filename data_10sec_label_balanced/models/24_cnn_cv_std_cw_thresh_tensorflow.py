import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# =====================
# تنظیمات
# =====================
DATA_DIR = os.path.join("..", "data")  # مسیر X.npy و Y.npy
RESULT_DIR = os.path.join("..", "results", "24_cnn_cv_std_cw_thresh_tensorflow")

N_SPLITS = 10
N_REPEATS = 4
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-3
VAL_SIZE = 0.2  # از train هر فولد جدا می‌کنیم برای انتخاب آستانه
THRESH_GRID = np.linspace(0.1, 0.9, 33)  # شبکه جستجوی آستانه

os.makedirs(RESULT_DIR, exist_ok=True)

# =====================
# بارگذاری داده
# =====================
X = np.load(os.path.join(DATA_DIR, "X.npy"))   # (N, T, C)
Y = np.load(os.path.join(DATA_DIR, "Y.npy"))   # (N,)
Y = Y.astype(int)

# =====================
# مدل CNN
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

        Flatten(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(LR), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =====================
# کمک‌تابع: استانداردسازی (z-score) روی کانال‌ها
# =====================
def standardize_fold(X_tr, X_va, X_te):
    # reshape به (N*T, C)
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
# کمک‌تابع: انتخاب آستانه بر اساس بهینه‌کردن F1 برای کلاس 0
# =====================
def choose_threshold_for_class0(val_probs, val_y, grid=THRESH_GRID):
    best_t, best_f1neg = 0.5, -1.0
    for t in grid:
        yhat = (val_probs > t).astype(int)
        # F1 برای کلاس 0: مثبت‌سازی "کلاس 0"
        f1_neg = f1_score((val_y == 0).astype(int), (yhat == 0).astype(int), zero_division=0)
        if f1_neg > best_f1neg:
            best_f1neg, best_t = f1_neg, t
    return best_t, best_f1neg

# =====================
# Cross Validation
# =====================
rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

all_rows = []    # برای CSV
all_reports = [] # گزارش‌های متنی
fold_idx = 1

for tr_idx, te_idx in rskf.split(X, Y):
    print(f"\n===== Fold {fold_idx} =====")
    X_tr_full, X_te = X[tr_idx], X[te_idx]
    Y_tr_full, Y_te = Y[tr_idx], Y[te_idx]

    # --- ساخت ولیدیشن از روی train فولد (stratified) برای انتخاب آستانه
    X_tr, X_va, Y_tr, Y_va = train_test_split(
        X_tr_full, Y_tr_full,
        test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=Y_tr_full
    )

    # --- استانداردسازی per-fold (fit روی Train، اعمال روی Val/Test)
    X_tr, X_va, X_te = standardize_fold(X_tr, X_va, X_te)

    # --- class_weight per-fold روی Train
    classes = np.unique(Y_tr)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=Y_tr)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}

    # --- مدل
    model = create_model((X.shape[1], X.shape[2]))

    # --- آموزش روی Train (بدون استفاده از Val داخلی Keras؛ ولیدیشن را خودمان داریم)
    model.fit(
        X_tr, Y_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        class_weight=class_weight_dict
    )

    # --- انتخاب آستانه با Val (هدف: بهبود F1 کلاس 0)
    val_probs = model.predict(X_va, verbose=0).ravel()
    best_t, best_f1neg = choose_threshold_for_class0(val_probs, Y_va)

    # --- ارزیابی روی Test فولد با آستانه‌ی برگزیده
    te_probs = model.predict(X_te, verbose=0).ravel()
    Y_pred = (te_probs > best_t).astype(int)

    # --- متریک‌ها
    acc  = accuracy_score(Y_te, Y_pred)
    prec = precision_score(Y_te, Y_pred, zero_division=0)
    rec  = recall_score(Y_te, Y_pred, zero_division=0)
    f1   = f1_score(Y_te, Y_pred, zero_division=0)

    # کلاس‌محور
    prec_c0 = precision_score((Y_te == 0).astype(int), (Y_pred == 0).astype(int), zero_division=0)
    rec_c0  = recall_score((Y_te == 0).astype(int), (Y_pred == 0).astype(int), zero_division=0)
    f1_c0   = f1_score((Y_te == 0).astype(int), (Y_pred == 0).astype(int), zero_division=0)

    prec_c1 = precision_score((Y_te == 1).astype(int), (Y_pred == 1).astype(int), zero_division=0)
    rec_c1  = recall_score((Y_te == 1).astype(int), (Y_pred == 1).astype(int), zero_division=0)
    f1_c1   = f1_score((Y_te == 1).astype(int), (Y_pred == 1).astype(int), zero_division=0)

    # --- گزارش متنی
    report = classification_report(Y_te, Y_pred, digits=4)
    all_reports.append(f"Fold {fold_idx} | best_threshold={best_t:.3f} | F1_neg(val)={best_f1neg:.4f}\n{report}\n")

    # --- Confusion Matrix و ذخیره تصویر
    cm = confusion_matrix(Y_te, Y_pred)
    plt.figure(figsize=(5.2, 4.6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['non-easy(0)','easy(1)'],
                yticklabels=['non-easy(0)','easy(1)'])
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion Matrix - Fold {fold_idx}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"cm_fold_{fold_idx:02d}.png"))
    plt.close()

    # --- ردیف CSV
    all_rows.append({
        "fold": fold_idx,
        "best_threshold": best_t,
        "acc": acc,
        "precision_macro": precision_score(Y_te, Y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(Y_te, Y_pred, average='macro', zero_division=0),
        "f1_macro": f1_score(Y_te, Y_pred, average='macro', zero_division=0),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "precision_c0": prec_c0,
        "recall_c0": rec_c0,
        "f1_c0": f1_c0,
        "precision_c1": prec_c1,
        "recall_c1": rec_c1,
        "f1_c1": f1_c1
    })

    fold_idx += 1

# =====================
# ذخیره نتایج
# =====================
import pandas as pd
df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_per_fold.csv"), index=False)

# خلاصه
mean = df[['acc','precision','recall','f1','precision_macro','recall_macro','f1_macro',
           'precision_c0','recall_c0','f1_c0','precision_c1','recall_c1','f1_c1']].mean()
std  = df[['acc','precision','recall','f1','precision_macro','recall_macro','f1_macro',
           'precision_c0','recall_c0','f1_c0','precision_c1','recall_c1','f1_c1']].std()

summary_lines = [
    f"Runs: {N_SPLITS * N_REPEATS}",
    f"Mean ± STD",
    f"Accuracy: {mean['acc']:.4f} ± {std['acc']:.4f}",
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
