import os, json, copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

# -------------------------
# پارامترها
# -------------------------
DATA_DIR       = os.path.join("..", "data")
RESULT_DIR     = os.path.join("..", "results", "06_cnn_groupkfold_thr_8feats_v3")
RANDOM_STATE   = 42
N_SPLITS       = 6
EPOCHS         = 120
BATCH_SIZE     = 16
LR             = 1e-3
MIN_C0_RATIO   = 0.20
THR_GRID       = np.linspace(0.30, 0.70, 41)
TOPK_CHECKPTS  = 3

os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------
# داده‌ها
# -------------------------
X = np.load(os.path.join(DATA_DIR, "X8.npy"))   # (N, 1500, 8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy"))
G = np.load(os.path.join(DATA_DIR, "G8.npy"))   # شناسه‌ی کاربر برای GroupKFold
T, C = X.shape[1], X.shape[2]
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}")

# -------------------------
# مدل
# -------------------------
def build_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=7, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.25),

        Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.25),

        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.25),

        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(LR), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------------------------
# کال‌بک: نگه‌داشتن بهترین k وزن
# -------------------------
import copy
class TopKWeights(tf.keras.callbacks.Callback):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.pool = []
    def on_epoch_end(self, epoch, logs=None):
        vloss = logs.get('val_loss', None)
        if vloss is None: return
        w = copy.deepcopy(self.model.get_weights())
        self.pool.append((vloss, w))
        self.pool.sort(key=lambda x: x[0])
        self.pool = self.pool[:self.k]
    def get_weights_list(self):
        return [w for _, w in self.pool]

def predict_ensemble(model, Xarr, weights_list, batch_size=256):
    probs = []
    for w in weights_list:
        model.set_weights(w)
        p = model.predict(Xarr, batch_size=batch_size, verbose=0).ravel()
        probs.append(p)
    return np.mean(np.stack(probs, axis=0), axis=0)

# -------------------------
# استانداردسازی
# -------------------------
from sklearn.model_selection import StratifiedGroupKFold
def fit_scaler(Xtr):
    sc = StandardScaler()
    sc.fit(Xtr.reshape(-1, C))
    return sc

def transform_with(sc, Xarr):
    X2 = Xarr.reshape(-1, C)
    X2 = sc.transform(X2)
    return X2.reshape(Xarr.shape[0], T, C)

# -------------------------
# آستانه
# -------------------------
def best_threshold_with_constraint(val_probs, val_true, grid=THR_GRID, min_c0_ratio=MIN_C0_RATIO):
    best_t, best_macro = 0.5, -1.0
    for t in grid:
        yhat = (val_probs > t).astype(int)
        if np.mean(yhat == 0) < min_c0_ratio:
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

# -------------------------
# Cross-user CV با inner-VAL گروه‌محور
# -------------------------
outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

all_val_probs = []
all_val_true  = []
fold_outputs  = []   # نگه‌داشتن (test_probs, Y_test) برای ارزیابی با آستانه‌ی سراسری

fold = 1
for tr_idx, te_idx in outer.split(X, Y, groups=G):
    print(f"\n===== Fold {fold} =====")
    X_tr_full, X_te = X[tr_idx], X[te_idx]
    Y_tr_full, Y_te = Y[tr_idx], Y[te_idx]
    G_tr_full       = G[tr_idx]

    # inner group-aware split
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_tr, inner_va = next(inner.split(X_tr_full, Y_tr_full, groups=G_tr_full))
    X_tr, Y_tr = X_tr_full[inner_tr], Y_tr_full[inner_tr]
    X_va, Y_va = X_tr_full[inner_va], Y_tr_full[inner_va]

    # scaling fit روی TRAIN
    sc = fit_scaler(X_tr)
    X_tr = transform_with(sc, X_tr)
    X_va = transform_with(sc, X_va)
    X_te = transform_with(sc, X_te)

    # class weights
    cw = compute_class_weight(class_weight='balanced', classes=np.unique(Y_tr), y=Y_tr)
    cw = {i: float(w) for i, w in enumerate(cw)}
    print("[CLASS WEIGHTS]", cw)

    # train
    model = build_model((T, C))
    topk = TopKWeights(k=TOPK_CHECKPTS)
    cbs = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=1),
        topk
    ]
    model.fit(X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_va, Y_va), verbose=0, class_weight=cw, callbacks=cbs)

    weights_list = topk.get_weights_list()
    if not weights_list:
        weights_list = [model.get_weights()]

    val_probs = predict_ensemble(model, X_va, weights_list)
    all_val_probs.append(val_probs)
    all_val_true.append(Y_va)

    te_probs = predict_ensemble(model, X_te, weights_list)
    fold_outputs.append((te_probs, Y_te))
    fold += 1

# آستانه‌ی سراسری
all_val_probs = np.concatenate(all_val_probs)
all_val_true  = np.concatenate(all_val_true)
t_global, macro_val = best_threshold_with_constraint(all_val_probs, all_val_true)
print(f"\n[GLOBAL] best_threshold={t_global:.3f} | MacroF1(val_all)={macro_val:.4f}")

# ارزیابی هر فولد با آستانه‌ی سراسری
import pandas as pd
rows, reports = [], []
for i, (te_probs, Y_te) in enumerate(fold_outputs, start=1):
    Y_pred = (te_probs > t_global).astype(int)
    acc  = accuracy_score(Y_te, Y_pred)
    prec = precision_score(Y_te, Y_pred, zero_division=0)
    rec  = recall_score(Y_te, Y_pred, zero_division=0)
    f1   = f1_score(Y_te, Y_pred, zero_division=0)
    f1m  = f1_score(Y_te, Y_pred, average='macro', zero_division=0)

    rep = classification_report(Y_te, Y_pred, digits=4)
    reports.append(f"Fold {i}\n{rep}\n")

    rows.append({"fold": i, "acc": acc, "precision": prec, "recall": rec, "f1": f1, "f1_macro": f1m})

    # confusion matrix
    cm = confusion_matrix(Y_te, Y_pred)
    plt.figure(figsize=(5.5,4.8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['non-easy','easy'],
                yticklabels=['non-easy','easy'])
    plt.title(f"Confusion Matrix - Fold {i} (thr={t_global:.3f})")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"cm_fold_{i:02d}.png")); plt.close()

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_per_fold.csv"), index=False)

mean = df.mean(numeric_only=True); std = df.std(numeric_only=True)
summary = [
    f"Global Threshold: {t_global:.3f}",
    f"Runs: {len(rows)}",
    f"MIN_C0_RATIO: {MIN_C0_RATIO}",
    f"TOPK_CHECKPOINTS: {TOPK_CHECKPTS}",
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
    f.write("\n".join(reports))

print("\n".join(summary))
print("Saved to:", RESULT_DIR)
