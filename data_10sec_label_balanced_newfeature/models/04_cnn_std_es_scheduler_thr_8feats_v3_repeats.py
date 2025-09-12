import os, json, copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
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
RESULT_DIR     = os.path.join("..", "results", "04_cnn_std_es_scheduler_thr_8feats_v3_repeats")
SEEDS          = [7, 17, 23, 42, 57]
TEST_SIZE      = 0.20
VAL_SIZE       = 0.20
EPOCHS         = 120
BATCH_SIZE     = 16
LR             = 1e-3
MIN_C0_RATIO   = 0.20                    # قید سهم حداقل کلاس 0 حین جست‌وجوی آستانه
THR_GRID       = np.linspace(0.30, 0.70, 41)
TOPK_CHECKPTS  = 3                       # تعداد بهترین وزن‌ها برای انسمبل

os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------
# داده‌ها
# -------------------------
X = np.load(os.path.join(DATA_DIR, "X8.npy"))   # (N, 1500, 8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy"))
T, C = X.shape[1], X.shape[2]
print(f"[INFO] X={X.shape}, Y={Y.shape}")

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
class TopKWeights(Callback):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.pool = []   # list of (val_loss, weights)

    def on_epoch_end(self, epoch, logs=None):
        vloss = logs.get('val_loss', None)
        if vloss is None: return
        weights = copy.deepcopy(self.model.get_weights())
        self.pool.append((vloss, weights))
        self.pool.sort(key=lambda x: x[0])
        self.pool = self.pool[:self.k]

    def get_weights_list(self):
        return [w for _, w in self.pool]

def predict_ensemble(model, Xarr, weights_list, batch_size=256):
    # میانگین‌گیری احتمال‌ها میان چند وزن برتر
    probs = []
    for w in weights_list:
        model.set_weights(w)
        p = model.predict(Xarr, batch_size=batch_size, verbose=0).ravel()
        probs.append(p)
    return np.mean(np.stack(probs, axis=0), axis=0)

# -------------------------
# استانداردسازی
# -------------------------
def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, C))
    return scaler

def transform_with(scaler, Xarr):
    X2 = Xarr.reshape(-1, C)
    X2 = scaler.transform(X2)
    return X2.reshape(Xarr.shape[0], T, C)

# -------------------------
# جست‌وجوی آستانه با قید سهم کلاس0
# -------------------------
def best_threshold_with_constraint(val_probs, val_true, grid=THR_GRID, min_c0_ratio=MIN_C0_RATIO):
    best_t, best_macro = 0.5, -1.0
    for t in grid:
        yhat = (val_probs > t).astype(int)
        # قید: حداقل درصد پیش‌بینی‌شده برای کلاس 0
        if np.mean(yhat == 0) < min_c0_ratio:
            continue
        macro = f1_score(val_true, yhat, average='macro', zero_division=0)
        if macro > best_macro:
            best_macro, best_t = macro, t
    # اگر هیچ آستانه‌ای قید را پاس نکرد، قید را کنار بگذار
    if best_macro < 0:
        for t in grid:
            yhat = (val_probs > t).astype(int)
            macro = f1_score(val_true, yhat, average='macro', zero_division=0)
            if macro > best_macro:
                best_macro, best_t = macro, t
    return best_t, best_macro

# -------------------------
# حلقه‌ی تکرار با چند seed
# -------------------------
all_rows = []
for seed in SEEDS:
    print(f"\n===== SEED {seed} =====")
    np.random.seed(seed); tf.random.set_seed(seed)

    # split
    X_tr_full, X_te, Y_tr_full, Y_te = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=seed, stratify=Y
    )
    X_tr, X_va, Y_tr, Y_va = train_test_split(
        X_tr_full, Y_tr_full, test_size=VAL_SIZE, random_state=seed, stratify=Y_tr_full
    )

    # scaling
    scaler = fit_scaler(X_tr)
    X_tr = transform_with(scaler, X_tr)
    X_va = transform_with(scaler, X_va)
    X_te = transform_with(scaler, X_te)

    # class weights
    cw = compute_class_weight(class_weight='balanced', classes=np.unique(Y_tr), y=Y_tr)
    cw = {i: float(w) for i, w in enumerate(cw)}
    print("[CLASS WEIGHTS]", cw)

    # model + callbacks
    model = build_model((T, C))
    topk_cb = TopKWeights(k=TOPK_CHECKPTS)
    cbs = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=1),
        topk_cb
    ]
    model.fit(X_tr, Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_va, Y_va), verbose=0, class_weight=cw, callbacks=cbs)

    # ensemble probs
    weights_list = topk_cb.get_weights_list()
    if not weights_list:
        weights_list = [model.get_weights()]  # fallback

    val_probs = predict_ensemble(model, X_va, weights_list)
    best_thr, val_macro = best_threshold_with_constraint(val_probs, Y_va)
    print(f"[THRESH] seed={seed} | thr={best_thr:.3f} | macroF1(val)={val_macro:.4f}")

    te_probs = predict_ensemble(model, X_te, weights_list)
    Y_pred = (te_probs > best_thr).astype(int)

    # metrics
    acc = accuracy_score(Y_te, Y_pred)
    prec = precision_score(Y_te, Y_pred, zero_division=0)
    rec  = recall_score(Y_te, Y_pred, zero_division=0)
    f1   = f1_score(Y_te, Y_pred, zero_division=0)
    f1m  = f1_score(Y_te, Y_pred, average='macro', zero_division=0)

    # save per-seed
    seed_dir = os.path.join(RESULT_DIR, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    rep = classification_report(Y_te, Y_pred, digits=4)
    with open(os.path.join(seed_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"[THRESH] {best_thr:.3f} (val_macroF1={val_macro:.4f})\n")
        f.write(rep)

    cm = confusion_matrix(Y_te, Y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["non-easy","easy"], yticklabels=["non-easy","easy"])
    plt.title(f"Confusion Matrix (seed={seed}, thr={best_thr:.3f})")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(os.path.join(seed_dir, "confusion_matrix.png")); plt.close()

    all_rows.append({
        "seed": seed, "thr": best_thr,
        "acc": acc, "precision": prec, "recall": rec, "f1": f1, "f1_macro": f1m
    })

# summary
import pandas as pd
df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_per_seed.csv"), index=False)
mean = df[["acc","precision","recall","f1","f1_macro"]].mean()
std  = df[["acc","precision","recall","f1","f1_macro"]].std()

summary_lines = [
    f"Seeds: {SEEDS}",
    f"MIN_C0_RATIO: {MIN_C0_RATIO}",
    f"TOPK_CHECKPOINTS: {TOPK_CHECKPTS}",
    "Mean ± STD over seeds",
    f"Accuracy: {mean['acc']:.4f} ± {std['acc']:.4f}",
    f"Precision: {mean['precision']:.4f} ± {std['precision']:.4f}",
    f"Recall: {mean['recall']:.4f} ± {std['recall']:.4f}",
    f"F1: {mean['f1']:.4f} ± {std['f1']:.4f}",
    f"F1 (macro): {mean['f1_macro']:.4f} ± {std['f1_macro']:.4f}",
]
with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\n".join(summary_lines))
print("Saved to:", RESULT_DIR)
