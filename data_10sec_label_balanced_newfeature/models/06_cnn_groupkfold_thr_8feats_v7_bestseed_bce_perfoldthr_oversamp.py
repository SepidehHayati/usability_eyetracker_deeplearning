import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             accuracy_score, precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.losses import BinaryCrossentropy

# -------------------------
# پارامترها
# -------------------------
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "06_cnn_groupkfold_thr_8feats_v7_bestseed_bce_perfoldthr_oversamp")
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE   = 42
N_SPLITS       = 6
SEEDS          = [7, 17, 23]       # کاندید seedها؛ در هر فولد بهترین انتخاب می‌شود
EPOCHS         = 140
BATCH_SIZE     = 16
LR             = 1e-3
TOPK_CHECKPTS  = 3

MIN_C0_RATIO   = 0.25             # کمی سخت‌گیرتر از v6 (=0.15)
LABEL_SMOOTH   = 0.02              # smoothing کوچک
OVERSAMP_EQUAL = True              # به نسبت 1:1 متعادل می‌کنیم

# -------------------------
# داده‌ها
# -------------------------
X = np.load(os.path.join(DATA_DIR, "X8.npy"))  # (N, 1500, 8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy"))
G = np.load(os.path.join(DATA_DIR, "G8.npy"))
T, C = X.shape[1], X.shape[2]
print(f"[INFO] X={X.shape}, Y={Y.shape}, G={G.shape}")

# -------------------------
# مدل (همان ساختار v6 با Dropout ملایم)
# -------------------------
def build_model(input_shape):
    m = Sequential([
        Input(shape=input_shape),
        Conv1D(32, 7, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.20),

        Conv1D(64, 5, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.20),

        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.20),

        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.30),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(LR),
              loss=BinaryCrossentropy(label_smoothing=LABEL_SMOOTH),
              metrics=['accuracy'])
    return m

# -------------------------
# کال‌بک: نگه‌داری بهترین k وزن
# -------------------------
class TopKWeights(Callback):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.pool = []  # list of (val_loss, weights)
    def on_epoch_end(self, epoch, logs=None):
        vloss = logs.get('val_loss')
        if vloss is None: return
        weights_copy = [w.copy() for w in self.model.get_weights()]
        self.pool.append((vloss, weights_copy))
        self.pool.sort(key=lambda x: x[0])
        self.pool = self.pool[:self.k]
    def get_weights_list(self):
        return [w for _, w in self.pool]

def predict_with_weights_list(model, Xarr, weights_list, batch_size=256):
    preds = []
    for w in weights_list:
        model.set_weights(w)
        p = model.predict(Xarr, batch_size=batch_size, verbose=0).ravel()
        preds.append(p)
    return np.mean(np.stack(preds, axis=0), axis=0)

# -------------------------
# Oversampling فقط روی TRAIN فولد
# -------------------------
def oversample_equal(Xtr, Ytr, jitter_std_frac=0.03, rng=None):
    """
    نمونه‌برداری با جایگذاری از کلاس اقلیت تا 1:1 شدن با اکثریت.
    روی نمونه‌های کپی، نویز گاوسی خیلی کوچک می‌زنیم (نسبت به std هر کانال).
    """
    if rng is None:
        rng = np.random.default_rng(123)
    # شمارش
    n0 = np.sum(Ytr == 0); n1 = np.sum(Ytr == 1)
    if n0 == 0 or n1 == 0:
        return Xtr, Ytr
    # کلاس اقلیت
    minority = 0 if n0 < n1 else 1
    maj_count = max(n0, n1)
    idx_min = np.where(Ytr == minority)[0]
    need = maj_count - len(idx_min)
    if need <= 0:
        return Xtr, Ytr

    extra_idx = rng.choice(idx_min, size=need, replace=True)
    X_extra = Xtr[extra_idx].copy()
    # نویز ملایم: σ = jitter_std_frac * std کانالی TRAIN
    ch_std = Xtr.reshape(-1, Xtr.shape[-1]).std(axis=0) + 1e-8
    noise = rng.normal(loc=0.0, scale=(jitter_std_frac * ch_std), size=X_extra.shape)
    X_extra = X_extra + noise
    Y_extra = Ytr[extra_idx]

    Xb = np.concatenate([Xtr, X_extra], axis=0)
    Yb = np.concatenate([Ytr, Y_extra], axis=0)
    p = rng.permutation(len(Yb))
    return Xb[p], Yb[p]

# -------------------------
# استانداردسازی per-fold
# -------------------------
def fit_scaler(Xtr):
    sc = StandardScaler()
    sc.fit(Xtr.reshape(-1, C))
    return sc

def transform_with(sc, Xarr):
    X2 = Xarr.reshape(-1, C)
    X2 = sc.transform(X2)
    return X2.reshape(Xarr.shape[0], T, C)

# -------------------------
# آستانهٔ بهینه با قید (گرید بر پایهٔ درصدی‌ها)
# -------------------------
def best_threshold_with_constraint(val_probs, val_true, min_c0_ratio=MIN_C0_RATIO):
    # گرید آستانه از 5th تا 95th percentiles
    p_lo, p_hi = np.percentile(val_probs, 5), np.percentile(val_probs, 95)
    grid = np.linspace(p_lo, p_hi, 41)
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
# Outer CV (Group-aware) + انتخاب بهترین seed در هر فولد
# -------------------------
outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

rows, reports = [], []
fold_thresholds, fold_best_seeds = [], []

fold = 1
for tr_idx, te_idx in outer.split(X, Y, groups=G):
    print(f"\n===== Fold {fold} =====")
    X_tr_full, X_te = X[tr_idx], X[te_idx]
    Y_tr_full, Y_te = Y[tr_idx], Y[te_idx]
    G_tr_full       = G[tr_idx]

    # inner split برای val (گروه‌محور)
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    inner_tr, inner_va = next(inner.split(X_tr_full, Y_tr_full, groups=G_tr_full))
    X_tr, Y_tr = X_tr_full[inner_tr], Y_tr_full[inner_tr]
    X_va, Y_va = X_tr_full[inner_va], Y_tr_full[inner_va]

    # ---- Oversampling فقط روی TRAIN داخلی ----
    X_tr_os, Y_tr_os = oversample_equal(X_tr, Y_tr, jitter_std_frac=0.03, rng=np.random.default_rng(1234))

    # استانداردسازی را فقط روی TRAIN oversampled fit می‌کنیم
    sc = fit_scaler(X_tr_os)
    X_tr_s = transform_with(sc, X_tr_os)
    X_va_s = transform_with(sc, X_va)
    X_te_s = transform_with(sc, X_te)

    # کلاس‌ویت روی TRAIN اصلی (نه oversampled) یا oversampled؟ هر دو قابل دفاع‌اند.
    # اینجا از TRAIN اصلی می‌گیریم تا weightها اغراق نشوند.
    cw = compute_class_weight(class_weight='balanced', classes=np.unique(Y_tr), y=Y_tr)
    class_weight_dict = {int(c): float(w) for c, w in zip(np.unique(Y_tr), cw)}
    print("[CLASS WEIGHTS]", class_weight_dict)

    # ----- رقابت seed ها؛ برای هر seed بهترین top-k و بهترین thr را می‌گیریم -----
    best_seed = None
    best_seed_thr = 0.5
    best_seed_macro = -1.0
    best_seed_te_probs = None

    for seed in SEEDS:
        np.random.seed(seed); tf.random.set_seed(seed)
        model = build_model((T, C))
        topk = TopKWeights(k=TOPK_CHECKPTS)
        cbs = [
            EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=0),
            topk
        ]
        model.fit(X_tr_s, Y_tr_os, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(X_va_s, Y_va), verbose=0,
                  callbacks=cbs, class_weight=class_weight_dict)

        weights_list = topk.get_weights_list() or [model.get_weights()]
        val_probs = predict_with_weights_list(model, X_va_s, weights_list)
        te_probs  = predict_with_weights_list(model, X_te_s, weights_list)

        t_seed, macro_val = best_threshold_with_constraint(val_probs, Y_va, min_c0_ratio=MIN_C0_RATIO)

        if macro_val > best_seed_macro:
            best_seed_macro = macro_val
            best_seed_thr   = t_seed
            best_seed_te_probs = te_probs
            best_seed = seed

    # ارزیابی با بهترین seed این فولد
    Y_pred = (best_seed_te_probs > best_seed_thr).astype(int)

    acc  = accuracy_score(Y_te, Y_pred)
    prec = precision_score(Y_te, Y_pred, zero_division=0)
    rec  = recall_score(Y_te, Y_pred, zero_division=0)
    f1   = f1_score(Y_te, Y_pred, zero_division=0)
    f1m  = f1_score(Y_te, Y_pred, average='macro', zero_division=0)

    rep = classification_report(Y_te, Y_pred, digits=4)
    reports.append(f"Fold {fold} | seed={best_seed} | thr={best_seed_thr:.3f} (val_macroF1={best_seed_macro:.4f})\n{rep}\n")

    rows.append({"fold": fold, "seed": best_seed, "thr": best_seed_thr,
                 "acc": acc, "precision": prec, "recall": rec, "f1": f1, "f1_macro": f1m})

    cm = confusion_matrix(Y_te, Y_pred)
    plt.figure(figsize=(5.5, 4.8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['non-easy','easy'], yticklabels=['non-easy','easy'])
    plt.title(f"Confusion Matrix - Fold {fold} (seed={best_seed}, thr={best_seed_thr:.3f})")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"cm_fold_{fold:02d}.png")); plt.close()

    fold_thresholds.append(best_seed_thr)
    fold_best_seeds.append(best_seed)
    fold += 1

# -------------------------
# خلاصه و ذخیره
# -------------------------
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULT_DIR, "metrics_per_fold.csv"), index=False)

mean = df.mean(numeric_only=True); std = df.std(numeric_only=True)
summary = [
    f"Runs: {len(df)}",
    f"SEEDS: {SEEDS}",
    f"MIN_C0_RATIO: {MIN_C0_RATIO}",
    f"LABEL_SMOOTH: {LABEL_SMOOTH}",
    f"TOPK_CHECKPOINTS: {TOPK_CHECKPTS}",
    f"Per-fold best seeds: {', '.join(map(str, fold_best_seeds))}",
    f"Per-fold thresholds: {', '.join([f'{t:.3f}' for t in fold_thresholds])}",
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
