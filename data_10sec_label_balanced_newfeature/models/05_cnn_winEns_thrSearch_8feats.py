# 05_cnn_winEns_thrSearch_8feats.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_recall_fscore_support
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =========================
#      Config & Paths
# =========================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "05_cnn_winEns_thrSearch_8feats")
os.makedirs(RESULT_DIR, exist_ok=True)

X_NAME, Y_NAME = "X8.npy", "Y8.npy"    # (N, 1500, 8)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Train/Val/Test & Window params
TEST_SIZE   = 0.2
VAL_SIZE    = 0.2
N_WINDOWS   = 6           # تعداد پنجره‌ها روی هر نمونه
WIN_LEN     = 300         # طول هر پنجره
AGG         = "median"    # تجمیع: 'mean' یا 'median'
LR          = 1e-3
EPOCHS      = 100
BATCH_SIZE  = 16

# Threshold search
THR_GRID       = np.linspace(0.05, 0.95, 91)  # گام 0.01
MIN_C0_RATIO   = 0.30                         # حداقل سهم پیش‌بینی کلاس صفر در ولیدیشن
OPTIMIZE_FOR   = "macro_f1"                   # یا "balanced"

# =========================
#          Load
# =========================
X = np.load(os.path.join(DATA_DIR, X_NAME))  # (N, T=1500, C=8)
Y = np.load(os.path.join(DATA_DIR, Y_NAME)).astype(int)

N, T, C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}")

# =========================
#     Train / Test split
# =========================
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=SEED, stratify=Y
)

# از train_full یک ولیدیشن جدا کنیم
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=VAL_SIZE, random_state=SEED, stratify=y_train_full
)

# =========================
#    Standardize per split
# =========================
def fit_transform_standardize(Xtr, Xva, Xte):
    """
    Standardize روی کانال‌ها:
    scaler روی (نمونه‌ها × زمان، کانال‌ها) فیت می‌شود.
    """
    T = Xtr.shape[1]; C = Xtr.shape[2]
    scaler = StandardScaler()
    Xtr2 = Xtr.reshape(-1, C)
    scaler.fit(Xtr2)

    Xtr_s = scaler.transform(Xtr2).reshape(Xtr.shape[0], T, C)
    Xva_s = scaler.transform(Xva.reshape(-1, C)).reshape(Xva.shape[0], T, C)
    Xte_s = scaler.transform(Xte.reshape(-1, C)).reshape(Xte.shape[0], T, C)
    return Xtr_s, Xva_s, Xte_s, scaler

X_train, X_val, X_test, scaler = fit_transform_standardize(X_train, X_val, X_test)

# =========================
#       Class Weights
# =========================
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print("[INFO] class_weight:", class_weight_dict)

# =========================
#        Build Model
# =========================
def build_model(n_channels, lr=1e-3):
    """
    ورودی با طول زمانی متغیر (None) تا مدل هم برای طول 1500 (آموزش)
    و هم برای طول 300 (پنجره‌ها در استنتاج) سازگار باشد.
    """
    model = Sequential([
        tf.keras.Input(shape=(None, n_channels)),  # NOTE: variable time length

        Conv1D(32, kernel_size=11, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model(C, lr=LR)
model.summary()

# =========================
#       Train Model
# =========================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    class_weight=class_weight_dict,
    verbose=1,
    callbacks=callbacks
)

# =========================
#   Windowing & Inference
# =========================
def make_windows(Xarr, win_len=300, n_windows=6):
    """
    Xarr: (N, T, C)
    برشِ منظم n_windows پنجره با طول win_len از طول T.
    """
    N, T, C = Xarr.shape
    if win_len > T:
        raise ValueError(f"win_len={win_len} > T={T}")
    if n_windows == 1:
        starts = [max(0, (T - win_len)//2)]
    else:
        # شروع‌ها را یکنواخت روی بازه [0, T-win_len] پخش می‌کنیم
        starts = np.linspace(0, T - win_len, n_windows).astype(int).tolist()

    # ساخت آرایه پنجره‌ها
    Xw_list = []
    idx_map = []   # (sample_index, window_index)
    for i in range(N):
        for j, s in enumerate(starts):
            Xw_list.append(Xarr[i, s:s+win_len, :])
            idx_map.append((i, j))
    Xw = np.stack(Xw_list, axis=0)  # (N*n_windows, win_len, C)
    return Xw, idx_map, starts

def predict_with_windows(model, Xarr, win_len=300, n_windows=6, agg="median", batch_size=256):
    """
    خروجی: probs شکل (N,)
    """
    Xw, idx_map, starts = make_windows(Xarr, win_len=win_len, n_windows=n_windows)
    probs_w = model.predict(Xw, batch_size=batch_size, verbose=0).ravel()

    # بازسازی به شکل (N, n_windows)
    N = Xarr.shape[0]
    W = n_windows
    M = np.full((N, W), np.nan, dtype=np.float32)
    for k, (i, j) in enumerate(idx_map):
        M[i, j] = probs_w[k]

    if agg == "mean":
        probs = np.nanmean(M, axis=1)
    else:
        probs = np.nanmedian(M, axis=1)
    return probs

# محاسبه احتمال روی VAL و TEST با پنجره‌ها
val_probs = predict_with_windows(model, X_val, win_len=WIN_LEN, n_windows=N_WINDOWS, agg=AGG)
test_probs = predict_with_windows(model, X_test, win_len=WIN_LEN, n_windows=N_WINDOWS, agg=AGG)

# =========================
#    Threshold Search
# =========================
def find_best_threshold(probs, y_true,
                        grid=THR_GRID,
                        min_c0_ratio=MIN_C0_RATIO,
                        optimize_for=OPTIMIZE_FOR):
    """
    ابتدا با قید سهم حداقلی کلاس 0، macro-F1 (یا balanced accuracy) را بیشینه می‌کنیم.
    اگر هیچ t با قید یافت نشد، قید را برمی‌داریم.
    """
    def score_fn(y_true, y_pred):
        if optimize_for == "balanced":
            # Balanced accuracy ~= میانگین Recall دو کلاس
            # از sklearn.metrics balanced_accuracy_score می‌شد استفاده کرد،
            # ولی برای پرهیز از وابستگی اضافی، از همان macro-recall هم می‌توان بهره برد.
            p, r, f, s = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            return r
        else:
            return f1_score(y_true, y_pred, average='macro', zero_division=0)

    best_t, best_s = None, -1.0
    # فاز 1: با قید
    for t in grid:
        y_hat = (probs >= t).astype(int)
        if len(np.unique(y_hat)) < 2:
            continue
        share_c0 = np.mean(y_hat == 0)
        if share_c0 < min_c0_ratio:
            continue
        s = score_fn(y_true, y_hat)
        if s > best_s:
            best_s, best_t = s, t

    # فاز 2: بدون قید اگر چیزی پیدا نشد
    if best_t is None:
        for t in grid:
            y_hat = (probs >= t).astype(int)
            if len(np.unique(y_hat)) < 2:
                continue
            s = score_fn(y_true, y_hat)
            if s > best_s:
                best_s, best_t = s, t

    return best_t, best_s

best_thr, best_score = find_best_threshold(val_probs, y_val,
                                           grid=THR_GRID,
                                           min_c0_ratio=MIN_C0_RATIO,
                                           optimize_for=OPTIMIZE_FOR)
print(f"[THR] best_thr={best_thr:.3f} | val_{OPTIMIZE_FOR}={best_score:.4f}")

# =========================
#       Evaluation
# =========================
y_pred = (test_probs >= best_thr).astype(int)

report = classification_report(y_test, y_pred, digits=4)
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["non-easy (0)", "easy (1)"],
            yticklabels=["non-easy (0)", "easy (1)"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (thr={best_thr:.3f}, agg={AGG}, W={N_WINDOWS}x{WIN_LEN})")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.close()

# =========================
#        Save Results
# =========================
# ذخیره گزارش متنی
with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Best Threshold (val, {OPTIMIZE_FOR}): {best_thr:.3f}\n")
    f.write(report)

# ذخیره تنظیمات و خلاصه
with open(os.path.join(RESULT_DIR, "run_summary.txt"), "w", encoding="utf-8") as f:
    f.write(
        "\n".join([
            f"X: {X.shape}, Y: {Y.shape}",
            f"Train/Val/Test sizes: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}",
            f"Windows: n_windows={N_WINDOWS}, win_len={WIN_LEN}, agg={AGG}",
            f"LR={LR}, epochs={EPOCHS}, batch_size={BATCH_SIZE}",
            f"class_weight={class_weight_dict}",
            f"Threshold grid: [{THR_GRID[0]:.2f}..{THR_GRID[-1]:.2f}] step~{THR_GRID[1]-THR_GRID[0]:.2f}",
            f"Min C0 share: {MIN_C0_RATIO}, optimize_for={OPTIMIZE_FOR}",
            f"Best thr (val): {best_thr:.3f} | best val {OPTIMIZE_FOR}: {best_score:.4f}"
        ])
    )

print("Saved to:", RESULT_DIR)
