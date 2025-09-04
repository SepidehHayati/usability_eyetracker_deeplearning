# 02_cnn_windowvote_thr_tta_v2.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -----------------------
# تنظیمات و مسیرها
# -----------------------
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "02_cnn_windowvote_thr_tta_v2")
os.makedirs(RESULT_DIR, exist_ok=True)

X_NAME = "X8.npy"   # (N, 1500, 8)
Y_NAME = "Y8.npy"   # (N,)

# پنجره‌گذاری و TTA
USE_WINDOWS   = True
N_WINDOWS     = 5            # مثلا 1500/5=300
AGG_WINDOWS   = "mean"       # mean یا median

USE_TTA       = True
TTA_CROP_LEN  = 1200         # طول هر کراپ TTA (<= T)
TTA_N_CROPS   = 3            # تعداد کراپ‌ها
AGG_TTA       = "mean"       # mean یا median

# آموزش
EPOCHS        = 100
BATCH_SIZE    = 16
LR            = 1e-3
RANDOM_STATE  = 42
VAL_SIZE      = 0.2          # نسبت از TRAIN که برای VAL کنار می‌گذاریم

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# -----------------------
# بارگذاری داده
# -----------------------
X = np.load(os.path.join(DATA_DIR, X_NAME))   # (N, T, C)
Y = np.load(os.path.join(DATA_DIR, Y_NAME)).astype(int)
N, T, C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}")

# -----------------------
# توابع کمکی پیش‌بینی
# -----------------------
def predict_with_windows(model, Xarr, n_windows=5, agg="mean", batch_size=256):
    """پنجره‌گذاری یکنواخت روی محور زمان و تجمیع خروجی‌ها"""
    N_, T_, C_ = Xarr.shape
    assert T_ % n_windows == 0, "طول توالی باید بر n_windows بخش‌پذیر باشد."
    Tw = T_ // n_windows
    Xw = Xarr.reshape(N_ * n_windows, Tw, C_)
    # خروجی مدل sigmoid است؛ همان احتمال را برمی‌گرداند
    probs_w = model.predict(Xw, batch_size=batch_size, verbose=0).ravel()
    probs_w = probs_w.reshape(N_, n_windows)
    if agg == "mean":
        return probs_w.mean(axis=1)
    elif agg == "median":
        return np.median(probs_w, axis=1)
    else:
        raise ValueError("Unsupported agg method")

def tta_crops_predict(model, Xarr, crop_len=1200, n_crops=3, agg="mean", batch_size=256):
    """Test-Time Augmentation با کراپ‌های زمانی و تجمیع نتایج"""
    N_, T_, C_ = Xarr.shape
    crop_len = min(crop_len, T_)
    if n_crops == 1 or crop_len == T_:
        starts = [0]
    else:
        if n_crops == 2:
            starts = [0, T_ - crop_len]
        else:
            gap = (T_ - crop_len) // (n_crops - 1) if (n_crops - 1) > 0 else 0
            starts = [i * gap for i in range(n_crops)]
            starts[-1] = T_ - crop_len

    all_probs = []
    for s in starts:
        Xe = Xarr[:, s:s+crop_len, :]
        probs = model.predict(Xe, batch_size=batch_size, verbose=0).ravel()  # خروجی probability
        all_probs.append(probs)

    all_probs = np.stack(all_probs, axis=1)  # (N, n_crops)
    if agg == "mean":
        return all_probs.mean(axis=1)
    elif agg == "median":
        return np.median(all_probs, axis=1)
    else:
        raise ValueError("Unsupported agg method")

def combined_predict(model, Xarr):
    """ابتدا (اختیاری) پنجره‌گذاری، سپس (اختیاری) TTA روی همان ورودی یا برعکس؟
       برای سادگی، هر کدام فعال بود، همان اعمال می‌شود و اگر هر دو فعال بود:
       - ابتدا TTA روی ورودی کامل و سپس پنجره‌گذاری روی هر کراپ منطقی نیست (دو بار تکه‌تکه می‌شود)
       بنابراین اینجا یکی را انتخاب می‌کنیم: اگر USE_WINDOWS=True، از window-voting استفاده می‌کنیم؛
       در غیر اینصورت اگر USE_TTA=True، از TTA استفاده می‌کنیم؛ و اگر هیچ‌کدام، از full-seq استفاده می‌کنیم.
    """
    if USE_WINDOWS:
        return predict_with_windows(model, Xarr, n_windows=N_WINDOWS, agg=AGG_WINDOWS, batch_size=256)
    elif USE_TTA:
        return tta_crops_predict(model, Xarr, crop_len=TTA_CROP_LEN, n_crops=TTA_N_CROPS, agg=AGG_TTA, batch_size=256)
    else:
        return model.predict(Xarr, batch_size=256, verbose=0).ravel()

# -----------------------
# تقسیم داده: train/val/test
# -----------------------
# ابتدا Train/Test
X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=RANDOM_STATE, stratify=Y
)

# سپس از Train بخشی را برای Val جدا می‌کنیم
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_full, Y_train_full, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=Y_train_full
)

print(f"[SPLIT] Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

# -----------------------
# کلاس‌ویت‌ها
# -----------------------
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print("[CLASS WEIGHTS]", class_weight_dict)

# -----------------------
# مدل: GAP برای سازگاری با طول متغیر
# -----------------------
def build_model(input_channels, lr=1e-3):
    model = Sequential([
        tf.keras.Input(shape=(None, input_channels)),  # طول زمان متغیر
        Conv1D(32, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        GlobalAveragePooling1D(),      # << جایگزین Flatten
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(1, activation='sigmoid') # خروجی احتمال
    ])
    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model(C, lr=LR)
model.summary()

callbacks = [
    EarlyStopping(patience=12, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.5, patience=6, min_lr=1e-6, monitor='val_loss', verbose=1)
]

history = model.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, Y_val),
    class_weight=class_weight_dict,
    verbose=1,
    callbacks=callbacks
)

# -----------------------
# انتخاب آستانه با بیشینه‌سازی Macro-F1 روی VAL
# -----------------------
val_probs = combined_predict(model, X_val)
grid = np.linspace(0.2, 0.8, 25)
best_t, best_f1 = 0.5, -1.0
for t in grid:
    yhat = (val_probs > t).astype(int)
    f1m = f1_score(Y_val, yhat, average='macro', zero_division=0)
    if f1m > best_f1:
        best_f1, best_t = f1m, t
print(f"[THRESHOLD] best_t={best_t:.3f} | MacroF1(val)={best_f1:.4f}")

# -----------------------
# ارزیابی روی TEST با همان روش پیش‌بینی
# -----------------------
test_probs = combined_predict(model, X_test)
Y_pred = (test_probs > best_t).astype(int)

rep = classification_report(Y_test, Y_pred, digits=4)
print(rep)

# ذخیره گزارش
with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(rep)

# ذخیره ماتریس درهم‌ریختگی
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["non-easy(0)", "easy(1)"], yticklabels=["non-easy(0)", "easy(1)"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (thr={best_t:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.close()

# ذخیره منحنی یادگیری
plt.figure(figsize=(7,5))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "loss_curve.png")); plt.close()

print("Saved to:", RESULT_DIR)
