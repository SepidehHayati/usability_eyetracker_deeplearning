import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =========================
# reproducibility (optional)
# =========================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# paths & data
# =========================
DATA_DIR   = os.path.join("..", "data")
RESULT_DIR = os.path.join("..", "results", "04_cnn_std_es_scheduler_thr_8feats_v2")
os.makedirs(RESULT_DIR, exist_ok=True)

X = np.load(os.path.join(DATA_DIR, "X8.npy"))   # (N, 1500, 8)
Y = np.load(os.path.join(DATA_DIR, "Y8.npy"))   # (N,)
print(f"[INFO] X={X.shape}, Y={Y.shape}")

# =========================
# split: test جدا؛ سپس val از train برای آستانه
# =========================
X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=SEED, stratify=Y
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_full, Y_train_full, test_size=0.2, random_state=SEED, stratify=Y_train_full
)

# =========================
# standardize (fit ONLY on train)
# =========================
T, C = X.shape[1], X.shape[2]
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, C))

def transform(arr):
    arr2 = arr.reshape(-1, C)
    arr2 = scaler.transform(arr2)
    return arr2.reshape(arr.shape[0], T, C)

X_train = transform(X_train)
X_val   = transform(X_val)
X_test  = transform(X_test)

# =========================
# class weights (balanced)
# =========================
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print("[CLASS WEIGHTS]", class_weight_dict)

# =========================
# model: Conv blocks -> GAP -> Dense
# (کمی قوی‌تر و GAP به‌جای Flatten)
# =========================
def build_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=7, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.30),

        Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.30),

        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.50),

        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model((T, C))
model.summary()

# =========================
# callbacks
# =========================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=1)
]

# =========================
# train
# =========================
history = model.fit(
    X_train, Y_train,
    epochs=150,
    batch_size=8,                 # <<— کوچکتر برای بهبود generalization روی دیتای کم
    validation_data=(X_val, Y_val),
    verbose=1,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# =========================
# threshold search (macro-F1) + حداقل سهم کلاس 0
# =========================
THR_GRID     = np.linspace(0.35, 0.75, 21)
MIN_C0_RATIO = 0.20  # کمینه سهم پیش‌بینی کلاس 0 تا کلاس 0 فراموش نشود

def best_threshold_with_constraint(val_probs, val_true, grid=THR_GRID, min_c0_ratio=MIN_C0_RATIO):
    best_t, best_macro = 0.5, -1.0
    for t in grid:
        yhat = (val_probs > t).astype(int)
        # اگر مدل فقط یک کلاس گفت، یا کلاس 0 کمتر از حداقل سهم بود، رد کن
        if len(np.unique(yhat)) < 2:
            continue
        if np.mean(yhat == 0) < min_c0_ratio:
            continue
        macro = f1_score(val_true, yhat, average='macro', zero_division=0)
        if macro > best_macro:
            best_macro, best_t = macro, t
    # اگر هیچ چیزی با قید نشد، بدون قید بهترین رو بگیر
    if best_macro < 0:
        for t in grid:
            yhat = (val_probs > t).astype(int)
            if len(np.unique(yhat)) < 2:
                continue
            macro = f1_score(val_true, yhat, average='macro', zero_division=0)
            if macro > best_macro:
                best_macro, best_t = macro, t
    return best_t, best_macro

val_probs = model.predict(X_val, verbose=0).ravel()
best_thr, best_macro = best_threshold_with_constraint(val_probs, Y_val)
print(f"[THRESHOLD] best_thr={best_thr:.3f} | macroF1(val)={best_macro:.4f}")

# =========================
# evaluate on test
# =========================
test_probs = model.predict(X_test, verbose=0).ravel()
Y_pred = (test_probs > best_thr).astype(int)

acc = accuracy_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred, digits=4)
print(report)
print(f"[TEST] acc={acc:.4f} | thr={best_thr:.3f}")

# =========================
# save: report + confusion matrix
# =========================
with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Best threshold (from val): {best_thr:.3f}\n")
    f.write(f"Accuracy (test): {acc:.4f}\n\n")
    f.write(report)

cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["non-easy", "easy"], yticklabels=["non-easy", "easy"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (thr={best_thr:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.close()

print("Saved to:", RESULT_DIR)
