import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -------------------------
# مسیر داده‌ها (X8/Y8)
# -------------------------
data_dir = os.path.join("..", "data")
X = np.load(os.path.join(data_dir, "X8.npy"))   # (N, 1500, 8)
Y = np.load(os.path.join(data_dir, "Y8.npy"))   # (N,)
print(f"[INFO] X={X.shape}, Y={Y.shape}")

# -------------------------
# split: test جدا؛ سپس از train یک val جدا برای تنظیم آستانه
# -------------------------
X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_full, Y_train_full, test_size=0.2, random_state=42, stratify=Y_train_full
)

# -------------------------
# نرمال‌سازی کانالی با StandardScaler (فقط fit روی TRAIN)
# -------------------------
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

# -------------------------
# class weights روی TRAIN
# -------------------------
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print("[CLASS WEIGHTS]", class_weight_dict)

# -------------------------
# مدل جمع‌وجور با GAP (کاهش اورفیت)
# -------------------------
def build_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=7, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),

        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model((T, C))
model.summary()

# -------------------------
# callbacks: EarlyStopping + ReduceLROnPlateau
# -------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=1)
]

# -------------------------
# آموزش
# -------------------------
history = model.fit(
    X_train, Y_train,
    epochs=120,
    batch_size=16,
    validation_data=(X_val, Y_val),
    verbose=1,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# -------------------------
# تنظیم آستانه بر اساس ولیدیشن (بهینه‌سازی macro-F1)
# -------------------------
val_probs = model.predict(X_val, verbose=0).ravel()
thr_grid = np.linspace(0.2, 0.8, 25)

best_thr, best_macro = 0.5, -1.0
for t in thr_grid:
    pred = (val_probs > t).astype(int)
    macro = f1_score(Y_val, pred, average='macro', zero_division=0)
    if macro > best_macro:
        best_macro, best_thr = macro, t

print(f"[THRESHOLD] best_thr={best_thr:.3f} | macroF1(val)={best_macro:.4f}")

# -------------------------
# ارزیابی روی تست با آستانه‌ی بهینه
# -------------------------
test_probs = model.predict(X_test, verbose=0).ravel()
Y_pred = (test_probs > best_thr).astype(int)

# -------------------------
# ذخیره نتایج مثل قبل
# -------------------------
result_dir = os.path.join("..", "results", "04_cnn_std_es_scheduler_thr_8feats")
os.makedirs(result_dir, exist_ok=True)

report = classification_report(Y_test, Y_pred, digits=4)
with open(os.path.join(result_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Best threshold (from val): {best_thr:.3f}\n")
    f.write(report)

cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["non-easy", "easy"], yticklabels=["non-easy", "easy"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (thr={best_thr:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
plt.close()

print(report)
print("Saved to:", result_dir)
