import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten,
    Dense, Dropout, BatchNormalization, Bidirectional, LSTM, GRU
)
from tensorflow.keras.optimizers import Adam

# ============== Config ==============
DATA_DIR   = os.path.join("..", "data")
X_NAME     = "X8.npy"   # (N, 1500, 8)
Y_NAME     = "Y8.npy"   # (N,)
TEST_SIZE  = 0.20
RANDOM_SEED= 42
EPOCHS     = 80
BATCH_SIZE = 16
LR         = 1e-3
THRESHOLD  = 0.5

# انتخاب مدل: 'cnn' | 'bilstm' | 'gru' | 'tcn'
MODEL_NAME = "cnn"
# تگ خروجی‌ها
RESULT_TAG = f"01_{MODEL_NAME}_balanced_split"

# ============ Load ============
X = np.load(os.path.join(DATA_DIR, X_NAME))   # (N, T, C)
Y = np.load(os.path.join(DATA_DIR, Y_NAME)).astype(int)
assert X.ndim == 3, "X must be (N, T, C)"
N, T, C = X.shape
print(f"[INFO] X={X.shape}, Y={Y.shape}")

# ============ Split ============
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, stratify=Y, random_state=RANDOM_SEED
)

# ============ Standardize per-feature using train stats ============
scaler = StandardScaler()
Xtr2 = X_train.reshape(-1, C)
scaler.fit(Xtr2)
def transform(arr):
    arr2 = arr.reshape(-1, C)
    arr2 = scaler.transform(arr2)
    return arr2.reshape(arr.shape[0], arr.shape[1], arr.shape[2])

X_train = transform(X_train)
X_test  = transform(X_test)

# ============ Class weights ============
classes = np.unique(y_train)
class_w = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_w)}
print("[INFO] class_weight:", class_weight_dict)

# ============ Models ============
def build_cnn(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
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
    return model

def build_bilstm(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid'),
    ])
    return model

def build_gru(input_shape):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        GRU(32),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid'),
    ])
    return model

def build_tcn(input_shape):
    # یک TCN ساده با کانولوشن‌های دیلاته
    model = Sequential([
        Conv1D(32, 3, padding='causal', activation='relu', dilation_rate=1, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(32, 3, padding='causal', activation='relu', dilation_rate=2),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(64, 3, padding='causal', activation='relu', dilation_rate=4),
        BatchNormalization(),
        Dropout(0.2),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

builders = {
    "cnn": build_cnn,
    "bilstm": build_bilstm,
    "gru": build_gru,
    "tcn": build_tcn
}
assert MODEL_NAME in builders, f"Unknown model '{MODEL_NAME}'"
model = builders[MODEL_NAME]((T, C))
model.compile(optimizer=Adam(LR), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ============ Train ============
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1,
    class_weight=class_weight_dict
)

# ============ Predict ============
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob > THRESHOLD).astype(int)

# ============ Save results ============
RESULT_DIR = os.path.join("..", "results", RESULT_TAG)
os.makedirs(RESULT_DIR, exist_ok=True)

# گزارش
report = classification_report(y_test, y_pred, digits=4)
with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)
print(report)

# ماتریس کانفیوژن
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["non-easy(0)", "easy(1)"],
            yticklabels=["non-easy(0)", "easy(1)"])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title(f"Confusion Matrix — {MODEL_NAME} (thr={THRESHOLD:.2f})")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.close()

# ذخیره احتمال‌ها و پیش‌بینی‌ها برای تحلیل‌های بعدی
np.save(os.path.join(RESULT_DIR, "y_test.npy"), y_test)
np.save(os.path.join(RESULT_DIR, "y_prob.npy"), y_prob)
np.save(os.path.join(RESULT_DIR, "y_pred.npy"), y_pred)

print("Saved to:", RESULT_DIR)
