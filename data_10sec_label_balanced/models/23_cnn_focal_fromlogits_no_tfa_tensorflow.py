import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ===================== تنظیمات قابل تغییر =====================
DATA_DIR = os.path.join("..", "data")  # مسیر داده‌ها
RESULT_DIR = os.path.join("..", "results", "23_cnn_focal_fromlogits_no_tfa_tensorflow")

ALPHA = 0.36   # ↓ برای تقویت کلاس 0 (negative)، این مقدار را کوچکتر کن (مثلاً 0.25 یا 0.1)
GAMMA = 2.0
EPOCHS = 100
BATCH_SIZE = 16
VAL_SPLIT = 0.2
LR = 1e-3
# =============================================================

# --------- Focal Loss سفارشی (از لاجیت) ----------
def focal_loss(alpha=0.25, gamma=2.0):
    """
    Binary Focal Loss با ورودی لاجیت (from_logits=True)
    alpha روی کلاس مثبت (label=1) اعمال می‌شود؛ وزن کلاس 0 عملاً (1 - alpha) است.
    اگر می‌خواهی کلاس 0 بیشتر دیده شود، ALPHA را کوچک‌تر انتخاب کن.
    """
    def loss(y_true, y_pred_logits):
        y_true = tf.cast(y_true, tf.float32)
        # Cross-entropy باینری با لاجیت
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits)
        # احتمال پیش‌بینی‌شده
        probs = tf.sigmoid(y_pred_logits)
        # pt = p_t
        pt = y_true * probs + (1.0 - y_true) * (1.0 - probs)
        # وزن‌دهی فوکال
        focal_weight = tf.pow(1.0 - pt, gamma)
        # اعمال alpha روی کلاس مثبت
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        loss_val = alpha_factor * focal_weight * ce
        return tf.reduce_mean(loss_val)
    return loss

# --------- بارگذاری داده ---------
X = np.load(os.path.join(DATA_DIR, "X.npy"))   # (N, T, 4)
Y = np.load(os.path.join(DATA_DIR, "Y.npy"))   # (N,)
# اطمینان از نوع 0/1
Y = Y.astype(np.float32)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# --------- مدل CNN (بدون سیگموید در خروجی) ---------
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=(X.shape[1], X.shape[2])),
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

    Dense(1)  # خروجی لاجیت
])

opt = Adam(LR)
model.compile(optimizer=opt, loss=focal_loss(alpha=ALPHA, gamma=GAMMA), metrics=['accuracy'])
model.summary()

# --------- callbacks ---------
callbacks = [
    EarlyStopping(patience=12, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.5, patience=6, monitor='val_loss', min_lr=1e-6, verbose=1)
]

# --------- آموزش ---------
history = model.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    verbose=1,
    callbacks=callbacks
)

# --------- پیش‌بینی و ارزیابی ---------
logits = model.predict(X_test)
probs = tf.sigmoid(logits).numpy().ravel()  # چون خروجی لاجیت است
Y_pred = (probs > 0.5).astype("int32")

os.makedirs(RESULT_DIR, exist_ok=True)

# گزارش متنی
report = classification_report(Y_test, Y_pred, digits=4)
with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

# ماتریس سردرگمی
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["non-easy", "easy"], yticklabels=["non-easy", "easy"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.close()

print("Saved results to:", RESULT_DIR)
print(report)

# --------- (اختیاری) جستجوی آستانه برای بهبود کلاس 0 ---------
# اگر دیدی با threshold=0.5 کلاس 0 هنوز کم‌شمار است، این بخش را فعال کن تا
# بهترین آستانه برای F1 کلاس 0 پیدا شود.
"""
from sklearn.metrics import f1_score
best_t, best_f1 = 0.5, -1
for t in np.linspace(0.05, 0.95, 19):
    y_hat = (probs > t).astype(int)
    f1_neg = f1_score((Y_test==0).astype(int), (y_hat==0).astype(int))
    if f1_neg > best_f1:
        best_f1, best_t = f1_neg, t
print("Best threshold for class 0 F1:", best_t, " | F1_neg:", best_f1)
"""
