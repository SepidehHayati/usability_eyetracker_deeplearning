# File: 20_cnn_dropout_bn_weighted.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# --------- مسیر ذخیره نتایج ----------
base_dir = os.path.dirname(os.path.dirname(__file__))  # مسیر اصلی پروژه
results_dir = os.path.join(base_dir, "results", "20_cnn_dropout_bn_weighted")
os.makedirs(results_dir, exist_ok=True)

# --------- بارگذاری داده‌ها ----------
X = np.load("../data/X.npy")
Y = np.load("../data/Y.npy")

# تبدیل برچسب‌ها به one-hot
Y_cat = to_categorical(Y, num_classes=2)

# --------- تقسیم داده‌ها به train, val, test ----------
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)

Y_train_cat = to_categorical(Y_train, num_classes=2)
Y_val_cat = to_categorical(Y_val, num_classes=2)
Y_test_cat = to_categorical(Y_test, num_classes=2)

# --------- محاسبه وزن کلاس‌ها ----------
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
class_weight_dict = dict(enumerate(class_weights))

# --------- تعریف مدل CNN ----------
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(filters=64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# --------- توقف زودهنگام ----------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# --------- آموزش مدل ----------
history = model.fit(
    X_train, Y_train_cat,
    validation_data=(X_val, Y_val_cat),
    epochs=50,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# --------- ارزیابی مدل ----------
preds = np.argmax(model.predict(X_test), axis=1)
report = classification_report(Y_test, preds, target_names=['Easy', 'Not Easy'])

# ذخیره گزارش طبقه‌بندی
with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# ماتریس آشفتگی
cm = confusion_matrix(Y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Easy', 'Not Easy'], yticklabels=['Easy', 'Not Easy'])
plt.title('CNN (Dropout + BN + Weighted)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
plt.close()

# ذخیره دقت
accuracy = np.mean(preds == Y_test)
with open(os.path.join(results_dir, "accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}")
