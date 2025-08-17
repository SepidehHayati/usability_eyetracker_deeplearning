import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# === مسیر داده‌ها ===
data_dir = os.path.join("..", "data")
X = np.load(os.path.join(data_dir, "X.npy"))   # (108, 1500, 4)
Y = np.load(os.path.join(data_dir, "Y.npy"))   # (108,)

# === آماده‌سازی پوشه نتایج ===
result_dir = os.path.join("..", "results", "09_cnn_class_weight_balanced_5foldcv")
os.makedirs(result_dir, exist_ok=True)

# === تعریف K-Fold ===
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_reports = []

# === اجرای 5-Fold CV ===
for fold, (train_idx, test_idx) in enumerate(kfold.split(X, Y), 1):
    print(f"Fold {fold} ...")

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # === class weights ===
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    # === تعریف مدل ===
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
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

    # === آموزش ===
    model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_split=0.2,
              verbose=0, class_weight=class_weight_dict)

    # === پیش‌بینی ===
    Y_pred = (model.predict(X_test) > 0.5).astype("int32")

    # === confusion matrix ===
    cm = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["non-easy", "easy"], yticklabels=["non-easy", "easy"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"confusion_matrix_fold{fold}.png"))
    plt.close()

    # === گزارش متنی ===
    report = classification_report(Y_test, Y_pred, digits=4)
    with open(os.path.join(result_dir, f"classification_report_fold{fold}.txt"), "w") as f:
        f.write(report)

    all_reports.append(report)

# === ذخیره همه گزارش‌ها در یک فایل نهایی ===
with open(os.path.join(result_dir, "all_folds_report.txt"), "w") as f:
    for i, r in enumerate(all_reports, 1):
        f.write(f"====== Fold {i} ======\n")
        f.write(r + "\n\n")

print("✅ All 5 folds completed. Results saved in 'results/09_cnn_class_weight_balanced_5foldcv'")
