import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# =====================
# مسیر داده‌ها
# =====================
data_dir = os.path.join("..", "data")
X = np.load(os.path.join(data_dir, "X.npy"))   # (108, 1500, 4)
Y = np.load(os.path.join(data_dir, "Y.npy"))   # (108,)

# =====================
# پارامترهای Cross Validation
# =====================
N_SPLITS = 10
N_REPEATS = 4  # برای تست سریع؛ می‌تونی بیشترش کنی (مثلاً 5 یا 10)
random_state = 42

rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=random_state)

# =====================
# تابع ساخت مدل CNN
# =====================
def create_model(input_shape):
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
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =====================
# لیست ذخیره نتایج
# =====================
all_reports = []
all_metrics = []

# =====================
# اجرای تکرارها
# =====================
fold_idx = 1
for train_idx, test_idx in rskf.split(X, Y):
    print(f"\n===== Fold {fold_idx} =====")
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    model = create_model((X.shape[1], X.shape[2]))

    history = model.fit(
        X_train, Y_train,
        epochs=50,
        batch_size=16,
        verbose=0,
        validation_split=0.2
    )

    Y_pred_prob = model.predict(X_test, verbose=0)
    Y_pred = (Y_pred_prob > 0.5).astype("int32")

    acc = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred, zero_division=0)
    rec = recall_score(Y_test, Y_pred, zero_division=0)
    f1 = f1_score(Y_test, Y_pred, zero_division=0)

    all_metrics.append([acc, prec, rec, f1])

    report = classification_report(Y_test, Y_pred, digits=4)
    all_reports.append(f"Fold {fold_idx}\n{report}\n")

    fold_idx += 1

# =====================
# محاسبه میانگین و انحراف معیار
# =====================
all_metrics = np.array(all_metrics)
mean_metrics = all_metrics.mean(axis=0)
std_metrics = all_metrics.std(axis=0)

summary_report = (
    f"\n=== Mean Results over {N_SPLITS*N_REPEATS} runs ===\n"
    f"Accuracy: {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}\n"
    f"Precision: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}\n"
    f"Recall: {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}\n"
    f"F1-score: {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}\n"
)

print(summary_report)

# =====================
# ذخیره نتایج
# =====================
result_dir = os.path.join("..", "results", "09_cnn_class_weight_Repeated Stratified K-Fold Evaluation")
os.makedirs(result_dir, exist_ok=True)

with open(os.path.join(result_dir, "all_folds_reports.txt"), "w") as f:
    f.write("\n".join(all_reports))
    f.write(summary_report)

print("Results saved in:", result_dir)
