import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from collections import Counter

# --------- مسیر ذخیره نتایج ----------
base_dir = os.path.dirname(os.path.dirname(__file__))
results_dir = os.path.join(base_dir, "results", "19_cnn_bilstm_weighted_smote_augmented_scheduler_f1focused")
os.makedirs(results_dir, exist_ok=True)

# --------- بارگذاری داده ----------
X = np.load(os.path.join(base_dir, "data", "X.npy"))  # (samples, time, features)
Y = np.load(os.path.join(base_dir, "data", "Y.npy"))  # (samples,)
print(f"Original class distribution: {Counter(Y)}")

# --------- تبدیل برچسب‌ها به one-hot ----------
Y_cat = to_categorical(Y, num_classes=2)

# --------- تقسیم داده‌ها ----------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_cat, test_size=0.2, random_state=42, stratify=Y
)

# --------- اعمال SMOTE روی داده‌های Train ----------
n_samples, timesteps, features = X_train.shape
X_train_flat = X_train.reshape((n_samples, timesteps * features))
Y_train_labels = np.argmax(Y_train, axis=1)

sm = SMOTE(random_state=42)
X_res, Y_res = sm.fit_resample(X_train_flat, Y_train_labels)
X_res = X_res.reshape((-1, timesteps, features))
Y_res_cat = to_categorical(Y_res, num_classes=2)
print(f"After SMOTE: {Counter(Y_res)}")

# --------- محاسبه وزن کلاس‌ها ----------
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_res), y=Y_res)
class_weight_dict = dict(enumerate(class_weights))

# --------- تعریف یادگیرنده با scheduler ----------
def lr_schedule(epoch, lr):
    return lr * 0.95  # کاهش تدریجی نرخ یادگیری

# --------- تعریف مدل ----------
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(timesteps, features)),
    MaxPooling1D(pool_size=2),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --------- آموزش مدل ----------
history = model.fit(
    X_res, Y_res_cat,
    epochs=40,
    batch_size=16,
    validation_data=(X_test, Y_test),
    class_weight=class_weight_dict,
    callbacks=[LearningRateScheduler(lr_schedule)]
)

# --------- پیش‌بینی ----------
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# --------- ماتریس درهم‌ریختگی ----------
cm = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Easy", "Not Easy"],
            yticklabels=["Easy", "Not Easy"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("CNN-BiLSTM Weighted SMOTE + Augmented + Scheduler")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()

# --------- گزارش طبقه‌بندی ----------
report = classification_report(Y_true, Y_pred_classes, target_names=["Easy", "Not Easy"])
with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# --------- دقت کلی ----------
acc = np.mean(Y_pred_classes == Y_true)
with open(os.path.join(results_dir, "accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {acc:.4f}")
