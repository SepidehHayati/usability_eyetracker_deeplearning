import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# مسیر پوشه فعلی اسکریپت
base_dir = os.path.dirname(os.path.abspath(__file__))

# مسیر داده‌ها
data_dir = os.path.abspath(os.path.join(base_dir, "../data"))
X = np.load(os.path.join(data_dir, "X.npy"))     # (108, 1500, 4)
Y = np.load(os.path.join(data_dir, "Y.npy"))     # (108,)

# تقسیم داده‌ها
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

# ساخت مدل MLP ساده
model = Sequential([
    Flatten(input_shape=(1500, 4)),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# آموزش مدل
model.fit(X_train, Y_train, epochs=30, batch_size=16, verbose=1)

# پیش‌بینی
Y_pred_probs = model.predict(X_test)
Y_pred = (Y_pred_probs > 0.5).astype("int32")

# ارزیابی
report = classification_report(Y_test, Y_pred, digits=4)
print(report)

# مسیر ذخیره خروجی
results_dir = os.path.abspath(os.path.join(base_dir, "../results/01_mlp_classifier_balanced"))
os.makedirs(results_dir, exist_ok=True)

# ذخیره گزارش متنی
with open(os.path.join(results_dir, "report.txt"), "w") as f:
    f.write(report)

# ذخیره confusion matrix به صورت تصویر
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()
