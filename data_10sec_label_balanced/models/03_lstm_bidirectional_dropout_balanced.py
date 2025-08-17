import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import matplotlib.pyplot as plt
import seaborn as sns

# تنظیم مسیر داده‌ها
data_dir = os.path.join("..", "data")
X = np.load(os.path.join(data_dir, "X.npy"))   # (108, 1500, 4)
Y = np.load(os.path.join(data_dir, "Y.npy"))   # (108,)

# تقسیم داده‌ها
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# تعریف مدل
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=False), input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# آموزش مدل
history = model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# پیش‌بینی و ارزیابی
Y_pred = (model.predict(X_test) > 0.5).astype("int32")

# مسیر ذخیره نتایج
result_dir = os.path.join("..", "results", "03_lstm_bidirectional_dropout_balanced")
os.makedirs(result_dir, exist_ok=True)

# ذخیره گزارش متنی
report = classification_report(Y_test, Y_pred, digits=4)
with open(os.path.join(result_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# ذخیره confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["non-easy", "easy"], yticklabels=["non-easy", "easy"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
plt.close()
