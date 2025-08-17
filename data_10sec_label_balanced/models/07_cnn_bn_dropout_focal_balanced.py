import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns

# focal loss function
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        return -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return focal_loss_fixed

# مسیر داده‌ها
data_dir = os.path.join("..", "data")
X = np.load(os.path.join(data_dir, "X.npy"))   # shape: (108, 1500, 4)
Y = np.load(os.path.join(data_dir, "Y.npy"))   # shape: (108,)

# تقسیم داده‌ها
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# تعریف مدل
model = Sequential([
    Conv1D(64, 5, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Conv1D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=focal_loss(gamma=2., alpha=.25), metrics=['accuracy'])
model.summary()

# آموزش مدل
history = model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# پیش‌بینی
Y_pred = (model.predict(X_test) > 0.5).astype("int32")

# مسیر نتایج
result_dir = os.path.join("..", "results", "07_cnn_bn_dropout_focal_balanced")
os.makedirs(result_dir, exist_ok=True)

# ذخیره classification report
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
