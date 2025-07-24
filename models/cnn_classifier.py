import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# بارگذاری داده‌ها
X = np.load("../data/X.npy")
Y = np.load("../data/Y.npy")

# تبدیل لیبل به one-hot
Y_cat = to_categorical(Y, num_classes=2)

# تقسیم داده‌ها
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_cat, test_size=0.2, random_state=42, stratify=Y
)

# تعریف مدل CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))  # چون ۲ کلاس داریم

# کامپایل و آموزش مدل
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=16, validation_data=(X_test, Y_test))

# ✅ محاسبه پیش‌بینی‌ها و نمایش confusion matrix
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true_classes = np.argmax(Y_test, axis=1)

cm = confusion_matrix(Y_true_classes, Y_pred_classes)

# رسم ماتریس درهم‌ریختگی
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Easy", "Not Easy"], yticklabels=["Easy", "Not Easy"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# گزارش دقیق‌تر
print("\nClassification Report:")
print(classification_report(Y_true_classes, Y_pred_classes, target_names=["Easy", "Not Easy"]))
