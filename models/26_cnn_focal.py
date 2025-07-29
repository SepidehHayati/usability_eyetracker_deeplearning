import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical

# تنظیم seed برای تکرارپذیری
np.random.seed(42)
tf.random.set_seed(42)

# تعریف Focal Loss
def focal_loss(gamma=2., alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(loss, axis=-1)
    return focal_loss_fixed

# 1. Load data
X = np.load('../data/X.npy')  # shape: (108, 1500, 4)
Y = np.load('../data/Y.npy')  # shape: (108,)

print(f"Original class distribution: {Counter(Y)}")

# 2. Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# 3. Encode labels
le = LabelEncoder()
Y_train_enc = le.fit_transform(Y_train)
Y_test_enc = le.transform(Y_test)

# 4. SMOTE (روی داده‌های فلت‌شده برای ساده‌سازی)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
sm = SMOTE(random_state=42, sampling_strategy=0.5)
X_res_flat, Y_res = sm.fit_resample(X_train_flat, Y_train_enc)
print(f"After SMOTE: {Counter(Y_res)}")

# بازگرداندن به شکل سری زمانی برای 1D-CNN
X_res = X_res_flat.reshape(-1, X_train.shape[1], X_train.shape[2])

# 5. One-hot
Y_res_cat = to_categorical(Y_res, num_classes=2)
Y_test_cat = to_categorical(Y_test_enc, num_classes=2)

# 6. Build model
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),  # (1500, 4)
    Conv1D(32, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Conv1D(16, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=focal_loss(gamma=2., alpha=0.75),
              metrics=['accuracy'])

# 7. Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 8. Train
history = model.fit(
    X_res, Y_res_cat,
    validation_data=(X_test, Y_test_cat),
    epochs=100,
    batch_size=8,
    callbacks=[lr_scheduler, early_stopping],
    verbose=2
)

# 9. Predict
Y_pred = model.predict(X_test)
Y_pred_labels = np.argmax(Y_pred, axis=1)

# 10. Evaluation
report = classification_report(Y_test_enc, Y_pred_labels, target_names=['Not Easy', 'Easy'])
cm = confusion_matrix(Y_test_enc, Y_pred_labels)

# 11. Save results
output_dir = '../results/26_cnn_focal'
os.makedirs(output_dir, exist_ok=True)

# Save classification report
with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
    f.write(report)

# Save confusion matrix as image
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0, 1], ['Not Easy', 'Easy'])
plt.yticks([0, 1], ['Not Easy', 'Easy'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()



print("Training & evaluation finished. Results saved.")