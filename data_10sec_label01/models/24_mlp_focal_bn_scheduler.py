import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import Counter

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

# ----------------------------
# 1. Load data
# ----------------------------
X = np.load('../data/X.npy')   # shape: (samples, time_steps, features)
Y = np.load('../data/Y.npy')   # shape: (samples,)
print(f"Original class distribution: {Counter(Y)}")

# ----------------------------
# 2. Train/test split
# ----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)

# ----------------------------
# 3. Sequence summarization: mean + std
# ----------------------------
X_train_mean = np.mean(X_train, axis=1)
X_train_std = np.std(X_train, axis=1)
X_train_flat = np.concatenate([X_train_mean, X_train_std], axis=1)

X_test_mean = np.mean(X_test, axis=1)
X_test_std = np.std(X_test, axis=1)
X_test_flat = np.concatenate([X_test_mean, X_test_std], axis=1)

# ----------------------------
# 4. Encode labels
# ----------------------------
le = LabelEncoder()
Y_train_enc = le.fit_transform(Y_train)
Y_test_enc = le.transform(Y_test)

# ----------------------------
# 5. SMOTE oversampling
# ----------------------------
sm = SMOTE(random_state=42)
X_res, Y_res = sm.fit_resample(X_train_flat, Y_train_enc)
print(f"After SMOTE: {Counter(Y_res)}")

# ----------------------------
# 6. One-hot encoding
# ----------------------------
Y_res_cat = to_categorical(Y_res, num_classes=2)
Y_test_cat = to_categorical(Y_test_enc, num_classes=2)

# ----------------------------
# 7. Compute class weights
# ----------------------------
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_res), y=Y_res)
class_weight_dict = dict(enumerate(class_weights))

# ----------------------------
# 8. Focal Loss
# ----------------------------
def focal_loss(gamma=2., alpha=.75):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.mean(K.sum(loss, axis=1))
    return loss

# ----------------------------
# 9. Build improved MLP model
# ----------------------------
model = Sequential([
    Input(shape=(X_res.shape[1],)),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

# ----------------------------
# 10. Callbacks
# ----------------------------
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# ----------------------------
# 11. Train
# ----------------------------
history = model.fit(
    X_res, Y_res_cat,
    validation_data=(X_test_flat, Y_test_cat),
    epochs=100,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=[lr_scheduler, early_stop],
    verbose=2
)

# ----------------------------
# 12. Predict & Evaluate
# ----------------------------
Y_pred = model.predict(X_test_flat)
Y_pred_labels = np.argmax(Y_pred, axis=1)

report = classification_report(Y_test_enc, Y_pred_labels, target_names=['Class 0', 'Class 1'])
cm = confusion_matrix(Y_test_enc, Y_pred_labels)

# ----------------------------
# 13. Save Results
# ----------------------------
output_dir = '../results/24_mlp_improved_focal_bn'
os.makedirs(output_dir, exist_ok=True)

# Save classification report
with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
    f.write(report)

# Save confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0, 1])
plt.yticks([0, 1])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

print("✅ Training & evaluation finished. Results saved to:", output_dir)
