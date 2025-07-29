import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# تنظیم seed برای تکرارپذیری
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load data
X = np.load('../data/X.npy')   # shape: (samples, time_steps, features)
Y = np.load('../data/Y.npy')   # shape: (samples,)

print(f"Original class distribution: {Counter(Y)}")

# 2. Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# 3. Convert sequence to summary features (mean & std)
X_train_mean = np.mean(X_train, axis=1)
X_train_std = np.std(X_train, axis=1)
X_train_flat = np.concatenate([X_train_mean, X_train_std], axis=1)  # shape: (samples, 120)

X_test_mean = np.mean(X_test, axis=1)
X_test_std = np.std(X_test, axis=1)
X_test_flat = np.concatenate([X_test_mean, X_test_std], axis=1)

# 4. Encode labels
le = LabelEncoder()
Y_train_enc = le.fit_transform(Y_train)
Y_test_enc = le.transform(Y_test)

# 5. SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_res, Y_res = smote_enn.fit_resample(X_train_flat, Y_train_enc)
print(f"After SMOTEENN: {Counter(Y_res)}")

# 6. One-hot
Y_res_cat = to_categorical(Y_res, num_classes=2)
Y_test_cat = to_categorical(Y_test_enc, num_classes=2)

# 7. Class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_res), y=Y_res)
class_weight_dict = dict(enumerate(class_weights))

# 8. Build model
model = Sequential([
    Input(shape=(X_res.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 9. Scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)

# 10. Train
history = model.fit(
    X_res, Y_res_cat,
    validation_data=(X_test_flat, Y_test_cat),
    epochs=50,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=[lr_scheduler],
    verbose=2
)

# 11. Predict
Y_pred = model.predict(X_test_flat)
Y_pred_labels = np.argmax(Y_pred, axis=1)

# 12. Evaluation
report = classification_report(Y_test_enc, Y_pred_labels, target_names=['Class 0', 'Class 1'])
cm = confusion_matrix(Y_test_enc, Y_pred_labels)

# 13. Save results
output_dir = '../results/25_mlp_smoteenn'
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
plt.xticks([0, 1])
plt.yticks([0, 1])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()



print("Training & evaluation finished. Results saved.")