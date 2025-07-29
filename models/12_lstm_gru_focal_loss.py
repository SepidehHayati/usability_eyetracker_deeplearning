import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# -------- Paths --------
base_dir = os.path.dirname(os.path.dirname(__file__))
results_dir = os.path.join(base_dir, "results", "18_cnn_lstm_weighted_smote_augmented_f1focused")
os.makedirs(results_dir, exist_ok=True)

# -------- Load Data --------
X = np.load(os.path.join(base_dir, "data", "X.npy"))
Y = np.load(os.path.join(base_dir, "data", "Y.npy"))

# -------- Data Augmentation (Gaussian Noise) --------
def augment_with_noise(X, noise_level=0.01):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=X.shape)
    return X + noise

X_augmented = augment_with_noise(X)
Y_augmented = Y.copy()

# Combine original + augmented
X_full = np.concatenate([X, X_augmented], axis=0)
Y_full = np.concatenate([Y, Y_augmented], axis=0)

# -------- Reshape for SMOTE --------
n_timesteps, n_features = X.shape[1], X.shape[2]
X_reshaped = X_full.reshape((X_full.shape[0], -1))

# -------- Apply SMOTE --------
sm = SMOTE(random_state=42)
X_resampled, Y_resampled = sm.fit_resample(X_reshaped, Y_full)

# Reshape back to 3D for CNN-LSTM
X_resampled = X_resampled.reshape((-1, n_timesteps, n_features))
Y_cat = to_categorical(Y_resampled, num_classes=2)

# -------- Train/Test Split --------
X_train, X_test, Y_train, Y_test = train_test_split(
    X_resampled, Y_cat, test_size=0.2, random_state=42, stratify=Y_resampled
)

# -------- Compute Class Weights --------
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_resampled), y=Y_resampled)
class_weights_dict = {i: class_weights[i] for i in range(2)}

# -------- Model Definition --------
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------- Train --------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=40,
    batch_size=32,
    class_weight=class_weights_dict,
    callbacks=[early_stop],
    verbose=1
)

# -------- Evaluation --------
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# -------- Confusion Matrix --------
cm = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Easy", "Not Easy"], yticklabels=["Easy", "Not Easy"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("CNN-LSTM Weighted SMOTE + Augmentation")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()

# -------- Classification Report --------
report = classification_report(Y_true, Y_pred_classes, target_names=["Easy", "Not Easy"])
with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# -------- Accuracy --------
acc = np.mean(Y_pred_classes == Y_true)
with open(os.path.join(results_dir, "accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {acc:.4f}")
