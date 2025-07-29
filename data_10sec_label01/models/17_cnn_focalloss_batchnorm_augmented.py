import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Create results directory if not exists
results_dir = "../results/17_cnn_focalloss_batchnorm_augmented"
os.makedirs(results_dir, exist_ok=True)

# ----- Load Data -----
X = np.load("../data/X.npy")
Y = np.load("../data/Y.npy")

# Convert labels to one-hot
Y_cat = to_categorical(Y, num_classes=2)

# ----- Split -----
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_cat, test_size=0.2, stratify=Y, random_state=42
)

# ----- Class Weights -----
class_weights_raw = compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
class_weights = {i: w for i, w in enumerate(class_weights_raw)}

# ----- Define CNN Model -----
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=X.shape[1:]),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),  # to reduce overfitting
    Dense(2, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, Y_test),
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ----- Evaluation -----
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(Y_true, Y_pred_classes)

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Easy', 'Not Easy'],
            yticklabels=['Easy', 'Not Easy'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()

# Save Classification Report
report = classification_report(Y_true, Y_pred_classes, target_names=["Easy", "Not Easy"])
with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# Save Accuracy
accuracy = np.mean(Y_pred_classes == Y_true)
with open(os.path.join(results_dir, "accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}")
