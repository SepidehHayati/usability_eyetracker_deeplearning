import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Create results directory if not exists
results_dir = "../results/16_cnn_classifier"
os.makedirs(results_dir, exist_ok=True)

# ----- Load Data -----
X = np.load("../data/X.npy")
Y = np.load("../data/Y.npy")

# Convert labels to one-hot encoding
Y_cat = to_categorical(Y, num_classes=2)

# ----- Train/Test Split -----
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_cat, test_size=0.2, random_state=42, stratify=Y
)

# ----- Define CNN Model -----
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=X.shape[1:]),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # Output layer for binary classification
])

# Compile and Train the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=16, validation_data=(X_test, Y_test))

# ----- Evaluation -----
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(Y_true, Y_pred_classes)

# Save Confusion Matrix as Image
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Easy', 'Not Easy'],
            yticklabels=['Easy', 'Not Easy'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "cnn_confusion.png"))
plt.close()

# Save Classification Report to Text File
report = classification_report(Y_true, Y_pred_classes, target_names=["Easy", "Not Easy"])
with open(os.path.join(results_dir, "cnn_classification.txt"), "w") as f:
    f.write(report)

# Save Accuracy to Text File
accuracy = np.mean(Y_pred_classes == Y_true)
with open(os.path.join(results_dir, "cnn_accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}")
