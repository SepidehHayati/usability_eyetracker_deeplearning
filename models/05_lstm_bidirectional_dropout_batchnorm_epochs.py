import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional

# ----- Set results directory relative to project root -----
base_dir = os.path.dirname(os.path.dirname(__file__))  # Go to project root
results_dir = os.path.join(base_dir, "results", "05_lstm_bidirectional_dropout_batchnorm_epochs")
os.makedirs(results_dir, exist_ok=True)

# ----- Load Data -----
X = np.load(os.path.join(base_dir, "data", "X.npy"))
Y = np.load(os.path.join(base_dir, "data", "Y.npy"))

# Convert labels to one-hot encoding (binary classification)
Y_cat = to_categorical(Y, num_classes=2)

# ----- Train/Test Split -----
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_cat, test_size=0.2, random_state=42, stratify=Y
)

# ----- Define LSTM Model -----
model = Sequential([
    Bidirectional(LSTM(units=64, return_sequences=True), input_shape=X.shape[1:]),
    BatchNormalization(),
    Dropout(0.5),
    LSTM(units=32),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # 2 classes: Easy vs Not Easy
])

# Compile and Train the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_data=(X_test, Y_test))

# ----- Evaluation -----
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# ----- Save Confusion Matrix -----
cm = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Easy", "Not Easy"],
            yticklabels=["Easy", "Not Easy"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Bidirectional LSTM + Dropout + BatchNorm + Epochs - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "lstm_confusion.png"))
plt.close()

# ----- Save Classification Report -----
report = classification_report(Y_true, Y_pred_classes, target_names=["Easy", "Not Easy"])
with open(os.path.join(results_dir, "lstm_classification.txt"), "w") as f:
    f.write(report)

# ----- Save Accuracy -----
accuracy = np.mean(Y_pred_classes == Y_true)
with open(os.path.join(results_dir, "lstm_accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}")
