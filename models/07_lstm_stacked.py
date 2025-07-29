import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ----- Set results directory -----
base_dir = os.path.dirname(os.path.dirname(__file__))  # Go to project root
results_dir = os.path.join(base_dir, "results", "07_lstm_stacked")
os.makedirs(results_dir, exist_ok=True)

# ----- Load Data -----
X = np.load(os.path.join(base_dir, "data", "X.npy"))
Y = np.load(os.path.join(base_dir, "data", "Y.npy"))
Y_cat = to_categorical(Y, num_classes=2)

# ----- Split -----
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_cat, test_size=0.2, stratify=Y, random_state=42
)

# ----- Define Stacked LSTM Model -----
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=X.shape[1:]),
    LSTM(32),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Optional: Early stopping if needed
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ----- Train -----
model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_data=(X_test, Y_test),
          callbacks=[early_stop], verbose=2)

# ----- Evaluate -----
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# ----- Confusion Matrix -----
cm = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Easy", "Not Easy"],
            yticklabels=["Easy", "Not Easy"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Stacked LSTM - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()

# ----- Classification Report -----
report = classification_report(Y_true, Y_pred_classes, target_names=["Easy", "Not Easy"])
with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# ----- Accuracy -----
accuracy = np.mean(Y_pred_classes == Y_true)
with open(os.path.join(results_dir, "accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}")
