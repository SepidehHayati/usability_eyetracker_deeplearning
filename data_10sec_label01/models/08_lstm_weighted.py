import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Create results directory if not exists
base_dir = os.path.dirname(os.path.dirname(__file__))  # برمی‌گرده به root پروژه
results_dir = os.path.join(base_dir, "results", "08_lstm_weighted")
os.makedirs(results_dir, exist_ok=True)

# ----- Load Data -----
X = np.load("../data/X.npy")
Y = np.load("../data/Y.npy")

# Convert labels to one-hot encoding (binary classification)
Y_cat = to_categorical(Y, num_classes=2)

# ----- Train/Test Split -----
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_cat, test_size=0.2, random_state=42, stratify=Y
)

# ----- Compute Class Weights -----
Y_labels = np.argmax(Y_cat, axis=1)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_labels), y=Y_labels)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Class Weights:", class_weights_dict)

# ----- Define LSTM Model -----
model = Sequential([
    LSTM(units=64, input_shape=X.shape[1:]),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # 2 classes: Easy vs Not Easy
])

# Compile and Train the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    X_train, Y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_test, Y_test),
    class_weight=class_weights_dict  # <-- اضافه شد
)

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
plt.title("LSTM (with Class Weights) - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "lstm_confusion.png"))
plt.close()

# Save Classification Report
report = classification_report(Y_true, Y_pred_classes, target_names=["Easy", "Not Easy"])
with open(os.path.join(results_dir, "lstm_classification.txt"), "w") as f:
    f.write(report)

# Save Accuracy
accuracy = np.mean(Y_pred_classes == Y_true)
with open(os.path.join(results_dir, "lstm_accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}")
