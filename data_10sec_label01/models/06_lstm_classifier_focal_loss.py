import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

# --------- Define Focal Loss ---------
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed

# --------- Paths ---------
base_dir = os.path.dirname(os.path.dirname(__file__))
results_dir = os.path.join(base_dir, "results", "06_lstm_classifier_focal_loss")
os.makedirs(results_dir, exist_ok=True)

# --------- Load Data ---------
X = np.load(os.path.join(base_dir, "data", "X.npy"))
Y = np.load(os.path.join(base_dir, "data", "Y.npy"))
Y_cat = to_categorical(Y, num_classes=2)

# --------- Train/Test Split ---------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_cat, test_size=0.2, random_state=42, stratify=Y
)

# --------- LSTM Model ---------
model = Sequential([
    LSTM(64, input_shape=X.shape[1:]),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss=focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=16, validation_data=(X_test, Y_test))

# --------- Evaluation ---------
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

# --------- Confusion Matrix ---------
cm = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Easy', 'Not Easy'],
            yticklabels=['Easy', 'Not Easy'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("LSTM (Focal Loss) - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "lstm_focal_confusion.png"))
plt.close()

# --------- Save Reports ---------
report = classification_report(Y_true, Y_pred_classes, target_names=["Easy", "Not Easy"])
with open(os.path.join(results_dir, "lstm_focal_classification.txt"), "w") as f:
    f.write(report)

accuracy = np.mean(Y_pred_classes == Y_true)
with open(os.path.join(results_dir, "lstm_focal_accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}")
