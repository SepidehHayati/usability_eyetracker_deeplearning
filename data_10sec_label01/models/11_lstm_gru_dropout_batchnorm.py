import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K

class FocalLoss(Loss):
    def __init__(self, gamma=2., alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = self.alpha * K.pow(1 - y_pred, self.gamma)
        return K.sum(weight * cross_entropy, axis=1)

# ---------- Paths ----------
base_dir = os.path.dirname(os.path.dirname(__file__))
results_dir = os.path.join(base_dir, "results", "11_lstm_gru_dropout_batchnorm")
os.makedirs(results_dir, exist_ok=True)

# ---------- Load Data ----------
X = np.load(os.path.join(base_dir, "data", "X.npy"))
Y = np.load(os.path.join(base_dir, "data", "Y.npy"))
Y_cat = to_categorical(Y, num_classes=2)

# ---------- Train/Test Split ----------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_cat, test_size=0.2, random_state=42, stratify=Y
)

# ---------- Model ----------
model = Sequential([
    Input(shape=X.shape[1:]),
    GRU(64, return_sequences=True),
    Dropout(0.3),
    GRU(64),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss=FocalLoss(), metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=30, batch_size=16, validation_data=(X_test, Y_test))

# ---------- Evaluation ----------
Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

cm = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Easy", "Not Easy"],
            yticklabels=["Easy", "Not Easy"])
plt.title("GRU + Dropout + BatchNorm + Focal Loss")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "gru_confusion.png"))
plt.close()

report = classification_report(Y_true, Y_pred_classes, target_names=["Easy", "Not Easy"])
with open(os.path.join(results_dir, "gru_classification.txt"), "w") as f:
    f.write(report)

accuracy = np.mean(Y_pred_classes == Y_true)
with open(os.path.join(results_dir, "gru_accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}")
