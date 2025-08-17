import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization,
    Layer, GlobalAveragePooling1D, Multiply
)
from tensorflow.keras.optimizers import Adam

# --- Simple attention layer ---
class SimpleAttention(Layer):
    def __init__(self):
        super(SimpleAttention, self).__init__()

    def call(self, inputs):
        score = tf.nn.softmax(inputs, axis=1)
        context = tf.reduce_sum(score * inputs, axis=1)
        return context

# --- Load Data ---
data_dir = os.path.join("..", "data")
X = np.load(os.path.join(data_dir, "X.npy"))
Y = np.load(os.path.join(data_dir, "Y.npy"))

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

# --- Model Definition ---
input_layer = Input(shape=(X.shape[1], X.shape[2]))

x = Conv1D(32, kernel_size=5, activation='relu')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

x = Conv1D(64, kernel_size=3, activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.3)(x)

x = SimpleAttention()(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Train ---
history = model.fit(
    X_train, Y_train, validation_split=0.2,
    epochs=100, batch_size=16, verbose=1
)

# --- Predict ---
Y_pred = (model.predict(X_test) > 0.5).astype("int32")

# --- Save Results ---
result_dir = os.path.join("..", "results", "15_cnn_attention_balanced")
os.makedirs(result_dir, exist_ok=True)

report = classification_report(Y_test, Y_pred, digits=4)
with open(os.path.join(result_dir, "classification_report.txt"), "w") as f:
    f.write(report)

cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=["non-easy", "easy"], yticklabels=["non-easy", "easy"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
plt.close()
