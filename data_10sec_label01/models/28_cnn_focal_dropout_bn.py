# models/27_cnn_focal_dropout_bn.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- Load Data ----------------------
X = np.load('../data/X.npy')
y = np.load('../data/Y.npy')

print("Original class distribution:", Counter(y))

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.22, random_state=42, stratify=y)

# ---------------------- Focal Loss ----------------------
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -keras.backend.mean(
            alpha * keras.backend.pow(1. - pt_1, gamma) * keras.backend.log(pt_1 + epsilon) +
            (1 - alpha) * keras.backend.pow(pt_0, gamma) * keras.backend.log(1. - pt_0 + epsilon)
        )
    return loss

# ---------------------- Model ----------------------
model = models.Sequential([
    layers.Input(shape=X.shape[1:]),
    layers.Conv1D(64, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Conv1D(64, 3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(1, activation='sigmoid')
])

# ---------------------- Compile ----------------------
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=focal_loss(gamma=2.0, alpha=0.25),
              metrics=['accuracy'])

# ---------------------- Training ----------------------
callbacks = [
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=3, min_lr=1e-6, verbose=1)
]

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=16,
                    callbacks=callbacks,
                    verbose=2)

# ---------------------- Evaluation ----------------------
y_pred = (model.predict(X_val) > 0.5).astype("int32")
report = classification_report(y_val, y_pred, digits=4)
print(report)

# ---------------------- Save Results ----------------------
output_dir = '../results/27_cnn_focal_dropout_bn'
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
    f.write(report)

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

print("Training & evaluation finished. Results saved.")
