# File: 21_cnn_gap_weighted.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# مسیر ذخیره نتایج
base_dir = os.path.dirname(os.path.dirname(__file__))
results_dir = os.path.join(base_dir, "results", "21_cnn_gap_weighted")
os.makedirs(results_dir, exist_ok=True)

# Load Data
X = np.load("../data/X.npy")
Y = np.load("../data/Y.npy")
Y_cat = to_categorical(Y, num_classes=2)

# Split Data
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)

Y_train_cat = to_categorical(Y_train, 2)
Y_val_cat = to_categorical(Y_val, 2)
Y_test_cat = to_categorical(Y_test, 2)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
class_weight_dict = dict(enumerate(class_weights))

# Model
model = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    Dropout(0.3),

    Conv1D(128, kernel_size=5, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train
model.fit(
    X_train, Y_train_cat,
    validation_data=(X_val, Y_val_cat),
    epochs=50,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# Evaluation
preds = np.argmax(model.predict(X_test), axis=1)
report = classification_report(Y_test, preds, target_names=['Easy', 'Not Easy'])

with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(Y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Easy', 'Not Easy'], yticklabels=['Easy', 'Not Easy'])
plt.title('CNN + GAP + Weighted')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
plt.close()
