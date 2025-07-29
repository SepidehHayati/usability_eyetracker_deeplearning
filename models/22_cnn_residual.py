# File: 22_cnn_residual.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten, Dense, Add
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --------- مسیر ذخیره نتایج ----------
base_dir = os.path.dirname(os.path.dirname(__file__))
results_dir = os.path.join(base_dir, "results", "22_cnn_residual")
os.makedirs(results_dir, exist_ok=True)

# ----- Load Data -----
X = np.load("../data/X.npy")
Y = np.load("../data/Y.npy")

# Convert labels to one-hot
Y_cat = to_categorical(Y, num_classes=2)

# ---- Train/Val/Test Split ----
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)

Y_train_cat = to_categorical(Y_train, num_classes=2)
Y_val_cat = to_categorical(Y_val, num_classes=2)
Y_test_cat = to_categorical(Y_test, num_classes=2)

# ---- Compute Class Weights ----
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
class_weight_dict = dict(enumerate(class_weights))

# ---- Residual Block Definition ----
def residual_block(x, filters, kernel_size):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Dropout(0.3)(x)
    return x

# ---- Model Definition ----
input_layer = Input(shape=(X.shape[1], X.shape[2]))
x = Conv1D(32, kernel_size=3, padding='same', activation='relu')(input_layer)
x = BatchNormalization()(x)
x = residual_block(x, 32, 3)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = residual_block(x, 64, 3)
x = MaxPooling1D(pool_size=2)(x)

x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
output_layer = Dense(2, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# ---- Callbacks ----
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)

# ---- Training ----
history = model.fit(
    X_train, Y_train_cat,
    validation_data=(X_val, Y_val_cat),
    epochs=50,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

# ---- Evaluation ----
preds = np.argmax(model.predict(X_test), axis=1)
report = classification_report(Y_test, preds, target_names=['Easy', 'Not Easy'])

with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
    f.write(report)

# ---- Confusion Matrix ----
cm = confusion_matrix(Y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Easy', 'Not Easy'], yticklabels=['Easy', 'Not Easy'])
plt.title('CNN Residual')
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
plt.close()
