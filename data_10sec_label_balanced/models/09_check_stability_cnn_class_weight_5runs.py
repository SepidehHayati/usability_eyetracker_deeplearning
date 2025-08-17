import os
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_model(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Flatten(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# مسیر داده
data_dir = os.path.join("..", "data")
X = np.load(os.path.join(data_dir, "X.npy"))
Y = np.load(os.path.join(data_dir, "Y.npy"))

n_runs = 5
reports = []

for run in range(n_runs):
    set_seed(42 + run)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=run)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X_train, Y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0, class_weight=class_weight_dict)

    Y_pred = (model.predict(X_test) > 0.5).astype("int32")
    report = classification_report(Y_test, Y_pred, output_dict=True)
    reports.append(report)

# محاسبه میانگین و انحراف معیار
import pandas as pd
metrics = ['precision', 'recall', 'f1-score']
labels = ['0', '1']
summary = {}

for label in labels:
    for metric in metrics:
        values = [rep[label][metric] for rep in reports]
        summary[f"{label}_{metric}_avg"] = np.mean(values)
        summary[f"{label}_{metric}_std"] = np.std(values)

accuracies = [rep['accuracy'] for rep in reports]
summary['accuracy_avg'] = np.mean(accuracies)
summary['accuracy_std'] = np.std(accuracies)

df_summary = pd.DataFrame([summary])
print(df_summary.to_string(index=False))
