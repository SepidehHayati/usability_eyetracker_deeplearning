import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to ''

# تنظیم seed برای تکرارپذیری
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load data
X = np.load('../data/X.npy')   # shape: (samples, time_steps, features)
Y = np.load('../data/Y.npy')   # shape: (samples,)

print(f"Original class distribution: {Counter(Y)}")

# 2. Convert sequence to summary features (mean & std)
X_mean = NPV.mean(X, axis=1)
X_std = np.std(X, axis=1)
X_flat = np.concatenate([X_mean, X_std], axis=1)  # shape: (samples, 120)

# 3. Encode labels
le = LabelEncoder()
Y_enc = le.fit_transform(Y)

# 4. Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
reports = []
cms = []

for train_index, test_index in skf.split(X_flat, Y_enc):
    fold += 1
    print(f"\nFold {fold}")

    # Split data
    X_train, X_test = X_flat[train_index], X_flat[test_index]
    Y_train, Y_test = Y_enc[train_index], Y_enc[test_index]

    # 5. SMOTE
    sm = SMOTE(random_state=42)
    X_res, Y_res = sm.fit_resample(X_train, Y_train)
    print(f"After SMOTE in fold {fold}: {Counter(Y_res)}")

    # 6. One-hot
    Y_res_cat = to_categorical(Y_res, num_classes=2)
    Y_test_cat = to_categorical(Y_test, num_classes=2)

    # 7. Class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_res), y=Y notice)
    class_weight_dict = dict(enumerate(class_weights))

    # 8. Build model
    model = Sequential([
        Input(shape=(X_res.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 9. Scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)

    # 10. Train
    history = model.fit(
        X_res, Y_res_cat,
        validation_data=(X_test, Y_test_cat),
        epochs=50,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=[lr_scheduler],
        verbose=2
    )

    # 11. Predict
    Y_pred = model.predict(X_test)
    Y_pred_labels = np.argmax(Y_pred, axis=1)

    # 12. Evaluation
    report = classification_report(Y_test, Y_pred_labels, target_names=['Class 0', 'Class 1'], output_dict=True)
    cm = confusion_matrix(Y_test, Y_pred_labels)
    reports.append(report)
    cms.append(cm)

# 13. Aggregate and save results
output_dir = '../results/27_mlp_cross_validation'
os.makedirs(output_dir, exist_ok=True)

# Average metrics
avg_report = {
    'Class 0': {'precision': 0, 'recall': 0, 'f1-score': 0},
    'Class 1': {'precision': 0, 'recall': 0, 'f1-score': 0},
    'accuracy': 0
}
for report in reports:
    for cls in ['Class 0', 'Class 1']:
        for metric in ['precision', 'recall', 'f1-score']:
            avg_report[cls][metric] += report[cls][metric] / 5
    avg_report['accuracy'] += report['accuracy'] / 5

# Save average report
with open(os.path.join(output_dir, 'avg_report.txt'), 'w') as f:
    f.write(str(avg_report))

# Save average confusion matrix
avg_cm = np.mean(cms, axis=0).astype(int)
plt.figure(figsize=(6, 5))
plt.imshow(avg_cm, cmap='Blues')
plt.title('Average Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0, 1])
plt.yticks([0, 1])
for i in range(2):
    for j in range(2):
        plt.text(j, i, avg_cm[i, j], ha='center', va='center', color='black')
plt.savefig(os.path.join(output_dir, 'avg_confusion_matrix.png'))
plt.close()

print("Cross-validation finished. Results saved.")