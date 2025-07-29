# 18_cnn_lstm_weighted_smote_augmented_f1focused.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os

# برای reproducibility
torch.manual_seed(42)
np.random.seed(42)

# مسیر ذخیره نتایج
model_name = "cnn_lstm_weighted_smote_augmented_f1focused"
results_dir = f"results/{model_name}"
os.makedirs(results_dir, exist_ok=True)

# --------------------- Load Data ---------------------
X = np.load('data/X.npy')  # شکل (samples, time, features)
Y = np.load('data/Y.npy')

# Normalize
scaler = StandardScaler()
X_shape = X.shape
X = scaler.fit_transform(X.reshape(-1, X_shape[-1])).reshape(X_shape)

# --------------------- Data Augmentation ---------------------
def augment_data(X, Y):
    noise = np.random.normal(0, 0.01, X.shape)
    X_noisy = X + noise
    return np.concatenate((X, X_noisy)), np.concatenate((Y, Y))

X_aug, Y_aug = augment_data(X, Y)

# --------------------- SMOTE ---------------------
X_flat = X_aug.reshape(X_aug.shape[0], -1)
smote = SMOTE(random_state=42)
X_res, Y_res = smote.fit_resample(X_flat, Y_aug)
X_res = X_res.reshape(-1, X_shape[1], X_shape[2])

# --------------------- Train/Test Split ---------------------
X_train, X_val, Y_train, Y_val = train_test_split(X_res, Y_res, test_size=0.2, stratify=Y_res, random_state=42)

# تبدیل به Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# --------------------- Model ---------------------
class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = x.permute(0, 2, 1)  # (B, T, F)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

model = CNN_LSTM(input_dim=X.shape[2], hidden_dim=64, num_classes=2)

# --------------------- Loss + Optimizer ---------------------
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --------------------- Training ---------------------
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb)
            all_preds.append(torch.argmax(preds, dim=1))
            all_labels.append(yb)
    return torch.cat(all_preds), torch.cat(all_labels)

num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        preds = model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# --------------------- Evaluation ---------------------
y_pred, y_true = evaluate(model, val_loader)
report = classification_report(y_true, y_pred, target_names=["Easy", "Not Easy"])
print(report)

# ذخیره classification report
with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# --------------------- Confusion Matrix ---------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Easy", "Not Easy"], yticklabels=["Easy", "Not Easy"])
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()
