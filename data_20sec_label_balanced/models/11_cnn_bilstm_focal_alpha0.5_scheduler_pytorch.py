import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# بارگذاری داده‌ها
data_dir = os.path.join("..", "data")
X = np.load(os.path.join(data_dir, "X.npy"))
Y = np.load(os.path.join(data_dir, "Y.npy"))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# تعریف Focal Loss با α=0.5
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        eps = 1e-8
        inputs = torch.clamp(inputs, eps, 1. - eps)
        BCE = - (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE
        return loss.mean()

# مدل CNN + BiLSTM
class CNNBiLSTMClassifier(nn.Module):
    def __init__(self):
        super(CNNBiLSTMClassifier, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(0.3)

        self.bilstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1,
                              batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop2(x)

        x = x.permute(0, 2, 1)  # (B, T, C)
        x, _ = self.bilstm(x)   # (B, T, 2H)
        x = x[:, -1, :]         # آخرین تایم‌استپ

        x = F.relu(self.bn3(self.fc1(x)))
        x = self.drop3(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)

# تنظیمات آموزش
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNBiLSTMClassifier().to(device)
criterion = FocalLoss(alpha=0.35, gamma=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Early stopping
best_loss = float('inf')
patience = 10
wait = 0

for epoch in range(100):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    scheduler.step(avg_loss)

    if avg_loss < best_loss:
        best_loss = avg_loss
        wait = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# ارزیابی مدل
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = (outputs > 0.5).int().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

# ذخیره نتایج
result_dir = os.path.join("..", "results", "11_cnn_bilstm_focal_alpha0.5_scheduler_pytorch")
os.makedirs(result_dir, exist_ok=True)

report = classification_report(all_labels, all_preds, digits=4)
with open(os.path.join(result_dir, "classification_report.txt"), "w") as f:
    f.write(report)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["non-easy", "easy"], yticklabels=["non-easy", "easy"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
plt.close()
