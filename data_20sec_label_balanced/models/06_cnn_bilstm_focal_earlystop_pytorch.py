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

# مسیر داده‌ها
data_dir = os.path.join("..", "data")
X = np.load(os.path.join(data_dir, "X.npy"))   # (108, 3000, 4)
Y = np.load(os.path.join(data_dir, "Y.npy"))   # (108,)

# تقسیم داده‌ها
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

# تبدیل به Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
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
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.3)

        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(0.5)
        self.fc = nn.Linear(64 * 2, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, 4, 3000) → (B, C, T)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (B, 32, L)
        x = self.drop1(x)
        x = x.permute(0, 2, 1)  # (B, L, 32) → برای LSTM
        out, _ = self.lstm(x)   # (B, L, 128)
        out = out[:, -1, :]     # فقط آخرین زمان
        out = self.drop2(out)
        out = torch.sigmoid(self.fc(out))
        return out.squeeze(1)

# آماده‌سازی آموزش
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_BiLSTM().to(device)
criterion = FocalLoss(alpha=0.75, gamma=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Early Stopping
best_loss = float('inf')
patience = 10
wait = 0

# آموزش مدل
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
result_dir = os.path.join("..", "results", "06_cnn_bilstm_focal_earlystop_pytorch")
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
