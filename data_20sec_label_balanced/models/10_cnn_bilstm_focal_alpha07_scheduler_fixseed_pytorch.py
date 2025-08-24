import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ثابت‌سازی نتایج
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seed(42)

# بارگذاری داده‌ها
data_dir = os.path.join("..", "data")
X = np.load(os.path.join(data_dir, "X.npy"))
Y = np.load(os.path.join(data_dir, "Y.npy"))

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

# تعریف focal loss متعادل‌شده
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = torch.tensor([0.5, 0.5])
        else:
            self.alpha = torch.tensor(alpha)

    def forward(self, inputs, targets):
        eps = 1e-8
        inputs = torch.clamp(inputs, eps, 1. - eps)
        targets = targets.long()
        alpha = self.alpha.to(inputs.device)

        loss = - alpha[targets] * ((1 - inputs) ** self.gamma) * torch.log(inputs)
        loss = torch.where(targets == 1, loss, - alpha[1 - targets] * (inputs ** self.gamma) * torch.log(1 - inputs))
        return loss.mean()

# تعریف مدل CNN + BiLSTM
class CNNBiLSTM(nn.Module):
    def __init__(self):
        super(CNNBiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.3)

        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64*2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (B, 32, T/2)
        x = self.drop1(x)
        x = x.permute(0, 2, 1)  # (B, T/2, C)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # آخرین تایم‌استپ
        x = self.drop2(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)

# آماده‌سازی آموزش
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNBiLSTM().to(device)
criterion = FocalLoss(alpha=[0.7, 0.3], gamma=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# تنظیم EarlyStopping
best_loss = float('inf')
wait = 0
patience = 10

# آموزش مدل
for epoch in range(100):
    model.train()
    total_loss = 0
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
        torch.save(model.state_dict(), "best_model.pt")
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

# ارزیابی
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
result_dir = os.path.join("..", "results", "10_cnn_bilstm_focal_alpha07_scheduler_fixseed_pytorch")
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
