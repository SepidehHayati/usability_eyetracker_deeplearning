import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ======================
# ثابت کردن seedها
# ======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ======================
# بارگذاری داده‌ها
# ======================
data_dir = os.path.join("..", "data")
X = np.load(os.path.join(data_dir, "X.npy"))   # (108, 3000, 4)
Y = np.load(os.path.join(data_dir, "Y.npy"))   # (108,)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # smaller batch size
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ======================
# Focal Loss با alpha برای هر کلاس
# ======================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else torch.tensor([0.5, 0.5])

    def forward(self, inputs, targets):
        eps = 1e-8
        inputs = torch.clamp(inputs, eps, 1. - eps)
        BCE = - (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        alpha_factor = torch.where(targets == 1, self.alpha[1], self.alpha[0])
        loss = alpha_factor * (1 - pt) ** self.gamma * BCE
        return loss.mean()

# ======================
# مدل CNN + BiLSTM
# ======================
class CNNBiLSTM(nn.Module):
    def __init__(self):
        super(CNNBiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=5)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(2)

        self.bilstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(64, 64)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.permute(0, 2, 1)  # (B, T, C)
        out, _ = self.bilstm(x)
        out = out[:, -1, :]
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = torch.sigmoid(self.fc2(out))
        return out.squeeze(1)

# ======================
# آموزش
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNBiLSTM().to(device)
criterion = FocalLoss(alpha=torch.tensor([0.9, 0.1]).to(device), gamma=2)  # class 0 مهم‌تر
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # lr کم‌تر + weight decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

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
    scheduler.step(avg_loss)
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

# ======================
# ارزیابی
# ======================
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
result_dir = os.path.join("..", "results", "09_cnn_bilstm_focal_fixseed_pytorch")
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
