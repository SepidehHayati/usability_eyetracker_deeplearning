import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

# ===============================
# مسیر داده‌ها (اصلاح شده بر اساس ساختار پروژه)
# ===============================
data_dir = os.path.join("..", "data", "X.npy")
X = np.load(os.path.join("..", "data", "X.npy"))
Y = np.load(os.path.join("..", "data", "Y.npy"))

# ===============================
# تقسیم داده‌ها
# ===============================
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# تبدیل به Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ===============================
# تعریف مدل CNN-LSTM
# ===============================
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=5),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, time)
        x = self.cnn(x)         # (batch, channels, time)
        x = x.permute(0, 2, 1)  # (batch, time, channels) -> برای LSTM
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.squeeze()

model = CNN_LSTM()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ===============================
# focal loss پیاده‌سازی ساده از
# ===============================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# ===============================
# آموزش مدل
# ===============================
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

# ===============================
# ارزیابی
# ===============================
model.eval()
y_pred, y_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        outputs = model(xb)
        y_pred.extend((outputs > 0.5).int().cpu().numpy())
        y_true.extend(yb.int().numpy())

# ===============================
# ذخیره نتایج
# ===============================
result_dir = os.path.join("..", "results", "18_cnn_lstm_focal_loss_balanced")
os.makedirs(result_dir, exist_ok=True)

# گزارش متنی
report = classification_report(y_true, y_pred, digits=4)
with open(os.path.join(result_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["non-easy", "easy"], yticklabels=["non-easy", "easy"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
plt.close()
