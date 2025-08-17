import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# ==== تنظیم مسیر دیتا و نتایج ====
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
results_dir = os.path.join(base_dir, 'results', '21_cnn_focalloss_from_model9')

os.makedirs(results_dir, exist_ok=True)

# ==== بارگذاری داده ====
X = np.load(os.path.join(data_dir, "X.npy"))   # (108, 1500, 4)
Y = np.load(os.path.join(data_dir, "Y.npy"))   # (108,)

# نرمال‌سازی داده
X = (X - X.mean()) / X.std()

# تقسیم داده
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# تبدیل به تنسور
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# ==== مدل ====
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, 4, 1500) → (B, Channels, Seq_len)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# ==== focal loss ====
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        focal_weight = (1 - probs) ** self.gamma
        if self.alpha is not None:
            alpha_factor = self.alpha.gather(0, targets)
            focal_weight = focal_weight * alpha_factor.unsqueeze(1)

        loss = -targets_one_hot * focal_weight * log_probs
        loss = loss.sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ==== آموزش ====
model = CNNModel()
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# ==== ارزیابی ====
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_classes = torch.argmax(y_pred, dim=1).numpy()
    y_true = y_test_tensor.numpy()

# ذخیره classification report
report = classification_report(y_true, y_pred_classes, digits=4)
with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# ذخیره confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['non-easy', 'easy'], yticklabels=['non-easy', 'easy'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()
