import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# تنظیمات اولیه
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = os.path.join("..", "data", "X.npy")
label_path = os.path.join("..", "data", "Y.npy")
X = np.load(data_path)
Y = np.load(label_path)

# تقسیم داده‌ها
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# تبدیل به Tensor
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

# تعریف focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return loss.mean()

# مدل CNN + Bidirectional LSTM
class CNN_BiLSTM_Model(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)

        self.bilstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(32 * 2, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.bilstm(x)
        h = torch.cat((h_n[0], h_n[1]), dim=1)
        x = torch.relu(self.bn3(self.fc1(h)))
        x = self.dropout3(x)
        x = torch.sigmoid(self.output(x))
        return x.view(-1)

# آموزش مدل
model = CNN_BiLSTM_Model().to(device)
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 100
batch_size = 16
train_size = X_train.shape[0]

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for i in range(0, train_size, batch_size):
        inputs = X_train[i:i+batch_size]
        labels = Y_train[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# ارزیابی
model.eval()
with torch.no_grad():
    preds = model(X_test)
    preds = (preds > 0.5).int().cpu().numpy()
    Y_true = Y_test.int().cpu().numpy()

# گزارش و ذخیره نتایج
results_dir = os.path.join("..", "results", "19_cnn_bilstm_focal_loss_balanced")
os.makedirs(results_dir, exist_ok=True)

report = classification_report(Y_true, preds, digits=4)
with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
    f.write(report)

cm = confusion_matrix(Y_true, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["non-easy", "easy"], yticklabels=["non-easy", "easy"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()
