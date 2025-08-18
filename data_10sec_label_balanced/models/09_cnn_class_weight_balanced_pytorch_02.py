import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# مسیر داده‌ها
data_dir = os.path.join("..", "data")
X = np.load(os.path.join(data_dir, "X.npy"))   # (108, 1500, 4)
Y = np.load(os.path.join(data_dir, "Y.npy"))   # (108,)

# تبدیل به تنسور PyTorch و تغییر شکل داده‌ها
X = torch.FloatTensor(X).permute(0, 2, 1)  # از (108, 1500, 4) به (108, 4, 1500)
Y = torch.FloatTensor(Y).unsqueeze(1)  # برای سازگاری با خروجی sigmoid (shape: [108, 1])

# تقسیم داده‌ها
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y.numpy())

# محاسبه class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train.numpy()), y=Y_train.numpy().flatten())
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# تعریف مدل CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 373, 64)  # محاسبه بعد خروجی بعد از pooling
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

model = CNN()

# تعریف optimizer و loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# لود دیتا به DataLoader
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)

# انتقال به دستگاه (GPU اگه موجود باشه)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# آموزش مدل با class weights
for epoch in range(100):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

# پیش‌بینی
model.eval()
X_test = X_test.to(device)
with torch.no_grad():
    Y_pred = model(X_test)
    Y_pred = (Y_pred > 0.5).float()

# تبدیل به numpy برای محاسبات بعدی
Y_test = Y_test.cpu().numpy()
Y_pred = Y_pred.cpu().numpy()

# مسیر ذخیره نتایج
result_dir = os.path.join("..", "results", "09_cnn_class_weight_balanced_pytorch_02")
os.makedirs(result_dir, exist_ok=True)

# ذخیره گزارش متنی
report = classification_report(Y_test, Y_pred, digits=4)
with open(os.path.join(result_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# ذخیره confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["non-easy", "easy"], yticklabels=["non-easy", "easy"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
plt.close()