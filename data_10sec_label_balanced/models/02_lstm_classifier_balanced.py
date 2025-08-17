import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

# مسیر داده‌ها
base_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base_dir, "data")
X = np.load(os.path.join(data_dir, "X.npy"))  # (108, 1500, 4)
Y = np.load(os.path.join(data_dir, "Y.npy"))  # (108,)

# تقسیم داده‌ها
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# تبدیل به تنسور
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
Y_test = torch.tensor(Y_test, dtype=torch.long)

# مدل LSTM ساده
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_classes=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # فقط خروجی آخر
        out = self.fc(out)
        return out

# آموزش مدل
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train.to(device))
    loss = criterion(outputs, Y_train.to(device))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# ارزیابی
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device)).argmax(dim=1).cpu().numpy()

report = classification_report(Y_test, y_pred, digits=4)
print(report)

# مسیر ذخیره نتایج
results_dir = os.path.join(base_dir, "results", "02_lstm_classifier_balanced")
os.makedirs(results_dir, exist_ok=True)

# ذخیره گزارش متنی
with open(os.path.join(results_dir, "report.txt"), "w") as f:
    f.write(report)

# ذخیره confusion matrix
cm = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()
