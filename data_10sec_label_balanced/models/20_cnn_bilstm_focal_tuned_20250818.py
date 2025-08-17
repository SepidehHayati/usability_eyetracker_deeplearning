
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Data loading
data_dir = os.path.join("..", "data")
X = np.load(os.path.join(data_dir, "X.npy"))   # (108, 1500, 4)
Y = np.load(os.path.join(data_dir, "Y.npy"))   # (108,)

X = X.astype(np.float32)
Y = Y.astype(np.int64)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

# Model definition
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(32 * 2, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)      # (B, 4, 1500) → (B, C_in, L)
        x = self.cnn(x)             # (B, 64, L)
        x = x.permute(0, 2, 1)      # (B, L, 64)
        _, (h_n, _) = self.lstm(x)  # h_n: (2, B, 32)
        h_n = torch.cat((h_n[0], h_n[1]), dim=1)  # (B, 64)
        out = self.fc(h_n)          # (B, 2)
        return out

# Focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_BiLSTM().to(device)
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor.to(device))
    loss = criterion(outputs, y_train_tensor.to(device))
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor.to(device))
    preds_labels = torch.argmax(preds, dim=1).cpu().numpy()

# Classification report
report = classification_report(y_test, preds_labels, digits=4)
print(report)

# Save classification report
results_dir = os.path.join("..", "results", "20_cnn_bilstm_focal_tuned_20250818")
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_test, preds_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["non-easy", "easy"], yticklabels=["non-easy", "easy"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()
