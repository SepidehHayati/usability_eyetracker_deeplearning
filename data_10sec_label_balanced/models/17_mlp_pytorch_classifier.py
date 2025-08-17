import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- Load Data ----------------------
data_dir = os.path.join("..", "data")
X = np.load(os.path.join(data_dir, "X.npy"))   # shape: (108, 1500, 4)
Y = np.load(os.path.join(data_dir, "Y.npy"))   # shape: (108,)

# Convert to torch tensors
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.long)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# ---------------------- DataLoader ----------------------
train_ds = TensorDataset(X_train, Y_train)
test_ds = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=16)

# ---------------------- Define MLP Model ----------------------
class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(1500 * 4, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

# ---------------------- Training ----------------------
model = MLPClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

# ---------------------- Evaluation ----------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

# ---------------------- Save Results ----------------------
result_dir = os.path.join("..", "results", "17_mlp_pytorch_classifier")
os.makedirs(result_dir, exist_ok=True)

# Save classification report
report = classification_report(all_labels, all_preds, digits=4)
with open(os.path.join(result_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# Save confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["non-easy", "easy"], yticklabels=["non-easy", "easy"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))
plt.close()
