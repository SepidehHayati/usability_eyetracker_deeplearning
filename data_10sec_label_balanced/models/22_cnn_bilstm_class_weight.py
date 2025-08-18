# 22_cnn_bilstm_class_weight.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# Paths
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
results_dir = os.path.join(os.path.dirname(__file__), "..", "results", "22_cnn_bilstm_class_weight")
os.makedirs(results_dir, exist_ok=True)

# Load data
X = np.load(os.path.join(data_dir, "X.npy"))  # (samples, timesteps, features)
Y = np.load(os.path.join(data_dir, "Y.npy"))  # (samples,)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)  # (B, F, T)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Model
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_features, hidden_size, num_classes):
        super(CNN_BiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # (B, 32, T)
        x = x.permute(0, 2, 1)  # (B, T, 32)
        out, _ = self.lstm(x)  # (B, T, 2*hidden)
        out = out[:, -1, :]  # Take last time step
        return self.fc(out)

model = CNN_BiLSTM(input_features=4, hidden_size=64, num_classes=2)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = torch.argmax(predictions, dim=1)

# Classification Report
report = classification_report(Y_test_tensor, predicted_labels, target_names=["non-easy", "easy"])
print(report)
with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(Y_test_tensor, predicted_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["non-easy", "easy"], yticklabels=["non-easy", "easy"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()
