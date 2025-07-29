import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================== CONFIG =====================
SEED = 42
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
D_MODEL = 64
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
DROPOUT = 0.1

os.makedirs('../results/28_transformer_weighted_classifier', exist_ok=True)

# =============== SET SEED ======================
torch.manual_seed(SEED)
np.random.seed(SEED)

# ========== LOAD DATA (X.npy, Y.npy) ==========
X = np.load("../data/X.npy")  # (108, 1500, 4)
Y = np.load("../data/Y.npy")   # (108,)



# ========== TRAIN-TEST SPLIT ==================
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =============== MODEL =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(4, D_MODEL)
        self.pos_encoder = PositionalEncoding(D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=NUM_HEADS, dropout=DROPOUT, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_ENCODER_LAYERS)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(D_MODEL, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(1)

# ============ LOSS FUNCTION ====================
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, outputs, targets):
        return self.bce(outputs, targets)

# =============== TRAIN =========================
def train_model(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# =============== EVAL ==========================
def eval_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            outputs = model(x_batch)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

# ============ RUN ==============================
model = TransformerClassifier()
criterion = WeightedBCELoss(pos_weight=2.0)  # چون کلاس 0 یک سومه
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    loss = train_model(model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

# ========== EVALUATION ========================
preds, labels = eval_model(model, test_loader)
y_pred_class = (preds > 0.5).astype(int)

report = classification_report(labels, y_pred_class, digits=4)
print(report)

# Save report
with open("results/transformer/report.txt", "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(labels, y_pred_class)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Easy', 'Not Easy'], yticklabels=['Easy', 'Not Easy'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Transformer Classifier - Confusion Matrix")
plt.savefig("results/transformer/confusion_matrix.png")
plt.close()
