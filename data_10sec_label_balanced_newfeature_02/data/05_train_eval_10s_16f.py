import os, json, time, argparse, random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, classification_report)

import matplotlib.pyplot as plt

# ===================== Paths (dataset fixed to your build) =====================
DATA_ROOT   = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature_02\data"
DATASET_DIR = os.path.join(DATA_ROOT, "dataset_10s_16f_scaled")  # from step 04

# ===================== Utils =====================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

class GazeDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()      # (N,1500,16)
        self.Y = torch.from_numpy(Y).float()      # (N,)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        x = self.X[idx].transpose(0,1)            # (16,1500) for Conv1d
        y = self.Y[idx]
        return x, y

# --------------------- Models ---------------------
class CNN1D(nn.Module):
    def __init__(self, in_ch=16, seq_len=1500, num_classes=1, pdrop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3), nn.ReLU(), nn.BatchNorm1d(64),
            nn.MaxPool1d(2),  # 750
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(128),
            nn.MaxPool1d(2),  # 375
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(pdrop),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):  # x: (B,16,1500)
        return self.net(x).squeeze(-1)

class LSTMHead(nn.Module):
    def __init__(self, in_ch=16, hidden=128, num_layers=2, bidir=True, pdrop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_ch, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=bidir, dropout=pdrop)
        dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Linear(128, 1)
        )
    def forward(self, x):  # x: (B,16,1500) -> transpose to (B,1500,16)
        x = x.transpose(1,2)
        out, _ = self.lstm(x)        # (B,1500,dim)
        feat = out.mean(dim=1)       # GAP over time
        return self.head(feat).squeeze(-1)

# --------------------- Training helpers ---------------------
def calc_pos_weight(y_train):
    # pos_weight = N_neg / N_pos  (for BCEWithLogitsLoss)
    pos = np.sum(y_train==1); neg = np.sum(y_train==0)
    return torch.tensor([ (neg / max(pos,1)) ], dtype=torch.float32)

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total, running = 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward(); opt.step()
        running += loss.item() * xb.size(0); total += xb.size(0)
    return running / total

@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    logits_all, y_all = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        logits_all.append(logits.cpu().numpy())
        y_all.append(yb.numpy())
    logits = np.concatenate(logits_all)
    y_true = np.concatenate(y_all).astype(int)
    y_prob = 1/(1+np.exp(-logits))
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc, cm=cm, y_true=y_true, y_prob=y_prob, y_pred=y_pred, report=report)

def save_confusion_matrix(cm, out_png):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix'); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center')
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

# --------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="A name for this run, used for folders.")
    ap.add_argument("--arch", choices=["cnn","lstm"], default="cnn")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- load dataset -----
    X = np.load(os.path.join(DATASET_DIR, "X_scaled_10s_16f.npy")) # (N,1500,16)
    Y = np.load(os.path.join(DATASET_DIR, "Y.npy"))                # (N,)
    idx_train = np.load(os.path.join(DATASET_DIR, "idx_train.npy")).astype(bool)
    idx_val   = np.load(os.path.join(DATASET_DIR, "idx_val.npy")).astype(bool)
    idx_test  = np.load(os.path.join(DATASET_DIR, "idx_test.npy")).astype(bool)
    meta_df   = pd.read_csv(os.path.join(DATASET_DIR, "meta_10s_16f.csv"))

    Xtr, Ytr = X[idx_train], Y[idx_train]
    Xva, Yva = X[idx_val],   Y[idx_val]
    Xte, Yte = X[idx_test],  Y[idx_test]

    train_ds = GazeDataset(Xtr, Ytr)
    val_ds   = GazeDataset(Xva, Yva)
    test_ds  = GazeDataset(Xte, Yte)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ----- model -----
    if args.arch == "cnn":
        model = CNN1D(in_ch=16).to(device)
    else:
        model = LSTMHead(in_ch=16).to(device)

    pos_weight = calc_pos_weight(Ytr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, factor=0.5, verbose=True)

    # ----- folders -----
    model_dir = Path(DATA_ROOT) / "models" / args.model_name
    ckpt_dir  = model_dir / "checkpoints"
    results_dir = Path(DATA_ROOT) / "results" / args.model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # save config
    config = vars(args).copy()
    config.update({"dataset_dir": DATASET_DIR, "pos_weight": float(pos_weight.item()), "arch": args.arch})
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ----- train loop with early stopping (best val F1) -----
    best_f1, best_state, best_epoch = -1.0, None, -1
    history = []
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_out = run_eval(model, val_loader, device)
        sched.step(val_out["f1"])
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_f1": val_out["f1"],
                        "val_acc": val_out["acc"], "val_auc": val_out["auc"]})
        dur = time.time()-t0
        print(f"Epoch {epoch:02d} | loss={tr_loss:.4f} | val_f1={val_out['f1']:.4f} | val_acc={val_out['acc']:.4f} | time={dur:.1f}s")

        if val_out["f1"] > best_f1:
            best_f1 = val_out["f1"]; best_epoch = epoch
            best_state = { "model": model.state_dict(),
                           "epoch": epoch, "val_metrics": {k:float(v) if not isinstance(v,np.ndarray) else None
                                                           for k,v in val_out.items()} }
            torch.save(best_state, ckpt_dir / "best.pt")
        # early stop
        if epoch - best_epoch >= args.patience:
            print(f"Early stop at epoch {epoch} (no improvement for {args.patience} epochs).")
            break

    # ----- load best and evaluate on test -----
    best = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best["model"])
    test_out = run_eval(model, test_loader, device)

    # ----- save results -----
    # metrics.json
    metrics = {
        "best_epoch": int(best["epoch"]),
        "val_best_f1": float(best["val_metrics"]["f1"]),
        "test_acc": float(test_out["acc"]),
        "test_prec": float(test_out["prec"]),
        "test_rec": float(test_out["rec"]),
        "test_f1": float(test_out["f1"]),
        "test_auc": float(test_out["auc"]),
    }
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # confusion matrix
    cm = test_out["cm"]
    np.savetxt(results_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    save_confusion_matrix(cm, results_dir / "confusion_matrix.png")

    # classification report (per-class)
    cr_df = pd.DataFrame(test_out["report"]).transpose()
    cr_df.to_csv(results_dir / "classification_report.csv")

    # per-sample predictions (attach meta rows of test set)
    test_idx_mask = np.load(os.path.join(DATASET_DIR, "idx_test.npy")).astype(bool)
    meta_test = meta_df[test_idx_mask].reset_index(drop=True).copy()
    meta_test["y_true"] = test_out["y_true"]
    meta_test["y_prob"] = test_out["y_prob"]
    meta_test["y_pred"] = test_out["y_pred"]
    meta_test.to_csv(results_dir / "test_predictions.csv", index=False)

    # training history
    pd.DataFrame(history).to_csv(results_dir / "training_history.csv", index=False)

    print(f"[DONE] Model saved to: {ckpt_dir/'best.pt'}")
    print(f"[DONE] Results saved to: {results_dir}")

if __name__ == "__main__":
    main()
