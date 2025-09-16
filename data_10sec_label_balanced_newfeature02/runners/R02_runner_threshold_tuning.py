# -*- coding: utf-8 -*-
"""
R02_runner_threshold_tuning.py
- 5-fold (Group) CV pipeline with train-only scaling
- EarlyStopping on F1@0.5
- Threshold tuning on validation to maximize F1 (per fold)
- Saves metrics at thr=0.5 and at best thr, confusion matrices, predictions, and fold summaries
"""

import os, sys, json, argparse, importlib.util, random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------------------- Robust root resolver --------------------
HERE = os.path.abspath(os.path.dirname(__file__))
CANDIDATE_ROOTS = [HERE, os.path.abspath(os.path.join(HERE, ".."))]

ROOT = None
for _root in CANDIDATE_ROOTS:
    if os.path.isdir(os.path.join(_root, "data")) and \
       os.path.isdir(os.path.join(_root, "models")) and \
       os.path.isdir(os.path.join(_root, "results")):
        ROOT = _root
        break
if ROOT is None:
    raise RuntimeError("Could not locate project root with data/models/results folders.")

DATA_DIR    = os.path.join(ROOT, "data")
MODELS_DIR  = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")

X_PATH     = os.path.join(DATA_DIR, "X_clean.npy")
Y_PATH     = os.path.join(DATA_DIR, "Y_clean.npy")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

CHANNELS = ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR'] + \
           [f"delta_{c}" for c in ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR']]

# -------------------- Utils --------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def _load_model_module_by_path(model_path: str):
    spec = importlib.util.spec_from_file_location("model_mod", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _load_model_module_by_name(model_name_or_file: str):
    fname = model_name_or_file if model_name_or_file.endswith(".py") else f"{model_name_or_file}.py"
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return _load_model_module_by_path(path), fname, path

def plot_confusion(cm, out_png, labels=(0,1), title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

class NpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32); self.y = y.astype(np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        x = self.X[idx]              # (T, C)
        x = np.transpose(x, (1,0))   # -> (C, T) for Conv1d
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

@torch.no_grad()
def _eval_probs(model, loader, device):
    """Return y_true, y_prob (no hard predictions here)."""
    model.eval()
    y_true, y_prob = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)                          # (B,1)
        prob = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        y_prob.append(prob); y_true.append(yb.numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)

def _metrics_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0,1])
    return {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "cm":cm, "thr":thr, "y_pred":y_pred}

def _find_best_threshold(y_true, y_prob, metric="f1"):
    thrs = np.linspace(0.0, 1.0, 101)
    best = None
    for t in thrs:
        m = _metrics_at_threshold(y_true, y_prob, t)
        score = m[metric]
        if (best is None) or (score > best[0]):
            best = (score, t, m)
    return best[1], best[2]  # best_thr, metrics_dict

# -------------------- Core Runner --------------------
def _run_training(mod, model_fname, cfg_overrides=None, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # Load data
    if not (os.path.exists(X_PATH) and os.path.exists(Y_PATH)):
        raise FileNotFoundError("X_clean.npy or Y_clean.npy not found in data/")
    X = np.load(X_PATH, mmap_mode="r")  # (N,1500,16)
    y = np.load(Y_PATH, mmap_mode="r")  # (N,)
    N, T, C = X.shape
    assert (T, C) == (1500, 16), f"Unexpected X shape: {X.shape}"

    # Results dir (exact model file name)
    results_subdir = os.path.join(RESULTS_DIR, model_fname)
    ensure_dir(results_subdir)

    # Hyperparams
    cfg = getattr(mod, "CONFIG", {"epochs":30, "batch_size":32, "lr":1e-3, "patience":7})
    if cfg_overrides:
        cfg = {**cfg, **{k:v for k,v in cfg_overrides.items() if v is not None}}

    # Save run config
    with open(os.path.join(results_subdir, "run_config.json"), "w") as f:
        json.dump({"model_file": model_fname, "cfg": cfg, "channels": CHANNELS}, f, indent=2)

    fold_metrics_best = []   # for global summary (metrics@best_thr)
    for k in range(1, 6):
        tr_idx = np.load(os.path.join(SPLITS_DIR, f"fold{k}_train_idx.npy"))
        va_idx = np.load(os.path.join(SPLITS_DIR, f"fold{k}_val_idx.npy"))
        mean   = np.load(os.path.join(SPLITS_DIR, f"fold{k}_mean.npy"))
        std    = np.load(os.path.join(SPLITS_DIR, f"fold{k}_std.npy"))
        std    = np.where(std < 1e-6, 1e-6, std)

        # Scaling per fold (train-only stats)
        Xtr = ((X[tr_idx] - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)).astype(np.float32)
        Xva = ((X[va_idx] - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)).astype(np.float32)
        ytr = y[tr_idx].astype(np.int64); yva = y[va_idx].astype(np.int64)

        train_loader = DataLoader(NpyDataset(Xtr, ytr), batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
        val_loader   = DataLoader(NpyDataset(Xva, yva), batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

        # Build model
        model = mod.build_model(input_channels=16, seq_len=1500).to(device)

        # Class imbalance → pos_weight
        pos = int((ytr==1).sum()); neg = int((ytr==0).sum())
        pos_weight = torch.tensor(neg/max(pos,1), dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

        best_f1_05, best_state, patience_left = -1.0, None, cfg["patience"]

        # ---- Train loop (early stop on F1@0.5) ----
        for epoch in range(1, cfg["epochs"]+1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)          # (B,C,T)
                yb = yb.to(device, non_blocking=True).float()  # (B,)
                optimizer.zero_grad()
                logits = model(xb).squeeze(-1)                 # (B,)
                loss = criterion(logits, yb)
                loss.backward(); optimizer.step()

            # quick val at thr=0.5 for monitoring/ES
            y_true_va, y_prob_va = _eval_probs(model, val_loader, device)
            m05 = _metrics_at_threshold(y_true_va, y_prob_va, 0.5)
            print(f"[fold {k}] epoch {epoch:02d} | loss={loss.item():.4f} | "
                  f"val@0.5: acc={m05['acc']:.3f} f1={m05['f1']:.3f} "
                  f"prec={m05['prec']:.3f} rec={m05['rec']:.3f}")

            if m05["f1"] > best_f1_05:
                best_f1_05 = m05["f1"]
                best_state = {n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
                patience_left = cfg["patience"]
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[fold {k}] early stopping triggered.")
                    break

        # ---- Load best state (by F1@0.5) and do Threshold Tuning ----
        fold_dir = os.path.join(results_subdir, f"fold{k}")
        ensure_dir(fold_dir)
        if best_state is not None:
            model.load_state_dict(best_state)

        # final eval on val
        y_true_va, y_prob_va = _eval_probs(model, val_loader, device)

        # 1) fixed thr=0.5
        m_fixed = _metrics_at_threshold(y_true_va, y_prob_va, 0.5)
        with open(os.path.join(fold_dir, "metrics_fixed.json"), "w") as f:
            json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k,v in m_fixed.items() if k not in ["cm","y_pred"]}, f, indent=2)
        plot_confusion(m_fixed["cm"], os.path.join(fold_dir, "confusion_matrix_0.5.png"))

        # 2) best threshold for max F1
        best_thr, m_best = _find_best_threshold(y_true_va, y_prob_va, metric="f1")
        with open(os.path.join(fold_dir, "metrics_best.json"), "w") as f:
            mb = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                  for k,v in m_best.items() if k not in ["cm","y_pred"]}
            mb["best_threshold"] = float(best_thr)
            json.dump(mb, f, indent=2)
        with open(os.path.join(fold_dir, "best_threshold.txt"), "w") as f:
            f.write(f"{best_thr:.4f}\n")
        plot_confusion(m_best["cm"], os.path.join(fold_dir, "confusion_matrix_best.png"))

        # predictions csv
        pd.DataFrame({
            "y_true": y_true_va,
            "y_prob": y_prob_va,
            "y_pred_0.5": m_fixed["y_pred"],
            "y_pred_best": m_best["y_pred"]
        }).to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)

        # for global summary, use metrics@best
        fold_metrics_best.append({"fold": k,
                                  "acc": float(m_best["acc"]),
                                  "prec": float(m_best["prec"]),
                                  "rec": float(m_best["rec"]),
                                  "f1": float(m_best["f1"])})

    # ----- Summary over folds (metrics@best) -----
    df = pd.DataFrame(fold_metrics_best)
    df.to_csv(os.path.join(results_subdir, "_summary_folds.csv"), index=False)
    avg = df[["acc","prec","rec","f1"]].mean().to_dict()
    std = df[["acc","prec","rec","f1"]].std().to_dict()
    with open(os.path.join(results_subdir, "_summary_avg_std.json"), "w") as f:
        json.dump({"avg": {k: float(v) for k,v in avg.items()},
                   "std": {k: float(v) for k,v in std.items()},
                   "note": "metrics@best threshold per fold"}, f, indent=2)
    print("[DONE] Results in:", results_subdir)

# -------------------- Public API (for models) --------------------
def run_with_model_path(model_path, epochs=None, batch_size=None, lr=None, patience=None, seed=42):
    mod = _load_model_module_by_path(model_path)
    model_fname = os.path.basename(model_path)   # e.g., 01_cnn_runner02.py
    cfg_overrides = {"epochs":epochs, "batch_size":batch_size, "lr":lr, "patience":patience}
    _run_training(mod, model_fname, cfg_overrides=cfg_overrides, seed=seed)

# -------------------- CLI entry (optional) --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="file in models/ (e.g., cnn1d or 01_cnn_runner02.py)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mod, model_fname, model_path = _load_model_module_by_name(args.model)
    cfg_overrides = {"epochs":args.epochs, "batch_size":args.batch_size, "lr":args.lr, "patience":args.patience}
    _run_training(mod, model_fname, cfg_overrides=cfg_overrides, seed=args.seed)

if __name__ == "__main__":
    main()
