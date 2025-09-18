# -*- coding: utf-8 -*-
# runners/R11_runner_accfirst_aug.py
# Accuracy-first + constraints (مثل R08) با بهبود آموزش:
# AdamW + CosineAnnealingLR + Augmentation زمانی (Train-only) + pos_weight قابل تنظیم

import os, json, argparse, importlib.util, random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- مسیرها ----------
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(ROOT, "data")
RESULTS_DIR= os.path.join(ROOT, "results")
X_PATH     = os.path.join(DATA_DIR, "X_clean.npy")
Y_PATH     = os.path.join(DATA_DIR, "Y_clean.npy")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

CHANNELS = ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR'] + \
           [f"delta_{c}" for c in ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR']]

# ---------- قیود (مشابه R08) ----------
HARD_MIN_F1   = 0.78
HARD_MIN_PREC = 0.82
SOFT_MIN_F1   = 0.75
SOFT_MIN_PREC = 0.78

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

# ---------- Augmentation های ساده فقط برای Train ----------
def time_mask_(x, max_len=30, n=2):
    # x: Tensor (B,C,T) — inplace
    B,C,T = x.shape
    if T <= 0 or max_len <= 0 or n <= 0: return
    for b in range(B):
        for _ in range(n):
            L = np.random.randint(10, max_len+1)
            s = np.random.randint(0, max(1, T-L))
            x[b, :, s:s+L] = 0.0

def add_gaussian_noise_(x, sigma=0.01):
    if sigma <= 0: return
    noise = torch.randn_like(x) * sigma
    x.add_(noise)

class DS(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32); self.y = y.astype(np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        x = np.transpose(self.X[idx], (1,0))  # (C,T)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

def _apply_train_aug(xb, use_aug=True):
    if not use_aug: return xb
    time_mask_(xb, max_len=30, n=2)
    add_gaussian_noise_(xb, sigma=0.01)
    return xb

@torch.no_grad()
def _eval_probs(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        prob = torch.sigmoid(model(xb)).squeeze(-1).detach().cpu().numpy()
        y_true.append(yb.numpy()); y_prob.append(prob)
    return np.concatenate(y_true), np.concatenate(y_prob)

def _metrics_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0,1])
    return {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "cm":cm, "thr":thr, "y_pred":y_pred}

def _choose_threshold_accfirst(y_true, y_prob,
                               hard_min_f1=HARD_MIN_F1, hard_min_prec=HARD_MIN_PREC,
                               soft_min_f1=SOFT_MIN_F1, soft_min_prec=SOFT_MIN_PREC):
    thrs = np.linspace(0.0, 1.0, 1001)
    cand = []
    for t in thrs:
        m = _metrics_at_threshold(y_true, y_prob, t)
        if (m["f1"] >= hard_min_f1) and (m["prec"] >= hard_min_prec):
            cand.append(m)
    if len(cand) > 0:
        cand.sort(key=lambda d: (d["acc"], d["prec"], -abs(d["thr"]-0.5)), reverse=True)
        return cand[0]["thr"], cand[0], True, False

    cand2 = []
    for t in thrs:
        m = _metrics_at_threshold(y_true, y_prob, t)
        if (m["f1"] >= soft_min_f1) and (m["prec"] >= soft_min_prec):
            cand2.append(m)
    if len(cand2) > 0:
        cand2.sort(key=lambda d: (d["acc"], d["prec"], -abs(d["thr"]-0.5)), reverse=True)
        return cand2[0]["thr"], cand2[0], False, True

    # fallback: بهترین Accuracy
    best = None
    for t in thrs:
        m = _metrics_at_threshold(y_true, y_prob, t)
        if (best is None) or (m["acc"] > best["acc"]) or (m["acc"]==best["acc"] and m["prec"]>best["prec"]):
            best = m
    return best["thr"], best, False, False

def run_with_model_path(model_path, epochs=None, batch_size=None, lr=None, patience=None,
                        seed=42, pos_weight=0.85, weight_decay=2e-4, use_aug=True):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    if not (os.path.exists(X_PATH) and os.path.exists(Y_PATH)):
        raise FileNotFoundError("X_clean.npy or Y_clean.npy not found in data/")

    X = np.load(X_PATH, mmap_mode="r")  # (N,1500,16)
    y = np.load(Y_PATH, mmap_mode="r")  # (N,)
    N, T, C = X.shape
    assert (T, C) == (1500, 16), f"Unexpected X shape: {X.shape}"

    # load model module
    mod = _load_model_module_by_path(model_path)
    model_fname = os.path.basename(model_path)

    # config (از مدل بخوان و override کن)
    cfg = getattr(mod, "CONFIG", {"epochs":35, "batch_size":32, "lr":1e-3, "patience":8})
    if epochs is not None: cfg["epochs"] = epochs
    if batch_size is not None: cfg["batch_size"] = batch_size
    if lr is not None: cfg["lr"] = lr
    if patience is not None: cfg["patience"] = patience

    results_subdir = os.path.join(RESULTS_DIR, f"{model_fname.replace('.py','')}__R11_accfirst_aug")
    ensure_dir(results_subdir)
    with open(os.path.join(results_subdir, "run_config.json"), "w") as f:
        json.dump({"runner":"R11_accfirst_aug",
                   "cfg": cfg,
                   "channels": CHANNELS,
                   "optim":"AdamW",
                   "sched":"CosineAnnealingLR",
                   "pos_weight": pos_weight,
                   "weight_decay": weight_decay,
                   "constraints":{"hard":{"f1":HARD_MIN_F1,"prec":HARD_MIN_PREC},
                                  "soft":{"f1":SOFT_MIN_F1,"prec":SOFT_MIN_PREC}},
                   "augment":{"enabled": use_aug, "time_mask":{"n":2,"max_len":30}, "gaussian_sigma":0.01}},
                  f, indent=2)

    class _DS(Dataset):
        def __init__(self, X, y): self.X=X.astype(np.float32); self.y=y.astype(np.int64)
        def __len__(self): return len(self.X)
        def __getitem__(self, i):
            x = np.transpose(self.X[i], (1,0))
            return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)

    fold_rows=[]
    for k in range(1,6):
        tr_idx = np.load(os.path.join(SPLITS_DIR, f"fold{k}_train_idx.npy"))
        va_idx = np.load(os.path.join(SPLITS_DIR, f"fold{k}_val_idx.npy"))
        mean   = np.load(os.path.join(SPLITS_DIR, f"fold{k}_mean.npy"))
        std    = np.load(os.path.join(SPLITS_DIR, f"fold{k}_std.npy"))
        std    = np.where(std < 1e-6, 1e-6, std)

        Xtr = ((X[tr_idx] - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)).astype(np.float32)
        Xva = ((X[va_idx] - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)).astype(np.float32)
        ytr = y[tr_idx].astype(np.int64); yva = y[va_idx].astype(np.int64)

        train_loader = DataLoader(_DS(Xtr, ytr), batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
        val_loader   = DataLoader(_DS(Xva, yva), batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

        # model + loss
        model = mod.build_model(input_channels=16, seq_len=1500).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=device))

        # AdamW + Cosine
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, cfg["epochs"]))

        best_f1_05, best_state, patience_left = -1.0, None, cfg["patience"]

        for epoch in range(1, cfg["epochs"]+1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True).float()
                if use_aug:
                    xb = _apply_train_aug(xb, use_aug=True)
                yb = yb.to(device, non_blocking=True).float()
                optimizer.zero_grad()
                logits = model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                loss.backward(); optimizer.step()
            scheduler.step()

            # quick val at 0.5
            yt, yp = _eval_probs(model, val_loader, device)
            m05 = _metrics_at_threshold(yt, yp, 0.5)
            print(f"[R11][fold {k}] epoch {epoch:02d} | loss={loss.item():.4f} | "
                  f"val@0.5: acc={m05['acc']:.3f} f1={m05['f1']:.3f} prec={m05['prec']:.3f} rec={m05['rec']:.3f}")

            if m05["f1"] > best_f1_05:
                best_f1_05 = m05["f1"]
                best_state = {n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
                patience_left = cfg["patience"]
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[R11][fold {k}] early stopping.")
                    break

        # load best
        if best_state is not None:
            model.load_state_dict(best_state)

        # انتخاب آستانه (Accuracy-first + constraints)
        yt, yp = _eval_probs(model, val_loader, device)
        thr, mb, hard_ok, soft_ok = _choose_threshold_accfirst(yt, yp)

        # ذخیره
        fold_dir = os.path.join(results_subdir, f"fold{k}")
        ensure_dir(fold_dir)
        with open(os.path.join(fold_dir, "metrics_best.json"), "w") as f:
            jb = {kk: (float(v) if isinstance(v, (float,int,np.floating,np.integer)) else v)
                  for kk,v in mb.items() if kk not in ["cm","y_pred"]}
            jb.update({"best_threshold": float(thr),
                       "hard_constraints_satisfied": bool(hard_ok),
                       "soft_constraints_used": bool(soft_ok),
                       "note": "R11_accfirst_aug"})
            json.dump(jb, f, indent=2)

        pd.DataFrame({"y_true": yt, "y_prob": yp, "y_pred_best": mb["y_pred"]}).to_csv(
            os.path.join(fold_dir, "predictions.csv"), index=False
        )

        fold_rows.append({"fold": k, "acc": float(mb["acc"]),
                          "prec": float(mb["prec"]), "rec": float(mb["rec"]), "f1": float(mb["f1"])})

    df = pd.DataFrame(fold_rows)
    df.to_csv(os.path.join(results_subdir, "_summary_folds.csv"), index=False)
    avg = df[["acc","prec","rec","f1"]].mean().to_dict()
    std = df[["acc","prec","rec","f1"]].std().to_dict()
    with open(os.path.join(results_subdir, "_summary_avg_std.json"), "w") as f:
        json.dump({"avg": {k: float(v) for k,v in avg.items()},
                   "std": {k: float(v) for k,v in std.items()},
                   "note": "R11: AdamW+CosineLR+Aug+pos_weight"}, f, indent=2)
    print("[DONE][R11] Results in:", results_subdir)

# ---- CLI (اختیاری) ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="مسیر models/<file>.py یا نام فایل")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos_weight", type=float, default=0.85)
    p.add_argument("--weight_decay", type=float, default=2e-4)
    p.add_argument("--no_aug", action="store_true")
    args = p.parse_args()

    model_path = args.model
    if not (os.path.isabs(model_path) or os.path.sep in model_path):
        model_path = os.path.join(ROOT, "models", model_path)
    if not model_path.endswith(".py"):
        model_path += ".py"
    run_with_model_path(model_path, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                        patience=args.patience, seed=args.seed,
                        pos_weight=args.pos_weight, weight_decay=args.weight_decay, use_aug=(not args.no_aug))

if __name__ == "__main__":
    main()
