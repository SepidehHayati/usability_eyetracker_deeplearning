# -*- coding: utf-8 -*-
# runners/R10_runner_balanced_swa_temp.py
# Runner with:
#  - Balanced threshold selection (penalize |Prec-Rec|)
#  - Temperature scaling (per-fold) on validation logits
#  - SWA (Stochastic Weight Averaging) optional
#
# Compatible with your project layout:
#   data/X_clean.npy, data/Y_clean.npy, data/splits/fold{k}_{train,val}_idx.npy, fold{k}_{mean,std}.npy
#   models/<your_model>.py (must expose build_model(input_channels=16, seq_len=1500) and optional CONFIG)
#
# Usage examples:
#   python -m runners.R10_runner_balanced_swa_temp --model models/28_tcn_runner06.py --epochs 45 --patience 10
#   python runners/R10_runner_balanced_swa_temp.py --model models/28_tcn_runner06.py

import os, sys, json, argparse, importlib.util, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------------ Paths ------------------------
HERE = os.path.abspath(os.path.dirname(__file__))
ROOT_CANDIDATES = [os.path.abspath(os.path.join(HERE, "..")), os.path.abspath(os.path.join(HERE, "."))]
ROOT = None
for r in ROOT_CANDIDATES:
    if all(os.path.isdir(os.path.join(r, d)) for d in ["data","models","results"]):
        ROOT = r; break
if ROOT is None:
    raise RuntimeError("Project root (with data/models/results) not found.")

DATA_DIR    = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
X_PATH      = os.path.join(DATA_DIR, "X_clean.npy")
Y_PATH      = os.path.join(DATA_DIR, "Y_clean.npy")
SPLITS_DIR  = os.path.join(DATA_DIR, "splits")

CHANNELS = ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR'] + \
           [f"delta_{c}" for c in ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR']]

# ------------------------ Config (defaults, can be overridden via model.CONFIG or CLI) ------------------------
DEFAULT_CFG = {
    "epochs": 35,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 8,
    "weight_decay": 0.0,
    "pos_weight": None,          # None -> auto (neg/pos); number -> fixed
    "use_swa": True,
    "swa_start_frac": 0.5,       # start SWA at epoch = max(5, int(frac*epochs))
    "swa_lr_mult": 0.5,
    "use_temp_scaling": True,
    "balanced_delta": 0.05,      # desired |Prec-Rec| <= delta
    "hard_min_f1": 0.78,         # optional hard constraint (applied after balancing objective)
    "hard_min_prec": 0.82,       # optional hard constraint
    "seed": 42,
}

# ------------------------ Utils ------------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def _load_model_module_by_path(model_path: str):
    spec = importlib.util.spec_from_file_location("model_mod", model_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

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
        # expected model input: (B, C, T)
        x = np.transpose(self.X[idx], (1,0))  # (C,T)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

@torch.no_grad()
def _eval_logits(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Return y_true, logits (before sigmoid)."""
    model.eval()
    ys, ls = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).squeeze(-1).detach().cpu().numpy()
        ls.append(logits); ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ls)

def _sigmoid_np(x): return 1.0 / (1.0 + np.exp(-x))
def _safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps); return np.log(p / (1-p))

def _metrics_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0,1])
    return {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "cm":cm, "thr":thr, "y_pred":y_pred}

def _choose_threshold_balanced(y_true, y_prob, delta=0.05,
                               hard_min_f1=None, hard_min_prec=None):
    thrs = np.linspace(0.0, 1.0, 1001)
    best = None; best_key = None

    def key(m):
        # هدف: Acc بالا و توازن Prec/Rec (penalty اگر اختلاف از delta بیشتر شد)
        penalty = max(0.0, abs(m["prec"] - m["rec"]) - delta)
        return (m["acc"] - 0.5 * penalty, m["f1"], -abs(m["prec"] - m["rec"]))

    # دو عبور: ابتدا با قیود سخت، اگر نشد بدون قید
    for pass_id in [0, 1]:
        best = None; best_key = None
        for t in thrs:
            m = _metrics_at_threshold(y_true, y_prob, t)
            if pass_id == 0:
                if (hard_min_f1 is not None) and (m["f1"] < hard_min_f1):  continue
                if (hard_min_prec is not None) and (m["prec"] < hard_min_prec): continue
            k = key(m)
            if (best is None) or (k > best_key):
                best, best_key = m, k
        if best is not None:
            return best["thr"], best
    # fallback (نباید به اینجا برسیم)
    return 0.5, _metrics_at_threshold(y_true, y_prob, 0.5)

def _fit_temperature(logits: np.ndarray, y_true: np.ndarray,
                     init_T=1.0, steps=200, lr=0.01) -> float:
    """Fit scalar temperature T by minimizing BCEWithLogits on val fold."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = torch.tensor([init_T], requires_grad=True, device=dev, dtype=torch.float32)
    y = torch.tensor(y_true, dtype=torch.float32, device=dev)
    logit_t = torch.tensor(logits, dtype=torch.float32, device=dev)
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=steps, line_search_fn="strong_wolfe")
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    def closure():
        opt.zero_grad()
        loss = bce(logit_t / torch.clamp(T, min=1e-3), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp(min=1e-3).cpu().item())

# ------------------------ Core Runner ------------------------
def run_with_model_path(model_path,
                        epochs=None, batch_size=None, lr=None, patience=None, seed=None,
                        use_swa=None, use_temp_scaling=None, balanced_delta=None,
                        hard_min_f1=None, hard_min_prec=None, pos_weight_override=None):
    set_seed(DEFAULT_CFG["seed"] if seed is None else seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # Load data
    if not (os.path.exists(X_PATH) and os.path.exists(Y_PATH)):
        raise FileNotFoundError("X_clean.npy or Y_clean.npy not found in data/")
    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(Y_PATH, mmap_mode="r")
    N, T, C = X.shape
    assert (T, C) == (1500, 16), f"Unexpected X shape: {X.shape}"

    # Load model module & config merge
    mod = _load_model_module_by_path(model_path)
    model_fname = os.path.basename(model_path)
    cfg = dict(DEFAULT_CFG)
    if hasattr(mod, "CONFIG") and isinstance(mod.CONFIG, dict):
        cfg.update({k: v for k, v in mod.CONFIG.items() if v is not None})
    # CLI overrides
    if epochs is not None:        cfg["epochs"] = epochs
    if batch_size is not None:    cfg["batch_size"] = batch_size
    if lr is not None:            cfg["lr"] = lr
    if patience is not None:      cfg["patience"] = patience
    if seed is not None:          cfg["seed"] = seed
    if use_swa is not None:       cfg["use_swa"] = bool(use_swa)
    if use_temp_scaling is not None: cfg["use_temp_scaling"] = bool(use_temp_scaling)
    if balanced_delta is not None:   cfg["balanced_delta"] = float(balanced_delta)
    if hard_min_f1 is not None:      cfg["hard_min_f1"] = float(hard_min_f1)
    if hard_min_prec is not None:    cfg["hard_min_prec"] = float(hard_min_prec)
    if pos_weight_override is not None: cfg["pos_weight"] = float(pos_weight_override)

    # results/<model_basename>__R10_balanced_swa_temp
    results_subdir = os.path.join(RESULTS_DIR, f"{model_fname.replace('.py','')}__R10_balanced_swa_temp")
    ensure_dir(results_subdir)
    with open(os.path.join(results_subdir, "run_config.json"), "w") as f:
        json.dump({"runner": "R10_balanced_swa_temp", "cfg": cfg, "channels": CHANNELS}, f, indent=2)

    # Small inner dataset class (normalized per-fold later)
    class DS(Dataset):
        def __init__(self, X, y):
            self.X = X.astype(np.float32); self.y = y.astype(np.int64)
        def __len__(self): return self.X.shape[0]
        def __getitem__(self, i):
            x = np.transpose(self.X[i], (1,0))
            return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)

    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

    fold_rows, thr_list = [], []
    for k in range(1, 6):
        tr_idx = np.load(os.path.join(SPLITS_DIR, f"fold{k}_train_idx.npy"))
        va_idx = np.load(os.path.join(SPLITS_DIR, f"fold{k}_val_idx.npy"))
        mean   = np.load(os.path.join(SPLITS_DIR, f"fold{k}_mean.npy"))
        std    = np.load(os.path.join(SPLITS_DIR, f"fold{k}_std.npy"))
        std    = np.where(std < 1e-6, 1e-6, std)

        Xtr = ((X[tr_idx] - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)).astype(np.float32)
        Xva = ((X[va_idx] - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)).astype(np.float32)
        ytr = y[tr_idx].astype(np.int64)
        yva = y[va_idx].astype(np.int64)

        train_loader = DataLoader(DS(Xtr, ytr), batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
        val_loader   = DataLoader(DS(Xva, yva), batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

        # Model + Loss
        model = mod.build_model(input_channels=16, seq_len=1500).to(device)
        # pos_weight policy
        if cfg["pos_weight"] is None or str(cfg["pos_weight"]).lower() == "auto":
            pos = int((ytr == 1).sum()); neg = int((ytr == 0).sum())
            pos_w = neg / max(pos, 1)
        else:
            pos_w = float(cfg["pos_weight"])
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device, dtype=torch.float32))

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        # SWA
        use_swa = bool(cfg.get("use_swa", True))
        if use_swa:
            swa_start = max(5, int(cfg["epochs"] * float(cfg.get("swa_start_frac", 0.5))))
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr=cfg["lr"] * float(cfg.get("swa_lr_mult", 0.5)))
        else:
            swa_start = 1_000_000  # effectively never

        best_f1_05, best_state, patience_left = -1.0, None, cfg["patience"]

        # -------- Training loop with early stop on F1@0.5 (simple & stable) --------
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).float()
                optimizer.zero_grad()
                logits = model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                loss.backward(); optimizer.step()

            # Update SWA
            if use_swa and epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()

            # quick val (F1@0.5)
            y_true_va, logits_va = _eval_logits(model, val_loader, device)
            prob_va = _sigmoid_np(logits_va)
            m05 = _metrics_at_threshold(y_true_va, prob_va, 0.5)
            print(f"[R10][fold {k}] epoch {epoch:02d} | loss={loss.item():.4f} | "
                  f"val@0.5: acc={m05['acc']:.3f} f1={m05['f1']:.3f} "
                  f"prec={m05['prec']:.3f} rec={m05['rec']:.3f}")

            if m05["f1"] > best_f1_05:
                best_f1_05 = m05["f1"]
                best_state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
                patience_left = cfg["patience"]
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[R10][fold {k}] early stopping.")
                    break

        # Use SWA weights if enabled
        if use_swa and best_state is not None:
            # load the last trained weights first
            model.load_state_dict(best_state, strict=False)
            # update BN stats for SWA model (best-effort; safe if no BN)
            try:
                update_bn(train_loader, swa_model, device=device)
                model = swa_model
                print(f"[R10][fold {k}] SWA applied (from epoch >= {swa_start}).")
            except Exception as e:
                print(f"[R10][fold {k}] SWA BN update skipped ({e}).")

        # Load best-state (pre-SWA) if SWA is off
        if (not use_swa) and (best_state is not None):
            model.load_state_dict(best_state)

        # -------- Final evaluation on val fold with Temperature Scaling + Balanced Threshold --------
        y_true_va, logits_va = _eval_logits(model, val_loader, device)

        # Temperature scaling
        if bool(cfg.get("use_temp_scaling", True)):
            T = _fit_temperature(logits_va, y_true_va, init_T=1.0, steps=200, lr=0.01)
            logits_cal = logits_va / max(T, 1e-3)
            prob_va = _sigmoid_np(logits_cal)
            print(f"[R10][fold {k}] Temperature scaling: T={T:.3f}")
        else:
            prob_va = _sigmoid_np(logits_va)

        thr, mb = _choose_threshold_balanced(
            y_true_va, prob_va,
            delta=cfg.get("balanced_delta", 0.05),
            hard_min_f1=cfg.get("hard_min_f1", None),
            hard_min_prec=cfg.get("hard_min_prec", None)
        )

        # Save fold results
        fold_dir = os.path.join(results_subdir, f"fold{k}")
        ensure_dir(fold_dir)

        with open(os.path.join(fold_dir, "metrics_best.json"), "w") as f:
            jb = {kk: (float(v) if isinstance(v, (float, np.floating, int, np.integer)) else v)
                  for kk, v in mb.items() if kk not in ["cm", "y_pred"]}
            jb.update({"best_threshold": float(thr), "note": "Balanced+TempScaling+SWA (R10)"})
            json.dump(jb, f, indent=2)

        plot_confusion(mb["cm"], os.path.join(fold_dir, "confusion_matrix_best.png"))
        pd.DataFrame({
            "y_true": y_true_va,
            "y_prob": prob_va,
            "y_pred_best": mb["y_pred"]
        }).to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)

        fold_rows.append({"fold": k, "acc": float(mb["acc"]), "prec": float(mb["prec"]),
                          "rec": float(mb["rec"]), "f1": float(mb["f1"])})
        thr_list.append(float(thr))

    # -------- Summary across folds --------
    df = pd.DataFrame(fold_rows)
    df.to_csv(os.path.join(results_subdir, "_summary_folds.csv"), index=False)
    avg = df[["acc","prec","rec","f1"]].mean().to_dict()
    std = df[["acc","prec","rec","f1"]].std().to_dict()
    thr_med = float(np.median(np.array(thr_list))) if len(thr_list) > 0 else 0.5

    with open(os.path.join(results_subdir, "_summary_avg_std.json"), "w") as f:
        json.dump({"avg": {k: float(v) for k, v in avg.items()},
                   "std": {k: float(v) for k, v in std.items()},
                   "deploy_threshold_median": thr_med,
                   "note": "R10 Balanced threshold + TempScaling + SWA"}, f, indent=2)

    with open(os.path.join(results_subdir, "deploy_threshold_median.txt"), "w") as f:
        f.write(f"{thr_med:.4f}\n")

    print("[DONE][R10] Results in:", results_subdir)
    print("AVG:", {k: round(v, 4) for k, v in avg.items()}, "| deploy_thr(median) =", round(thr_med, 3))


# ------------------------ CLI ------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to models/<file>.py or absolute path")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--use_swa", type=int, default=None, help="1/0 to enable/disable SWA")
    p.add_argument("--use_temp_scaling", type=int, default=None, help="1/0 to enable/disable temperature scaling")
    p.add_argument("--balanced_delta", type=float, default=None, help="target |Prec-Rec| tolerance")
    p.add_argument("--hard_min_f1", type=float, default=None)
    p.add_argument("--hard_min_prec", type=float, default=None)
    p.add_argument("--pos_weight", type=float, default=None)
    args = p.parse_args()

    model_path = args.model
    if not (os.path.isabs(model_path) or os.path.sep in model_path):
        model_path = os.path.join(ROOT, "models", model_path)
    if not model_path.endswith(".py"):
        model_path += ".py"
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    run_with_model_path(
        model_path,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
        seed=args.seed, use_swa=args.use_swa, use_temp_scaling=args.use_temp_scaling,
        balanced_delta=args.balanced_delta, hard_min_f1=args.hard_min_f1, hard_min_prec=args.hard_min_prec,
        pos_weight_override=args.pos_weight
    )

if __name__ == "__main__":
    main()
