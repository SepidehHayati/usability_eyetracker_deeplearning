# -*- coding: utf-8 -*-
# runners/R09_runner_multiseed_acc_first.py
# Bagging روی چند seed + انتخاب آستانه با اولویت Accuracy (قیود سخت→نرم→آزاد)

import os, sys, json, argparse, importlib.util, random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ----- constraints (قابل تغییر) -----
HARD_MIN_F1   = 0.78
HARD_MIN_PREC = 0.82
SOFT_MIN_F1   = 0.75
SOFT_MIN_PREC = 0.78

# ----- paths -----
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(ROOT, "data")
RESULTS_DIR= os.path.join(ROOT, "results")
X_PATH     = os.path.join(DATA_DIR, "X_clean.npy")
Y_PATH     = os.path.join(DATA_DIR, "Y_clean.npy")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

CHANNELS = ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR'] + \
           [f"delta_{c}" for c in ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR']]

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

def plot_confusion(cm, out_png, labels=(0,1), title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)

class DS(Dataset):
    def __init__(self, X, y): self.X=X.astype(np.float32); self.y=y.astype(np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        x = self.X[i]              # (T,C)
        x = np.transpose(x, (1,0)) # (C,T)
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)

@torch.no_grad()
def _eval_probs(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        prob = torch.sigmoid(model(xb)).squeeze(-1).cpu().numpy()
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

def _choose_threshold_accfirst(y_true, y_prob):
    thrs = np.linspace(0.0, 1.0, 1001)
    cand = [ _metrics_at_threshold(y_true, y_prob, t) for t in thrs
             if _metrics_at_threshold(y_true, y_prob, t)["f1"]  >= HARD_MIN_F1
             and _metrics_at_threshold(y_true, y_prob, t)["prec"]>= HARD_MIN_PREC ]
    if cand:
        cand.sort(key=lambda d: (d["acc"], d["prec"], -abs(d["thr"]-0.5)), reverse=True)
        return cand[0]["thr"], cand[0], True, False

    cand2 = [ _metrics_at_threshold(y_true, y_prob, t) for t in thrs
              if _metrics_at_threshold(y_true, y_prob, t)["f1"]  >= SOFT_MIN_F1
              and _metrics_at_threshold(y_true, y_prob, t)["prec"]>= SOFT_MIN_PREC ]
    if cand2:
        cand2.sort(key=lambda d: (d["acc"], d["prec"], -abs(d["thr"]-0.5)), reverse=True)
        return cand2[0]["thr"], cand2[0], False, True

    best = None
    for t in thrs:
        m = _metrics_at_threshold(y_true, y_prob, t)
        if (best is None) or (m["acc"] > best["acc"]) or (m["acc"]==best["acc"] and m["prec"]>best["prec"]):
            best = m
    return best["thr"], best, False, False

def run_with_model_path(model_path, seeds=(42,123,999), epochs=None, batch_size=None, lr=None, patience=None, seed=None):
    # seed تکی را در این رانر نادیده می‌گیریم؛ از لیست seeds استفاده می‌کنیم
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device} | seeds={list(seeds)}")

    if not (os.path.exists(X_PATH) and os.path.exists(Y_PATH)):
        raise FileNotFoundError("X_clean.npy or Y_clean.npy not found in data/")

    X = np.load(X_PATH, mmap_mode="r"); y = np.load(Y_PATH, mmap_mode="r")
    N, T, C = X.shape
    assert (T, C) == (1500, 16), f"Unexpected X shape: {X.shape}"

    mod = _load_model_module_by_path(model_path)
    model_fname = os.path.basename(model_path)
    cfg = getattr(mod, "CONFIG", {"epochs":30, "batch_size":32, "lr":1e-3, "patience":7})
    if epochs   is not None: cfg["epochs"]   = epochs
    if batch_size is not None: cfg["batch_size"] = batch_size
    if lr is not None: cfg["lr"] = lr
    if patience is not None: cfg["patience"] = patience
    wd = float(getattr(mod, "CONFIG", {}).get("weight_decay", 0.0))

    results_subdir = os.path.join(RESULTS_DIR, model_fname + "__R09ms")
    ensure_dir(results_subdir)
    with open(os.path.join(results_subdir, "run_config.json"), "w") as f:
        json.dump({"runner":"R09_multiseed_accfirst",
                   "cfg": cfg, "channels": CHANNELS,
                   "constraints":{"hard":{"f1":HARD_MIN_F1,"prec":HARD_MIN_PREC},
                                  "soft":{"f1":SOFT_MIN_F1,"prec":SOFT_MIN_PREC}},
                   "seeds": list(seeds)}, f, indent=2)

    class MakeDS(Dataset):
        def __init__(self, X, y): self.X=X.astype(np.float32); self.y=y.astype(np.int64)
        def __len__(self): return self.X.shape[0]
        def __getitem__(self, i):
            x = self.X[i]; x = np.transpose(x, (1,0))
            return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)

    rows = []
    for k in range(1, 6):
        tr_idx = np.load(os.path.join(SPLITS_DIR, f"fold{k}_train_idx.npy"))
        va_idx = np.load(os.path.join(SPLITS_DIR, f"fold{k}_val_idx.npy"))
        mean   = np.load(os.path.join(SPLITS_DIR, f"fold{k}_mean.npy"))
        std    = np.load(os.path.join(SPLITS_DIR, f"fold{k}_std.npy"))
        std    = np.where(std < 1e-6, 1e-6, std)

        Xtr = ((X[tr_idx] - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)).astype(np.float32)
        Xva = ((X[va_idx] - mean.reshape(1,1,-1)) / std.reshape(1,1,-1)).astype(np.float32)
        ytr = y[tr_idx].astype(np.int64); yva = y[va_idx].astype(np.int64)

        train_loader = DataLoader(MakeDS(Xtr, ytr), batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
        val_loader   = DataLoader(MakeDS(Xva, yva), batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

        probs_per_seed = []
        for sd in seeds:
            set_seed(sd)
            model = mod.build_model(input_channels=16, seq_len=1500).to(device)

            pos = int((ytr==1).sum()); neg = int((ytr==0).sum())
            pos_w_cfg = getattr(mod, "CONFIG", {}).get("pos_weight", None)
            if pos_w_cfg is None:
                pos_weight = torch.tensor(neg/max(pos,1), dtype=torch.float32, device=device)
            else:
                pos_weight = torch.tensor(float(pos_w_cfg), dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=wd)

            best_f1_05, best_state, patience_left = -1.0, None, cfg["patience"]
            for epoch in range(1, cfg["epochs"]+1):
                model.train()
                for xb, yb in train_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True).float()
                    optimizer.zero_grad()
                    logits = model(xb).squeeze(-1)
                    loss = criterion(logits, yb)
                    loss.backward(); optimizer.step()

                # monitor ساده
                y_true_va, y_prob_va = _eval_probs(model, val_loader, device)
                m05 = _metrics_at_threshold(y_true_va, y_prob_va, 0.5)
                print(f"[fold {k} | seed {sd}] ep {epoch:02d} | loss={loss.item():.4f} | "
                      f"val@0.5 acc={m05['acc']:.3f} f1={m05['f1']:.3f} prec={m05['prec']:.3f} rec={m05['rec']:.3f}")

                if m05["f1"] > best_f1_05:
                    best_f1_05 = m05["f1"]
                    best_state = {n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
                    patience_left = cfg["patience"]
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(f"[fold {k} | seed {sd}] early stopping.")
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            # ذخیره‌ی احتمال‌های ولیدیشن این seed
            _, y_prob_best = _eval_probs(model, val_loader, device)
            probs_per_seed.append(y_prob_best)

            # اختیاری: ذخیرهٔ پیش‌بینی‌های هر seed
            fold_dir = os.path.join(results_subdir, f"fold{k}")
            ensure_dir(fold_dir)
            pd.DataFrame({"y_true": y_true_va, "y_prob": y_prob_best}).to_csv(
                os.path.join(fold_dir, f"predictions_seed{sd}.csv"), index=False
            )

        # Ensemble روی seedها
        y_prob_ens = np.mean(np.stack(probs_per_seed, axis=0), axis=0)
        thr, mb, hard_ok, soft_ok = _choose_threshold_accfirst(y_true_va, y_prob_ens)

        fold_dir = os.path.join(results_subdir, f"fold{k}")
        ensure_dir(fold_dir)
        pd.DataFrame({"y_true": y_true_va, "y_prob_ens": y_prob_ens,
                      "y_pred_best": mb["y_pred"]}).to_csv(os.path.join(fold_dir, "predictions_ens.csv"), index=False)

        with open(os.path.join(fold_dir, "metrics_ens.json"), "w") as f:
            jb = {kk: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                  for kk, v in mb.items() if kk not in ["cm","y_pred"]}
            jb.update({
                "best_threshold": float(thr),
                "hard_constraints_satisfied": bool(hard_ok),
                "soft_constraints_used": bool(soft_ok),
                "note": "R09_multiseed_accfirst_ensemble"
            })
            json.dump(jb, f, indent=2)
        plot_confusion(mb["cm"], os.path.join(fold_dir, "confusion_matrix_ens.png"))

        rows.append({"fold": k, "acc": float(mb["acc"]),
                     "prec": float(mb["prec"]), "rec": float(mb["rec"]), "f1": float(mb["f1"])})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_subdir, "_summary_folds.csv"), index=False)
    avg = df[["acc","prec","rec","f1"]].mean().to_dict()
    std = df[["acc","prec","rec","f1"]].std().to_dict()
    with open(os.path.join(results_subdir, "_summary_avg_std.json"), "w") as f:
        json.dump({"avg": {k: float(v) for k,v in avg.items()},
                   "std": {k: float(v) for k,v in std.items()},
                   "note": "metrics@best (multiseed ensemble, accuracy-first constraints)"},
                  f, indent=2)
    print("[DONE] Results in:", results_subdir)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42,123,999])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    args = p.parse_args()

    if os.path.sep in args.model or args.model.endswith(".py"):
        model_path = args.model if args.model.endswith(".py") else args.model + ".py"
        if not os.path.isabs(model_path):
            model_path = os.path.join(ROOT, "models", model_path)
    else:
        model_path = os.path.join(ROOT, "models", args.model + ".py")

    run_with_model_path(model_path,
                        seeds=args.seeds,
                        epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, patience=args.patience)

if __name__ == "__main__":
    main()
