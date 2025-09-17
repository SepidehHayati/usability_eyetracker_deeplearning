# -*- coding: utf-8 -*-
# runners/R06_runner_soft_fallback.py
import os, sys, json, argparse, importlib.util, random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

MIN_F1 = 0.75
MIN_PREC = 0.75
RELAXED_F1 = 0.70  # ← قید نرم

HERE = os.path.abspath(os.path.dirname(__file__))
CANDIDATE_ROOTS = [HERE, os.path.abspath(os.path.join(HERE, ".."))]
ROOT = None
for _r in CANDIDATE_ROOTS:
    if all(os.path.isdir(os.path.join(_r, d)) for d in ["data","models","results"]):
        ROOT = _r; break
if ROOT is None:
    raise RuntimeError("Project root not found.")

DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
X_PATH = os.path.join(DATA_DIR, "X_clean.npy")
Y_PATH = os.path.join(DATA_DIR, "Y_clean.npy")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

CHANNELS = ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR'] + \
           [f"delta_{c}" for c in ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR']]

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def _load_model_module_by_path(model_path: str):
    spec = importlib.util.spec_from_file_location("model_mod", model_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def _load_model_module_by_name(name_or_file: str):
    fname = name_or_file if name_or_file.endswith(".py") else f"{name_or_file}.py"
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path): raise FileNotFoundError(path)
    return _load_model_module_by_path(path), fname, path

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

class NpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32); self.y = y.astype(np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        x = np.transpose(self.X[idx], (1,0))  # (C,T)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

@torch.no_grad()
def _eval_probs(model, loader, device):
    model.eval(); yt, yp = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        pb = torch.sigmoid(model(xb)).squeeze(-1).cpu().numpy()
        yp.append(pb); yt.append(yb.numpy())
    return np.concatenate(yt), np.concatenate(yp)

def _metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred, labels=[0,1]),
        "thr": thr, "y_pred": y_pred
    }

def _choose_threshold(y_true, y_prob, min_f1=MIN_F1, min_prec=MIN_PREC, relaxed_f1=RELAXED_F1):
    thrs = np.linspace(0.0, 1.0, 1001)

    # 1) سخت: F1>=min_f1 و Prec>=min_prec
    hard = []
    best_acc = (-1, None, None)
    for t in thrs:
        m = _metrics(y_true, y_prob, t)
        if (m["acc"] > best_acc[0]): best_acc = (m["acc"], t, m)
        if (m["f1"] >= min_f1) and (m["prec"] >= min_prec):
            hard.append((m["acc"], t, m))
    if hard:
        hard.sort(key=lambda z: (z[0], z[2]["f1"], z[2]["prec"]), reverse=True)
        return hard[0][1], hard[0][2], True, False

    # 2) نرم: F1>=relaxed_f1 و Prec>=min_prec
    soft = []
    for t in thrs:
        m = _metrics(y_true, y_prob, t)
        if (m["f1"] >= relaxed_f1) and (m["prec"] >= min_prec):
            soft.append((m["acc"], t, m))
    if soft:
        soft.sort(key=lambda z: (z[0], z[2]["f1"], z[2]["prec"]), reverse=True)
        return soft[0][1], soft[0][2], False, True

    # 3) fallback: بهترین Acc بدون قید
    return best_acc[1], best_acc[2], False, False

def _run(mod, model_fname, cfg_over=None, seed=42):
    set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.load(X_PATH, mmap_mode="r"); y = np.load(Y_PATH, mmap_mode="r")
    N,T,C = X.shape; assert (T,C)==(1500,16)

    out_dir = os.path.join(RESULTS_DIR, model_fname); ensure_dir(out_dir)
    cfg = getattr(mod,"CONFIG",{"epochs":30,"batch_size":32,"lr":1e-3,"patience":7})
    if cfg_over: cfg = {**cfg, **{k:v for k,v in cfg_over.items() if v is not None}}
    wd = float(cfg.get("weight_decay", 0.0))

    with open(os.path.join(out_dir,"run_config.json"),"w") as f:
        json.dump({"runner":"R06_soft_fallback","cfg":cfg,
                   "min_f1":MIN_F1,"min_prec":MIN_PREC,"relaxed_f1":RELAXED_F1}, f, indent=2)

    class DS(Dataset):
        def __init__(self, X,y): self.X=X; self.y=y
        def __len__(self): return len(self.X)
        def __getitem__(self,i):
            x = np.transpose(self.X[i],(1,0))
            return torch.from_numpy(x).float(), torch.tensor(self.y[i],dtype=torch.long)

    fold_rows=[]
    for k in range(1,6):
        tr = np.load(os.path.join(SPLITS_DIR,f"fold{k}_train_idx.npy"))
        va = np.load(os.path.join(SPLITS_DIR,f"fold{k}_val_idx.npy"))
        mean = np.load(os.path.join(SPLITS_DIR,f"fold{k}_mean.npy"))
        std  = np.load(os.path.join(SPLITS_DIR,f"fold{k}_std.npy")); std=np.where(std<1e-6,1e-6,std)

        Xtr=((X[tr]-mean.reshape(1,1,-1))/std.reshape(1,1,-1)).astype(np.float32)
        Xva=((X[va]-mean.reshape(1,1,-1))/std.reshape(1,1,-1)).astype(np.float32)
        ytr=y[tr].astype(np.int64); yva=y[va].astype(np.int64)

        dl_tr=DataLoader(DS(Xtr,ytr),batch_size=cfg["batch_size"],shuffle=True,num_workers=0)
        dl_va=DataLoader(DS(Xva,yva),batch_size=cfg["batch_size"],shuffle=False,num_workers=0)

        model = mod.build_model(input_channels=16, seq_len=1500).to(dev)

        # pos_weight policy
        pw_cfg = cfg.get("pos_weight","auto")
        if pw_cfg=="auto":
            pos=int((ytr==1).sum()); neg=int((ytr==0).sum()); pw=neg/max(pos,1)
        elif pw_cfg is None: pw=1.0
        else: pw=float(pw_cfg)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device=dev))
        optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=wd)

        best_f1=-1; best_state=None; patience=cfg["patience"]
        for ep in range(1,cfg["epochs"]+1):
            model.train()
            for xb,yb in dl_tr:
                xb=xb.to(dev); yb=yb.to(dev).float()
                optim.zero_grad(); l=criterion(model(xb).squeeze(-1), yb)
                l.backward(); optim.step()
            yt, yp = _eval_probs(model, dl_va, dev)
            f1 = f1_score(yt, (yp>=0.5).astype(int), zero_division=0)
            if f1>best_f1:
                best_f1=f1; best_state={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
                patience=cfg["patience"]
            else:
                patience-=1
                if patience<=0: break

        fold_dir=os.path.join(out_dir,f"fold{k}"); ensure_dir(fold_dir)
        if best_state is not None: model.load_state_dict(best_state)

        yt, yp = _eval_probs(model, dl_va, dev)
        # fixed 0.5
        mf = _metrics(yt, yp, 0.5)
        with open(os.path.join(fold_dir,"metrics_fixed.json"),"w") as f:
            json.dump({k:(float(v) if isinstance(v,(np.floating,np.integer)) else v)
                       for k,v in mf.items() if k not in ["cm","y_pred"]}, f, indent=2)
        plot_confusion(mf["cm"], os.path.join(fold_dir,"confusion_matrix_0.5.png"))

        thr, mb, hard_ok, soft_ok = _choose_threshold(yt, yp, MIN_F1, MIN_PREC, RELAXED_F1)
        meta = {"best_threshold":float(thr),
                "hard_constraints_satisfied":bool(hard_ok),
                "soft_constraints_used":bool(soft_ok),
                "min_f1":float(MIN_F1), "min_prec":float(MIN_PREC), "relaxed_f1":float(RELAXED_F1)}
        with open(os.path.join(fold_dir,"metrics_best.json"),"w") as f:
            jb={k:(float(v) if isinstance(v,(np.floating,np.integer)) else v)
                for k,v in mb.items() if k not in ["cm","y_pred"]}
            jb.update(meta); json.dump(jb,f,indent=2)
        with open(os.path.join(fold_dir,"best_threshold.txt"),"w") as f: f.write(f"{thr:.4f}\n")
        plot_confusion(mb["cm"], os.path.join(fold_dir,"confusion_matrix_best.png"))

        pd.DataFrame({"y_true":yt,"y_prob":yp,"y_pred_0.5":(yp>=0.5).astype(int),"y_pred_best":mb["y_pred"]})\
            .to_csv(os.path.join(fold_dir,"predictions.csv"), index=False)

        fold_rows.append({"fold":k,"acc":float(mb["acc"]),"prec":float(mb["prec"]),
                          "rec":float(mb["rec"]),"f1":float(mb["f1"])})

    df=pd.DataFrame(fold_rows)
    df.to_csv(os.path.join(out_dir,"_summary_folds.csv"), index=False)
    avg=df[["acc","prec","rec","f1"]].mean().to_dict()
    std=df[["acc","prec","rec","f1"]].std().to_dict()
    with open(os.path.join(out_dir,"_summary_avg_std.json"),"w") as f:
        json.dump({"avg":{k:float(v) for k,v in avg.items()},
                   "std":{k:float(v) for k,v in std.items()},
                   "note":"metrics@best with soft fallback (see config in run_config.json)"}, f, indent=2)
    print("[DONE] Results in:", out_dir)

def run_with_model_path(model_path, **kw):
    mod=_load_model_module_by_path(model_path); fname=os.path.basename(model_path)
    _run(mod, fname, cfg_over=kw, seed=kw.get("seed",42))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_f1", type=float, default=None)
    p.add_argument("--min_prec", type=float, default=None)
    p.add_argument("--relaxed_f1", type=float, default=None)
    args=p.parse_args()
    global MIN_F1, MIN_PREC, RELAXED_F1
    if args.min_f1 is not None: MIN_F1=float(args.min_f1)
    if args.min_prec is not None: MIN_PREC=float(args.min_prec)
    if args.relaxed_f1 is not None: RELAXED_F1=float(args.relaxed_f1)
    mod,fname,_=_load_model_module_by_name(args.model)
    _run(mod, fname, cfg_over={"epochs":args.epochs,"batch_size":args.batch_size,"lr":args.lr,"patience":args.patience}, seed=args.seed)

if __name__=="__main__":
    main()
