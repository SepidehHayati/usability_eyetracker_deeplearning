# 14_ensemble_grid_accfirst.py
# Grid-search وزن‌ها + انتخاب آستانه با اولویت Accuracy و قیود F1/Precision/Recall (نسخه سریع و بردارسازی‌شده)

import os, json, itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT        = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")

MODEL_RESULTS = [
    "11_resnet1d_runner06.py",
    "10_cnn_bn_runner06.py",
    "18_resnet1d_accfirst_runner08.py",
]

WEIGHT_RANGE = [1, 2, 3, 4]
THRS         = np.linspace(0.0, 1.0, 501)

# قیود
HARD_MIN_F1,   HARD_MIN_PREC,   HARD_MIN_REC = 0.78, 0.82, 0.75
SOFT_MIN_F1,   SOFT_MIN_PREC,   SOFT_MIN_REC = 0.75, 0.78, 0.65

def _find_prob_column(df: pd.DataFrame) -> str:
    for c in ["y_prob", "y_prob_ens"]:
        if c in df.columns:
            return c
    prob_cols = [c for c in df.columns if c.startswith("y_prob")]
    if not prob_cols:
        raise ValueError(f"Could not find probability column in {df.columns.tolist()}")
    return prob_cols[0]

def load_fold_probs(model_dirname: str, fold: int):
    p1 = os.path.join(RESULTS_DIR, model_dirname, f"fold{fold}", "predictions.csv")
    p2 = os.path.join(RESULTS_DIR, model_dirname, f"fold{fold}", "predictions_ens.csv")
    path = p1 if os.path.exists(p1) else p2
    if not os.path.exists(path):
        raise FileNotFoundError(f"No predictions file for {model_dirname} fold{fold}")
    df = pd.read_csv(path)
    prob_col = _find_prob_column(df)
    return df["y_true"].values.astype(np.int64), df[prob_col].values.astype(np.float64)

def _metrics_arrays_from_counts(TP, FP, FN, TN):
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where((TP+FP)>0, TP/(TP+FP), 0.0)
        rec  = np.where((TP+FN)>0, TP/(TP+FN), 0.0)
        acc  = (TP+TN) / (TP+FP+FN+TN)
        f1   = np.where((prec+rec)>0, 2*prec*rec/(prec+rec), 0.0)
    return acc, prec, rec, f1

def choose_thr_accfirst_vec(y_true: np.ndarray, y_prob: np.ndarray, thrs: np.ndarray = THRS):
    y_true = y_true.astype(np.int64)
    pos = (y_true == 1); neg = ~pos
    P = int(pos.sum()); N = int(y_true.size)

    pred_mat = (y_prob[None, :] >= thrs[:, None])  # (T, N)
    TP = (pred_mat[:, pos]).sum(axis=1).astype(np.float64)
    FP = (pred_mat[:, neg]).sum(axis=1).astype(np.float64)
    FN = P - TP
    TN = (N - P) - FP

    acc, prec, rec, f1 = _metrics_arrays_from_counts(TP, FP, FN, TN)

    def _pick_best(idx):
        # مرتب‌سازی: اول Acc، بعد Prec، بعد نزدیکی به 0.5
        key = np.stack([acc[idx], prec[idx], -np.abs(thrs[idx]-0.5)], axis=1)
        best_i = idx[np.lexsort((key[:,2], key[:,1], key[:,0]))][-1]
        return dict(acc=float(acc[best_i]), prec=float(prec[best_i]), rec=float(rec[best_i]),
                    f1=float(f1[best_i]), thr=float(thrs[best_i]))

    hard_idx = np.where((f1>=HARD_MIN_F1) & (prec>=HARD_MIN_PREC) & (rec>=HARD_MIN_REC))[0]
    if hard_idx.size: return _pick_best(hard_idx), True, False

    soft_idx = np.where((f1>=SOFT_MIN_F1) & (prec>=SOFT_MIN_PREC) & (rec>=SOFT_MIN_REC))[0]
    if soft_idx.size: return _pick_best(soft_idx), False, True

    # بدون قید: بیشینه Acc (تساوی → Prec بالاتر)
    key_all = np.stack([acc, prec], axis=1)
    best_i = np.lexsort((key_all[:,1], key_all[:,0]))[-1]
    return dict(acc=float(acc[best_i]), prec=float(prec[best_i]), rec=float(rec[best_i]),
                f1=float(f1[best_i]), thr=float(thrs[best_i])), False, False

def main():
    folds_data = {}
    for k in range(1, 6):
        probs = []
        yref = None
        for md in MODEL_RESULTS:
            yt, yp = load_fold_probs(md, k)
            if yref is None: yref = yt
            else:
                if not np.array_equal(yref, yt):
                    raise ValueError(f"y_true mismatch between models on fold{k}")
            probs.append(yp)
        folds_data[k] = (yref, np.stack(probs, axis=0))  # (M, Nfold)

    M = len(MODEL_RESULTS)
    weight_grids = list(itertools.product(WEIGHT_RANGE, repeat=M))

    rows = []
    out_root = os.path.join(RESULTS_DIR, "14_ensemble_grid_accfirst")
    os.makedirs(out_root, exist_ok=True)

    for k in range(1, 6):
        y_true, P = folds_data[k]  # P: (M, N)
        best = None
        best_w = None
        best_prob = None

        for w in weight_grids:
            w = np.array(w, dtype=np.float64); w = w / w.sum()
            y_prob_ens = np.average(P, axis=0, weights=w)
            mb, hard_ok, soft_ok = choose_thr_accfirst_vec(y_true, y_prob_ens, thrs=THRS)
            key = (mb["acc"], mb["prec"])
            if (best is None) or (key > (best["acc"], best["prec"])):
                best = {**mb, "hard": bool(hard_ok), "soft": bool(soft_ok)}
                best_w = w.copy()
                best_prob = y_prob_ens.copy()

        out_fold = os.path.join(out_root, f"fold{k}")
        os.makedirs(out_fold, exist_ok=True)
        pd.DataFrame({"y_true": y_true, "y_prob_ens": best_prob}).to_csv(os.path.join(out_fold, "predictions_ens.csv"), index=False)
        with open(os.path.join(out_fold, "metrics_ens.json"), "w") as f:
            json.dump({
                "acc":best["acc"], "prec":best["prec"], "rec":best["rec"], "f1":best["f1"], "thr":best["thr"],
                "hard_constraints_satisfied": best["hard"],
                "soft_constraints_used": best["soft"],
                "weights": best_w.tolist(),
                "models": MODEL_RESULTS
            }, f, indent=2)

        rows.append({"fold":k, "acc":best["acc"], "prec":best["prec"], "rec":best["rec"], "f1":best["f1"]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "_summary_folds.csv"), index=False)
    avg = df[["acc","prec","rec","f1"]].mean().to_dict()
    std = df[["acc","prec","rec","f1"]].std().to_dict()
    with open(os.path.join(out_root, "_summary_avg_std.json"), "w") as f:
        json.dump({"avg":{k:float(v) for k,v in avg.items()},
                   "std":{k:float(v) for k,v in std.items()},
                   "note":"Grid ensemble (accuracy-first) with F1/Precision/Recall constraints",
                   "models": MODEL_RESULTS,
                   "weight_range": WEIGHT_RANGE,
                   "thresholds": {"count": int(len(THRS))}}, f, indent=2)
    print("[DONE]", df)
    print("AVG:", {k: round(float(v),4) for k,v in avg.items()})

if __name__ == "__main__":
    main()
