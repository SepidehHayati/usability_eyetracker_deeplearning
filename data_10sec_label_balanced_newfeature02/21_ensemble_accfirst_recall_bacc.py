# 21_ensemble_accfirst_recall_bacc.py
# Grid ensemble با قیود F1/Precision/Recall + fallback = بیشینه Balanced Accuracy
# انتخاب وزن‌ها با کلید (Acc, BAcc, F1, Rec, Prec) تا FP کنترل شود.

import os, json, itertools
import numpy as np
import pandas as pd

ROOT        = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")

# مدل‌هایی که در انسمبل می‌خواهی
MODEL_RESULTS = [
    "11_resnet1d_runner06.py",
    "10_cnn_bn_runner06.py",
    "18_resnet1d_accfirst_runner08.py",
]

# وزن صفر مجاز (حذف مدل در هر فولد)، برای سرعت می‌تونی موقتاً [0,1,2] بذاری
WEIGHT_RANGE = [0, 1, 2, 3, 4]
THRS         = np.linspace(0.0, 1.0, 501)

# قیود
HARD_MIN_F1,   HARD_MIN_PREC,   HARD_MIN_REC = 0.78, 0.82, 0.75
SOFT_MIN_F1,   SOFT_MIN_PREC,   SOFT_MIN_REC = 0.75, 0.78, 0.65

def _find_prob_column(df: pd.DataFrame) -> str:
    for c in ["y_prob", "y_prob_ens"]:
        if c in df.columns: return c
    for c in df.columns:
        if str(c).startswith("y_prob"): return c
    raise ValueError(f"Could not find probability column in {df.columns.tolist()}")

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
        rec  = np.where((TP+FN)>0, TP/(TP+FN), 0.0)          # TPR
        tnr  = np.where((TN+FP)>0, TN/(TN+FP), 0.0)          # TNR (specificity)
        acc  = (TP+TN) / (TP+FP+FN+TN)
        f1   = np.where((prec+rec)>0, 2*prec*rec/(prec+rec), 0.0)
        bacc = 0.5*(rec + tnr)
    return acc, prec, rec, f1, tnr, bacc

def choose_thr_accfirst_vec(y_true: np.ndarray, y_prob: np.ndarray, thrs: np.ndarray = THRS):
    y_true = y_true.astype(np.int64)
    pos = (y_true == 1); neg = ~pos
    P = int(pos.sum()); N = int(y_true.size)

    pred_mat = (y_prob[None, :] >= thrs[:, None])  # (T, N)
    TP = (pred_mat[:, pos]).sum(axis=1).astype(np.float64)
    FP = (pred_mat[:, neg]).sum(axis=1).astype(np.float64)
    FN = P - TP
    TN = (N - P) - FP

    acc, prec, rec, f1, tnr, bacc = _metrics_arrays_from_counts(TP, FP, FN, TN)

    # 1) HARD constraints
    hard_idx = np.where((f1>=HARD_MIN_F1) & (prec>=HARD_MIN_PREC) & (rec>=HARD_MIN_REC))[0]
    if hard_idx.size:
        key = np.stack([acc[hard_idx], bacc[hard_idx], prec[hard_idx], -np.abs(thrs[hard_idx]-0.5)], axis=1)
        best_i = hard_idx[np.lexsort((key[:,3], key[:,2], key[:,1], key[:,0]))][-1]
        return dict(acc=float(acc[best_i]), prec=float(prec[best_i]), rec=float(rec[best_i]),
                    f1=float(f1[best_i]), thr=float(thrs[best_i]), bacc=float(bacc[best_i])), True, False

    # 2) SOFT constraints
    soft_idx = np.where((f1>=SOFT_MIN_F1) & (prec>=SOFT_MIN_PREC) & (rec>=SOFT_MIN_REC))[0]
    if soft_idx.size:
        key = np.stack([acc[soft_idx], bacc[soft_idx], prec[soft_idx], -np.abs(thrs[soft_idx]-0.5)], axis=1)
        best_i = soft_idx[np.lexsort((key[:,3], key[:,2], key[:,1], key[:,0]))][-1]
        return dict(acc=float(acc[best_i]), prec=float(prec[best_i]), rec=float(rec[best_i]),
                    f1=float(f1[best_i]), thr=float(thrs[best_i]), bacc=float(bacc[best_i])), False, True

    # 3) Fallback: **بیشینه Balanced Accuracy** (تساوی → Accuracy بالاتر → بعد Precision)
    key_all = np.stack([bacc, acc, prec], axis=1)
    best_i = np.lexsort((key_all[:,2], key_all[:,1], key_all[:,0]))[-1]
    return dict(acc=float(acc[best_i]), prec=float(prec[best_i]), rec=float(rec[best_i]),
                f1=float(f1[best_i]), thr=float(thrs[best_i]), bacc=float(bacc[best_i])), False, False

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
    best_weights_per_fold = {}
    out_root = os.path.join(RESULTS_DIR, "21_ensemble_accfirst_recall_bacc")
    os.makedirs(out_root, exist_ok=True)

    for k in range(1, 6):
        y_true, P = folds_data[k]
        best_tuple = None  # (acc, bacc, f1, rec, prec, metrics_dict, hard_ok, soft_ok, weights, y_prob_ens)

        for w in weight_grids:
            w = np.array(w, dtype=np.float64)
            if w.sum() == 0:   # همه صفر؟ بی‌معنا
                continue
            w = w / w.sum()

            y_prob_ens = np.average(P, axis=0, weights=w)
            mb, hard_ok, soft_ok = choose_thr_accfirst_vec(y_true, y_prob_ens, thrs=THRS)

            # کلید انتخاب وزن: Acc → BAcc → F1 → Rec → Prec
            key = (mb["acc"], mb["bacc"], mb["f1"], mb["rec"], mb["prec"])
            if (best_tuple is None) or (key > (best_tuple[0], best_tuple[1], best_tuple[2], best_tuple[3], best_tuple[4])):
                best_tuple = (mb["acc"], mb["bacc"], mb["f1"], mb["rec"], mb["prec"],
                              mb, hard_ok, soft_ok, w.copy(), y_prob_ens.copy())

        _, _, _, _, _, mb, hard_ok, soft_ok, w_best, y_prob_ens = best_tuple
        best_weights_per_fold[k] = w_best.tolist()

        out_fold = os.path.join(out_root, f"fold{k}")
        os.makedirs(out_fold, exist_ok=True)
        pd.DataFrame({"y_true": y_true, "y_prob_ens": y_prob_ens}).to_csv(
            os.path.join(out_fold, "predictions_ens.csv"), index=False
        )
        with open(os.path.join(out_fold, "metrics_ens.json"), "w") as f:
            json.dump({
                "acc": mb["acc"], "prec": mb["prec"], "rec": mb["rec"], "f1": mb["f1"], "thr": mb["thr"],
                "bacc": mb["bacc"],
                "hard_constraints_satisfied": bool(hard_ok),
                "soft_constraints_used": bool(soft_ok),
                "weights": w_best.tolist(),
                "models": MODEL_RESULTS
            }, f, indent=2)

        rows.append({"fold": k, "acc": mb["acc"], "prec": mb["prec"], "rec": mb["rec"], "f1": mb["f1"], "bacc": mb["bacc"]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "_summary_folds.csv"), index=False)
    avg = df[["acc","prec","rec","f1","bacc"]].mean().to_dict()
    std = df[["acc","prec","rec","f1","bacc"]].std().to_dict()
    with open(os.path.join(out_root, "_summary_avg_std.json"), "w") as f:
        json.dump({
            "avg": {k: float(v) for k,v in avg.items()},
            "std": {k: float(v) for k,v in std.items()},
            "note": "Grid ensemble; fallback=max Balanced Accuracy; weight key=(Acc,BAcc,F1,Rec,Prec)",
            "models": MODEL_RESULTS,
            "weight_range": WEIGHT_RANGE,
            "thresholds": {"count": int(len(THRS)), "min": float(THRS[0]), "max": float(THRS[-1])}
        }, f, indent=2)

    print("[DONE] results →", out_root)
    print(df)
    print("AVG:", {k: round(float(v),4) for k,v in avg.items()})

if __name__ == "__main__":
    main()
