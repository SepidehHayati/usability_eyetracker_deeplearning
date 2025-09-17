# 23_ensemble_accfirst_fprpen.py
# Ensemble با:
# - قیود HARD/SOFT (F1/Prec/Rec)
# - fallback: بیشینه‌سازی (Accuracy - alpha*FPR)
# - weight grid پیوسته (گام 0.1)، وزن صفر مجاز
# - ترکیب به روش logit-averaging
# - کلید انتخاب وزن: (Acc, BAcc, F1, Rec, Prec)

import os, json, itertools
import numpy as np
import pandas as pd

ROOT        = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")

MODEL_RESULTS = [
    "11_resnet1d_runner06.py",
    "10_cnn_bn_runner06.py",
    "18_resnet1d_accfirst_runner08.py",
]

THRS = np.linspace(0.0, 1.0, 501)

# قیود
HARD_MIN_F1,   HARD_MIN_PREC,   HARD_MIN_REC = 0.78, 0.82, 0.75
SOFT_MIN_F1,   SOFT_MIN_PREC,   SOFT_MIN_REC = 0.75, 0.78, 0.65

# پارامتر جریمهٔ FP در fallback
ALPHA_FPR_PENALTY = 0.15

EPS = 1e-6

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
    y_true = df["y_true"].values.astype(np.int64)
    y_prob = df[prob_col].values.astype(np.float64)
    y_prob = np.clip(y_prob, EPS, 1.0 - EPS)
    return y_true, y_prob

def _metrics_arrays_from_counts(TP, FP, FN, TN):
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where((TP+FP)>0, TP/(TP+FP), 0.0)
        rec  = np.where((TP+FN)>0, TP/(TP+FN), 0.0)           # TPR
        tnr  = np.where((TN+FP)>0, TN/(TN+FP), 0.0)           # TNR
        fpr  = 1.0 - tnr
        acc  = (TP+TN) / (TP+FP+FN+TN)
        f1   = np.where((prec+rec)>0, 2*prec*rec/(prec+rec), 0.0)
        bacc = 0.5*(rec + tnr)
    return acc, prec, rec, f1, tnr, bacc, fpr

def choose_thr_accfirst_vec(y_true: np.ndarray, y_prob: np.ndarray, thrs: np.ndarray = THRS):
    y_true = y_true.astype(np.int64)
    pos = (y_true == 1); neg = ~pos
    P = int(pos.sum()); N = int(y_true.size)

    pred_mat = (y_prob[None, :] >= thrs[:, None])  # (T, N)
    TP = (pred_mat[:, pos]).sum(axis=1).astype(np.float64)
    FP = (pred_mat[:, neg]).sum(axis=1).astype(np.float64)
    FN = P - TP
    TN = (N - P) - FP

    acc, prec, rec, f1, tnr, bacc, fpr = _metrics_arrays_from_counts(TP, FP, FN, TN)

    # 1) HARD constraints
    hard_idx = np.where((f1>=HARD_MIN_F1) & (prec>=HARD_MIN_PREC) & (rec>=HARD_MIN_REC))[0]
    if hard_idx.size:
        key = np.stack([acc[hard_idx], bacc[hard_idx], prec[hard_idx], -np.abs(thrs[hard_idx]-0.5)], axis=1)
        best_i = hard_idx[np.lexsort((key[:,3], key[:,2], key[:,1], key[:,0]))][-1]
        return dict(acc=float(acc[best_i]), prec=float(prec[best_i]), rec=float(rec[best_i]),
                    f1=float(f1[best_i]), bacc=float(bacc[best_i]), fpr=float(fpr[best_i]),
                    thr=float(thrs[best_i])), True, False

    # 2) SOFT constraints
    soft_idx = np.where((f1>=SOFT_MIN_F1) & (prec>=SOFT_MIN_PREC) & (rec>=SOFT_MIN_REC))[0]
    if soft_idx.size:
        key = np.stack([acc[soft_idx], bacc[soft_idx], prec[soft_idx], -np.abs(thrs[soft_idx]-0.5)], axis=1)
        best_i = soft_idx[np.lexsort((key[:,3], key[:,2], key[:,1], key[:,0]))][-1]
        return dict(acc=float(acc[best_i]), prec=float(prec[best_i]), rec=float(rec[best_i]),
                    f1=float(f1[best_i]), bacc=float(bacc[best_i]), fpr=float(fpr[best_i]),
                    thr=float(thrs[best_i])), False, True

    # 3) Fallback: maximize (Acc - alpha * FPR)
    score = acc - ALPHA_FPR_PENALTY * fpr
    # تساوی → اول BAcc، بعد Prec
    key = np.stack([score, bacc, prec], axis=1)
    best_i = np.lexsort((key[:,2], key[:,1], key[:,0]))[-1]
    return dict(acc=float(acc[best_i]), prec=float(prec[best_i]), rec=float(rec[best_i]),
                f1=float(f1[best_i]), bacc=float(bacc[best_i]), fpr=float(fpr[best_i]),
                thr=float(thrs[best_i])), False, False

def _gen_weight_grid_fine(M: int, step: float = 0.1):
    vals = np.round(np.arange(0.0, 1.0 + step/2, step), 3)  # 0.0..1.0
    seen, weights = set(), []
    for w in itertools.product(vals, repeat=M):
        if all(v == 0.0 for v in w):  # همه صفر؟
            continue
        w = np.array(w, dtype=np.float64)
        s = w.sum()
        if s <= 0:
            continue
        w = w / s
        key = tuple(np.round(w, 3).tolist())
        if key in seen:
            continue
        seen.add(key); weights.append(w)
    return weights

def main():
    # 1) لود احتمال‌ها
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

    # 2) logit-averaging
    def _p2logit(p): return np.log(p/(1.0 - p))
    folds_logits = {k: _p2logit(folds_data[k][1]) for k in folds_data}  # (M, Nfold)

    # 3) شبکه وزن
    M = len(MODEL_RESULTS)
    weight_grids = _gen_weight_grid_fine(M, step=0.1)

    rows = []
    out_root = os.path.join(RESULTS_DIR, "23_ensemble_accfirst_fprpen")
    os.makedirs(out_root, exist_ok=True)

    for k in range(1, 6):
        y_true, _ = folds_data[k]
        L = folds_logits[k]  # (M, N)
        best_tuple = None  # (acc, bacc, f1, rec, prec, metrics, hard, soft, w_best, y_prob_ens)

        for w in weight_grids:
            l_ens = np.tensordot(w, L, axes=(0, 0))      # (N,)
            y_prob_ens = 1.0 / (1.0 + np.exp(-l_ens))    # sigmoid

            mb, hard_ok, soft_ok = choose_thr_accfirst_vec(y_true, y_prob_ens, thrs=THRS)

            # کلید انتخاب وزن: Acc → BAcc → F1 → Rec → Prec
            key = (mb["acc"], mb["bacc"], mb["f1"], mb["rec"], mb["prec"])
            if (best_tuple is None) or (key > (best_tuple[0], best_tuple[1], best_tuple[2], best_tuple[3], best_tuple[4])):
                best_tuple = (mb["acc"], mb["bacc"], mb["f1"], mb["rec"], mb["prec"],
                              mb, hard_ok, soft_ok, w.copy(), y_prob_ens.copy())

        _, _, _, _, _, mb, hard_ok, soft_ok, w_best, y_prob_ens = best_tuple

        out_fold = os.path.join(out_root, f"fold{k}")
        os.makedirs(out_fold, exist_ok=True)
        pd.DataFrame({"y_true": y_true, "y_prob_ens": y_prob_ens}).to_csv(
            os.path.join(out_fold, "predictions_ens.csv"), index=False
        )
        with open(os.path.join(out_fold, "metrics_ens.json"), "w") as f:
            json.dump({
                "acc": mb["acc"], "prec": mb["prec"], "rec": mb["rec"], "f1": mb["f1"],
                "bacc": mb["bacc"], "fpr": mb["fpr"], "thr": mb["thr"],
                "hard_constraints_satisfied": bool(hard_ok),
                "soft_constraints_used": bool(soft_ok),
                "weights": np.round(w_best, 3).tolist(),
                "models": MODEL_RESULTS,
                "alpha_fpr_penalty": ALPHA_FPR_PENALTY
            }, f, indent=2)

        rows.append({"fold": k, "acc": mb["acc"], "prec": mb["prec"], "rec": mb["rec"], "f1": mb["f1"], "bacc": mb["bacc"], "fpr": mb["fpr"]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "_summary_folds.csv"), index=False)
    avg = df[["acc","prec","rec","f1","bacc","fpr"]].mean().to_dict()
    std = df[["acc","prec","rec","f1","bacc","fpr"]].std().to_dict()

    with open(os.path.join(out_root, "_summary_avg_std.json"), "w") as f:
        json.dump({
            "avg": {k: float(v) for k,v in avg.items()},
            "std": {k: float(v) for k,v in std.items()},
            "note": "Logit-averaging ensemble; fallback=max (Acc - alpha*FPR); weight key=(Acc,BAcc,F1,Rec,Prec)",
            "models": MODEL_RESULTS,
            "thresholds": {"count": int(len(THRS)), "min": float(THRS[0]), "max": float(THRS[-1])},
            "alpha_fpr_penalty": ALPHA_FPR_PENALTY
        }, f, indent=2)

    print("[DONE] results →", out_root)
    print(df)
    print("AVG:", {k: round(float(v),4) for k,v in avg.items()})

if __name__ == "__main__":
    main()
