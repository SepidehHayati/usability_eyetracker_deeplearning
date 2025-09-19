# 33_stacking_meta_lr_accfirst.py
# OOF Stacking (LogisticRegression on logits) + Accuracy-first thresholding with constraints
# Reads: results/<model>/foldK/predictions.csv  (columns: y_true, y_prob)
# Writes: results/33_stacking_meta_lr_accfirst/*

import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")

# همان 5 مدل برتر (اسم پوشه‌ها = اسم فایل مدل)
MODELS = [
    "29_tcn_wide_runner06.py",
    "31b_tcn_wide112_gnspdrop_runner06.py",
    "11_resnet1d_runner06.py",
    "10_cnn_bn_runner06.py",
    "18_resnet1d_accfirst_runner08.py",
]

OUT_DIR = os.path.join(RESULTS_DIR, "33_stacking_meta_lr_accfirst")
os.makedirs(OUT_DIR, exist_ok=True)

# قیود و هدف
HARD_MIN_F1, HARD_MIN_PREC = 0.75, 0.80   # می‌توانی طبق نیاز سخت‌تر/نرم‌تر کنی
SOFT_MIN_F1, SOFT_MIN_PREC = 0.72, 0.78
THRS = np.linspace(0.0, 1.0, 1001)
OBJECTIVE = ("acc", "f1", "bacc", "rec", "prec")

def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def _safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def _balanced_accuracy(cm):
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return 0.5 * (tpr + tnr)

def _metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec":  recall_score(y_true, y_pred, zero_division=0),
        "f1":   f1_score(y_true, y_pred, zero_division=0),
        "bacc": _balanced_accuracy(cm),
        "thr":  float(thr),
        "cm":   cm,
        "y_pred": y_pred,
    }

def _choose_thr_accfirst(y_true, y_prob):
    best_hard, best_soft, best_free = None, None, None
    key = lambda m: (m["acc"], m["f1"], m["bacc"], m["rec"], m["prec"])
    for t in THRS:
        m = _metrics(y_true, y_prob, t)
        if (m["f1"]>=HARD_MIN_F1) and (m["prec"]>=HARD_MIN_PREC):
            if (best_hard is None) or (key(m) > key(best_hard)): best_hard = m
        if (m["f1"]>=SOFT_MIN_F1) and (m["prec"]>=SOFT_MIN_PREC):
            if (best_soft is None) or (key(m) > key(best_soft)): best_soft = m
        if (best_free is None) or (key(m) > key(best_free)): best_free = m
    return (best_hard or best_soft or best_free)

def _load_fold_matrix(fold):
    """X: [n_samples, n_models] (logits), y: [n_samples]"""
    mats, yt = [], None
    for m in MODELS:
        pth = os.path.join(RESULTS_DIR, m, f"fold{fold}", "predictions.csv")
        df = pd.read_csv(pth)
        if yt is None:
            yt = df["y_true"].values.astype(int)
        else:
            assert np.array_equal(yt, df["y_true"].values.astype(int)), f"y_true mismatch in {m}/fold{fold}"
        mats.append(_safe_logit(df["y_prob"].values.astype(float)))
    X = np.vstack(mats).T  # [N, M]
    return yt, X

def main():
    # جمع‌آوری OOF برای استکینگ: برای هر فولد، متا روی سایر فولدها train و روی فولد فعلی test می‌شود.
    all_rows, all_thrs = [], []
    for k in range(1,6):
        # data for this fold
        y_va, X_va = _load_fold_matrix(k)
        # OOF train = تمام فولدهای دیگر
        X_tr_list, y_tr_list = [], []
        for j in range(1,6):
            if j==k: continue
            y_j, X_j = _load_fold_matrix(j)
            X_tr_list.append(X_j); y_tr_list.append(y_j)
        X_tr = np.vstack(X_tr_list)
        y_tr = np.concatenate(y_tr_list)

        # متا-مدل: LR با L2 (ridge)، کلاس‌ویت balanced تا FP کنترل شود
        meta = LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs"
        )
        meta.fit(X_tr, y_tr)

        # پیش‌بینی روی fold k
        prob_va = _sigmoid(meta.decision_function(X_va))
        m = _choose_thr_accfirst(y_va, prob_va)

        all_thrs.append(m["thr"])
        all_rows.append({
            "fold": k, "acc": m["acc"], "prec": m["prec"], "rec": m["rec"],
            "f1": m["f1"], "bacc": m["bacc"], "thr": m["thr"]
        })

        # ذخیره جزئیات فولد
        outk = os.path.join(OUT_DIR, f"fold{k}")
        os.makedirs(outk, exist_ok=True)
        pd.DataFrame({"y_true": y_va, "y_prob_meta": prob_va, "y_pred_best": m["y_pred"]})\
          .to_csv(os.path.join(outk, "pred_meta.csv"), index=False)
        with open(os.path.join(outk, "metrics_best.json"), "w") as f:
            json.dump({k:(float(v) if not isinstance(v, (list,np.ndarray)) else None) for k,v in m.items()
                       if k not in ["cm","y_pred"]}, f, indent=2)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT_DIR, "summary_folds.csv"), index=False)
    avg = df[["acc","prec","rec","f1","bacc"]].mean().to_dict()
    std = df[["acc","prec","rec","f1","bacc"]].std().to_dict()

    # آستانه‌ی توصیه‌شده برای استقرار: مدین آستانه‌ها (پایدارتر از per-fold)
    thr_deploy = float(np.median(np.array(all_thrs)))

    with open(os.path.join(OUT_DIR, "summary_avg_std.json"), "w") as f:
        json.dump({"avg": {k: float(v) for k,v in avg.items()},
                   "std": {k: float(v) for k,v in std.items()},
                   "deploy_threshold_median": thr_deploy,
                   "note": "OOF stacking with LR on logits + accuracy-first thresholding"},
                  f, indent=2)

    print("FOLDS:\n", df)
    print("AVG:", {k: round(v,4) for k,v in avg.items()}, "| deploy_thr(median) =", round(thr_deploy,3))

if __name__ == "__main__":
    main()
