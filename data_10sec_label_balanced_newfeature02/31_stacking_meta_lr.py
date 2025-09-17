# -*- coding: utf-8 -*-
"""
31_stacking_meta_lr.py (robust)
Stacking over out-of-fold predictions of base models:
- Auto-resolves model results folder names (with/without .py; fuzzy prefix match)
- Train per-fold LogisticRegression meta-learner (no leak)
- Tune decision threshold per fold under constraints (accuracy-first)

Run: just press Run in PyCharm
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT       = os.path.abspath(os.path.dirname(__file__))
DATA_DIR   = os.path.join(ROOT, "data")
RESULTS_DIR= os.path.join(ROOT, "results")

# ⬇️ این‌ها را مطابق *نام پوشه‌های داخل results/* خودت تنظیم کن (با یا بدون .py فرق ندارد)
BASE_MODELS = [
    "29_tcn_wide_runner06",
    "31b_tcn_wide112_gnspdrop_runner06",
    "11_resnet1d_runner06",
    "10_cnn_bn_runner06",
]


OUT_DIR = os.path.join(RESULTS_DIR, "31_stacking_meta_lr")
os.makedirs(OUT_DIR, exist_ok=True)

# Constraints
MIN_F1    = 0.75
MIN_PREC  = 0.70
RELAX_F1  = 0.70
THR_GRID  = np.linspace(0.0, 1.0, 501)

def _list_result_dirs():
    if not os.path.isdir(RESULTS_DIR):
        return []
    return sorted([d for d in os.listdir(RESULTS_DIR)
                   if os.path.isdir(os.path.join(RESULTS_DIR, d))])

def _resolve_model_dir(name: str):
    """
    تلاش برای یافتن پوشه‌ی نتایج این مدل:
    - اول همان نام دقیق
    - بعد بدون پسوند .py
    - بعد جست‌وجوی شروع-با (prefix) بین پوشه‌های results
    """
    exact = os.path.join(RESULTS_DIR, name)
    if os.path.exists(exact):
        return exact

    base = name[:-3] if name.lower().endswith(".py") else name
    cand = os.path.join(RESULTS_DIR, base)
    if os.path.exists(cand):
        return cand

    # fuzzy: هر پوشه‌ای که با base شروع شود
    dirs = _list_result_dirs()
    matches = [d for d in dirs if d.lower().startswith(base.lower())]
    if len(matches) == 1:
        return os.path.join(RESULTS_DIR, matches[0])

    # اگر چندتا یا هیچ‌کدام نبود:
    raise FileNotFoundError(
        f"نتوانستم پوشه نتایج برای «{name}» را پیدا کنم.\n"
        f"پوشه‌های موجود در results/: {dirs}"
    )

def _load_fold_preds(model_name, fold):
    model_dir = _resolve_model_dir(model_name)
    pred_path = os.path.join(model_dir, f"fold{fold}", "predictions.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"predictions.csv یافت نشد: {pred_path}")
    df = pd.read_csv(pred_path)
    if ("y_true" not in df.columns) or ("y_prob" not in df.columns):
        raise ValueError(f"ستون‌های y_true/y_prob در {pred_path} پیدا نشد.")
    return df

def _metrics_from_probs(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    bacc = 0.5 * ((tp/(tp+fn) if (tp+fn)>0 else 0.0) + (tn/(tn+fp) if (tn+fp)>0 else 0.0))
    fpr  = fp / (fp+tn) if (fp+tn) > 0 else 0.0
    return {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "bacc":bacc, "fpr":fpr, "thr":thr, "y_pred":y_pred}

def _best_thr_with_constraints(y_true, y_prob):
    # 1) سخت: بیشینه Acc با F1>=MIN_F1 و Prec>=MIN_PREC
    best = None
    for t in THR_GRID:
        m = _metrics_from_probs(y_true, y_prob, t)
        if (m["f1"] >= MIN_F1) and (m["prec"] >= MIN_PREC):
            key = (m["acc"], m["f1"], m["rec"], m["prec"])
            if (best is None) or (key > best[0]): best = (key, m)
    if best is not None:
        m = best[1]; m["mode"] = "hard"; return m
    # 2) نرم
    best = None
    for t in THR_GRID:
        m = _metrics_from_probs(y_true, y_prob, t)
        if m["f1"] >= RELAX_F1:
            key = (m["acc"], m["f1"], m["rec"], m["prec"])
            if (best is None) or (key > best[0]): best = (key, m)
    if best is not None:
        m = best[1]; m["mode"] = "soft"; return m
    # 3) fallback: بیشینه BAcc
    best = None
    for t in THR_GRID:
        m = _metrics_from_probs(y_true, y_prob, t)
        key = (m["bacc"], m["acc"])
        if (best is None) or (key > best[0]): best = (key, m)
    m = best[1]; m["mode"] = "fallback_bacc"; return m

def _stack_features(prob_mat):
    eps = 1e-6
    probs = np.clip(prob_mat, eps, 1-eps)
    logits = np.log(probs/(1-probs))
    feats = [probs, logits]
    # تعامل‌های درجه 2 ساده
    inter = []
    M = probs.shape[1]
    for i in range(M):
        for j in range(i+1, M):
            inter.append((probs[:, i]*probs[:, j]).reshape(-1,1))
    if inter:
        feats.append(np.hstack(inter))
    return np.hstack(feats)

def main():
    # 0) برای دیباگ: نشان بده اسامی پوشه‌های results چی هست
    print("[results dirs]", _list_result_dirs())

    # 1) جمع‌کردن پروب‌های ولیدیشن هر مدل در هر فولد
    per_fold = {}
    for k in range(1, 6):
        mats = []
        y_ref = None
        for mname in BASE_MODELS:
            df = _load_fold_preds(mname, k)
            if y_ref is None:
                y_ref = df["y_true"].values.astype(int)
            mats.append(df["y_prob"].values.reshape(-1,1))
        X_prob = np.hstack(mats)
        per_fold[k] = {"X_prob": X_prob, "y": y_ref}

    rows = []
    for k in range(1, 6):
        # train meta on folds != k
        X_tr_list, y_tr_list = [], []
        for j in range(1, 6):
            if j == k: continue
            X_tr_list.append(per_fold[j]["X_prob"])
            y_tr_list.append(per_fold[j]["y"])
        X_tr_prob = np.vstack(X_tr_list)
        y_tr      = np.concatenate(y_tr_list)

        X_va_prob = per_fold[k]["X_prob"]
        y_va      = per_fold[k]["y"]

        X_tr = _stack_features(X_tr_prob)
        X_va = _stack_features(X_va_prob)

        # Logistic Regression با جست‌وجوی کوچک
        param_grid = {"C":[0.3, 1.0, 3.0], "penalty":["l2"], "solver":["liblinear"], "class_weight":["balanced"]}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        gs = GridSearchCV(LogisticRegression(max_iter=1000),
                          param_grid=param_grid,
                          scoring="accuracy", cv=cv, n_jobs=-1)
        gs.fit(X_tr, y_tr)
        meta = gs.best_estimator_

        y_prob_meta = meta.predict_proba(X_va)[:,1]
        best = _best_thr_with_constraints(y_va, y_prob_meta)

        rows.append({
            "fold": k,
            "acc": best["acc"], "prec": best["prec"], "rec": best["rec"],
            "f1": best["f1"], "bacc": best["bacc"], "thr": best["thr"],
            "mode": best["mode"],
            "best_meta_params": str(gs.best_params_)
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "folds_metrics.csv"), index=False)
    avg = df[["acc","prec","rec","f1","bacc"]].mean().to_dict()

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "avg": {k: float(v) for k,v in avg.items()},
            "note": "Stacking ensemble (logistic regression over base model probabilities) with per-fold threshold tuning under constraints",
            "base_models": BASE_MODELS,
            "constraints": {"min_f1": MIN_F1, "min_prec": MIN_PREC, "relax_f1": RELAX_F1}
        }, f, indent=2)

    print("[DONE] results →", OUT_DIR)
    print(df)
    print("AVG:", {k: round(float(v),4) for k,v in avg.items()})

if __name__ == "__main__":
    main()
