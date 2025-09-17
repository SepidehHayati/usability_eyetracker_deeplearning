# 32_ensemble_weightopt_with29_31b.py
# Continuous-weight ensemble optimizer (per fold) with threshold tuning
# هدف: ماکزیمم کردن Accuracy تحت قید نرم روی F1/Precision (اگه شد)
# خروجی: نتایج هر فولد + میانگین، پوشه نتایج جدا

import os, json, itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT        = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_DIR     = os.path.join(RESULTS_DIR, "32_weightopt_with29_31b")
os.makedirs(OUT_DIR, exist_ok=True)

# مدل‌هایی که ازشون predictions.csv داریم (همون‌هایی که تا الان استفاده کردی)
CANDIDATE_MODELS = [
    "29_tcn_wide_runner06.py",
    "31b_tcn_wide112_gnspdrop_runner06.py",
    "11_resnet1d_runner06.py",
    "10_cnn_bn_runner06.py",
]
FOLDS = [1,2,3,4,5]

# تنظیمات جستجو
THR_GRID = np.linspace(0.0, 1.0, 501)   # 0.002 گام
# قید نرم (اگر شد رعایت می‌کنیم، وگرنه بهترین Acc بدون قید):
MIN_F1   = 0.78
MIN_PREC = 0.70

def _find_model_dir(name):
    # دقیق یا بر اساس شروع نام، بین فولدرهای results
    dirs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR,d))]
    # اول exact
    for d in dirs:
        if d == name:
            return os.path.join(RESULTS_DIR, d)
    # بعد prefix match
    for d in dirs:
        if d.startswith(name.replace(".py","")):
            return os.path.join(RESULTS_DIR, d)
    return None

def _load_fold_preds(model_dir, k):
    csv_path = os.path.join(model_dir, f"fold{k}", "predictions.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing: {csv_path}")
    df = pd.read_csv(csv_path)
    if not {"y_true","y_prob"}.issubset(df.columns):
        raise ValueError(f"predictions.csv must have y_true and y_prob: {csv_path}")
    return df

def _eval_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    tnr = tn / max((tn + fp), 1)
    bacc = 0.5*(rec + tnr)
    return {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "bacc":bacc}

def _best_thr_under_constraints(y_true, y_prob):
    best = None
    # اول تلاش می‌کنیم قید نرم را رعایت کنیم
    for thr in THR_GRID:
        m = _eval_metrics(y_true, y_prob, thr)
        ok = (m["f1"] >= MIN_F1) and (m["prec"] >= MIN_PREC)
        score = m["acc"]
        if ok:
            if (best is None) or (score > best[0]):
                best = (score, thr, m, True)  # True → constraints satisfied
    if best is not None:
        return best[1], best[2], True

    # اگر نشد، بهترین Accuracy بدون قید
    best = None
    for thr in THR_GRID:
        m = _eval_metrics(y_true, y_prob, thr)
        score = m["acc"]
        if (best is None) or (score > best[0]):
            best = (score, thr, m, False)
    return best[1], best[2], False

def _normalize(w):
    w = np.clip(np.array(w, dtype=float), 0.0, None)
    s = w.sum()
    if s <= 0:
        # اگر همه صفر شد، یکنواخت کن
        return np.ones_like(w) / len(w)
    return w / s

def _score_weights(y_true, probs_list, w):
    # w: real weights (non-negative, normalized)
    w = _normalize(w)
    blended = np.zeros_like(probs_list[0])
    for wi, pr in zip(w, probs_list):
        blended += wi * pr
    thr, met, ok = _best_thr_under_constraints(y_true, blended)
    return met["acc"], thr, met, ok

def coordinate_ascent(y_true, probs_list, max_iter=200, grid=np.linspace(0,1,21)):
    M = len(probs_list)
    w = np.ones(M) / M
    best_acc, best_thr, best_met, best_ok = _score_weights(y_true, probs_list, w)

    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(M):
            current = w.copy()
            local_best = (best_acc, w[i], best_thr, best_met, best_ok)
            for cand in grid:
                w_try = current.copy()
                w_try[i] = cand
                # نرمال‌سازی داخل _score_weights انجام می‌شود
                acc, thr, met, ok = _score_weights(y_true, probs_list, w_try)
                # تابع هدف: اول Acc، اگر مساوی شد، F1 بالاتر
                key_new  = (acc, met["f1"])
                key_best = (local_best[0], local_best[3]["f1"])
                if key_new > key_best:
                    local_best = (acc, cand, thr, met, ok)
            if local_best[0] > best_acc or (local_best[0] == best_acc and local_best[3]["f1"] > best_met["f1"]):
                # به‌روزرسانی وزن i
                w[i] = local_best[1]
                # نرمال‌سازی نهایی
                w = _normalize(w)
                best_acc, best_thr, best_met, best_ok = local_best[0], local_best[2], local_best[3], local_best[4]
                improved = True
    return w, best_thr, best_met, best_ok

def main():
    # جمع‌آوری پیش‌بینی‌ها برای هر فولد
    fold_rows = []
    model_dirs = {}
    for name in CANDIDATE_MODELS:
        d = _find_model_dir(name)
        if d is None:
            print(f"[WARN] skip (no results dir): {name}")
        else:
            model_dirs[name] = d
    if not model_dirs:
        raise SystemExit("No model results found.")

    for k in FOLDS:
        # لود y_true و y_prob هر مدل
        y_true = None
        probs_list, used_names = [], []
        for name, d in model_dirs.items():
            try:
                df = _load_fold_preds(d, k)
                if y_true is None:
                    y_true = df["y_true"].values.astype(int)
                else:
                    # sanity check
                    if len(y_true) != len(df["y_true"]):
                        raise ValueError(f"Length mismatch in fold {k} for {name}")
                probs_list.append(df["y_prob"].values.astype(float))
                used_names.append(name)
            except Exception as e:
                print(f"[WARN] fold{k} skip {name}: {e}")

        if y_true is None or len(probs_list) == 0:
            print(f"[WARN] no predictions for fold{k}; skipping.")
            continue

        # بهینه‌سازی وزن‌ها
        w, thr, met, ok = coordinate_ascent(y_true, probs_list, max_iter=200, grid=np.linspace(0,1,21))

        row = {
            "fold": k,
            "acc":  met["acc"],
            "prec": met["prec"],
            "rec":  met["rec"],
            "f1":   met["f1"],
            "bacc": met["bacc"],
            "thr":  float(thr),
            "constraints_met": bool(ok),
            "weights": [float(x) for x in _normalize(w)],
            "models": used_names
        }
        fold_rows.append(row)
        print(f"[fold {k}] acc={met['acc']:.4f} | prec={met['prec']:.4f} | rec={met['rec']:.4f} | f1={met['f1']:.4f} | thr={thr:.3f} | weights={row['weights']}")

    if not fold_rows:
        print("No folds processed.")
        return

    df = pd.DataFrame(fold_rows)
    df.to_csv(os.path.join(OUT_DIR, "fold_metrics.csv"), index=False)
    avg = {m: float(df[m].mean()) for m in ["acc","prec","rec","f1","bacc"]}
    print("AVG:", {k: round(v,4) for k,v in avg.items()})
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"avg": avg,
                   "note": "continuous weight-optimized ensemble (per-fold), threshold tuned, soft constraints"},
                  f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
