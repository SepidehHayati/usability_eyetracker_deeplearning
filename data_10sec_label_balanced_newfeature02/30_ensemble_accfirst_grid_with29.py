# 30_fast_ensemble_accfirst_with29_31b.py
# Fast grid ensemble with accuracy-first selection + constraints
# - Adds 31b model alongside 29/11/10/18
# - Prunes search space (few thresholds + small weight grid + max 2 active models)
# - Reads: results/<model>/foldK/predictions.csv
# - Saves: results/<OUTPUT_DIR_NAME>/*

import os, json, itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ----------------- CONFIG -----------------
ROOT = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")

# مدل‌ها (نام پوشه نتایج دقیقاً برابر با نام فایل مدل)
MODELS = [
    "29_tcn_wide_runner06.py",                 # بهترین تک‌مدل فعلی
    "31b_tcn_wide112_gnspdrop_runner06.py",    # مدل 31b (حدود 77%)
    "11_resnet1d_runner06.py",
    "10_cnn_bn_runner06.py",
    "18_resnet1d_accfirst_runner08.py",
]

# دامنه وزن‌ها (صفر یعنی حذف مدل). رِنج‌ها را کوچک نگه می‌داریم برای سرعت.
WEIGHT_RANGE_PER_MODEL = {
    "29_tcn_wide_runner06.py":               [0, 1, 2, 3],   # می‌توانی بعداً تا 5 ببری اگر سریع بود
    "31b_tcn_wide112_gnspdrop_runner06.py":  [0, 1, 2],
    "11_resnet1d_runner06.py":               [0, 1, 2],
    "10_cnn_bn_runner06.py":                 [0, 1, 2],
    "18_resnet1d_accfirst_runner08.py":      [0, 1, 2],
}

# محدودیت تعداد مدل‌های فعال در هر ترکیب (برای سرعت)
MAX_ACTIVE_MODELS = 2

# آستانه‌ها (کمش کردیم تا سریع شود)
THRESHOLDS = np.linspace(0.0, 1.0, 201)   # قبلاً 601 بود

# محدودیت‌های سخت (می‌توانی تغییر دهی)
MIN_F1   = 0.75
MIN_PREC = None
MIN_REC  = None

# ترتیب هدف (برای شکستن تساوی‌ها). Accuracy اولویت اول.
OBJECTIVE_KEY = ("acc", "f1", "bacc", "rec", "prec")

# اگر هیچ ترکیبی محدودیت سخت را پاس نکند، fallback بر اساس کدام معیار:
FALLBACK_MODE = "bacc"   # "bacc" یا "f1"

# نوع ترکیب: "logit" یا "prob" (لاجیت معمولاً بهتر)
COMBINE_MODE = "logit"

# خروجی
OUTPUT_DIR_NAME = "30_fast_ensemble_accfirst_with29_31b"

# (اختیاری) در صورت نیاز می‌توانی coarse prune را فعال کنی:
USE_COARSE_PRUNE = False         # True -> سریع‌تر، ولی ریسک حذف برخی ترکیب‌های خوب
COARSE_THR = 0.5
COARSE_MARGIN = 0.00             # اگر acc در thr=0.5 < best_single_acc - margin → skip
# ------------------------------------------


def _sigmoid(x): return 1 / (1 + np.exp(-x))

def _safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def _balanced_accuracy(cm):
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (tpr + tnr)

def _metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0,1])
    bacc = _balanced_accuracy(cm)
    return {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "bacc":bacc, "cm":cm, "thr":thr}

def _satisfy_hard(m):
    if (MIN_F1   is not None) and (m["f1"]   < MIN_F1):   return False
    if (MIN_PREC is not None) and (m["prec"] < MIN_PREC): return False
    if (MIN_REC  is not None) and (m["rec"]  < MIN_REC):  return False
    return True

def _score_tuple(m):
    return tuple(m[k] for k in OBJECTIVE_KEY)

def _combine_probs(prob_list, weights, mode="logit"):
    w = np.array(weights, dtype=float)
    if np.all(w == 0): return None
    if mode == "prob":
        return np.average(np.vstack(prob_list), axis=0, weights=w)
    else:
        logits = [ _safe_logit(p) for p in prob_list ]
        lw = np.average(np.vstack(logits), axis=0, weights=w)
        return _sigmoid(lw)

def _load_fold_preds(model_file, fold):
    pred_csv = os.path.join(RESULTS_DIR, model_file, f"fold{fold}", "predictions.csv")
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"Missing predictions: {pred_csv}")
    df = pd.read_csv(pred_csv)
    if "y_prob" not in df.columns or "y_true" not in df.columns:
        raise ValueError(f"predictions.csv must contain y_true,y_prob columns → {pred_csv}")
    return df["y_true"].values.astype(int), df["y_prob"].values.astype(float)

def _best_single_baseline(y_true, probs, thresholds):
    # بهترین تک مدل بین MODELS با constraints: برای یونیت تست‌های coarse prune
    best = None; best_tuple = None
    for i, p in enumerate(probs):
        for thr in thresholds:
            m = _metrics(y_true, p, thr)
            if _satisfy_hard(m):
                t = _score_tuple(m)
            else:
                # fallback
                key = "bacc" if FALLBACK_MODE == "bacc" else "f1"
                t = (m[key], m["acc"], m["f1"], m["rec"], m["prec"])
            if (best is None) or (t > best_tuple):
                best, best_tuple = m, t
    return best  # dict

def _grid_for_fold(fold, out_dir):
    # Load all model predictions for this fold
    y_true, p0 = _load_fold_preds(MODELS[0], fold)
    probs_list = [p0]
    for m in MODELS[1:]:
        yt, pi = _load_fold_preds(m, fold)
        assert np.array_equal(y_true, yt), f"y_true mismatch in fold{fold} for {m}"
        probs_list.append(pi)

    # Baseline (برای prune اختیاری)
    baseline = _best_single_baseline(y_true, probs_list, THRESHOLDS)

    # Build weight grids per model
    weight_ranges = [WEIGHT_RANGE_PER_MODEL[m] for m in MODELS]
    candidates = []

    for weights in itertools.product(*weight_ranges):
        # محدودیت تعداد مدل‌های فعال
        if np.count_nonzero(weights) == 0 or np.count_nonzero(weights) > MAX_ACTIVE_MODELS:
            continue

        ens_prob = _combine_probs(probs_list, weights, mode=COMBINE_MODE)
        if ens_prob is None: continue

        # (اختیاری) prune سریع بر اساس thr=0.5
        if USE_COARSE_PRUNE:
            m_coarse = _metrics(y_true, ens_prob, COARSE_THR)
            if m_coarse["acc"] < (baseline["acc"] - COARSE_MARGIN):
                continue

        # Sweep thresholds
        hard_best = None; hard_best_tuple = None
        fallback_best = None; fallback_best_tuple = None

        for thr in THRESHOLDS:
            m = _metrics(y_true, ens_prob, thr)
            if _satisfy_hard(m):
                t = _score_tuple(m)
                if (hard_best is None) or (t > hard_best_tuple):
                    hard_best, hard_best_tuple = m, t
            else:
                key = "bacc" if FALLBACK_MODE == "bacc" else "f1"
                t = (m[key], m["acc"], m["f1"], m["rec"], m["prec"])
                if (fallback_best is None) or (t > fallback_best_tuple):
                    fallback_best, fallback_best_tuple = m, t

        chosen = hard_best if hard_best is not None else fallback_best
        chosen_mode = "hard" if hard_best is not None else f"fallback_{FALLBACK_MODE}"

        candidates.append({
            "weights": list(weights),
            "metrics": chosen,
            "mode": chosen_mode
        })

    # Pick the best candidate by OBJECTIVE_KEY
    best = None; best_tuple = None
    for c in candidates:
        t = _score_tuple(c["metrics"])
        if (best is None) or (t > best_tuple):
            best, best_tuple = c, t

    # Save fold detail
    fold_dir = os.path.join(out_dir, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    rec = {
        "fold": fold,
        "weights": best["weights"],
        "mode": best["mode"],
        "thr": float(best["metrics"]["thr"]),
        "acc": float(best["metrics"]["acc"]),
        "prec": float(best["metrics"]["prec"]),
        "rec": float(best["metrics"]["rec"]),
        "f1": float(best["metrics"]["f1"]),
        "bacc": float(best["metrics"]["bacc"]),
        "models": MODELS,
        "combine_mode": COMBINE_MODE,
        "objective": list(OBJECTIVE_KEY),
        "hard_constraints": {"min_f1": MIN_F1, "min_prec": MIN_PREC, "min_rec": MIN_REC}
    }
    with open(os.path.join(fold_dir, "chosen.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    pd.DataFrame([{
        "fold": fold,
        "acc": rec["acc"], "prec": rec["prec"], "rec": rec["rec"],
        "f1": rec["f1"], "bacc": rec["bacc"],
        "thr": rec["thr"], "mode": rec["mode"],
        "weights": str(rec["weights"])
    }]).to_csv(os.path.join(fold_dir, "summary.csv"), index=False)

    return rec

def main():
    out_dir = os.path.join(RESULTS_DIR, OUTPUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for k in range(1, 6):
        rec = _grid_for_fold(k, out_dir)
        all_rows.append({
            "fold": rec["fold"],
            "acc": rec["acc"], "prec": rec["prec"],
            "rec": rec["rec"], "f1": rec["f1"], "bacc": rec["bacc"],
            "thr": rec["thr"], "mode": rec["mode"],
            "weights": str(rec["weights"])
        })

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(out_dir, "summary_folds.csv"), index=False)

    avg = df[["acc","prec","rec","f1","bacc"]].mean().to_dict()
    print(f"[DONE] results → {out_dir}")
    print(df)
    print("AVG:", {k: round(v, 4) for k, v in avg.items()})

    with open(os.path.join(out_dir, "summary_avg.json"), "w", encoding="utf-8") as f:
        json.dump({
            "avg": avg,
            "note": "Fast grid ensemble (accuracy-first) with 29 & 31b; max 2 active models",
            "models": MODELS,
            "combine_mode": COMBINE_MODE,
            "weight_ranges": WEIGHT_RANGE_PER_MODEL,
            "thresholds": {"count": len(THRESHOLDS),
                           "min": float(THRESHOLDS.min()),
                           "max": float(THRESHOLDS.max())},
            "objective": list(OBJECTIVE_KEY),
            "hard_constraints": {"min_f1": MIN_F1, "min_prec": MIN_PREC, "min_rec": MIN_REC},
            "fallback": FALLBACK_MODE,
            "prune": {"max_active_models": MAX_ACTIVE_MODELS,
                      "use_coarse": USE_COARSE_PRUNE,
                      "coarse_thr": COARSE_THR,
                      "coarse_margin": COARSE_MARGIN}
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
