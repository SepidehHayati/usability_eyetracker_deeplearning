# 24_ensemble_pickbest_perfold.py
# ایده: برای هر فولد، بین چند مدل موجود همان مدلی را انتخاب کن که
# با "تنظیم آستانه تحت قیود" بیشترین Accuracy را می‌دهد.
# قیود: HARD/SOFT مثل قبل؛ اگر هیچ آستانه‌ای پاس نشد، fallback = بیشینه Balanced Accuracy.
# (می‌تونی fallback را به Acc - alpha*FPR هم تغییر بده اگر خواستی.)

import os, json
import numpy as np
import pandas as pd

ROOT        = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")

# مدل‌هایی که می‌خواهیم میانشان انتخاب کنیم (همان سه تای قبلی)
MODEL_RESULTS = [
    "11_resnet1d_runner06.py",
    "10_cnn_bn_runner06.py",
    "18_resnet1d_accfirst_runner08.py",
]

THRS = np.linspace(0.0, 1.0, 501)

# قیود
HARD_MIN_F1,   HARD_MIN_PREC,   HARD_MIN_REC = 0.78, 0.82, 0.75
SOFT_MIN_F1,   SOFT_MIN_PREC,   SOFT_MIN_REC = 0.75, 0.78, 0.65

EPS = 1e-6

def _find_prob_column(df: pd.DataFrame) -> str:
    for c in ["y_prob", "y_prob_ens"]:
        if c in df.columns: return c
    for c in df.columns:
        if str(c).startswith("y_prob"): return c
    raise ValueError(f"Could not find probability column in {df.columns.tolist()}")

def load_fold_probs(model_dirname: str, fold: int):
    # predictions از هر مدل: results/<model>/foldk/predictions.csv  یا predictions_ens.csv
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

def _metrics_from_counts(TP, FP, FN, TN):
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0    # TPR
    tnr  = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    acc  = (TP + TN) / (TP + FP + FN + TN)
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    bacc = 0.5 * (rec + tnr)
    return acc, prec, rec, f1, bacc

def choose_thr_accfirst_vec(y_true: np.ndarray, y_prob: np.ndarray, thrs: np.ndarray = THRS):
    y_true = y_true.astype(np.int64)
    pos = (y_true == 1); neg = ~pos
    P = int(pos.sum()); N = int(y_true.size)

    best_hard = None
    best_soft = None
    best_fallback = None

    for t in thrs:
        y_pred = (y_prob >= t).astype(np.int64)
        TP = int(((y_pred == 1) & pos).sum())
        FP = int(((y_pred == 1) & neg).sum())
        FN = int(((y_pred == 0) & pos).sum())
        TN = int(((y_pred == 0) & neg).sum())

        acc, prec, rec, f1, bacc = _metrics_from_counts(TP, FP, FN, TN)

        row = {"acc":acc, "prec":prec, "rec":rec, "f1":f1, "bacc":bacc, "thr":float(t)}
        # HARD
        if (f1>=HARD_MIN_F1) and (prec>=HARD_MIN_PREC) and (rec>=HARD_MIN_REC):
            key = (acc, bacc, prec, -abs(t-0.5))
            if (best_hard is None) or (key > best_hard[0]):
                best_hard = (key, row)
        # SOFT
        if (f1>=SOFT_MIN_F1) and (prec>=SOFT_MIN_PREC) and (rec>=SOFT_MIN_REC):
            key = (acc, bacc, prec, -abs(t-0.5))
            if (best_soft is None) or (key > best_soft[0]):
                best_soft = (key, row)
        # FALLBACK (max BAcc → Acc → Prec)
        key_fb = (bacc, acc, prec)
        if (best_fallback is None) or (key_fb > best_fallback[0]):
            best_fallback = (key_fb, row)

    if best_hard is not None:
        row = best_hard[1]; row["mode"] = "hard"; return row
    if best_soft is not None:
        row = best_soft[1]; row["mode"] = "soft"; return row
    row = best_fallback[1]; row["mode"] = "fallback_bacc"; return row

def main():
    rows = []
    picked = {}
    out_root = os.path.join(RESULTS_DIR, "24_ensemble_pickbest_perfold")
    os.makedirs(out_root, exist_ok=True)

    for k in range(1, 5+1):
        best_tuple = None  # (acc, bacc, f1, rec, prec, model, row)
        best_row = None
        best_model = None

        for md in MODEL_RESULTS:
            y_true, y_prob = load_fold_probs(md, k)
            row = choose_thr_accfirst_vec(y_true, y_prob, thrs=THRS)
            key = (row["acc"], row["bacc"], row["f1"], row["rec"], row["prec"])
            if (best_tuple is None) or (key > best_tuple):
                best_tuple = key
                best_row = row
                best_model = md

        picked[k] = {"model": best_model, **best_row}

        rows.append({
            "fold": k,
            "model": best_model,
            "acc": best_row["acc"], "prec": best_row["prec"], "rec": best_row["rec"],
            "f1": best_row["f1"], "bacc": best_row["bacc"], "thr": best_row["thr"],
            "mode": best_row["mode"]
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "_summary_folds.csv"), index=False)
    avg = df[["acc","prec","rec","f1","bacc"]].mean().to_dict()
    std = df[["acc","prec","rec","f1","bacc"]].std().to_dict()

    with open(os.path.join(out_root, "_summary_avg_std.json"), "w") as f:
        json.dump({
            "avg": {k: float(v) for k,v in avg.items()},
            "std": {k: float(v) for k,v in std.items()},
            "note": "Pick-best-single-model per fold with threshold tuning under constraints",
            "candidates": MODEL_RESULTS
        }, f, indent=2)

    print("[DONE] results →", out_root)
    print(df)
    print("AVG:", {k: round(float(v),4) for k,v in avg.items()})

if __name__ == "__main__":
    main()
