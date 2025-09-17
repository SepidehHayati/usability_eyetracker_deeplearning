# 25_tabular_rf_runner06.py
# Classic tabular model with summary features + per-fold training & threshold tuning (HARD/SOFT)

import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- مسیرها: از پوشهٔ models یک پوشه برو بالا تا به ریشهٔ پروژه برسی
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR   = os.path.join(ROOT, "data")
RESULTS_DIR= os.path.join(ROOT, "results")

X_PATH     = os.path.join(DATA_DIR, "X_clean.npy")
Y_PATH     = os.path.join(DATA_DIR, "Y_clean.npy")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

MODEL_NAME = os.path.basename(__file__)  # e.g., "25_tabular_rf_runner06.py"
OUT_DIR    = os.path.join(RESULTS_DIR, MODEL_NAME)

THRS = np.linspace(0.0, 1.0, 501)

# Constraints
HARD_MIN_F1,   HARD_MIN_PREC,   HARD_MIN_REC = 0.78, 0.82, 0.75
SOFT_MIN_F1,   SOFT_MIN_PREC,   SOFT_MIN_REC = 0.75, 0.78, 0.65

def feat_one_channel(x1d: np.ndarray):
    m   = float(np.mean(x1d))
    sd  = float(np.std(x1d) + 1e-8)
    med = float(np.median(x1d))
    mn  = float(np.min(x1d))
    mx  = float(np.max(x1d))
    q25 = float(np.percentile(x1d, 25))
    q75 = float(np.percentile(x1d, 75))
    iqr = float(q75 - q25)
    t = np.arange(x1d.shape[0], dtype=np.float32)
    try:
        slope = float(np.polyfit(t, x1d, 1)[0])
    except Exception:
        slope = 0.0
    return [m, sd, med, mn, mx, iqr, slope]

def feat_sample(x: np.ndarray):
    # x: (T, C=16)
    feats = []
    for c in range(x.shape[1]):
        feats.extend(feat_one_channel(x[:, c]))
    return np.array(feats, dtype=np.float32)  # 16 * 7 = 112

def build_features(X: np.ndarray):
    F = [feat_sample(X[i]) for i in range(X.shape[0])]
    return np.stack(F, axis=0)  # (N, 112)

def metrics_from_counts(TP, FP, FN, TN):
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
    acc  = (TP+TN)/(TP+FP+FN+TN)
    f1   = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
    tnr  = TN/(TN+FP) if (TN+FP)>0 else 0.0
    bacc = 0.5*(rec+tnr)
    return acc, prec, rec, f1, bacc

def choose_thr_accfirst_vec(y_true: np.ndarray, y_prob: np.ndarray, thrs: np.ndarray = THRS):
    y_true = y_true.astype(np.int64)
    pos = (y_true == 1); neg = ~pos

    best_hard = None
    best_soft = None
    best_fb   = None

    for t in thrs:
        y_pred = (y_prob >= t).astype(np.int64)
        TP = int(((y_pred==1) & pos).sum())
        FP = int(((y_pred==1) & neg).sum())
        FN = int(((y_pred==0) & pos).sum())
        TN = int(((y_pred==0) & neg).sum())

        acc, prec, rec, f1, bacc = metrics_from_counts(TP, FP, FN, TN)
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
        if (best_fb is None) or (key_fb > best_fb[0]):
            best_fb = (key_fb, row)

    if best_hard is not None:
        r = best_hard[1]; r["mode"]="hard"; return r
    if best_soft is not None:
        r = best_soft[1]; r["mode"]="soft"; return r
    r = best_fb[1]; r["mode"]="fallback_bacc"; return r

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not (os.path.exists(X_PATH) and os.path.exists(Y_PATH)):
        print("[ERR] X_PATH:", X_PATH)
        print("[ERR] Y_PATH:", Y_PATH)
        raise FileNotFoundError("X_clean.npy or Y_clean.npy not found in data/")
    X = np.load(X_PATH, mmap_mode="r")  # (N,1500,16)
    y = np.load(Y_PATH, mmap_mode="r")  # (N,)
    print("[INFO] X:", X.shape, "Y:", y.shape)
    print("[INFO] building features...")
    F_all = build_features(X)  # (N,112)
    print("[INFO] Features:", F_all.shape)

    rows = []
    for k in range(1, 6):
        tr_idx = np.load(os.path.join(SPLITS_DIR, f"fold{k}_train_idx.npy"))
        va_idx = np.load(os.path.join(SPLITS_DIR, f"fold{k}_val_idx.npy"))

        Xtr, Xva = F_all[tr_idx], F_all[va_idx]
        ytr, yva = y[tr_idx].astype(np.int64), y[va_idx].astype(np.int64)

        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xva_s = scaler.transform(Xva)

        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42
        )
        clf.fit(Xtr_s, ytr)
        y_prob = clf.predict_proba(Xva_s)[:, 1]

        row = choose_thr_accfirst_vec(yva, y_prob, thrs=THRS)

        fold_dir = os.path.join(OUT_DIR, f"fold{k}")
        os.makedirs(fold_dir, exist_ok=True)
        pd.DataFrame({"y_true": yva, "y_prob": y_prob}).to_csv(
            os.path.join(fold_dir, "predictions.csv"), index=False
        )
        with open(os.path.join(fold_dir, "metrics_best.json"), "w") as f:
            json.dump(row, f, indent=2)

        rows.append({"fold": k, **row})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "_summary_folds.csv"), index=False)
    avg = df[["acc","prec","rec","f1","bacc"]].mean().to_dict()
    std = df[["acc","prec","rec","f1","bacc"]].std().to_dict()
    with open(os.path.join(OUT_DIR, "_summary_avg_std.json"), "w") as f:
        json.dump({
            "avg": {k: float(v) for k,v in avg.items()},
            "std": {k: float(v) for k,v in std.items()},
            "note": "RandomForest on summary features; per-fold scaler & threshold tuning (HARD/SOFT)"
        }, f, indent=2)

    print("[DONE] results →", OUT_DIR)
    print(df)
    print("AVG:", {k: round(float(v),4) for k,v in avg.items()})

if __name__ == "__main__":
    main()
