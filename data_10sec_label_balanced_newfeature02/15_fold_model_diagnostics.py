# 15_fold_model_diagnostics.py
import os, json, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")

MODEL_RESULTS = [
    "11_resnet1d_runner06.py",
    "10_cnn_bn_runner06.py",
    "18_resnet1d_accfirst_runner08.py",
]

THRS = np.linspace(0, 1, 501)

def load_fold_probs(model_dirname, fold):
    p1 = os.path.join(RESULTS_DIR, model_dirname, f"fold{fold}", "predictions.csv")
    p2 = os.path.join(RESULTS_DIR, model_dirname, f"fold{fold}", "predictions_ens.csv")
    path = p1 if os.path.exists(p1) else p2
    df = pd.read_csv(path)
    y_true = df["y_true"].values.astype(int)
    prob_col = "y_prob" if "y_prob" in df.columns else ("y_prob_ens" if "y_prob_ens" in df.columns else None)
    if prob_col is None:
        for c in df.columns:
            if c.startswith("y_prob"): prob_col = c; break
    y_prob = df[prob_col].values.astype(float)
    return y_true, y_prob

def best_f1(y_true, y_prob):
    best = None
    for t in THRS:
        y_pred = (y_prob >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec= precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        key = f1
        if (best is None) or (key > best[0]): best = (key, t, acc, prec, rec, f1)
    return best  # (f1, thr, acc, prec, rec, f1)

def main():
    rows = []
    for k in range(1, 6):
        for md in MODEL_RESULTS:
            yt, yp = load_fold_probs(md, k)
            f1, thr, acc, prec, rec, _ = best_f1(yt, yp)
            rows.append({"fold":k, "model":md, "thr_bestF1":thr, "acc":acc, "prec":prec, "rec":rec, "f1":f1})
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "15_fold_model_diagnostics.csv")
    df.to_csv(out, index=False)
    print(df)
    print(f"[SAVED] {out}")

if __name__ == "__main__":
    main()
