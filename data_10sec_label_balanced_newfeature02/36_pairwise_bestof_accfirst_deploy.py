# 36_pairwise_bestof_accfirst_deploy.py
# Pairwise Best-of Ensemble (LogitMix) + Accuracy-first thresholding
# Candidates: 29, 31b, 10, 18  |  Outputs → results/36_pairwise_bestof_accfirst_deploy/*
import os, json, itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT        = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
OUT_DIR     = os.path.join(RESULTS_DIR, "36_pairwise_bestof_accfirst_deploy")

# ---- candidates (exact folder name or .py; resolver handles both) ----
CAND_MODELS = [
    "29_tcn_wide_runner06.py",
    "31b_tcn_wide112_gnspdrop_runner06.py",
    "10_cnn_bn_runner06.py",
    "18_resnet1d_accfirst_runner08.py",
]

# ---- search/config ----
THRESHOLDS = np.linspace(0.0, 1.0, 1001)   # threshold sweep
WEIGHTS    = np.linspace(0.0, 1.0, 11)     # w for m1; (1-w) for m2 in logit space
MIN_F1, MIN_PREC = 0.75, 0.78              # accuracy-first constraints
OBJECTIVE = ("acc", "f1", "bacc", "rec", "prec")
FALLBACK_KEY = "bacc"                      # if constraints fail → maximize bacc

# ---------------- utils ----------------
def _list_result_dirs():
    if not os.path.isdir(RESULTS_DIR): return []
    return sorted([d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))])

def _resolve_model_dir(name: str):
    """Try exact → strip .py → case-insensitive prefix; return absolute path or None."""
    dirs = _list_result_dirs()
    if name in dirs: return os.path.join(RESULTS_DIR, name)
    base = name[:-3] if name.lower().endswith(".py") else name
    if base in dirs: return os.path.join(RESULTS_DIR, base)
    low = base.lower()
    cand = [d for d in dirs if d.lower().startswith(low)]
    if len(cand) == 1: return os.path.join(RESULTS_DIR, cand[0])
    return None

def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def _safe_logit(p, eps=1e-6): p = np.clip(p, eps, 1-eps); return np.log(p/(1-p))

def _balanced_accuracy(cm):
    tn, fp, fn, tp = cm.ravel()
    tpr = tp/(tp+fn) if (tp+fn)>0 else 0.0
    tnr = tn/(tn+fp) if (tn+fp)>0 else 0.0
    return 0.5*(tpr+tnr)

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
    }

def _choose_thr_accfirst(y_true, y_prob):
    best_hard, best_fb = None, None
    key = lambda m: tuple(m[k] for k in OBJECTIVE)
    for t in THRESHOLDS:
        m = _metrics(y_true, y_prob, t)
        if (m["f1"]>=MIN_F1) and (m["prec"]>=MIN_PREC):
            if (best_hard is None) or (key(m) > key(best_hard)): best_hard = m
        if (best_fb is None) or ((m[FALLBACK_KEY], m["acc"]) > (best_fb[FALLBACK_KEY], best_fb["acc"])):
            best_fb = m
    if best_hard is not None: m=best_hard; m["mode"]="hard"; return m
    m=best_fb; m["mode"]=f"fallback_{FALLBACK_KEY}"; return m

def _find_prob_column(df: pd.DataFrame) -> str:
    for c in ["y_prob","y_prob_ens"]:
        if c in df.columns: return c
    cand = [c for c in df.columns if str(c).startswith("y_prob")]
    if cand: return cand[0]
    raise ValueError(f"Could not find probability column in {df.columns.tolist()}")

def _load_fold_preds(model_dir, fold):
    p1 = os.path.join(model_dir, f"fold{fold}", "predictions.csv")
    p2 = os.path.join(model_dir, f"fold{fold}", "predictions_ens.csv")
    path = p1 if os.path.exists(p1) else p2
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing predictions for fold{fold}: {path}")
    df = pd.read_csv(path)
    y_true = df["y_true"].values.astype(int)
    y_prob = df[_find_prob_column(df)].values.astype(float)
    return y_true, y_prob

def _best_single(y_true, y_prob):
    return _choose_thr_accfirst(y_true, y_prob)

def _best_pair_logit(y_true, p1, p2):
    best = None; best_key = None
    l1, l2 = _safe_logit(p1), _safe_logit(p2)
    for w in WEIGHTS:
        lw = w*l1 + (1.0-w)*l2
        m = _choose_thr_accfirst(y_true, _sigmoid(lw))
        k = tuple(m[k] for k in OBJECTIVE)
        if (best is None) or (k > best_key):
            best, best_key = {"w": float(w), **m}, k
    return best

# ---------------- main ----------------
def main():
    print("[results dirs]", _list_result_dirs())

    # resolve available candidates
    resolved, missing = {}, []
    for m in CAND_MODELS:
        d = _resolve_model_dir(m)
        (resolved if d else missing).__setitem__(m, d) if d else None
        if d is None: missing.append(m)
    if missing: print("[WARN] Not found (skip):", missing)
    if not resolved:
        raise SystemExit("No candidate model results found. Check RESULTS_DIR or generate base predictions first.")
    cands = list(resolved.keys())
    pairs = list(itertools.combinations(cands, 2))

    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []
    for fold in range(1, 6):
        # load all predictions for this fold
        y_true = None
        probs = {}
        for m, d in resolved.items():
            try:
                yt, yp = _load_fold_preds(d, fold)
                if y_true is None: y_true = yt
                else:
                    assert np.array_equal(y_true, yt), f"y_true mismatch in fold{fold} ({m})"
                probs[m] = yp
            except Exception as e:
                print(f"[WARN] fold{fold} skip {m}: {e}")

        if (y_true is None) or (len(probs)==0):
            print(f"[WARN] fold{fold}: no predictions available; skip.")
            continue

        best_candidate, best_key = None, None

        # singles
        for m, yp in probs.items():
            ms = _best_single(y_true, yp)
            rec = {"fold": fold, "type": "single", "model": m, **ms, "weights": [1.0], "pair": None}
            k = tuple(ms[k] for k in OBJECTIVE)
            if (best_candidate is None) or (k > best_key): best_candidate, best_key = rec, k

        # pairs (logit mix)
        for m1, m2 in pairs:
            if (m1 not in probs) or (m2 not in probs): continue
            ms = _best_pair_logit(y_true, probs[m1], probs[m2])
            rec = {"fold": fold, "type": "pair", "model": f"{m1}+{m2}",
                   **ms, "weights": [ms["w"], 1.0-ms["w"]], "pair": [m1, m2]}
            k = tuple(ms[k] for k in OBJECTIVE)
            if (best_candidate is None) or (k > best_key): best_candidate, best_key = rec, k

        all_rows.append(best_candidate)

    if not all_rows: raise SystemExit("No folds processed.")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT_DIR, "summary_folds.csv"), index=False)

    avg = df[["acc","prec","rec","f1","bacc"]].mean().to_dict()
    thr_deploy = float(np.median(df["thr"].astype(float)))
    out = {
        "avg": {k: float(v) for k,v in avg.items()},
        "deploy_threshold_median": thr_deploy,
        "note": "Pairwise Best-of (single + logit-mix pairs) with accuracy-first thresholding (MIN_F1=0.75, MIN_PREC=0.78)",
        "candidates": cands,
        "resolved_map": {m: os.path.basename(resolved[m]) for m in resolved},
    }
    with open(os.path.join(OUT_DIR, "chosen.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[DONE] results → {OUT_DIR}")
    print(df)
    print("AVG:", {k: round(v,4) for k,v in avg.items()}, "| deploy_thr(median) =", round(thr_deploy,3))

if __name__ == "__main__":
    main()
