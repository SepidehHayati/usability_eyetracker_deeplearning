# -*- coding: utf-8 -*-
# runners/R07_ensemble_from_results.py
import os, json, argparse
import numpy as np
import pandas as pd

# از Runner06 منطق انتخاب آستانه و محاسبه متریک را قرض می‌گیریم
import runners.R06_runner_soft_fallback as r06

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_fold_preds(result_dir, fold_k):
    """predictions.csv را از یک مدل برای یک فولد می‌خواند."""
    csv_path = os.path.join(result_dir, f"fold{fold_k}", "predictions.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    # انتظار ستون‌ها: y_true, y_prob, y_pred_0.5, y_pred_best
    return df["y_true"].to_numpy(), df["y_prob"].to_numpy(), csv_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=False,
                   help="نام فولدرهای نتایج مدل‌ها در results/ ، به‌صورت پیش‌فرض دو مدل خوب فعلی")
    p.add_argument("--min_prec", type=float, default=0.80)
    p.add_argument("--min_f1",   type=float, default=0.75)
    p.add_argument("--relaxed_f1", type=float, default=0.70)
    p.add_argument("--weights", nargs="*", type=float, default=None,
                   help="وزن هر مدل برای میانگین گرفتن احتمال‌ها؛ پیش‌فرض وزن برابر")
    args = p.parse_args()

    # ریشهٔ پروژه
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RESULTS_DIR = os.path.join(ROOT, "results")

    # لیست پیش‌فرض مدل‌ها (نام فولدر نتایج که دقیقاً برابر نام فایل مدل است)
    default_models = [
        "11_resnet1d_runner06.py",
        "10_cnn_bn_runner06.py",
    ]
    model_dirs = args.models if args.models else default_models
    result_paths = [os.path.join(RESULTS_DIR, m) for m in model_dirs]
    for rp in result_paths:
        if not os.path.isdir(rp):
            raise FileNotFoundError(f"Result dir not found: {rp}")

    # وزن‌ها
    if args.weights:
        w = np.array(args.weights, dtype=float)
        assert len(w) == len(result_paths), "weights length must match #models"
    else:
        w = np.ones(len(result_paths), dtype=float)
    w = w / w.sum()

    # خروجی Ensemble
    out_name = "E01_ensemble_" + "_".join([os.path.splitext(m)[0] for m in model_dirs])
    OUT_DIR = os.path.join(RESULTS_DIR, out_name)
    ensure_dir(OUT_DIR)

    # ست‌کردن قیود برای Runner06
    r06.MIN_PREC   = float(args.min_prec)
    r06.MIN_F1     = float(args.min_f1)
    r06.RELAXED_F1 = float(args.relaxed_f1)

    fold_rows = []
    for k in range(1, 5+1):
        # خواندن پیش‌بینی‌ها از همه مدل‌ها
        y_trues = []
        y_probs = []
        srcs = []
        for rp in result_paths:
            yt, yp, path_csv = load_fold_preds(rp, k)
            y_trues.append(yt)
            y_probs.append(yp)
            srcs.append(path_csv)

        # اطمینان از یکسان بودن y_true
        for i in range(1, len(y_trues)):
            if not np.array_equal(y_trues[0], y_trues[i]):
                raise ValueError(f"Mismatch y_true between models at fold {k}:\n{srcs[0]}\n{srcs[i]}")
        y_true = y_trues[0]

        # Ensemble احتمال‌ها
        probs_stack = np.stack(y_probs, axis=1)  # (N, n_models)
        y_prob_ens = (probs_stack * w.reshape(1, -1)).sum(axis=1)

        # انتخاب آستانه با منطق Runner06
        thr, mb, hard_ok, soft_ok = r06._choose_threshold(
            y_true, y_prob_ens, r06.MIN_F1, r06.MIN_PREC, r06.RELAXED_F1
        )

        # ذخیرهٔ متریک‌ها و خروجی‌ها
        fold_dir = os.path.join(OUT_DIR, f"fold{k}")
        ensure_dir(fold_dir)
        with open(os.path.join(fold_dir, "metrics_best.json"), "w") as f:
            jb = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                  for k, v in mb.items() if k not in ["cm", "y_pred"]}
            jb.update({
                "best_threshold": float(thr),
                "hard_constraints_satisfied": bool(hard_ok),
                "soft_constraints_used": bool(soft_ok),
                "models": model_dirs,
                "weights": w.tolist()
            })
            json.dump(jb, f, indent=2)

        # ذخیرهٔ پیش‌بینی‌های Ensemble
        pd.DataFrame({
            "y_true": y_true,
            "y_prob_ens": y_prob_ens,
            "y_pred_best": mb["y_pred"]
        }).to_csv(os.path.join(fold_dir, "predictions_ensemble.csv"), index=False)

        # ثبت برای خلاصه
        fold_rows.append({
            "fold": k,
            "acc": float(mb["acc"]),
            "prec": float(mb["prec"]),
            "rec": float(mb["rec"]),
            "f1": float(mb["f1"])
        })

    df = pd.DataFrame(fold_rows)
    df.to_csv(os.path.join(OUT_DIR, "_summary_folds.csv"), index=False)
    avg = df[["acc","prec","rec","f1"]].mean().to_dict()
    std = df[["acc","prec","rec","f1"]].std().to_dict()
    with open(os.path.join(OUT_DIR, "_summary_avg_std.json"), "w") as f:
        json.dump({"avg": {k: float(v) for k, v in avg.items()},
                   "std": {k: float(v) for k, v in std.items()},
                   "note": "Ensemble from saved predictions with Runner06 constraints"}, f, indent=2)

    print("[DONE] Ensemble results in:", OUT_DIR)

if __name__ == "__main__":
    main()
