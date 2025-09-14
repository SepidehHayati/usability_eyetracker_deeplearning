import os
import numpy as np
import pandas as pd

# ---------- Target folder to QC ----------
# برای QC روی ویژگی‌های خام:
# features_dir = r"...\gaze_features"
# برای QC روی فایل‌های cleaned:
# features_dir = r"...\gaze_features_cleaned"
features_dir = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature_02\data\gaze_features"

report_dir = os.path.join(features_dir, "_reports_qc")
os.makedirs(report_dir, exist_ok=True)

# ---------- Columns ----------
BASE  = ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR']
DELTA = [f'delta_{c}' for c in BASE]
ALL   = BASE + DELTA
TARGET_LEN = 1500

# خام‌هایی که ثابت بودنشان طبیعی‌ست
ALLOW_CONST_RAW = {'BKDUR', 'FPOGD'}
# آستانه‌ی هشدار اسپایک دلتا
WORST_DELTA_WARN = 0.30

def robust_outlier_rate(x: pd.Series, n_sigmas=5.0):
    x = pd.to_numeric(x, errors='coerce')
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return 0.0, 0, len(x)
    k = 1.4826 * mad * n_sigmas
    mask = (x - med).abs() > k
    cnt = int(mask.sum())
    return float(cnt / max(len(x),1)), cnt, len(x)

files = sorted([f for f in os.listdir(features_dir)
                if f.endswith("_features.csv") or f.endswith("_cleaned.csv")])

print(f"[INFO] Found {len(files)} feature files to QC in: {features_dir}")

summary_rows, percol_rows = [], []

for fname in files:
    fpath = os.path.join(features_dir, fname)
    try:
        df = pd.read_csv(fpath)
        shape_ok   = (df.shape[0] == TARGET_LEN and df.shape[1] >= 16)
        missing_cols = list(set(ALL) - set(df.columns))
        order_ok   = False
        if not missing_cols:
            df = df[ALL].copy()
            order_ok = True

        df = df.replace([np.inf, -np.inf], np.nan)
        n_cells = df.size
        n_nans  = int(df.isna().sum().sum())
        nan_ratio = n_nans / n_cells if n_cells else 0.0

        constant_cols = [c for c in ALL if df[c].nunique(dropna=True) <= 1]

        # delta outliers
        worst_rate = 0.0
        for dc in DELTA:
            r, _, _ = robust_outlier_rate(df[dc], n_sigmas=5.0)
            percol_rows.append({"file": fname, "column": dc, "outlier_rate": r})
            worst_rate = max(worst_rate, r)

        nonallowed_const_raw = [c for c in constant_cols if (c in BASE and c not in ALLOW_CONST_RAW)]
        allowed_const_raw    = [c for c in constant_cols if (c in BASE and c in ALLOW_CONST_RAW)]
        const_deltas_due_to_allowed_raw = [c for c in constant_cols if (c in DELTA and c.replace('delta_','') in ALLOW_CONST_RAW)]

        flag, reasons = "OK", []
        if missing_cols:
            flag = "FAIL"; reasons.append(f"missing_cols={missing_cols}")
        if not shape_ok:
            flag = "FAIL"; reasons.append(f"shape={df.shape} expected (1500,16)")
        if nonallowed_const_raw:
            flag = "FAIL"; reasons.append(f"constant_raw_nonallowed={nonallowed_const_raw}")
        if allowed_const_raw:
            reasons.append(f"allowed_const_raw={allowed_const_raw}")
        if const_deltas_due_to_allowed_raw:
            reasons.append(f"delta_from_allowed_raw={const_deltas_due_to_allowed_raw}")
        if nan_ratio > 0.15:
            flag = "FAIL"; reasons.append(f"nan_ratio={nan_ratio:.3f}")
        if flag == "OK":
            if nan_ratio > 0.0:
                flag = "WARN"; reasons.append(f"nan_ratio={nan_ratio:.3f}")
            if worst_rate > WORST_DELTA_WARN:
                flag = "WARN"; reasons.append(f"delta_outlier_rate_max={worst_rate:.3f}")

        summary_rows.append({
            "file": fname, "rows": df.shape[0], "cols": df.shape[1],
            "shape_ok": int(shape_ok), "order_ok": int(order_ok),
            "nan_ratio": nan_ratio,
            "n_constant_cols": len(constant_cols),
            "constant_cols": ";".join(constant_cols),
            "delta_outlier_rate_max": worst_rate,
            "flag": flag, "reasons": "; ".join(reasons)
        })
        print(f"[{flag}] {fname} | nan={nan_ratio:.3f} | const={len(constant_cols)} | worstΔ={worst_rate:.3f}")

    except Exception as e:
        summary_rows.append({
            "file": fname, "rows": None, "cols": None, "shape_ok": 0, "order_ok": 0,
            "nan_ratio": None, "n_constant_cols": None, "constant_cols": None,
            "delta_outlier_rate_max": None, "flag": "ERROR", "reasons": str(e)
        })
        print(f"[ERROR] {fname}: {e}")

summary_df = pd.DataFrame(summary_rows)
percol_df  = pd.DataFrame(percol_rows)

summary_csv = os.path.join(report_dir, "02a_qc_summary_10s_16f.csv")
percol_csv  = os.path.join(report_dir, "02b_qc_delta_outliers_10s_16f.csv")
summary_df.to_csv(summary_csv, index=False)
percol_df.to_csv(percol_csv, index=False)

print(f"[DONE] summary → {summary_csv}")
print(f"[DONE] per-column delta outliers → {percol_csv}")
