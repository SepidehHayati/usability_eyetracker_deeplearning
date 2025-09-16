# -*- coding: utf-8 -*-
"""
Robust cleaning for gaze feature files:
- Coordinates (FPOGX/FPOGY/BPOGX/BPOGY): hard clip to [0, 1]
- Pupils (LPD/RPD) + durations (FPOGD/BKDUR): file-level winsorize by quantiles
- Local spike suppression with rolling median + MAD (pupil/duration by default)
- Recompute deltas from cleaned base features
- Save per-file cleaning report + a global report CSV
"""

import os
import numpy as np
import pandas as pd

# ========= CONFIG =========
INPUT_DIR  = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data\gaze_features"
OUTPUT_DIR = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data\gaze_features_cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_FEATS   = ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR']
DELTA_FEATS  = [f'delta_{f}' for f in BASE_FEATS]
ALL_EXPECTED = BASE_FEATS + DELTA_FEATS
N_ROWS       = 1500   # بر اساس استخراج

# Winsorize quantiles (file-level; leakage-free نسبت به split)
Q_LO, Q_HI   = 0.01, 0.99

# Spike suppression (local)
ROLL_WIN     = 7          # window size (samples)
MAD_K        = 6.0        # threshold multiplier
EPS          = 1e-9
ZERO_AS_MISSING_PUPIL = True   # 0 در مردمک فاقد معنی فیزیکی → NA

# Apply spike suppression to which groups
SUPPRESS_SPIKES_ON = {
    "pupil": True,
    "duration": True,
    "coords": False,   # coords already hard-clipped to [0,1]
}

# ========= HELPERS =========
def winsorize_series(s: pd.Series, q_lo=Q_LO, q_hi=Q_HI):
    x = s.astype(float).copy()
    if x.notna().sum() < 10:
        return x, 0, 0
    lo, hi = x.quantile([q_lo, q_hi])
    # اگر lo==hi (سیگنال ثابت)، تغییری نده
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return x, 0, 0
    below = (x < lo).sum()
    above = (x > hi).sum()
    x = x.clip(lo, hi)
    return x, int(below), int(above)

def rolling_spike_suppress(s: pd.Series, win=ROLL_WIN, k=MAD_K):
    """
    Replace points that deviate from rolling median by > k * rolling MAD
    with the rolling median (centered).
    """
    if s.notna().sum() < 5:
        return s, 0
    med = s.rolling(win, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(win, center=True, min_periods=1).median()
    mask = (s - med).abs() > (k * (mad + EPS))
    out = s.copy()
    out[mask] = med[mask]
    return out, int(mask.sum())

def recompute_deltas(df_clean_base: pd.DataFrame):
    out = {}
    for f in BASE_FEATS:
        d = df_clean_base[f].astype(float).diff().fillna(0.0)
        out[f'delta_{f}'] = d.astype(np.float32)
    return pd.DataFrame(out)

# ========= MAIN CLEAN LOOP =========
global_reports = []  # جمع‌بندی همه فایل‌ها

for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith("_features.csv"):
        continue

    in_path  = os.path.join(INPUT_DIR, fname)
    out_path = os.path.join(OUTPUT_DIR, fname.replace("_features.csv", "_cleaned.csv"))
    rep_path = os.path.join(OUTPUT_DIR, fname.replace("_features.csv", "_clean_report.csv"))

    try:
        df = pd.read_csv(in_path)
        # sanity checks
        for c in BASE_FEATS:
            if c not in df.columns:
                raise ValueError(f"{fname}: missing base feature column '{c}'")
        if len(df) != N_ROWS:
            raise ValueError(f"{fname}: expected {N_ROWS} rows, got {len(df)}")

        # فقط با نسخهٔ base کار می‌کنیم، دلتاها را در انتها می‌سازیم
        base = df[BASE_FEATS].copy()
        base = base.apply(pd.to_numeric, errors='coerce')
        base = base.ffill().bfill()

        # دسته‌بندی‌ها
        coord_feats = ['FPOGX','FPOGY','BPOGX','BPOGY']
        pupil_feats = ['LPD','RPD']
        dur_feats   = ['FPOGD','BKDUR']

        report_rows = []

        # 1) Coordinates: hard-clip [0,1]
        for f in coord_feats:
            s = base[f].astype(float)
            before = s.copy()
            s = s.clip(0, 1)
            changed = int((before != s).sum())
            base[f] = s
            report_rows.append([f, 'coords_clip_01', changed, len(s), changed/len(s)*100])

            # (اختیاری) سرکوب اسپایک محلی روی coords خاموش است
            if SUPPRESS_SPIKES_ON["coords"]:
                s2, spikes = rolling_spike_suppress(s, ROLL_WIN, MAD_K)
                base[f] = s2
                report_rows.append([f, 'coords_spike_suppress', spikes, len(s2), spikes/len(s2)*100])

        # 2) Pupils: zeros->NA (اختیاری) + ffill/bfill + winsorize + spike suppression
        for f in pupil_feats:
            s = base[f].astype(float)
            if ZERO_AS_MISSING_PUPIL:
                zeros = int((s == 0).sum())
                s = s.replace(0, np.nan)
                report_rows.append([f, 'pupil_zero_to_na', zeros, len(s), zeros/len(s)*100])

            s = s.ffill().bfill()

            s_w, below, above = winsorize_series(s, Q_LO, Q_HI)
            base[f] = s_w
            tot = int((s != s_w).sum())
            report_rows.append([f, 'pupil_winsorize_q', tot, len(s_w), tot/len(s_w)*100])

            if SUPPRESS_SPIKES_ON["pupil"]:
                s2, spikes = rolling_spike_suppress(base[f], ROLL_WIN, MAD_K)
                base[f] = s2
                report_rows.append([f, 'pupil_spike_suppress', spikes, len(s2), spikes/len(s2)*100])

        # 3) Durations: non-negative + winsorize + (optional) spike suppression
        for f in dur_feats:
            s = base[f].astype(float)
            s = s.clip(lower=0)  # مدت زمان منفی ندارد
            s_w, below, above = winsorize_series(s, Q_LO, Q_HI)
            base[f] = s_w
            tot = int((s != s_w).sum())
            report_rows.append([f, 'dur_winsorize_q', tot, len(s_w), tot/len(s_w)*100])

            if SUPPRESS_SPIKES_ON["duration"]:
                s2, spikes = rolling_spike_suppress(base[f], ROLL_WIN, MAD_K)
                base[f] = s2
                report_rows.append([f, 'dur_spike_suppress', spikes, len(s2), spikes/len(s2)*100])

        # ایمنی نهایی
        base = base.ffill().bfill()

        # 4) Recompute deltas from cleaned base
        deltas = recompute_deltas(base)

        # 5) Assemble & save
        out_df = pd.concat([base.astype(np.float32), deltas], axis=1)
        # ترتیب ستون‌ها را ثابت نگه دار
        out_df = out_df[[*BASE_FEATS, *[f'delta_{f}' for f in BASE_FEATS]]]
        out_df.to_csv(out_path, index=False)
        print(f"[CLEAN] {fname} → {out_path}  shape={out_df.shape}")

        # 6) File-level report
        rep = pd.DataFrame(report_rows, columns=["feature","action","num_changed","n","pct_changed"])
        rep.to_csv(rep_path, index=False)
        # برای گزارش کل:
        rep_summary = rep.groupby("feature", as_index=False)["num_changed","n"].sum()
        rep_summary["file"] = fname
        global_reports.append(rep_summary.assign(pct_changed=rep_summary["num_changed"]/rep_summary["n"]*100))

    except Exception as e:
        print(f"[ERROR] {fname}: {e}")

# 7) Global report
if global_reports:
    glb = pd.concat(global_reports, ignore_index=True)
    glb = glb[["file","feature","num_changed","n","pct_changed"]]
    glb.to_csv(os.path.join(OUTPUT_DIR, "_cleaning_report_ALL.csv"), index=False)
    print(f"[REPORT] Saved global report → {os.path.join(OUTPUT_DIR, '_cleaning_report_ALL.csv')}")
print("[OK] Cleaning completed.")
