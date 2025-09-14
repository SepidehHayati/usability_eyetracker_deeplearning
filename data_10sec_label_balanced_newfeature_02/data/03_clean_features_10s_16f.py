import os
import numpy as np
import pandas as pd

# ---------- Paths ----------
data_root  = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature_02\data"
input_dir  = os.path.join(data_root, "gaze_features")
output_dir = os.path.join(data_root, "gaze_features_cleaned")
mask_dir   = os.path.join(data_root, "gaze_features_masks")
report_dir = os.path.join(output_dir, "_reports_clean")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# ---------- Columns ----------
BASE  = ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR']
DELTA = [f'delta_{c}' for c in BASE]
ALL   = BASE + DELTA
TARGET_LEN = 1500

DESPIKE_COLS = {'FPOGX','FPOGY','LPD','RPD'}      # فقط این‌ها دی‌اسپایک می‌شن
ALLOW_CONST_RAW = {'BKDUR', 'FPOGD'}              # ثابت بودنشان طبیعی است
HAMPEL_WINDOW = 11
HAMPEL_SIGMAS = 8.0
MAX_REPLACE_FRAC = 0.05                           # حداکثر 5% جایگزینی در هر ستون

def hampel_mask_on_diff(s: pd.Series, window=HAMPEL_WINDOW, n_sigmas=HAMPEL_SIGMAS) -> pd.Series:
    x  = pd.to_numeric(s, errors='coerce').astype('float64')
    dx = x.diff()
    med = dx.rolling(window, center=True).median()
    mad = (dx - med).abs().rolling(window, center=True).median()
    k = 1.4826 * mad * n_sigmas
    mask = (dx - med).abs() > k
    return mask.fillna(False)

def cap_mask_by_fraction(mask: pd.Series, s: pd.Series, frac: float) -> pd.Series:
    max_k = int(frac * len(mask))
    k = int(mask.sum())
    if k <= max_k or max_k == 0:
        return mask
    dx_abs = s.diff().abs()
    top_idx = dx_abs.nlargest(max_k).index
    new_mask = pd.Series(False, index=mask.index)
    new_mask.loc[top_idx] = True
    return new_mask

def clean_one_file(fpath: str):
    df = pd.read_csv(fpath)
    miss = set(ALL) - set(df.columns)
    if miss:
        raise ValueError(f"Missing expected columns: {miss}")
    df = df[ALL].copy()

    n_in = len(df)
    if n_in > TARGET_LEN:
        df = df.iloc[:TARGET_LEN].copy(); padded_rows = 0
    elif n_in < TARGET_LEN:
        need = TARGET_LEN - n_in
        last = df.iloc[[-1]].copy()
        df = pd.concat([df, pd.concat([last]*need, ignore_index=True)], ignore_index=True)
        padded_rows = need
    else:
        padded_rows = 0

    df_raw = df[BASE].copy()
    spike_counts = {c:0 for c in BASE}
    repl_fracs   = {c:0.0 for c in BASE}

    for c in BASE:
        if c not in DESPIKE_COLS:
            continue
        m = hampel_mask_on_diff(df_raw[c])
        m = cap_mask_by_fraction(m, df_raw[c], MAX_REPLACE_FRAC)
        spike_counts[c] = int(m.sum())
        repl_fracs[c]   = spike_counts[c] / TARGET_LEN
        df_raw.loc[m, c] = np.nan

    row_has_nan_before_interp = df_raw.isna().any(axis=1).values.astype(np.uint8)

    before_nan = int(df_raw.isna().sum().sum())
    df_interp = df_raw.interpolate(method='linear', limit=5, limit_direction='both')
    df_interp = df_interp.ffill().bfill()
    after_nan  = int(df_interp.isna().sum().sum())

    for c in ['FPOGX','FPOGY']:
        s = df_interp[c].rolling(3, center=True).median()
        df_interp[c] = s.fillna(df_interp[c])

    # ستون‌هایی که دی‌اسپایک نشدن را از df اصلی برگردان
    for c in BASE:
        if c not in DESPIKE_COLS:
            df_interp[c] = df[c]

    out = df_interp.copy()
    for c in BASE:
        out[f'delta_{c}'] = out[c].diff().fillna(0.0)

    for c in ALL:
        ql, qh = out[c].quantile([0.005, 0.995])
        if pd.notna(ql) and pd.notna(qh) and qh > ql:
            out[c] = out[c].clip(ql, qh)

    out = out[ALL].astype('float32')

    mask_row = row_has_nan_before_interp.copy()
    if padded_rows > 0:
        mask_row[-padded_rows:] = 1

    constant_raw = [c for c in BASE if out[c].nunique(dropna=False) <= 1]

    stats = {
        "n_in": n_in, "padded_rows": padded_rows, "constant_raw": constant_raw,
        "spike_counts": spike_counts, "replace_fracs": repl_fracs,
        "nans_before_interp": before_nan, "nans_after_interp": after_nan
    }
    return out, mask_row, stats

files = sorted([f for f in os.listdir(input_dir) if f.endswith("_features.csv")])

meta = []
for fname in files:
    in_path  = os.path.join(input_dir, fname)
    out_name = fname.replace("_features.csv", "_cleaned.csv")
    out_path = os.path.join(output_dir, out_name)
    mask_path = os.path.join(mask_dir, fname.replace("_features.csv", "_mask.csv"))

    try:
        cleaned, mask_row, st = clean_one_file(in_path)
        max_frac = max(st["replace_fracs"].values()) if st["replace_fracs"] else 0.0
        nonallowed_const = [c for c in st["constant_raw"] if c not in ALLOW_CONST_RAW]
        flag = "OK"
        if nonallowed_const:
            flag = "FAIL"
        elif max_frac > MAX_REPLACE_FRAC:
            flag = "WARN"

        cleaned.to_csv(out_path, index=False)
        pd.DataFrame({"interp_or_padded": mask_row.astype(int)}).to_csv(mask_path, index=False)

        meta.append({
            "file_in": fname, "file_out": out_name,
            "rows_in": st["n_in"], "rows_out": len(cleaned), "cols_out": cleaned.shape[1],
            "flag": flag, "constant_raw": ";".join(st["constant_raw"]),
            "spike_sum": int(sum(st["spike_counts"].values())),
            "max_replace_frac": float(max_frac),
            "nans_before_interp": st["nans_before_interp"], "nans_after_interp": st["nans_after_interp"],
            "padded_rows": st["padded_rows"], "mask_ones": int(mask_row.sum())
        })
        print(f"[{flag}] {fname} → {out_name} | max_replace_frac={max_frac:.3f} | spikes_sum={sum(st['spike_counts'].values())} | const_raw={st['constant_raw']}")

    except Exception as e:
        meta.append({
            "file_in": fname, "file_out": None, "rows_in": None, "rows_out": None,
            "cols_out": None, "flag": "ERROR", "constant_raw": "",
            "spike_sum": None, "max_replace_frac": None,
            "nans_before_interp": None, "nans_after_interp": None,
            "padded_rows": None, "mask_ones": None, "error": str(e)
        })
        print(f"[ERROR] {fname}: {e}")

meta_df = pd.DataFrame(meta)
meta_csv = os.path.join(report_dir, "03_clean_meta_10s_16f.csv")
meta_df.to_csv(meta_csv, index=False)
print(f"[DONE] Cleaned files → {output_dir}")
print(f"[DONE] Masks → {mask_dir}")
print(f"[DONE] Clean meta → {meta_csv}")
