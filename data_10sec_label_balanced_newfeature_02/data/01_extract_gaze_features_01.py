import os
import numpy as np
import pandas as pd

# ---------- Paths ----------
data_root     = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature_02\data"
input_folder  = os.path.join(data_root, "gaze_files")
output_folder = os.path.join(data_root, "gaze_features")
report_dir    = os.path.join(output_folder, "_reports_extract")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# ---------- Settings ----------
BASE_FEATURES  = ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR']
DELTA_FEATURES = [f'delta_{f}' for f in BASE_FEATURES]
ALL_FEATURES   = BASE_FEATURES + DELTA_FEATURES
TARGET_LEN     = 1500
INPUT_SUFFIX   = "_gaze.csv"
OUTPUT_SUFFIX  = "_features.csv"

meta_rows = []

def load_keep_base_cols(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df[BASE_FEATURES].apply(pd.to_numeric, errors='coerce')
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

def fix_length(df: pd.DataFrame, target_len: int):
    n = len(df)
    if n >= target_len:
        return df.iloc[:target_len].copy(), 0, n - target_len
    need = target_len - n
    last = df.iloc[[-1]].copy() if n > 0 else pd.DataFrame([{c:0.0 for c in BASE_FEATURES}])
    pad_block = pd.concat([last]*need, ignore_index=True)
    fixed = pd.concat([df, pad_block], ignore_index=True)
    return fixed, need, 0

def build_deltas(df_raw: pd.DataFrame) -> pd.DataFrame:
    out = df_raw.copy()
    for f in BASE_FEATURES:
        out[f'delta_{f}'] = out[f].diff().fillna(0.0)
    out = out[ALL_FEATURES].astype('float32')
    return out

files = sorted([f for f in os.listdir(input_folder) if f.endswith(INPUT_SUFFIX)])
print(f"[INFO] Found {len(files)} gaze files.")

for fname in files:
    in_path  = os.path.join(input_folder, fname)
    out_name = fname.replace(INPUT_SUFFIX, OUTPUT_SUFFIX)
    out_path = os.path.join(output_folder, out_name)
    try:
        df_raw = load_keep_base_cols(in_path)
        df_fixed, pad_count, trunc_count = fix_length(df_raw, TARGET_LEN)
        feat_df = build_deltas(df_fixed)
        feat_df.to_csv(out_path, index=False)

        meta_rows.append({
            "input_file": fname,
            "output_file": out_name,
            "rows_in": len(df_raw),
            "rows_out": len(feat_df),
            "padded_rows": pad_count,
            "truncated_rows": trunc_count,
            "n_cols_out": feat_df.shape[1],
            "columns_ok": int(list(feat_df.columns) == ALL_FEATURES)
        })
        print(f"[OK] {fname} → {out_name} | shape={feat_df.shape} | pad={pad_count} trunc={trunc_count}")
    except Exception as e:
        meta_rows.append({
            "input_file": fname, "output_file": None, "rows_in": None, "rows_out": None,
            "padded_rows": None, "truncated_rows": None, "n_cols_out": None,
            "columns_ok": 0, "error": str(e)
        })
        print(f"[ERROR] {fname}: {e}")

meta_df = pd.DataFrame(meta_rows)
meta_csv = os.path.join(report_dir, "01_features_meta_10s_16f.csv")
meta_df.to_csv(meta_csv, index=False)
print(f"[DONE] Meta → {meta_csv}")
