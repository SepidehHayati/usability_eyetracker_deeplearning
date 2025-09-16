# -*- coding: utf-8 -*-
"""
Map labels → cleaned feature files, and build X_clean.npy / Y_clean.npy
- فقط فایل‌های *_cleaned.csv خوانده می‌شوند (reportها نادیده گرفته می‌شوند)
- مچ بر اساس کلید (user, website_norm, task_id) → مستقل از ترتیب فایل‌ها
- خروجی: manifest_clean.csv + X_clean.npy (N,1500,16) + Y_clean.npy (N,)
"""

import os, re, glob, sys
import numpy as np
import pandas as pd

# ===== PATHS (ویرایش در صورت نیاز) =====
LABELS_PATH  = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data\label_balanced.xlsx"
LABEL_SHEET  = "Distribution of Tasks"
CLEAN_DIR    = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data\gaze_features_cleaned"
OUT_DIR      = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== CONSTANTS =====
BASE_FEATS   = ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR']
DELTA_FEATS  = [f'delta_{f}' for f in BASE_FEATS]
ALL_COLS     = BASE_FEATS + DELTA_FEATS
N_EXPECTED   = 1500  # rows per file

# ===== UTILS =====
def norm_site(s: str) -> str:
    """ lowercase + حذف هرچیز غیرحرفی/عددی برای هم‌سان‌سازی اسم سایت """
    return re.sub(r'[^a-z0-9]+', '', str(s).strip().lower())

# پترن نام فایل Clean شده (فاصله/زیرخط/خط‌تیره پشتیبانی می‌شود)
# مثال‌ها: caltech_t1_user7_cleaned.csv  |  Princeton_T1_User 12_cleaned.csv
PATTERN = re.compile(
    r'(?P<site>[A-Za-z0-9\-]+)[_\-\s]*t(?P<task>\d+)[_\-\s]*user[_\s]*(?P<user>\d+)_cleaned\.csv$',
    re.I
)

# ===== 1) Read labels and normalize website =====
labels = pd.read_excel(LABELS_PATH, sheet_name=LABEL_SHEET)
needed = {'user','website','task_id','label'}
if not needed.issubset(set(labels.columns)):
    raise ValueError(f"Excel must contain columns: {needed}")

labels['website_norm'] = labels['website'].map(norm_site)
labels['user'] = labels['user'].astype(int)
labels['task_id'] = labels['task_id'].astype(int)
print(f"[labels] rows={len(labels)}  dist={labels['label'].value_counts().to_dict()}")

# ===== 2) Scan cleaned files only =====
rows = []
for fp in sorted(glob.glob(os.path.join(CLEAN_DIR, "*_cleaned.csv"))):
    fn = os.path.basename(fp)
    m = PATTERN.search(fn)
    if not m:
        # اگر الگو متفاوت بود، پیام بده تا regex را اصلاح کنیم
        print(f"[WARN] cannot parse cleaned filename: {fn}")
        continue
    site = norm_site(m.group('site'))
    task = int(m.group('task'))
    user = int(m.group('user'))
    rows.append({"filename": fn, "path": fp, "website_norm": site, "task_id": task, "user": user})

files_df = pd.DataFrame(rows)
if files_df.empty:
    print("[ERROR] no *_cleaned.csv parsed. Check CLEAN_DIR or filename pattern.")
    sys.exit(1)
print(f"[cleaned_files] parsed={len(files_df)}")

# ===== 3) Join with labels (exact key) =====
merged = files_df.merge(
    labels[['user','task_id','website_norm','label']],
    on=['user','task_id','website_norm'],
    how='outer',
    indicator=True
)

print(merged['_merge'].value_counts())
if (merged['_merge'] != 'both').any():
    left_only  = merged.loc[merged['_merge']=='left_only',  ['filename','user','task_id','website_norm']]
    right_only = merged.loc[merged['_merge']=='right_only', ['user','task_id','website_norm']]
    if not left_only.empty:
        print("\n[Unmatched cleaned files] (label not found):")
        print(left_only.to_string(index=False))
    if not right_only.empty:
        print("\n[Unmatched label rows] (cleaned file not found):")
        print(right_only.to_string(index=False))
    sys.exit("[ERROR] unmatched rows exist. Fix names or regex and rerun.")

matched = merged[merged['_merge']=='both'].drop(columns=['_merge']).copy()
matched = matched.sort_values(['user','website_norm','task_id']).reset_index(drop=True)

# مانيفست پایدار
manifest_path = os.path.join(OUT_DIR, "manifest_clean.csv")
matched.to_csv(manifest_path, index=False, encoding="utf-8")
print(f"[manifest] saved → {manifest_path}  rows={len(matched)}")

# ===== 4) Build X/Y from cleaned files =====
X_list, Y_list = [], []

bad = []
for fp in matched['path']:
    df = pd.read_csv(fp)
    # اطمینان از وجود و ترتیب ستون‌ها
    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        bad.append((os.path.basename(fp), f"missing cols: {missing}"))
        continue
    arr = df[ALL_COLS].to_numpy(dtype=np.float32)
    if arr.shape != (N_EXPECTED, len(ALL_COLS)):
        bad.append((os.path.basename(fp), f"bad shape: {arr.shape}"))
        continue
    X_list.append(arr)

Y_list = matched['label'].astype(np.int64).to_list()

if bad:
    print("[ERROR] problems in cleaned files:")
    for name, msg in bad:
        print("  -", name, "→", msg)
    sys.exit(1)

X = np.stack(X_list, axis=0)  # (N, 1500, 16)
y = np.array(Y_list, dtype=np.int64)

X_path = os.path.join(OUT_DIR, "X_clean.npy")
Y_path = os.path.join(OUT_DIR, "Y_clean.npy")
np.save(X_path, X)
np.save(Y_path, y)

print(f"[OK] Saved X_clean.npy → {X_path}  shape={X.shape}, dtype={X.dtype}")
print(f"[OK] Saved Y_clean.npy → {Y_path}  shape={y.shape},  dtype={y.dtype}")
print(f"[Check] label balance: {dict(pd.Series(y).value_counts())}")
