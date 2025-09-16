import os, re
import numpy as np
import pandas as pd

# ================== Paths (hard-coded) ==================
DATA_ROOT  = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature_02\data"
FEAT_DIR   = os.path.join(DATA_ROOT, "gaze_features_cleaned")   # ورودی: فایل‌های *_cleaned.csv
MASK_DIR   = os.path.join(DATA_ROOT, "gaze_features_masks")     # اختیاری: ماسک ردیفی
LABEL_PATH = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature_02\data\label_balanced.xlsx"
OUT_DIR    = os.path.join(DATA_ROOT, "dataset_10s_16f_scaled")
os.makedirs(OUT_DIR, exist_ok=True)

# ================== Settings ==================
BASE  = ['FPOGX','FPOGY','LPD','RPD','FPOGD','BPOGX','BPOGY','BKDUR']
DELTA = [f'delta_{c}' for c in BASE]
ALL   = BASE + DELTA
TARGET_LEN = 1500
NFEAT = 16
EPS = 1e-8

# Split ratios (by unique users)
TEST_FRAC = 0.20
VAL_FRAC  = 0.20
SEED = 42

# Scaling: "standard" (mean/std) یا "minmax"
SCALE_TYPE = "standard"

# ================== Helpers ==================
def parse_filename(fname: str):
    """
    Expected: <site>_t<task>_user<space?> <id>_(features|cleaned).csv
    مثال: 'Princeton_T1_User 12_cleaned.csv' یا 'caltech_t2_user12_cleaned.csv'
    """
    stem = os.path.splitext(fname)[0]
    m = re.match(r'^(?P<site>[A-Za-z0-9]+)_t(?P<task>\d+)_user\s*(?P<user>\d+)_(?:features|cleaned)$',
                 stem, flags=re.I)
    if not m:
        return None
    return m.group('site').strip().lower(), int(m.group('task')), int(m.group('user'))

def load_one_csv(path):
    df = pd.read_csv(path)
    miss = set(ALL) - set(df.columns)
    if miss:
        raise ValueError(f"{os.path.basename(path)} missing columns: {sorted(miss)}")
    df = df[ALL].astype('float32')
    if df.shape != (TARGET_LEN, NFEAT):
        raise ValueError(f"{os.path.basename(path)} bad shape {df.shape}, expected {(TARGET_LEN, NFEAT)}")
    return df

def read_labels(label_path: str):
    if label_path.lower().endswith((".xlsx",".xls")):
        labels = pd.read_excel(label_path)
    else:
        labels = pd.read_csv(label_path)

    # نرمال‌سازی نام ستون‌ها
    cols_lower = {c.lower(): c for c in labels.columns}
    need = {"website","task_id","user","label"}
    if not need.issubset(set(cols_lower.keys())):
        raise ValueError(f"Labels must have columns {need}, got {list(labels.columns)}")

    labels = labels.rename(columns={
        cols_lower['website']: 'website',
        cols_lower['task_id']: 'task_id',
        cols_lower['user']:    'user',
        cols_lower['label']:   'label',
    })
    labels['website'] = labels['website'].astype(str).str.strip().str.lower()
    labels['task_id'] = labels['task_id'].astype(int)
    labels['user']    = labels['user'].astype(int)

    def _to01(v):
        if isinstance(v, str):
            t = v.strip().lower()
            if t in ('easy','1','true','yes','ez'): return 1
            if t in ('not easy','hard','0','false','no','ne','not_easy','not-easy'): return 0
        return int(v)
    labels['label'] = labels['label'].apply(_to01).astype(int)
    return labels

# ================== 1) Read labels ==================
labels = read_labels(LABEL_PATH)

# ================== 2) Collect cleaned files & map ==================
files = sorted([f for f in os.listdir(FEAT_DIR) if f.endswith("_cleaned.csv")])
if not files:
    raise RuntimeError(f"No *_cleaned.csv found in {FEAT_DIR}")

X_list, Y_list, G_list, M_list, meta_rows = [], [], [], [], []
skipped, dup_warn = 0, 0

for fname in files:
    parsed = parse_filename(fname)
    if not parsed:
        print(f"[SKIP] Unparsable name: {fname}")
        skipped += 1
        continue
    site, task, user = parsed
    hit = labels[(labels['website']==site) & (labels['task_id']==task) & (labels['user']==user)]
    if hit.empty:
        print(f"[SKIP] No label for {fname} ({site}, t{task}, u{user})")
        skipped += 1
        continue
    if len(hit) > 1:
        print(f"[WARN] Duplicate rows in labels for {site}-t{task}-u{user}; taking first.")
        dup_warn += 1
    y = int(hit['label'].iloc[0])

    df = load_one_csv(os.path.join(FEAT_DIR, fname))
    X_list.append(df.values)  # (1500,16)
    Y_list.append(y)
    G_list.append(user)

    # row mask (optional)
    mpath = os.path.join(MASK_DIR, fname.replace("_cleaned.csv", "_mask.csv"))
    if os.path.exists(mpath):
        m = pd.read_csv(mpath)['interp_or_padded'].astype('uint8').values
    else:
        m = np.zeros(TARGET_LEN, dtype='uint8')
    M_list.append(m)

    meta_rows.append({
        "idx": len(X_list)-1, "filename": fname,
        "website": site, "task": task, "user": user, "label": y
    })

if not X_list:
    raise RuntimeError("No matched samples. Check LABEL_PATH or filenames.")

X = np.stack(X_list).astype('float32')   # (N,1500,16)
Y = np.array(Y_list, dtype=np.int64)     # (N,)
G = np.array(G_list, dtype=np.int64)     # (N,)
M = np.stack(M_list).astype('uint8')     # (N,1500)
meta = pd.DataFrame(meta_rows)

print(f"[INFO] Collected: X{X.shape}, Y{Y.shape}, G{G.shape} | skipped={skipped}, dup_warn={dup_warn}")
print(f"[INFO] Class balance (Y==1): {float((Y==1).mean()):.3f}")

# ================== 3) Group split by user ==================
rng = np.random.default_rng(SEED)
users = np.unique(G)
n_users = len(users)
n_test = max(1, int(round(TEST_FRAC * n_users)))
n_val  = max(1, int(round(VAL_FRAC  * n_users)))

test_users = rng.choice(users, size=n_test, replace=False)
remain = np.setdiff1d(users, test_users, assume_unique=True)
val_users = rng.choice(remain, size=n_val, replace=False)
train_users = np.setdiff1d(remain, val_users, assume_unique=True)

train_idx = np.isin(G, train_users)
val_idx   = np.isin(G, val_users)
test_idx  = np.isin(G, test_users)

print(f"[SPLIT] users: total={n_users} | train={len(train_users)} val={len(val_users)} test={len(test_users)}")
print(f"[SPLIT] samples: train={train_idx.sum()} val={val_idx.sum()} test={test_idx.sum()}")

# ================== 4) Fit scaler on TRAIN only ==================
X_train_flat = X[train_idx].reshape(-1, NFEAT)

if SCALE_TYPE == "standard":
    mean = X_train_flat.mean(axis=0)
    std  = X_train_flat.std(axis=0, ddof=0)
    std  = np.where(std < EPS, 1.0, std)
    def apply_scale(Z): return (Z - mean) / std
    scaler = {"type":"standard","mean":mean,"std":std}
    scaler_name = "scaler_standard_16f_trainonly.npz"
elif SCALE_TYPE == "minmax":
    mn = X_train_flat.min(axis=0)
    mx = X_train_flat.max(axis=0)
    span = np.where((mx - mn) < EPS, 1.0, (mx - mn))
    def apply_scale(Z): return (Z - mn) / span
    scaler = {"type":"minmax","min":mn,"max":mx}
    scaler_name = "scaler_minmax_16f_trainonly.npz"
else:
    raise ValueError("Unknown SCALE_TYPE")

X_scaled = apply_scale(X.astype('float32')).astype('float32')

# ================== 5) Save outputs ==================
np.save(os.path.join(OUT_DIR, "X_scaled_10s_16f.npy"), X_scaled)
np.save(os.path.join(OUT_DIR, "Y.npy"), Y)
np.save(os.path.join(OUT_DIR, "G.npy"), G)
np.save(os.path.join(OUT_DIR, "M_mask.npy"), M)

np.save(os.path.join(OUT_DIR, "idx_train.npy"), train_idx.astype('uint8'))
np.save(os.path.join(OUT_DIR, "idx_val.npy"),   val_idx.astype('uint8'))
np.save(os.path.join(OUT_DIR, "idx_test.npy"),  test_idx.astype('uint8'))

np.savez(os.path.join(OUT_DIR, scaler_name), **scaler)

meta['split'] = np.where(train_idx, 'train', np.where(val_idx, 'val', 'test'))
meta.to_csv(os.path.join(OUT_DIR, "meta_10s_16f.csv"), index=False)

print(f"[DONE] Saved dataset & scaler to: {OUT_DIR}")
