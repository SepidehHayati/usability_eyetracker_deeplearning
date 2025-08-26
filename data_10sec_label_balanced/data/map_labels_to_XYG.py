import os
import numpy as np
import pandas as pd

# ===== مسیرها =====
feature_dir = r"..\data\gaze_features"
label_path  = r"..\data\label_balanced.xlsx"
out_dir     = r"..\data"

os.makedirs(out_dir, exist_ok=True)

# ===== خواندن لیبل‌ها =====
labels_df = pd.read_excel(label_path)
# انتظار ستون‌ها: website | task_id | user | label

X_list, Y_list, G_list, meta_rows = [], [], [], []

for filename in sorted(os.listdir(feature_dir)):
    if not filename.endswith(".csv"):
        continue

    base = filename[:-4]  # remove .csv
    parts = base.split("_")  # مثال: caltech_t2_user5_features
    # انتظار فرمت: website, t{task}, user{num}, features
    if len(parts) != 4 or parts[-1].lower() != "features":
        print(f"[WARN] Invalid filename format: {filename}")
        continue

    website = parts[0].lower().strip()
    task    = int(parts[1][1:])   # t2 -> 2
    user    = int(parts[2][4:])   # user5 -> 5

    # مپ‌کردن لیبل
    match = labels_df[
        (labels_df["website"].str.lower().str.strip() == website) &
        (labels_df["task_id"] == task) &
        (labels_df["user"] == user)
    ]

    if match.empty:
        print(f"[WARN] No label for {filename}")
        continue

    label = int(match["label"].values[0])

    # خواندن ویژگی‌ها
    df = pd.read_csv(os.path.join(feature_dir, filename))
    X_list.append(df.values)  # (<=1500, 4) — فقط delta_* ها
    Y_list.append(label)
    G_list.append(user)       # گروه = user id

    meta_rows.append({
        "filename": filename,
        "website": website,
        "task": task,
        "user": user,
        "label": label,
        "n_rows": len(df)
    })

# آرایه‌ها
X = np.array(X_list, dtype=np.float32)  # (N, T, 4)
Y = np.array(Y_list, dtype=np.int64)    # (N,)
G = np.array(G_list, dtype=np.int64)    # (N,)  <- user id

print("X:", X.shape, "Y:", Y.shape, "G:", G.shape)

# ذخیره
np.save(os.path.join(out_dir, "X2.npy"), X)
np.save(os.path.join(out_dir, "Y2.npy"), Y)
np.save(os.path.join(out_dir, "G1.npy"), G)

# ذخیره meta برای کنترل
pd.DataFrame(meta_rows).to_csv(os.path.join(out_dir, "meta.csv"), index=False)
print(f"[OK] Saved X2.npy, Y2.npy, G1.npy, meta.csv in {out_dir}")
