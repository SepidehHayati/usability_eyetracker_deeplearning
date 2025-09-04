import os
import re
import numpy as np
import pandas as pd

# مسیرها
feature_dir = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature\data\gaze_features"
label_path  = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature\data\label_balanced.xlsx"
out_dir     = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature\data"
os.makedirs(out_dir, exist_ok=True)

# خواندن لیبل‌ها
labels_df = pd.read_excel(label_path)
# انتظار ستون‌ها: website | task_id | user | label

X_list, Y_list, G_list, meta_rows = [], [], [], []

def parse_filename(fname):
    """
    مثال: 'Princeton_T1_User 12_features.csv' یا 'caltech_t2_user5_features.csv'
    خروجی: website(پایین‌حرف), task(int), user(int)
    """
    base = fname[:-4]  # remove .csv
    parts = base.split("_")
    if len(parts) < 4 or parts[-1].lower() != "features":
        raise ValueError(f"Invalid filename format: {fname}")

    website = parts[0].strip().lower()
    # عددِ task را از هر چیزی مثل 't1' یا 'T3' می‌گیریم
    task_str = parts[1]
    m_task = re.search(r"\d+", task_str)
    if not m_task:
        raise ValueError(f"Cannot parse task from: {fname}")
    task = int(m_task.group())

    # عددِ user را از هر چیزی مثل 'user5' یا 'User 12' می‌گیریم
    user_str = parts[2]
    m_user = re.search(r"\d+", user_str)
    if not m_user:
        raise ValueError(f"Cannot parse user from: {fname}")
    user = int(m_user.group())

    return website, task, user

# پیمایش فایل‌های ویژگی
for filename in sorted(os.listdir(feature_dir)):
    if not filename.endswith(".csv"):
        continue

    try:
        website, task, user = parse_filename(filename)

        # مپ به لیبل
        match = labels_df[
            (labels_df["website"].str.lower().str.strip() == website) &
            (labels_df["task_id"] == task) &
            (labels_df["user"] == user)
        ]
        if match.empty:
            print(f"[WARN] No label for {filename}")
            continue

        label = int(match["label"].values[0])

        # بارگذاری همه‌ی ستون‌های delta (الان 8 ستونه) — بدون دستکاری
        df = pd.read_csv(os.path.join(feature_dir, filename))

        # اطمینان از شکل 1500×8 (اگر تولید قبلی رعایت کرده باشه همین است)
        X_list.append(df.values.astype(np.float32))
        Y_list.append(label)
        G_list.append(user)

        meta_rows.append({
            "filename": filename,
            "website": website,
            "task": task,
            "user": user,
            "label": label,
            "n_rows": len(df),
            "n_cols": df.shape[1]
        })

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")

# تبدیل به آرایه
X = np.array(X_list, dtype=np.float32)   # (N, 1500, 8)
Y = np.array(Y_list, dtype=np.int64)     # (N,)
G = np.array(G_list, dtype=np.int64)     # (N,)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("G shape:", G.shape)
print("Class balance (Y==1):", np.mean(Y==1))

# ذخیره
np.save(os.path.join(out_dir, "X8.npy"), X)
np.save(os.path.join(out_dir, "Y8.npy"), Y)
np.save(os.path.join(out_dir, "G8.npy"), G)

pd.DataFrame(meta_rows).to_csv(os.path.join(out_dir, "meta_features_8cols.csv"), index=False)

print("[OK] Saved: X8.npy, Y8.npy, G8.npy, meta_features_8cols.csv")
