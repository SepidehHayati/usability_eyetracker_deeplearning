import os
import re
import numpy as np
import pandas as pd

# مسیرها
feature_dir = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data\gaze_features_normalized"
label_path = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data\label_balanced.xlsx"
out_dir = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data"
os.makedirs(out_dir, exist_ok=True)

# خواندن لیبل‌ها
labels_df = pd.read_excel(label_path)
print("Columns in labels_df:", labels_df.columns.tolist())  # دیباگ برای چک کردن ستون‌ها

X_list, Y_list, G_list, meta_rows = [], [], [], []

def parse_filename(fname):
    """
    مثال: 'caltech_t1_user7_normalized.csv' یا 'stanford_t3_user 4_normalized.csv'
    خروجی: website(پایین‌حرف), task(int), user(int)
    """
    base = fname[:-14]  # remove _normalized.csv
    parts = base.split("_")
    if len(parts) < 3:
        print(f"[WARN] Invalid filename format (too few parts): {fname}")
        return None, None, None

    website = parts[0].strip().lower()
    task_str = parts[1]
    m_task = re.search(r"\d+", task_str)
    if not m_task:
        print(f"[WARN] Cannot parse task from: {fname}")
        return None, None, None
    task = int(m_task.group())

    user_str = parts[2]  # ممکنه فاصله داشته باشه (مثل user 4)
    m_user = re.search(r"\d+", user_str)
    if not m_user:
        print(f"[WARN] Cannot parse user from: {fname} (user_str={user_str})")
        return None, None, None
    user = int(m_user.group())

    print(f"Parsed: {fname} -> website={website}, task={task}, user={user}")  # دیباگ
    return website, task, user

# پیمایش فایل‌های ویژگی
print(f"Looking for files in: {feature_dir}")
for filename in sorted(os.listdir(feature_dir)):
    print(f"Checking file: {filename}")  # دیباگ برای چک کردن فایل‌ها
    if not filename.endswith("_normalized.csv"):
        continue

    try:
        website, task, user = parse_filename(filename)
        if website is None or task is None or user is None:
            continue

        # مپ به لیبل
        match = labels_df[
            (labels_df["website"].str.lower().str.strip() == website) &
            (labels_df["task_id"] == task) &
            (labels_df["user"] == user)
        ]
        if match.empty:
            print(f"[WARN] No label for {filename} (website={website}, task={task}, user={user})")
            continue

        label = int(match["label"].values[0])

        # بارگذاری همه‌ی ستون‌ها (الان 16 ستونه: 8 خام + 8 دلتا)
        df = pd.read_csv(os.path.join(feature_dir, filename))

        # اطمینان از شکل 1500×16
        if df.shape != (1500, 16):
            print(f"[WARN] Invalid shape for {filename}: {df.shape}")
            continue

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
X = np.array(X_list, dtype=np.float32)   # (N, 1500, 16)
Y = np.array(Y_list, dtype=np.int64)     # (N,)
G = np.array(G_list, dtype=np.int64)     # (N,)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("G shape:", G.shape)
print("Class balance (Y==1):", np.mean(Y==1) if len(Y) > 0 else "N/A (no data)")

# ذخیره
np.save(os.path.join(out_dir, "X16_normalized.npy"), X)
np.save(os.path.join(out_dir, "Y16_normalized.npy"), Y)
np.save(os.path.join(out_dir, "G16_normalized.npy"), G)

pd.DataFrame(meta_rows).to_csv(os.path.join(out_dir, "meta_features_16cols_normalized.csv"), index=False)

print("[OK] Saved: X16_normalized.npy, Y16_normalized.npy, G16_normalized.npy, meta_features_16cols_normalized.csv")