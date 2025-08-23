import os
import numpy as np
import pandas as pd

# مسیر پوشه‌ها
feature_dir = r"C:\Users\sepideh\PycharmProjects\Thesis\data_20sec_label01\data\gaze_features"
label_path = r"C:\Users\sepideh\PycharmProjects\Thesis\data_20sec_label_balanced/data/label_balanced.xlsx"


# خواندن فایل لیبل
labels_df = pd.read_excel(label_path)

X_list = []
Y_list = []

num_timesteps = 3000  # تعداد زمان‌های ثابت (20 ثانیه)
num_features = 4      # delta_FPOGX, delta_FPOGY, delta_LPD, delta_RPD

for filename in os.listdir(feature_dir):
    if filename.endswith(".csv"):
        try:
            base = filename.replace(".csv", "")
            parts = base.split("_")

            if len(parts) == 4 and parts[-1].lower() == "features":
                website = parts[0].lower().strip()
                task = int(parts[1][1:])          # "t2" → 2
                user = int(parts[2][4:])          # "user5" → 5
            else:
                print(f" Invalid filename format: {filename}")
                continue

            match = labels_df[
                (labels_df["website"].str.lower().str.strip() == website) &
                (labels_df["task_id"] == task) &
                (labels_df["user"] == user)
            ]

            if match.empty:
                print(f" No label found for {filename}")
                continue

            label = match["label"].values[0]
            df = pd.read_csv(os.path.join(feature_dir, filename)).values  # shape: (n, 4)

            # صفرپَد اگر کمتر از 3000 ردیف باشد
            if df.shape[0] < num_timesteps:
                pad_len = num_timesteps - df.shape[0]
                pad = np.zeros((pad_len, num_features))
                df_padded = np.vstack([df, pad])
            else:
                df_padded = df[:num_timesteps]  # اگر بیشتر بود، فقط 3000 ردیف نگه دار

            X_list.append(df_padded)
            Y_list.append(label)

        except Exception as e:
            print(f" Error in {filename}: {e}")

# تبدیل به آرایه numpy
X = np.array(X_list)  # shape: (108, 3000, 4)
Y = np.array(Y_list)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# ذخیره‌سازی
np.save("X.npy", X)
np.save("Y.npy", Y)
