import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# مسیر پوشه و فایل‌ها
feature_dir = r"C:\Users\spdhy\PycharmProjects\gaze_pytorch_project\data_10sec_label_balanced\data\gaze_features"
label_path = r"C:\Users\spdhy\PycharmProjects\gaze_pytorch_project\data_10sec_label_balanced\data\label_balanced.xlsx"

# خواندن فایل لیبل
labels_df = pd.read_excel(label_path)

X_list = []
Y_list = []

# پیمایش فایل‌های ویژگی
for filename in os.listdir(feature_dir):
    if filename.endswith(".csv"):
        try:
            # استخراج اطلاعات از نام فایل (مثال: caltech_t2_user5_features.csv)
            base = filename.replace(".csv", "")
            parts = base.split("_")

            if len(parts) == 4 and parts[-1].lower() == "features":
                website = parts[0].lower().strip()
                task = int(parts[1][1:])          # "t2" → 2
                user = int(parts[2][4:])          # "user5" → 5
            else:
                print(f" Invalid filename format: {filename}")
                continue

            # پیدا کردن لیبل مربوطه از فایل لیبل
            match = labels_df[
                (labels_df["website"].str.lower().str.strip() == website) &
                (labels_df["task_id"] == task) &
                (labels_df["user"] == user)
            ]

            if match.empty:
                print(f" No label found for {filename}")
                continue

            label = match["label"].values[0]

            # بارگیری داده ویژگی
            df = pd.read_csv(os.path.join(feature_dir, filename))
            X_list.append(df.values)
            Y_list.append(label)

        except Exception as e:
            print(f" Error in {filename}: {e}")

# تبدیل لیست‌ها به numpy array
X = np.array(X_list)
Y = np.array(Y_list)

print(" X shape:", X.shape)  # (samples, time steps, features)
print(" Y shape:", Y.shape)  # (samples,)

# اگر لیبل‌ها عددی هستند:
# Y_cat = to_categorical(Y, num_classes=2)

# می‌تونی بعدش اینو استفاده کنی برای آموزش:
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y_cat, test_size=0.2, random_state=42, stratify=Y)
np.save("C:\Users\spdhy\PycharmProjects\gaze_pytorch_project\data_10sec_label_balanced\data\X.npy", X)
np.save("C:\Users\spdhy\PycharmProjects\gaze_pytorch_project\data_10sec_label_balanced\data\Y.npy", Y)
