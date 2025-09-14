import os
import pandas as pd

# پوشه‌های ورودی و خروجی
input_folder = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data\gaze_features"
output_folder = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data\gaze_features_normalized"
os.makedirs(output_folder, exist_ok=True)

# همه ویژگی‌ها (8 خام + 8 دلتا)
base_features = ['FPOGX', 'FPOGY', 'LPD', 'RPD', 'FPOGD', 'BPOGX', 'BPOGY', 'BKDUR']
delta_features = [f'delta_{f}' for f in base_features]
all_features = base_features + delta_features  # 16 ویژگی

# پیمایش تمام فایل‌های پردازش‌شده
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith("_features.csv"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace("_features.csv", "_normalized.csv"))

        try:
            # خواندن فایل
            df = pd.read_csv(input_path)

            # نرمال‌سازی برای هر ویژگی
            df_normalized = df.copy()
            for feature in all_features:
                min_val = df[feature].min()
                max_val = df[feature].max()
                if max_val > min_val:  # جلوگیری از تقسیم بر صفر
                    df_normalized[feature] = (df[feature] - min_val) / (max_val - min_val)
                else:
                    df_normalized[feature] = 0  # اگه همه مقادیر یکسان باشن

            # ذخیره فایل نرمال‌شده با همون اسم قبلی
            df_normalized.to_csv(output_path, index=False)
            print(f"Normalized: {filename} → {output_path}")

        except Exception as e:
            print(f"Error in file {filename}: {e}")

print("[OK] Normalization completed for all files!")