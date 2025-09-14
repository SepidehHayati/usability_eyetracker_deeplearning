import os
import pandas as pd

# پوشه‌های ورودی و خروجی
input_folder = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data\gaze_files"
output_folder = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data\gaze_features"
os.makedirs(output_folder, exist_ok=True)

# ۸ ویژگی پایه و تعداد ردیف‌هایی که می‌خوایم نگه‌داریم
base_features = ['FPOGX', 'FPOGY', 'LPD', 'RPD', 'FPOGD', 'BPOGX', 'BPOGY', 'BKDUR']
num_rows = 1500

# پیمایش تمام فایل‌های gaze
for filename in os.listdir(input_folder):
    if filename.endswith("_gaze.csv"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace("_gaze.csv", "_features.csv"))

        try:
            df = pd.read_csv(input_path)

            # فقط 1500 ردیف اول
            df_limited = df.iloc[:num_rows].copy()

            # محاسبه دلتا برای هر ویژگی
            for feature in base_features:
                df_limited[f'delta_{feature}'] = df_limited[feature].diff().fillna(0)

            # ذخیره هم ویژگی‌های خام و هم دلتاها
            delta_cols = [f'delta_{f}' for f in base_features]
            df_limited[base_features + delta_cols].to_csv(output_path, index=False)

            print(f"Processed: {filename} → {output_path} ({df_limited.shape[0]} rows, {df_limited.shape[1]} cols)")

        except Exception as e:
            print(f"Error in file {filename}: {e}")