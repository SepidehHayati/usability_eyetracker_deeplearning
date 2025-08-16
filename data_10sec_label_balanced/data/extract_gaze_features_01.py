import os
import pandas as pd

# Input and output folders
# پوشه‌های ورودی و خروجی
input_folder = r"C:\Users\spdhy\PycharmProjects\gaze_pytorch_project\data\gaze_files"
output_folder = r"C:\Users\spdhy\PycharmProjects\gaze_pytorch_project\data\gaze_features"
os.makedirs(output_folder, exist_ok=True)

# Helper function to find the time column
# تابع کمکی برای یافتن ستون زمان
def find_time_column(columns):
    for col in columns:
        if col.startswith("TIME("):
            return col
    raise ValueError("Time column not found!")

# Features to extract
# ویژگی‌هایی که باید استخراج شوند
features = ['FPOGX', 'FPOGY', 'LPD', 'RPD']
num_rows = 1500  # تعداد ثابت رکوردهایی که می‌خوایم نگه‌داریم

# Iterate through all gaze files
# پیمایش تمام فایل‌های gaze
for filename in os.listdir(input_folder):
    if filename.endswith("_gaze.csv"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace("_gaze.csv", "_features.csv"))

        try:
            df = pd.read_csv(input_path)

            # فقط 1500 ردیف اول رو نگه دار
            df_limited = df.iloc[:num_rows].copy()

            for feature in features:
                df_limited[f'delta_{feature}'] = df_limited[feature].diff().fillna(0)

            delta_df = df_limited[[f'delta_{f}' for f in features]]
            delta_df.to_csv(output_path, index=False)

            print(f"Processed: {filename} → {output_path} ({delta_df.shape[0]} rows)")

        except Exception as e:
            print(f"Error in file {filename}: {e}")
