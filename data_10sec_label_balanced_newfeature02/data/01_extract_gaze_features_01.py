# -*- coding: utf-8 -*-
"""
استخراج 8 ویژگی پایه + دلتا (جمعاً 16 کانال) از فایل‌های gaze
- همه فایل‌ها 1500 ردیفی هستند (بر اساس گفته شما)، پس پد لازم نداریم.
- هشدار قدیمی pandas حذف شده: از ffill/bfill جدید استفاده می‌کنیم.
"""

import os
import pandas as pd
import numpy as np

# === مسیرها را مطابق پروژه‌ات تنظیم کن ===
INPUT_FOLDER  = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data\gaze_files"
OUTPUT_FOLDER = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature02\data\gaze_features"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ۸ ویژگی پایه
BASE_FEATURES = ['FPOGX', 'FPOGY', 'LPD', 'RPD', 'FPOGD', 'BPOGX', 'BPOGY', 'BKDUR']
NUM_ROWS = 1500       # همه فایل‌ها باید همین طول را داشته باشند
CLIP_FPOG_01 = True  # اگر مختصات نگاه نرمال شده‌اند، True بگذار

for filename in sorted(os.listdir(INPUT_FOLDER)):
    if not filename.lower().endswith("_gaze.csv"):
        continue

    in_path  = os.path.join(INPUT_FOLDER, filename)
    out_path = os.path.join(OUTPUT_FOLDER, filename.replace("_gaze.csv", "_features.csv"))

    try:
        # فقط ستون‌های لازم را بخوان
        df = pd.read_csv(in_path, usecols=BASE_FEATURES, low_memory=False)

        # کنترل طول
        if len(df) != NUM_ROWS:
            raise ValueError(f"{filename}: expected {NUM_ROWS} rows, got {len(df)}")

        # به عدد تبدیل و رفع NA
        for c in BASE_FEATURES:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.ffill().bfill()

        # (اختیاری) کلیپ مختصات نگاه اگر در بازه [0,1] هستند
        if CLIP_FPOG_01:
            if 'FPOGX' in df.columns: df['FPOGX'] = df['FPOGX'].clip(0, 1)
            if 'FPOGY' in df.columns: df['FPOGY'] = df['FPOGY'].clip(0, 1)

        # دلتاها
        for f in BASE_FEATURES:
            df[f'delta_{f}'] = df[f].diff().fillna(0.0)

        # خروجی با ترتیب ثابت و نوع float32
        all_cols = BASE_FEATURES + [f'delta_{f}' for f in BASE_FEATURES]
        out = df[all_cols].astype(np.float32)

        out.to_csv(out_path, index=False)
        print(f"OK: {filename} -> {out_path}  shape={out.shape}")

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")

print("DONE: feature extraction.")
