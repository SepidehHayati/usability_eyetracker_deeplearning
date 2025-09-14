import os, shutil

base = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature_02\data"

# gaze_features
gf = os.path.join(base, "gaze_features")
gf_qc = os.path.join(gf, "_reports_qc"); gf_ext = os.path.join(gf, "_reports_extract")
os.makedirs(gf_qc, exist_ok=True); os.makedirs(gf_ext, exist_ok=True)

for name in ["features_qc_summary.csv","features_qc_deltas_outliers.csv","02a_qc_summary_10s_16f.csv","02b_qc_delta_outliers_10s_16f.csv"]:
    src = os.path.join(gf, name)
    if os.path.exists(src): shutil.move(src, os.path.join(gf_qc, os.path.basename(src)))

for name in ["features_meta.csv","01_features_meta_10s_16f.csv"]:
    src = os.path.join(gf, name)
    if os.path.exists(src): shutil.move(src, os.path.join(gf_ext, os.path.basename(src)))

# gaze_features_cleaned
gfc = os.path.join(base, "gaze_features_cleaned")
gfc_rep = os.path.join(gfc, "_reports_clean")
os.makedirs(gfc_rep, exist_ok=True)
for name in ["clean_meta_10s_16f.csv","03_clean_meta_10s_16f.csv"]:
    src = os.path.join(gfc, name)
    if os.path.exists(src): shutil.move(src, os.path.join(gfc_rep, os.path.basename(src)))

print("✅ old reports moved into _reports_* subfolders.")
