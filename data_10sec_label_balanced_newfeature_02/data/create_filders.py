import os

DATA_ROOT = r"C:\Users\sepideh\PycharmProjects\Thesis\data_10sec_label_balanced_newfeature_02"
model_name = "cnn_v1"   # ← هر اسمی خواستی

models_dir  = os.path.join(DATA_ROOT, "models", model_name)
results_dir = os.path.join(DATA_ROOT, "results", model_name)
ckpt_dir    = os.path.join(models_dir, "checkpoints")

for d in [models_dir, results_dir, ckpt_dir]:
    os.makedirs(d, exist_ok=True)

print("Models →", models_dir)
print("Results →", results_dir)
print("Checkpoints →", ckpt_dir)
