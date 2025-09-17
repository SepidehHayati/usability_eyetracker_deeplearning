# models/13_zz_ensemble_launcher.py
import os, sys

# پروژه: .../data_10sec_label_balanced_newfeature02/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# رانر Ensemble را وارد می‌کنیم
import runners.R07_ensemble_from_results as ens

# 🔧 تنظیمات Ensemble (در صورت نیاز همینجا ادیت کن)
MODELS      = ["11_resnet1d_runner06.py", "10_cnn_bn_runner06.py"]  # نام پوشه‌های نتایج در results/
WEIGHTS     = [1, 2]      # وزن هر مدل در میانگین احتمال‌ها (مثلاً [2,1] یعنی ResNet وزن دو برابر)
MIN_PREC    = 0.82        # قید دقت مثبت (Precision)
MIN_F1      = 0.75        # قید F1
RELAXED_F1  = 0.70        # قید نرم برای fallback

if __name__ == "__main__":
    # چون ens.main از argparse استفاده می‌کنه، آرگومان‌ها رو دستی تزریق می‌کنیم:
    sys.argv = ["_"]
    if MODELS:
        sys.argv += ["--models", *MODELS]
    if WEIGHTS:
        sys.argv += ["--weights", *map(str, WEIGHTS)]
    sys.argv += ["--min_prec", str(MIN_PREC),
                 "--min_f1",   str(MIN_F1),
                 "--relaxed_f1", str(RELAXED_F1)]
    ens.main()
