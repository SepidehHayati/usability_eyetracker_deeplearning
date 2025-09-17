# models/33_bilstm_runner06.py
# BiLSTM (با downsample زمانی + attention) سازگار با رانرهای R06/R08
import os, sys, argparse, importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------ تنظیمات آموزش (توسط رانر خوانده می‌شود) ------------
CONFIG = {
    "epochs":   35,
    "batch_size": 32,
    "lr":       1e-3,
    "patience": 8,
    # اگر رانری داری که weight decay را می‌خواند:
    # "weight_decay": 1e-4,
}

# ------------ مدل ------------
class BiLSTMWithAttn(nn.Module):
    """
    ورودی: x با شکل (B, C, T)
    خروجی: logits با شکل (B, 1)
    - ابتدا با Conv1d کمی T را کم می‌کنیم تا BiLSTM سبک‌تر شود (1500 -> 375)
    - سپس BiLSTM دو لایه‌ی دوسویه
    - attention pooling روی خروجی زمانی
    """
    def __init__(self, in_ch=16, ds_channels=64, ds_stride_total=4, lstm_hidden=128, lstm_layers=2, attn_dim=128, dropout=0.3):
        super().__init__()
        # دو کانولوشن با stride=2 → مجموعاً 4x downsample زمانی: 1500 → 375
        self.down = nn.Sequential(
            nn.Conv1d(in_ch, ds_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(ds_channels, ds_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=ds_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if lstm_layers > 1 else 0.0,
        )
        # attention: (B,T,2H) -> (B,T,1)
        self.attn = nn.Sequential(
            nn.Linear(2*lstm_hidden, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )
        # head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2*lstm_hidden, 1)  # logits
        )

    def forward(self, x):  # x: (B, C, T)
        # downsample زمانی
        z = self.down(x)              # (B, C', T')
        z = z.transpose(1, 2)         # (B, T', C') برای LSTM (batch_first=True)
        y, _ = self.lstm(z)           # (B, T', 2H)

        # attention pooling
        scores = self.attn(y)         # (B, T', 1)
        alpha = torch.softmax(scores, dim=1)  # (B, T', 1)
        pooled = (alpha * y).sum(dim=1)       # (B, 2H)

        logits = self.head(pooled)    # (B, 1)
        return logits


def build_model(input_channels=16, seq_len=1500):
    # اگر خواستی هایپرها را سریع تست کنی، اینجا تغییر بده
    return BiLSTMWithAttn(
        in_ch=input_channels,
        ds_channels=64,
        lstm_hidden=128,
        lstm_layers=2,
        attn_dim=128,
        dropout=0.3,
    )

# ------------ لودر رانر (با import-by-path) ------------
def _load_runner_module():
    """
    رانر را از مسیر فایل لود می‌کند.
    ترتیب candidates تعیین‌کنندهٔ ترجیح است.
    اگر می‌خواهی حتماً R08 (accuracy-first) استفاده شود، آن دو خط را بیا بالا.
    """
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        # 1) R06 نرم با fallback (اگر موجود است)
        os.path.join(ROOT, "runners", "R08_runner_acc_first.py"),
        os.path.join(ROOT, "R08_runner_acc_first.py"),

        # 2) R08 accuracy-first (اگر موجود است)
        os.path.join(ROOT, "runners", "R08_runner_acc_first.py"),
        os.path.join(ROOT, "R08_runner_acc_first.py"),
    ]

    for path in candidates:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("runner", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"[INFO] runner loaded from: {path}")
            return mod

    # دیباگ دوستانه:
    roots = []
    try:
        roots = os.listdir(ROOT)
    except Exception:
        pass
    runners_dir = os.path.join(ROOT, "runners")
    runners_list = []
    if os.path.isdir(runners_dir):
        try:
            runners_list = os.listdir(runners_dir)
        except Exception:
            pass

    raise FileNotFoundError(
        "هیچ رانری پیدا نشد. نام/مسیر را چک کن.\n"
        + "\n".join(f" - {p}" for p in candidates)
        + f"\nفایل‌های ROOT: {roots}\n"
        + f"فایل‌های runners/: {runners_list}"
    )

# ------------ اجرای مستقیم فایل مدل ------------
if __name__ == "__main__":
    # برای دسترسی به ریشهٔ پروژه و runners
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)

    runner = _load_runner_module()

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # نوت: اگر می‌خواهی با رانر دیگری اجرا شود، فقط ترتیب کاندیدها را در _load_runner_module عوض کن.
    runner.run_with_model_path(
        __file__,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed
    )
