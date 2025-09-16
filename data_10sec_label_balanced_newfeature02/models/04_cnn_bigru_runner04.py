# models/04_cnn_bigru_runner04.py
import torch
import torch.nn as nn

# هایپرپارامترها (Runner 04 وزن‌دهی L2 را از اینجا می‌خوانَد)
CONFIG = {
    "epochs": 50,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 10,
    "weight_decay": 1e-4,   # ← توسط R04 به Adam پاس می‌شود
    "dropout": 0.30,
    "rnn_hidden": 96,
    "rnn_layers": 2,        # اگر 1 کنی، dropout داخلی GRU غیرفعال می‌شود (رفتار PyTorch)
    "rnn_dropout": 0.20,    # فقط وقتی rnn_layers>1 اعمال می‌شود
    "bidirectional": True
}

class CNN_BiGRU(nn.Module):
    """
    Frontend CNN برای استخراج الگوهای محلی + BiGRU برای وابستگی‌های بلندمدت.
    ورودی: (B, C=16, T=1500) → خروجی logits با شکل (B, 1)
    """
    def __init__(self, in_ch=16, cfg: dict = None):
        super().__init__()
        cfg = cfg or CONFIG
        drop = float(cfg.get("dropout", 0.30))
        hid  = int(cfg.get("rnn_hidden", 96))
        nl   = int(cfg.get("rnn_layers", 2))
        rnn_do = float(cfg.get("rnn_dropout", 0.20))
        bi   = bool(cfg.get("bidirectional", True))

        # ---- CNN blocks ----
        self.block1 = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)             # 1500 -> 750
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)             # 750 -> 375
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
            # طول توالی همینجا 375 می‌مونه
        )

        # ---- BiGRU ----
        self.gru = nn.GRU(
            input_size=128, hidden_size=hid,
            num_layers=nl, batch_first=True,
            bidirectional=bi, dropout=(rnn_do if nl > 1 else 0.0)
        )
        rnn_out_ch = hid * (2 if bi else 1)  # چون bidirectional=True معمولاً 2*hidden می‌شود

        # ---- Head (temporal pooling + MLP) ----
        # از هر دو Pooling (mean/max) روی خروجی زمان‌دار GRU استفاده می‌کنیم
        head_in = rnn_out_ch * 2  # concat(mean, max)
        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 1)  # logits
        )

    def forward(self, x):  # x: (B, C, T)
        # CNN
        x = self.block1(x)                  # (B, 32, 750)
        x = self.block2(x)                  # (B, 64, 375)
        x = self.block3(x)                  # (B, 128, 375)

        # برای GRU نیاز به (B, T, F) داریم
        x = x.transpose(1, 2)               # (B, 375, 128)

        # GRU
        y, _ = self.gru(x)                  # (B, 375, rnn_out_ch)

        # Temporal pooling
        y_mean = y.mean(dim=1)              # (B, rnn_out_ch)
        y_max, _ = y.max(dim=1)             # (B, rnn_out_ch)
        z = torch.cat([y_mean, y_max], dim=1)  # (B, 2*rnn_out_ch)

        # Head → logits
        logits = self.head(z)               # (B,1)
        return logits

def build_model(input_channels=16, seq_len=1500):
    # seq_len نیازی به استفاده ندارد ولی امضا را با رانر سازگار نگه می‌داریم
    return CNN_BiGRU(in_ch=input_channels, cfg=CONFIG)

# ---- اجرای مستقیم فایل مدل (PyCharm Run) ----
if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)

    # 🔧 رانر 04 (دقیقاً مطابق نام فایلی که ساختی):
    import runners.R04_runner_acc_constrained_wd as runner
    # یا اگر با حروف کوچک ذخیره کرده‌ای:
    # import runners.runner_04_acc_constrained_wd as runner

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    runner.run_with_model_path(
        __file__,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed
    )
