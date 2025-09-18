# models/29_tcn_wide_runner06.py
# TCN (Temporal Convolutional Network) عریض با WeightNorm + Residual
# سازگار با رانر R10 (Balanced threshold + TempScaling + SWA)

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- پچ دائمی weight_norm (رفع خطای deepcopy و FutureWarning) ---
try:
    from torch.nn.utils.parametrizations import weight_norm  # PyTorch ≥ 2.x
except Exception:
    from torch.nn.utils import weight_norm                   # fallback برای نسخه‌های قدیمی

# هایپرپارامترهای پیش‌فرض مدل (رانر R10 این‌ها را merge می‌کند؛ با CLI هم قابل override هستند)
CONFIG = {
    "epochs": 45,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 10,
    "dropout": 0.20,
    "weight_decay": 0.0,
    "pos_weight": None,  # None → auto (neg/pos)؛ بخوای می‌تونی 0.9 بذاری
}

def same_length_padding(kernel_size, dilation):
    # طول ثابت با stride=1: padding = (k-1)*d
    return (kernel_size - 1) * dilation

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dilation=1, dropout=0.2):
        super().__init__()
        pad = same_length_padding(kernel_size, dilation)

        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation, bias=True))
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation, bias=True))
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv1(x)
        out = out[..., :x.size(-1)]              # trim tail for exact same length
        out = self.relu(self.bn1(out))
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[..., :x.size(-1)]
        out = self.relu(self.bn2(out))
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """
    نسخهٔ wide: channels=112, kernel_size=7, levels=5 با دیلیشن‌های 1,2,4,8,16
    ورودی: (B, C=16, T=1500) → خروجی لاجیت (B,1)
    """
    def __init__(self, in_ch=16, channels=112, levels=5, kernel_size=7, dropout=0.2):
        super().__init__()
        layers = []
        c_in = in_ch
        for i in range(levels):
            d = 2 ** i
            layers.append(TemporalBlock(c_in, channels, kernel_size, d, dropout))
            c_in = channels
        self.net = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, 1)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B,C,T)
        h = self.net(x)             # (B,channels,T)
        z = self.gap(h).squeeze(-1) # (B,channels)
        return self.head(z)         # (B,1) logits

def build_model(input_channels=16, seq_len=1500):
    return TCN(
        in_ch=input_channels,
        channels=112,                 # wide
        levels=5,
        kernel_size=7,
        dropout=CONFIG.get("dropout", 0.2)
    )

# ====================== اجرای مستقیم با دکمهٔ Run (PyCharm) ======================
if __name__ == "__main__":
    import os, sys, argparse
    # ریشهٔ پروژه را به sys.path اضافه می‌کنیم تا runners/ پیدا شود
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    try:
        import runners.R10_runner_balanced_swa_temp as runner
    except ModuleNotFoundError:
        # اگر PyCharm working dir را متفاوت گذاشته بود
        from runners.R10_runner_balanced_swa_temp import run_with_model_path as _run
        runner = type("R", (), {"run_with_model_path": _run})

    p = argparse.ArgumentParser()
    # هایپرها (اگر تو Run Configuration چیزی وارد نکنی، همین دیفالت‌ها استفاده می‌شوند)
    p.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    p.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    p.add_argument("--lr", type=float, default=CONFIG["lr"])
    p.add_argument("--patience", type=int, default=CONFIG["patience"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos_weight", type=float, default=None)  # None → auto

    # پارامترهای رانر R10 (دیفالت‌ها مناسب شروع هستند؛ بعداً می‌تونی عوضشان کنی)
    p.add_argument("--use_swa", type=int, default=0)                 # 0=خاموش (سازگار با وزن‌نرم قدیمی)
    p.add_argument("--use_temp_scaling", type=int, default=1)        # 1=روشن
    p.add_argument("--balanced_delta", type=float, default=0.05)     # |Prec-Rec| ≤ 0.05
    p.add_argument("--hard_min_f1", type=float, default=0.75)        # قیود ملایم برای تعادل
    p.add_argument("--hard_min_prec", type=float, default=0.78)

    args = p.parse_args()

    # اجرای رانر R10 با همین فایل مدل
    runner.run_with_model_path(
        __file__,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        use_swa=args.use_swa,
        use_temp_scaling=args.use_temp_scaling,
        balanced_delta=args.balanced_delta,
        hard_min_f1=args.hard_min_f1,
        hard_min_prec=args.hard_min_prec,
        pos_weight_override=args.pos_weight
    )
