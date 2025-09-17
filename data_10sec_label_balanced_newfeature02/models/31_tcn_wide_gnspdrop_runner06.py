# models/31_tcn_wide_gnspdrop_runner06.py
# TCN (channels=96, kernel=7) + GroupNorm + SpatialDropout1D + GELU
# همون ایدهٔ مدل 29، با نرمال‌سازی/رگولاریزیشن بهتر.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

CONFIG = {"epochs": 35, "batch_size": 32, "lr": 1e-3, "patience": 9, "dropout": 0.2}

def same_length_padding(kernel_size, dilation):
    return (kernel_size - 1) * dilation

class SpatialDropout1D(nn.Module):
    """کانال‌-دراپ‌اوت برای ورودی (B, C, T)"""
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        # تبدیل به (B,C,T,1) تا dropout2d کانالی عمل کند
        x = x.unsqueeze(-1)
        x = F.dropout2d(x, p=self.p, training=True)
        return x.squeeze(-1)

class GN(nn.Module):
    """GroupNorm با تعداد گروه معقول بر اساس تعداد کانال‌ها"""
    def __init__(self, num_channels, max_groups=16):
        super().__init__()
        # تعداد گروه باید مقسوم‌علیهِ num_channels باشد
        groups = 1
        for g in range(min(max_groups, num_channels), 0, -1):
            if num_channels % g == 0:
                groups = g
                break
        self.gn = nn.GroupNorm(groups, num_channels)
    def forward(self, x):
        return self.gn(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dilation=1, dropout=0.2):
        super().__init__()
        pad = same_length_padding(kernel_size, dilation)

        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation))
        self.norm1 = GN(out_ch)
        self.act1  = nn.GELU()
        self.drop1 = SpatialDropout1D(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation))
        self.norm2 = GN(out_ch)
        self.act2  = nn.GELU()
        self.drop2 = SpatialDropout1D(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv1(x);  out = out[..., :x.size(-1)]
        out = self.act1(self.norm1(out))
        out = self.drop1(out)

        out = self.conv2(out); out = out[..., :x.size(-1)]
        out = self.act2(self.norm2(out))
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return F.gelu(out + res)

class TCN(nn.Module):
    def __init__(self, in_ch=16, channels=96, levels=5, kernel_size=7, dropout=0.2):
        super().__init__()
        layers = []
        c_in = in_ch
        for i in range(levels):
            dilation = 2 ** i   # 1,2,4,8,16
            layers.append(TemporalBlock(c_in, channels, kernel_size, dilation, dropout))
            c_in = channels
        self.net = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # سر سبک با یک لایهٔ میانی کوچک
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):  # x: (B,C,T)
        h = self.net(x)
        z = self.gap(h).squeeze(-1)   # (B,channels)
        return self.head(z)           # (B,1) logits

def build_model(input_channels=16, seq_len=1500):
    return TCN(in_ch=input_channels,
               channels=96,       # همان 29 (می‌تونی بعداً 112 تست کنی)
               levels=5,
               kernel_size=7,     # همان 29
               dropout=CONFIG.get("dropout", 0.2))

# ---- اجرای مستقیم فایل با رانر ----
if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    try:
        import runners.R08_runner_acc_first as runner
    except ModuleNotFoundError:
        import R08_runner_acc_first as runner

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
