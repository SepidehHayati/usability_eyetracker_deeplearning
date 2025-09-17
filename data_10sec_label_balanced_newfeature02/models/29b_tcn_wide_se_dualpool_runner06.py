# models/29b_tcn_wide_se_dualpool_runner06.py
# TCN (channels=96, k=7) با SE داخل بلاک‌ها + SpatialDropout1d + (GAP||GMP) head
# سازگار با R08_runner_acc_first (و فالبک به همان در صورت import محلی)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

CONFIG = {"epochs": 35, "batch_size": 32, "lr": 1e-3, "patience": 8, "dropout": 0.2}

def same_length_padding(kernel_size, dilation):
    return (kernel_size - 1) * dilation

class SE1d(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):  # x: (B,C,T)
        s = self.avg(x).squeeze(-1)         # (B,C)
        w = self.fc(s).unsqueeze(-1)        # (B,C,1)
        return x * w                        # scale per-channel

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dilation=1, dropout=0.2):
        super().__init__()
        pad = same_length_padding(kernel_size, dilation)
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation))
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation))
        self.bn2   = nn.BatchNorm1d(out_ch)

        # کانال‌-دراپ‌آوت (spatial dropout روی بعد کانال)
        self.sdrop = nn.Dropout2d(dropout)

        self.se    = SE1d(out_ch, reduction=8)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = out[..., :x.size(-1)]
        out = self.relu(self.bn1(out))
        out = self.sdrop(out)          # drop whole channels

        out = self.conv2(out)
        out = out[..., :x.size(-1)]
        out = self.relu(self.bn2(out))
        out = self.se(out)             # کانال‌های مهم تقویت شوند
        out = self.sdrop(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_ch=16, channels=96, levels=5, kernel_size=7, dropout=0.2):
        super().__init__()
        layers = []
        C_in = in_ch
        for i in range(levels):
            dilation = 2 ** i
            layers.append(TemporalBlock(C_in, channels, kernel_size, dilation, dropout))
            C_in = channels
        self.net = nn.Sequential(*layers)

        # Head: GAP + GMP → concat → MLP کوچک → 1
        self.head = nn.Sequential(
            nn.Linear(2 * channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):  # (B,C,T)
        h = self.net(x)                                   # (B,channels,T)
        z_avg = F.adaptive_avg_pool1d(h, 1).squeeze(-1)   # (B,channels)
        z_max = F.adaptive_max_pool1d(h, 1).squeeze(-1)   # (B,channels)
        z = torch.cat([z_avg, z_max], dim=1)              # (B,2*channels)
        return self.head(z)                               # (B,1) logits

def build_model(input_channels=16, seq_len=1500):
    return TCN(in_ch=input_channels,
               channels=96,
               levels=5,
               kernel_size=7,
               dropout=CONFIG.get("dropout", 0.2))

# ---- اجرای مستقیم فایل با رانر ----
if __name__ == "__main__":
    import os, sys, argparse, importlib.util
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    # تلاش برای لود R08 (و اگر نبود همان نام در ریشه)
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
