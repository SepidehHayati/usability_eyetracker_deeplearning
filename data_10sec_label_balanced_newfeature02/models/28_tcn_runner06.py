# models/28_tcn_runner06.py
# TCN (Temporal Convolutional Network) سبک با WeightNorm + Residual
# سازگار با runnerهای فعلی (R08 پیش‌فرض)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

CONFIG = {"epochs": 35, "batch_size": 32, "lr": 1e-3, "patience": 8, "dropout": 0.2}

def same_length_padding(kernel_size, dilation):
    # طول ثابت با stride=1: padding = (k-1)*d
    return (kernel_size - 1) * dilation

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, dropout=0.2):
        super().__init__()
        pad = same_length_padding(kernel_size, dilation)
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation))
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation))
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = out[..., :x.size(-1)]  # trim padding tail (حفظ طول)
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
    سطوح دیلاته با receptive field رو به رشد:
    dilations = 1, 2, 4, 8, 16  (۵ بلاک)
    """
    def __init__(self, in_ch=16, channels=64, levels=5, kernel_size=5, dropout=0.2):
        super().__init__()
        layers = []
        C_in = in_ch
        for i in range(levels):
            dilation = 2 ** i
            layers.append(TemporalBlock(C_in, channels, kernel_size, dilation, dropout))
            C_in = channels
        self.net = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, 1)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):  # x: (B,C,T)
        h = self.net(x)           # (B,channels,T)
        z = self.gap(h).squeeze(-1)
        return self.head(z)       # (B,1) logits

def build_model(input_channels=16, seq_len=1500):
    return TCN(in_ch=input_channels,
               channels=64,      # می‌تونی بعداً 96/128 تست کنی
               levels=5,
               kernel_size=5,
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
