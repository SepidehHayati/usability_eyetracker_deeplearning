# models/27_inceptiontime_runner06.py
# InceptionTime (light) برای سری زمانی 1D + GlobalAvgPool
# خروجی: logits (B,1)  |  سازگار با runnerهای فعلی (R08 پیش‌فرض)

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {"epochs": 35, "batch_size": 32, "lr": 1e-3, "patience": 8, "dropout": 0.2}

def same_padding(k):  # برای Conv1d با stride=1
    return k // 2

class InceptionModule1D(nn.Module):
    """
    Bottleneck 1x1 → سه شاخه‌ی کانولوشن با کرنل‌های بزرگ/متوسط/کوچک → شاخه‌ی MaxPool → کانکت
    """
    def __init__(self, in_ch, out_ch, kernel_sizes=(9, 19, 39), bottleneck_ch=32):
        super().__init__()
        use_bottleneck = in_ch > 1
        self.bottleneck = nn.Conv1d(in_ch, bottleneck_ch, kernel_size=1) if use_bottleneck else None
        bch = bottleneck_ch if use_bottleneck else in_ch

        b1 = nn.Conv1d(bch, out_ch // 4, kernel_size=kernel_sizes[0], padding=same_padding(kernel_sizes[0]))
        b2 = nn.Conv1d(bch, out_ch // 4, kernel_size=kernel_sizes[1], padding=same_padding(kernel_sizes[1]))
        b3 = nn.Conv1d(bch, out_ch // 4, kernel_size=kernel_sizes[2], padding=same_padding(kernel_sizes[2]))
        b4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch // 4, kernel_size=1)
        )
        self.branches = nn.ModuleList([b1, b2, b3, b4])
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # (B,C,T)
        if self.bottleneck is not None:
            xb = self.bottleneck(x)
        else:
            xb = x
        y1 = self.branches[0](xb)
        y2 = self.branches[1](xb)
        y3 = self.branches[2](xb)
        y4 = self.branches[3](x)   # دقت: شاخه‌ی pool مستقیماً از x می‌آید (طبق پیاده‌سازی‌های متداول)
        y = torch.cat([y1, y2, y3, y4], dim=1)
        y = self.bn(y)
        return self.relu(y)

class InceptionBlock(nn.Module):
    """
    سه ماژول اینسپشن پشت سر هم + اتصال رزدوال (1x1 conv برای مچ کانال)
    """
    def __init__(self, in_ch, out_ch, kernel_sizes=(9,19,39), bottleneck_ch=32):
        super().__init__()
        self.inc1 = InceptionModule1D(in_ch,  out_ch, kernel_sizes, bottleneck_ch)
        self.inc2 = InceptionModule1D(out_ch, out_ch, kernel_sizes, bottleneck_ch)
        self.inc3 = InceptionModule1D(out_ch, out_ch, kernel_sizes, bottleneck_ch)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm1d(out_ch)
            )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.inc1(x)
        out = self.inc2(out)
        out = self.inc3(out)
        out = out + self.shortcut(x)
        out = self.bn(out)
        return self.relu(out)

class InceptionTime1D(nn.Module):
    def __init__(self, in_ch=16, width=64, dropout=0.2):
        super().__init__()
        # سه بلاک با عرض ثابت (می‌توانی بعداً عرض را زیاد کنی)
        self.block1 = InceptionBlock(in_ch,   width)
        self.block2 = InceptionBlock(width,   width)
        self.block3 = InceptionBlock(width,   width)
        self.dropout = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(width, 1)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):  # x: (B,C,T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        z = self.gap(x).squeeze(-1)  # (B, width)
        return self.head(z)          # (B,1) logits

def build_model(input_channels=16, seq_len=1500):
    return InceptionTime1D(in_ch=input_channels, width=64, dropout=CONFIG.get("dropout", 0.2))

# ---- اجرای مستقیم فایل مدل با رانر ---
if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    # رانر پیشنهادی:
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
