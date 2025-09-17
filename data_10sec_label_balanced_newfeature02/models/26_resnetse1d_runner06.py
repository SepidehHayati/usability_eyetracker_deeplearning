# models/26_resnetse1d_runner06.py
# ResNet1D + Squeeze-Excitation + Attention Pooling + Multi-Scale Stem
# خروجی: logits با شکل (B,1)
# CONFIG با اپوک/بتچ/لرن‌ریت استاندارد و dropout نسبتاً محافظه‌کارانه

import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {"epochs": 30, "batch_size": 32, "lr": 1e-3, "patience": 8, "dropout": 0.25}

# --------- Building Blocks ---------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation روی کانال‌ها (میانگین در زمان)."""
    def __init__(self, C, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(C, max(1, C // r))
        self.fc2 = nn.Linear(max(1, C // r), C)

    def forward(self, x):            # x: (B, C, T)
        s = self.avg(x).squeeze(-1)  # (B, C)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1)  # (B, C, 1)
        return x * s                 # channel-wise reweight

class ResBlock(nn.Module):
    """Basic Residual Block: Conv-BN-ReLU - Conv-BN + SE + skip."""
    def __init__(self, C):
        super().__init__()
        self.conv1 = nn.Conv1d(C, C, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(C)
        self.conv2 = nn.Conv1d(C, C, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(C)
        self.se    = SEBlock(C, r=8)

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = F.relu(out, inplace=True)
        out = self.conv2(out); out = self.bn2(out)
        out = self.se(out)
        out = F.relu(out + identity, inplace=True)
        return out

class AttnPool1D(nn.Module):
    """Attention Pooling روی بعد زمان: وزن‌دهی نرم و جمع وزنی."""
    def __init__(self, C):
        super().__init__()
        self.score = nn.Conv1d(C, 1, kernel_size=1)

    def forward(self, x):            # x: (B, C, T)
        a = self.score(x)            # (B, 1, T)
        w = torch.softmax(a, dim=-1) # (B, 1, T)
        z = (x * w).sum(dim=-1)      # (B, C)
        return z

# --------- Model ---------
class ResNetSE1D(nn.Module):
    """
    Stem چند-مقیاس (k=3/5/9) → فشرده‌سازی 1x1 → استیج‌ها (Residual + SE)
    بین استیج‌ها MaxPool برای کوچک‌کردن T. در انتها AttnPool و Head.
    """
    def __init__(self, in_ch=16, drop_p=0.25):
        super().__init__()
        stem_out_each = 24  # تعداد فیلتر هر شاخه‌ی مولتی‌اسکیل
        self.stem3 = nn.Sequential(
            nn.Conv1d(in_ch, stem_out_each, kernel_size=3, padding=1),
            nn.BatchNorm1d(stem_out_each),
            nn.ReLU(inplace=True)
        )
        self.stem5 = nn.Sequential(
            nn.Conv1d(in_ch, stem_out_each, kernel_size=5, padding=2),
            nn.BatchNorm1d(stem_out_each),
            nn.ReLU(inplace=True)
        )
        self.stem9 = nn.Sequential(
            nn.Conv1d(in_ch, stem_out_each, kernel_size=9, padding=4),
            nn.BatchNorm1d(stem_out_each),
            nn.ReLU(inplace=True)
        )
        stem_total = stem_out_each * 3  # 72
        self.stem_proj = nn.Sequential(
            nn.Conv1d(stem_total, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: C=64, دو بلاک
        self.stage1 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.pool1  = nn.MaxPool1d(2)   # 1500 -> 750

        # Stage 2: افزایش کانال به 128 و دو بلاک
        self.to128  = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(ResBlock(128), ResBlock(128))
        self.pool2  = nn.MaxPool1d(2)   # 750 -> 375

        # Stage 3: C=128، یک بلاک + Dropout سبک
        self.stage3 = ResBlock(128)
        self.dropout = nn.Dropout(drop_p)

        # Attention pooling + Head
        self.attn = AttnPool1D(128)
        self.head = nn.Linear(128, 1)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):  # x: (B, C_in, T)
        s = torch.cat([self.stem3(x), self.stem5(x), self.stem9(x)], dim=1)
        s = self.stem_proj(s)

        s = self.stage1(s)
        s = self.pool1(s)

        s = self.to128(s)
        s = self.stage2(s)
        s = self.pool2(s)

        s = self.stage3(s)
        s = self.dropout(s)

        z = self.attn(s)          # (B, 128)
        logits = self.head(z)     # (B, 1)
        return logits

def build_model(input_channels=16, seq_len=1500):
    return ResNetSE1D(in_ch=input_channels, drop_p=CONFIG.get("dropout", 0.25))

# ---- اجرای مستقیم فایل مدل با رانر ----
if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)

    # 👇 اگر نام رانر تو فرق دارد، همین import را تغییر بده (مثلاً runners.R06_runner_accfirst)
    try:
        import runners.R08_runner_acc_first as runner
    except Exception:
        # در بعضی پروژه‌ها رانر کنار ریشه است و بدون پکیج فراخوانی می‌شود:
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
