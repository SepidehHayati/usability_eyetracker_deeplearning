# models/16_dualstream_resnet_runner06.py
import torch
import torch.nn as nn

# هدف: کاهش FP و بهبود Acc با تفکیک جریانِ فیچرهای Base و Delta
CONFIG = {
    "epochs": 50,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 12,
    "weight_decay": 3e-4,
    "pos_weight": 0.8,   # کمی محافظه‌کارتر نسبت به مثبت‌ها
    "dropout": 0.20
}

# ---------- Blocks ----------
class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
    def forward(self, x):
        idn = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            idn = self.downsample(x)
        return self.relu(out + idn)

class Stream(nn.Module):
    """یک ResNet کوچک برای یک زیرمجموعه از کانال‌ها (Base یا Delta)"""
    def __init__(self, in_ch, width=48, drop=0.20):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(inplace=True),
            nn.MaxPool1d(2)  # 1500 -> 750
        )
        self.layer1 = nn.Sequential(
            BasicBlock1D(width, width, stride=1),
            BasicBlock1D(width, width, stride=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock1D(width, width*2, stride=2),  # 750 -> 375
            BasicBlock1D(width*2, width*2, stride=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock1D(width*2, width*2, stride=1),
            BasicBlock1D(width*2, width*2, stride=1),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):              # x: (B, Cin, T)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)             # (B, 2W, T')
        a = self.gap(x)                # (B, 2W, 1)
        m = self.gmp(x)                # (B, 2W, 1)
        x = torch.cat([a, m], dim=1)   # (B, 4W, 1)
        x = self.drop(x)
        return x                       # (B, 4W, 1)

class DualStreamResNet1D(nn.Module):
    """
    ورودی: (B, 16, 1500) → split: base=(B,8,1500), delta=(B,8,1500)
    دو ResNet کوچک موازی → GAP+GMP هرکدام → concat → Linear→1
    """
    def __init__(self, in_ch_total=16, drop=0.20):
        super().__init__()
        assert in_ch_total == 16, "expected 16 channels (8 base + 8 delta)"
        self.base_stream  = Stream(in_ch=8, width=48, drop=drop)
        self.delta_stream = Stream(in_ch=8, width=48, drop=drop)
        # هر استریم خروجی 4W=192 کانالی می‌دهد → مجموع 384
        self.head = nn.Sequential(
            nn.Flatten(),          # (B, 384)
            nn.Dropout(drop),
            nn.Linear(384, 1)      # logits
        )

    def forward(self, x):           # x: (B,16,T)
        xb = x[:, :8, :]            # Base channels (ترتیب: اول 8 تا)
        xd = x[:, 8:, :]            # Delta channels (8 تای بعدی)
        fb = self.base_stream(xb)   # (B,192,1)
        fd = self.delta_stream(xd)  # (B,192,1)
        f  = torch.cat([fb, fd], dim=1)  # (B,384,1)
        return self.head(f)         # (B,1)

def build_model(input_channels=16, seq_len=1500):
    return DualStreamResNet1D(in_ch_total=input_channels, drop=CONFIG.get("dropout", 0.20))

# اجرای مستقیم با Runner06
if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    import runners.R06_runner_soft_fallback as runner

    # کمی سخت‌گیرانه برای Precision تا FP کم شود و Acc بالا برود
    runner.MIN_PREC   = 0.82
    runner.MIN_F1     = 0.75
    runner.RELAXED_F1 = 0.70

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
