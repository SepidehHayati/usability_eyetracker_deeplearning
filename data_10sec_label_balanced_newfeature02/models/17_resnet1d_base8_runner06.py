# models/17_resnet1d_base8_runner06.py
import torch
import torch.nn as nn

# هدف: استفاده فقط از 8 ویژگی خام (بدون دلتا) برای کاهش نویز و FP
CONFIG = {
    "epochs": 45,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 10,
    "weight_decay": 3e-4,
    "pos_weight": 0.8,   # کمی محافظه‌کارتر نسبت به مثبت‌ها
    "dropout": 0.20
}

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

class ResNet1D_Base8_Core(nn.Module):
    """ResNet1D برای 8 کانال خام"""
    def __init__(self, in_ch=8, drop=0.20):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(2)  # 1500→750
        )
        self.layer1 = nn.Sequential(
            BasicBlock1D(64, 64, stride=1),
            BasicBlock1D(64, 64, stride=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock1D(64, 128, stride=2),  # 750→375
            BasicBlock1D(128, 128, stride=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock1D(128, 128, stride=1),
            BasicBlock1D(128, 128, stride=1),
        )
        self.gap  = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(128, 1))

    def forward(self, x):  # x: (B, 8, T)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)    # (B,128,1)
        return self.head(x)

class ResNet1D_Base8_Wrapper(nn.Module):
    """Wrapper که ورودی 16 کاناله می‌گیرد، فقط 8 کانال خامِ اول را استفاده می‌کند."""
    def __init__(self, drop=0.20):
        super().__init__()
        self.core = ResNet1D_Base8_Core(in_ch=8, drop=drop)
    def forward(self, x):         # x: (B, 16, T)
        x8 = x[:, :8, :]          # فقط 8 کانال خام
        return self.core(x8)

def build_model(input_channels=16, seq_len=1500):
    return ResNet1D_Base8_Wrapper(drop=CONFIG.get("dropout", 0.20))

# اجرای مستقیم با Runner06
if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    import runners.R06_runner_soft_fallback as runner

    # کمی سخت‌گیرانه تا FP کم شود و Acc بالا برود
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
