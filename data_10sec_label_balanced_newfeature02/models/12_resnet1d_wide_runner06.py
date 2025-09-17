# models/12_resnet1d_wide_runner06.py
import torch
import torch.nn as nn

CONFIG = {
    "epochs": 50,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 12,
    "weight_decay": 5e-4,  # کمی قوی‌تر
    "pos_weight": 0.8,     # اندکی فشار کمتر به مثبت‌ها
    "dropout": 0.15
}

class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class ResNet1D_Wide(nn.Module):
    """
    stem: Conv7x1(80) → BN → ReLU → MaxPool(2)    T:1500→750
    layer1: 2×BasicBlock(80)                       T:750
    layer2: 2×BasicBlock(160, stride=2)           T:750→375
    layer3: 2×BasicBlock(160)                      T:375
    GAP + GMP concat → Dropout → Linear(320→1)
    """
    def __init__(self, in_ch=16, drop=0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 80, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(80),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        self.layer1 = nn.Sequential(
            BasicBlock1D(80, 80, stride=1),
            BasicBlock1D(80, 80, stride=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock1D(80, 160, stride=2),
            BasicBlock1D(160, 160, stride=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock1D(160, 160, stride=1),
            BasicBlock1D(160, 160, stride=1),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),        # (B, 160*2) = 320
            nn.Dropout(drop),
            nn.Linear(320, 1)    # logits
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        a = self.gap(x)   # (B,160,1)
        m = self.gmp(x)   # (B,160,1)
        x = torch.cat([a, m], dim=1)  # (B,320,1)
        return self.head(x)

def build_model(input_channels=16, seq_len=1500):
    return ResNet1D_Wide(in_ch=input_channels, drop=CONFIG.get("dropout", 0.15))

# اجرای مستقیم با Runner 06 (soft fallback)
if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    import runners.R06_runner_soft_fallback as runner

    # قیود آستانه: حفظ دقت با کنترل FP
    runner.MIN_PREC   = 0.80
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
