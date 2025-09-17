# models/14_resnet1d_tuned_runner06.py
import torch
import torch.nn as nn

# هدف این تیون: Precision بالاتر (کاهش FP) و Acc بالاتر بدون افت F1
CONFIG = {
    "epochs": 60,          # اپوک بیشتر؛ EarlyStopping جلو overfit را می‌گیرد
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 12,        # صبر بیشتر برای رسیدن به نقطه بهتر
    "weight_decay": 7e-4,  # WD کمی قوی‌تر از قبل
    "pos_weight": 0.6,     # فشار مثبت‌گویی را کم می‌کند → Precision بالا می‌رود
    "dropout": 0.15
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
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out

class ResNet1D(nn.Module):
    """
    همان معماری موفق قبلی:
    stem(64) → layer1(64×2) → layer2(128×2, stride=2) → layer3(128×2) → GAP → Dropout → Linear
    """
    def __init__(self, in_ch=16, drop=0.15):
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

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x)    # (B,128,1)
        return self.head(x)  # (B,1) logits

def build_model(input_channels=16, seq_len=1500):
    return ResNet1D(in_ch=input_channels, drop=CONFIG.get("dropout", 0.15))

if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    # همان رانر نرم با fallback چندمرحله‌ای
    import runners.R06_runner_soft_fallback as runner

    # کمی قیود را نگه می‌داریم تا Precision پایین اجازه نگیرد
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
