# models/06_mkse_cnn_runner04.py
import torch
import torch.nn as nn

# هایپرپارامترها (Runner 04 مقدار weight_decay را از اینجا می‌خوانَد)
CONFIG = {
    "epochs": 45,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 10,
    "weight_decay": 1e-4,
    "dropout": 0.30
}

# ---------- Squeeze-and-Excitation for 1D ----------
class SE1d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hid, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):  # (B,C,T)
        w = self.pool(x)      # (B,C,1)
        w = self.fc(w)        # (B,C,1)
        return x * w          # channel-wise reweight

# ---------- Multi-Kernel Convolution Block ----------
class MKBlock1d(nn.Module):
    """
    چند شاخه‌ی موازی با کرنل‌های مختلف (۳/۷/۱۱) + همجوسازی 1x1 + BatchNorm + ReLU + SE.
    optionally downsample = MaxPool1d(2)
    """
    def __init__(self, in_ch, fuse_ch, ks_list=(3,7,11), use_pool=True):
        super().__init__()
        branches = []
        for k in ks_list:
            pad = k // 2
            branches.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, fuse_ch, kernel_size=k, padding=pad, bias=False),
                    nn.BatchNorm1d(fuse_ch),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.Sequential(
            nn.Conv1d(fuse_ch * len(ks_list), fuse_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(fuse_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SE1d(fuse_ch, reduction=16)
        self.use_pool = use_pool
        if use_pool:
            self.pool = nn.MaxPool1d(2)

    def forward(self, x):  # (B, C_in, T)
        outs = [b(x) for b in self.branches]     # list of (B,fuse_ch,T)
        y = torch.cat(outs, dim=1)               # (B, fuse_ch*len(ks), T)
        y = self.fuse(y)                         # (B, fuse_ch, T)
        y = self.se(y)                           # (B, fuse_ch, T)
        if self.use_pool:
            y = self.pool(y)                     # downsample
        return y

class MKSE_CNN1D(nn.Module):
    """
    ورودی: (B, C=16, T=1500)
    Block1: MK(16->64), pool → T: 1500→750
    Block2: MK(64->96), pool → T: 750→375
    Block3: MK(96->128), no pool → T: 375
    GAP: AdaptiveAvgPool1d(1) → (B,128,1)
    Head: Dropout + Linear(128→1)
    """
    def __init__(self, in_ch=16, drop=0.30):
        super().__init__()
        self.block1 = MKBlock1d(in_ch=in_ch, fuse_ch=64, ks_list=(3,7,11), use_pool=True)
        self.block2 = MKBlock1d(in_ch=64,   fuse_ch=96, ks_list=(3,7,11), use_pool=True)
        self.block3 = MKBlock1d(in_ch=96,   fuse_ch=128, ks_list=(3,7,11), use_pool=False)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),          # (B,128)
            nn.Dropout(drop),
            nn.Linear(128, 1)      # logits
        )

    def forward(self, x):  # x: (B,C,T)
        x = self.block1(x)   # (B,64,750)
        x = self.block2(x)   # (B,96,375)
        x = self.block3(x)   # (B,128,375)
        x = self.gap(x)      # (B,128,1)
        return self.head(x)  # (B,1)

def build_model(input_channels=16, seq_len=1500):
    return MKSE_CNN1D(in_ch=input_channels, drop=CONFIG.get("dropout", 0.30))

# ---- اجرای مستقیم فایل مدل (PyCharm Run) ----
if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    # رانر 04 (acc constrained + weight_decay)
    import runners.R04_runner_acc_constrained_wd as runner

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
