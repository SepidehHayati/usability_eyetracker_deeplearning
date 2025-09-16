# models/04_cnn_bnwd_runner04.py
import torch
import torch.nn as nn

CONFIG = {
    "epochs": 40,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 10,
    "weight_decay": 1e-4,   # خوانده می‌شود توسط runner_04
    "dropout": 0.30
}

class CNN1D_BN(nn.Module):
    def __init__(self, in_ch=16, drop=0.30):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)          # 1500 -> 750
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)          # 750 -> 375
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # -> (B,128,1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),            # (B,128)
            nn.Dropout(drop),
            nn.Linear(128, 1)        # logits (B,1)
        )

    def forward(self, x):            # x: (B, C, T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)

def build_model(input_channels=16, seq_len=1500):
    return CNN1D_BN(in_ch=input_channels, drop=CONFIG.get("dropout", 0.30))

if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    # 🔧 رانر جدید:
    import runners.R04_runner_acc_constrained_wd as runner
    # یا اگر با نام پیشنهادی من ذخیره کردی:
    # import runners.runner_04_acc_constrained_wd as runner

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
