# models/09_cnn_runner06.py
import torch
import torch.nn as nn

# همون CNN ساده که قبلاً جواب بهتری می‌داد
CONFIG = {
    "epochs": 30,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 7,
    "weight_decay": 1e-4,  # کمی منظم‌کننده
    "pos_weight": None     # گرایش به مثبت‌ها رو کم می‌کنه
}

class CNN1D(nn.Module):
    def __init__(self, in_ch=16):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),   # 1500->750
            nn.Conv1d(32, 64, kernel_size=7, padding=3),    nn.ReLU(), nn.MaxPool1d(2),   # 750->375
            nn.Conv1d(64,128, kernel_size=5, padding=2),    nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):  # x: (B,C,T)
        return self.head(self.feat(x))  # (B,1) logits

def build_model(input_channels=16, seq_len=1500):
    return CNN1D(in_ch=input_channels)

# اجرای مستقیم با Runner 06 (soft fallback)
if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    import runners.R06_runner_soft_fallback as runner

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
