# models/05_cnn_bigru_runner05.py
import torch
import torch.nn as nn

CONFIG = {
    "epochs": 50,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 10,
    "weight_decay": 5e-4,   # ↑ سخت‌گیری بیشتر روی وزن‌ها (کاهش overfit/overshoot)
    "dropout": 0.40,        # ↑ کمی بیشتر برای جلوگیری از overfit و مثبت دیدن بیش از حد
    "rnn_hidden": 64,       # ↓ کوچک‌تر برای نرم‌تر شدن تصمیم‌ها
    "rnn_layers": 1,        # ↓ تک‌لایه (dropout داخلی GRU=0 می‌شود)
    "rnn_dropout": 0.0,
    "bidirectional": True,
    "pos_weight": None      # ← وزن‌دهی کلاس را خاموش کن (از حالت auto خارج شو)
}

class CNN_BiGRU(nn.Module):
    def __init__(self, in_ch=16, cfg: dict = None):
        super().__init__()
        cfg = cfg or CONFIG
        drop = float(cfg.get("dropout", 0.40))
        hid  = int(cfg.get("rnn_hidden", 64))
        nl   = int(cfg.get("rnn_layers", 1))
        rnn_do = float(cfg.get("rnn_dropout", 0.0))
        bi   = bool(cfg.get("bidirectional", True))

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
            nn.ReLU()
        )

        self.gru = nn.GRU(
            input_size=128, hidden_size=hid,
            num_layers=nl, batch_first=True,
            bidirectional=bi, dropout=(rnn_do if nl > 1 else 0.0)
        )
        rnn_out_ch = hid * (2 if bi else 1)

        self.head = nn.Sequential(
            nn.Linear(rnn_out_ch * 2, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.block1(x)          # (B,32,750)
        x = self.block2(x)          # (B,64,375)
        x = self.block3(x)          # (B,128,375)
        x = x.transpose(1, 2)       # (B,375,128)
        y, _ = self.gru(x)          # (B,375, rnn_out_ch)
        y_mean = y.mean(dim=1)
        y_max, _ = y.max(dim=1)
        z = torch.cat([y_mean, y_max], dim=1)
        return self.head(z)         # (B,1) logits

def build_model(input_channels=16, seq_len=1500):
    return CNN_BiGRU(in_ch=input_channels, cfg=CONFIG)

if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    import runners.R05_runner_acc_prec_constrained as runner  # ← Runner 05

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
