# models/15_tcn_runner06.py
import torch
import torch.nn as nn

# هدف: افزایش Acc بدون خراب کردن F1 با پوشش زمانی بلندتر
CONFIG = {
    "epochs": 45,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 10,
    "weight_decay": 3e-4,
    "pos_weight": None,   # تمایل به تعادل؛ مثبت‌گویی افراطی نداشته باشیم
    "dropout": 0.20
}

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=9, d=1, dropout=0.2):
        super().__init__()
        pad = (k - 1) * d // 2  # same length
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=d, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, dilation=d, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.down = None
        if in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(out_ch))

    def forward(self, x):
        y = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        y = self.drop2(self.relu2(self.bn2(self.conv2(y))))
        if self.down is not None:
            x = self.down(x)
        return y + x

class TCN1D(nn.Module):
    """
    دایلیشن‌ها: 1,2,4,8,16,32,64,128  با kernel=9  → receptive field ~ 1 + 8*sum(d) = 1 + 8*255 ≈ 2041 > 1500
    طول ثابت نگه داشته می‌شود؛ در انتها GAP و Linear.
    """
    def __init__(self, in_ch=16, base_ch=64, k=9, dropout=0.20):
        super().__init__()
        dils = [1,2,4,8,16,32,64,128]
        chs = [base_ch]*len(dils)

        layers = []
        c_in = in_ch
        for c_out, d in zip(chs, dils):
            layers.append(TemporalBlock(c_in, c_out, k=k, d=d, dropout=dropout))
            c_in = c_out
        self.tcn = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_ch, 1)
        )

    def forward(self, x):  # x: (B,C,T)
        x = self.tcn(x)     # (B,base_ch,T)
        x = self.gap(x)     # (B,base_ch,1)
        return self.head(x) # (B,1) logits

def build_model(input_channels=16, seq_len=1500):
    return TCN1D(in_ch=input_channels, base_ch=64, k=9, dropout=CONFIG.get("dropout", 0.20))

if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    import runners.R06_runner_soft_fallback as runner

    # قیدها: تعادل Precision/Recall را حفظ می‌کنند
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
