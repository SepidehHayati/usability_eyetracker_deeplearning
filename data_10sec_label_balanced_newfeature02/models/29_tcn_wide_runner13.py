# models/29_tcn_wide_runner13.py
# TCN wide (همان معماری 29)، برای اجرا با Runner 13

import torch
import torch.nn as nn
try:
    from torch.nn.utils.parametrizations import weight_norm
except Exception:
    from torch.nn.utils import weight_norm

CONFIG = {
    "epochs": 45,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 10,
    "dropout": 0.20,
    "weight_decay": 2e-4,
    "pos_weight": None,
}

def same_length_padding(kernel_size, dilation):
    return (kernel_size - 1) * dilation

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dilation=1, dropout=0.2):
        super().__init__()
        pad = same_length_padding(kernel_size, dilation)
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation, bias=True))
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.act1  = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation, bias=True))
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act2  = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x);  out = out[..., :x.size(-1)]
        out = self.act1(self.bn1(out)); out = self.drop1(out)
        out = self.conv2(out); out = out[..., :x.size(-1)]
        out = self.act2(self.bn2(out)); out = self.drop2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.act2(out + res)

class TCNWide(nn.Module):
    def __init__(self, in_ch=16, channels=112, levels=5, kernel_size=7, dropout=0.2):
        super().__init__()
        layers, c_in = [], in_ch
        for i in range(levels):
            d = 2 ** i
            layers.append(TemporalBlock(c_in, channels, kernel_size, d, dropout))
            c_in = channels
        self.net = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(channels, 1)
        nn.init.kaiming_normal_(self.head.weight, nonlinearity="relu")
        if self.head.bias is not None: nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = self.net(x); z = self.gap(h).squeeze(-1)
        return self.head(z)

def build_model(input_channels=16, seq_len=1500):
    return TCNWide(
        in_ch=input_channels,
        channels=112,
        levels=5,
        kernel_size=7,
        dropout=CONFIG.get("dropout", 0.2)
    )

# ===== Run via PyCharm (calls Runner 13) =====
if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    try:
        import runners.R13_runner_accfirst_minimal as runner
    except ModuleNotFoundError:
        from runners.R13_runner_accfirst_minimal import run_with_model_path as _run
        runner = type("R", (), {"run_with_model_path": _run})

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    p.add_argument("--batch_size", type=int, default=CONFIG["batch_size"])
    p.add_argument("--lr", type=float, default=CONFIG["lr"])
    p.add_argument("--patience", type=int, default=CONFIG["patience"])
    p.add_argument("--seed", type=int, default=42)

    # R13 options
    p.add_argument("--pos_weight_mode", default="auto")   # 'auto' یا عدد مثل 0.9
    p.add_argument("--weight_decay", type=float, default=CONFIG["weight_decay"])
    p.add_argument("--optim", default="adamw")            # 'adamw' یا 'adam'
    p.add_argument("--no_sched", action="store_true")     # خاموش کردن scheduler
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--use_aug", action="store_true")      # پیش‌فرض خاموش

    args = p.parse_args()

    runner.run_with_model_path(
        __file__,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
        pos_weight_mode=args.pos_weight_mode,
        weight_decay=args.weight_decay,
        optim_name=args.optim,
        use_scheduler=(not args.no_sched),
        warmup_epochs=args.warmup_epochs,
        grad_clip=args.grad_clip,
        use_aug=args.use_aug
    )
