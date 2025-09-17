# models/34_tcn_deeper_gn_eca_runner08.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

CONFIG = {
    "epochs": 45,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 10,
    "weight_decay": 3e-4,
    "pos_weight": 0.9,   # دیتاست متعادله → کمی زیر 1 برای مهار FP
    "dropout": 0.20
}

def _same_pad(k, d): return (k - 1) * d

class SpatialDropout1D(nn.Module):
    def __init__(self, p=0.2): super().__init__(); self.p = p
    def forward(self, x):
        if (not self.training) or self.p == 0.0: return x
        x = x.unsqueeze(-1)
        x = F.dropout2d(x, p=self.p, training=True)
        return x.squeeze(-1)

class GN(nn.Module):
    def __init__(self, c, max_groups=16):
        super().__init__()
        g = 1
        for gg in range(min(max_groups, c), 0, -1):
            if c % gg == 0: g = gg; break
        self.gn = nn.GroupNorm(g, c)
    def forward(self, x): return self.gn(x)

class SE1D(nn.Module):
    """Squeeze-Excitation سبک روی کانال‌ها"""
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Linear(c, max(1, c // r))
        self.fc2 = nn.Linear(max(1, c // r), c)
    def forward(self, x):                # x: (B,C,T)
        s = x.mean(dim=-1)               # (B,C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))   # (B,C)
        return x * s.unsqueeze(-1)

class TemporalBlock(nn.Module):
    def __init__(self, cin, cout, k=9, d=1, p_drop=0.2):
        super().__init__()
        pad = _same_pad(k, d)
        self.conv1 = weight_norm(nn.Conv1d(cin, cout, k, padding=pad, dilation=d))
        self.norm1 = GN(cout)
        self.drop1 = SpatialDropout1D(p_drop)

        self.conv2 = weight_norm(nn.Conv1d(cout, cout, k, padding=pad, dilation=d))
        self.norm2 = GN(cout)
        self.drop2 = SpatialDropout1D(p_drop)

        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else None
        self.se   = SE1D(cout)
        self.act  = nn.GELU()

    def forward(self, x):
        out = self.conv1(x); out = out[..., :x.size(-1)]
        out = self.act(self.norm1(out)); out = self.drop1(out)

        out = self.conv2(out); out = out[..., :x.size(-1)]
        out = self.act(self.norm2(out)); out = self.drop2(out)

        res = x if self.down is None else self.down(x)
        out = self.act(out + res)
        out = self.se(out)
        return out

class TCN(nn.Module):
    def __init__(self, in_ch=16, channels=112, levels=8, k=9, p_drop=0.2):
        super().__init__()
        layers = []
        c = in_ch
        for i in range(levels):
            d = 2 ** i            # 1,2,4,...,128  → RF بزرگ
            layers.append(TemporalBlock(c, channels, k=k, d=d, p_drop=p_drop))
            c = channels
        self.net = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(channels, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d): nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if isinstance(m, nn.Linear): nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):          # x: (B,C,T)
        h = self.net(x)
        z = self.gap(h).squeeze(-1)
        return self.head(z)        # logits

def build_model(input_channels=16, seq_len=1500):
    return TCN(in_ch=input_channels,
               channels=112, levels=8, k=9,
               p_drop=CONFIG.get("dropout", 0.2))

# اجرای مستقیم با R08 (Accuracy-first + constraints)
if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    import runners.R08_runner_acc_first as runner

    # کمی سخت‌تر برای Precision تا Acc بالا بره
    runner.HARD_MIN_F1   = 0.78
    runner.HARD_MIN_PREC = 0.84
    runner.SOFT_MIN_F1   = 0.75
    runner.SOFT_MIN_PREC = 0.80

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    runner.run_with_model_path(
        __file__,
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        patience=a.patience, seed=a.seed
    )
