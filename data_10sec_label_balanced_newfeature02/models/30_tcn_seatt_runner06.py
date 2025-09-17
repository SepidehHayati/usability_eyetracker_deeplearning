# models/30_tcn_seatt_runner06.py
# TCN پهن با Squeeze-Excitation و Attention Pooling
# سازگار با R08_runner_acc_first (Threshold tuning + constraints)

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

CONFIG = {"epochs": 40, "batch_size": 32, "lr": 1e-3, "patience": 10, "dropout": 0.25}

def same_length_padding(kernel_size, dilation):
    return (kernel_size - 1) * dilation

class SE1D(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        hidden = max(4, ch // r)
        self.fc1 = nn.Linear(ch, hidden)
        self.fc2 = nn.Linear(hidden, ch)

    def forward(self, x):  # x: (B,C,T)
        s = x.mean(dim=-1)                    # (B,C)
        s = F.relu(self.fc1(s), inplace=True) # (B,hidden)
        s = torch.sigmoid(self.fc2(s))        # (B,C)
        s = s.unsqueeze(-1)                   # (B,C,1)
        return x * s

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dilation=1, dropout=0.25, use_se=True):
        super().__init__()
        pad = same_length_padding(kernel_size, dilation)
        self.conv1 = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation))
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                           padding=pad, dilation=dilation))
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
        self.relu = nn.ReLU(inplace=True)
        self.se = SE1D(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = out[..., :x.size(-1)]
        out = self.relu(self.bn1(out))
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[..., :x.size(-1)]
        out = self.relu(self.bn2(out))
        out = self.dropout(out)

        out = self.se(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalAttentionPool(nn.Module):
    """ توجه زمانی: وزن‌دهی به گام‌های زمانی به‌صورت آموختنی """
    def __init__(self, ch):
        super().__init__()
        self.proj = nn.Linear(ch, 1, bias=False)

    def forward(self, x):             # x: (B,C,T)
        xt = x.transpose(1, 2)        # (B,T,C)
        scores = self.proj(xt).squeeze(-1)          # (B,T)
        alpha = torch.softmax(scores, dim=-1)       # (B,T)
        z = torch.bmm(xt.transpose(1, 2), alpha.unsqueeze(-1)).squeeze(-1)  # (B,C)
        return z

class TCN_SE_Att(nn.Module):
    def __init__(self, in_ch=16, channels=128, levels=5, kernel_size=7, dropout=0.25):
        super().__init__()
        layers = []
        c_in = in_ch
        for i in range(levels):
            dilation = 2 ** i
            layers.append(TemporalBlock(c_in, channels, kernel_size, dilation, dropout, use_se=True))
            c_in = channels
        self.tcn = nn.Sequential(*layers)
        self.att = TemporalAttentionPool(channels)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels, 1)
        )
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):   # x: (B,C,T)
        h = self.tcn(x)     # (B,channels,T)
        z = self.att(h)     # (B,channels)
        return self.head(z) # (B,1) logits

def build_model(input_channels=16, seq_len=1500):
    return TCN_SE_Att(in_ch=input_channels,
                      channels=128,    # پهن‌تر از مدل 29
                      levels=5,
                      kernel_size=7,   # همان کرنل مؤثر مدل 29
                      dropout=CONFIG.get("dropout", 0.25))

# ---- اجرای مستقیم فایل با رانر ----
if __name__ == "__main__":
    import os, sys, argparse
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    try:
        import runners.R08_runner_acc_first as runner
    except ModuleNotFoundError:
        import R08_runner_acc_first as runner

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
