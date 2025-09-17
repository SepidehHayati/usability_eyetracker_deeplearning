# models/08_mkse_cnn_runner06.py
import torch
import torch.nn as nn

CONFIG = {
    "epochs": 45,
    "batch_size": 32,
    "lr": 1e-3,
    "patience": 10,
    "weight_decay": 1e-4,
    "dropout": 0.30,
    "pos_weight": None
}

class SE1d(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        h=max(1,c//r)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.fc=nn.Sequential(nn.Conv1d(c,h,1), nn.ReLU(True), nn.Conv1d(h,c,1), nn.Sigmoid())
    def forward(self,x):
        w=self.fc(self.pool(x)); return x*w

class MKBlock1d(nn.Module):
    def __init__(self,in_ch,fuse_ch,ks_list=(3,7,11),use_pool=True):
        super().__init__()
        self.branches=nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_ch,fuse_ch,k,padding=k//2,bias=False), nn.BatchNorm1d(fuse_ch), nn.ReLU(True))
            for k in ks_list
        ])
        self.fuse=nn.Sequential(nn.Conv1d(fuse_ch*len(ks_list), fuse_ch,1,bias=False), nn.BatchNorm1d(fuse_ch), nn.ReLU(True))
        self.se=SE1d(fuse_ch,16); self.use_pool=use_pool
        if use_pool: self.pool=nn.MaxPool1d(2)
    def forward(self,x):
        y=torch.cat([b(x) for b in self.branches], dim=1)
        y=self.fuse(y); y=self.se(y)
        return self.pool(y) if self.use_pool else y

class MKSE_CNN1D(nn.Module):
    def __init__(self,in_ch=16,drop=0.30):
        super().__init__()
        self.b1=MKBlock1d(in_ch,64,(3,7,11),True)   # 1500→750
        self.b2=MKBlock1d(64,96,(3,7,11),True)      # 750→375
        self.b3=MKBlock1d(96,128,(3,7,11),False)    # 375
        self.gap=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(128,1))
    def forward(self,x):
        x=self.b1(x); x=self.b2(x); x=self.b3(x); x=self.gap(x)
        return self.head(x)

def build_model(input_channels=16, seq_len=1500):
    return MKSE_CNN1D(in_ch=input_channels, drop=CONFIG.get("dropout",0.30))

if __name__=="__main__":
    import os, sys, argparse
    ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, ROOT)
    import runners.R06_runner_soft_fallback as runner

    p=argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args=p.parse_args()

    runner.run_with_model_path(__file__,
                               epochs=args.epochs,
                               batch_size=args.batch_size,
                               lr=args.lr,
                               patience=args.patience,
                               seed=args.seed)
