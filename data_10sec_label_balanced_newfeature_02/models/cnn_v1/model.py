# models/cnn_v1/model.py
import torch
import torch.nn as nn

class GazeNetCNN(nn.Module):
    """ساده و تمیز برای ورودی (B, 16, 1500)"""
    def __init__(self, in_ch=16, num_classes=1, pdrop=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3), nn.ReLU(), nn.BatchNorm1d(64),
            nn.MaxPool1d(2),  # 1500 -> 750
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(128),
            nn.MaxPool1d(2),  # 750 -> 375
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1),  # -> (B,256,1)
            nn.Flatten(),             # -> (B,256)
            nn.Dropout(pdrop),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Linear(128, num_classes)  # خروجی logits
        )

    def forward(self, x):   # x: (B,16,1500)
        return self.net(x).squeeze(-1)

# --------- اینترفیس استاندارد برای ترینر ---------
def get_input_layout():
    """layout ورودی مورد انتظار مدل: 'channels_first' یا 'time_first'"""
    return "channels_first"

def build_model():
    """ترینر این تابع رو صدا می‌زنه و یک nn.Module می‌گیره."""
    return GazeNetCNN(in_ch=16, num_classes=1, pdrop=0.25)
