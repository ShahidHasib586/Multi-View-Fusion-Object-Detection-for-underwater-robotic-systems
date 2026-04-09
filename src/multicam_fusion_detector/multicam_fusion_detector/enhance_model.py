import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, stride=s, padding=p, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return F.silu(x)

class TinyCurveNet(nn.Module):
    # compatible with your checkpoint structure: returns y, gain, gamma
    def __init__(self, gain_min=0.6, gain_max=2.0, gamma_min=0.6, gamma_max=2.2):
        super().__init__()
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )
        self.b1 = DepthwiseSeparableConv(16, 24, s=2)
        self.b2 = DepthwiseSeparableConv(24, 32, s=2)
        self.b3 = DepthwiseSeparableConv(32, 48, s=2)
        self.b4 = DepthwiseSeparableConv(48, 64, s=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 6)
        )

    def _map_range(self, x, lo, hi):
        x = torch.sigmoid(x)
        return lo + (hi - lo) * x

    def forward(self, x):
        feat = self.stem(x)
        feat = self.b1(feat)
        feat = self.b2(feat)
        feat = self.b3(feat)
        feat = self.b4(feat)

        v = self.pool(feat)
        p = self.head(v)

        gain = self._map_range(p[:, 0:3], self.gain_min, self.gain_max).view(-1,3,1,1)
        gamma = self._map_range(p[:, 3:6], self.gamma_min, self.gamma_max).view(-1,3,1,1)

        y = torch.clamp(gain * x, 0.0, 1.0)
        y = torch.pow(y + 1e-6, gamma)
        y = torch.clamp(y, 0.0, 1.0)
        return y, gain, gamma

def load_enhancer(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TinyCurveNet()
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model
