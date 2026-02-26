import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.down1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        x = self.up1(x)
        return self.out(x)
