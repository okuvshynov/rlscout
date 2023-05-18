import torch.nn as nn
from collections import OrderedDict

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), # TODO: bias=False ?
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), # TODO: bias=False ?
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return self.relu(out)
    
class ResidualBlocks(nn.Module):
    def __init__(self, channels, nblocks):
        super(ResidualBlocks, self).__init__()
        self.blocks = nn.Sequential(OrderedDict(
            (f'res_block_{i}', ResidualBlock(channels, channels)) for i in range(nblocks))
        )

    def forward(self, x):
        return self.blocks(x)

class ActionValueModel(nn.Module):
    def __init__(self, n=6, m=6, channels=64, nblocks=6):
        super(ActionValueModel, self).__init__()
        self.residual_tower = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            ResidualBlocks(channels=channels, nblocks=nblocks)
        )
        self.action = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * m * n, m * n),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        w = self.residual_tower(x)
        return self.action(w)
