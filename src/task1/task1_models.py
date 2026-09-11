"""Checkpoint-compatible architectures extracted from the executed Task 1 runtime.

ResNet topology follows He et al. (2016); the 3x3 stem preserves thumbnail detail.
No training or pretrained downloads run when this module is imported.
"""
import torch
from torch import nn
from torch.nn import functional as F
N_CLASSES = 124

class PlainCNN(nn.Module):
    """VGG-style stack sized for 60x80 inputs. The reference architecture."""

    def __init__(self, n_classes=None, width=32, dropout=0.3):
        super().__init__()
        n_classes = n_classes or N_CLASSES

        def block(in_channels, out_channels, pool=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3, width),                 # 80x60 -> 40x30
            block(width, width * 2),         # 40x30 -> 20x15
            block(width * 2, width * 4),     # 20x15 -> 10x7
            block(width * 4, width * 8, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(width * 8, n_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))

class BasicBlock(nn.Module):
    """Standard two-convolution residual block with an optional projection shortcut."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # A shortcut can only be added to the block output if both match in shape, so a
        # change of stride or width needs a 1x1 projection to bring it into line.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)

class SmallResNet(nn.Module):
    """ResNet-18 topology with a stride-1 3x3 stem, sized for 60x80 inputs."""

    def __init__(self, n_classes=None, width=64, blocks=(2, 2, 2, 2), dropout=0.3):
        super().__init__()
        n_classes = n_classes or N_CLASSES
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )
        stages, in_channels = [], width
        for stage_index, n_blocks in enumerate(blocks):
            out_channels = width * (2 ** stage_index)
            for block_index in range(n_blocks):
                # Downsample once per stage, at its first block, and never in stage 0.
                stride = 2 if (block_index == 0 and stage_index > 0) else 1
                stages.append(BasicBlock(in_channels, out_channels, stride))
                in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, n_classes)

    def forward(self, x):
        x = self.pool(self.stages(self.stem(x))).flatten(1)
        return self.fc(self.dropout(x))
