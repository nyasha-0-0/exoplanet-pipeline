"""
model.py — Dual-View 1D CNN with Multi-Scale Inception Blocks
"""

import torch
import torch.nn as nn


class MultiScaleBlock(nn.Module):
    """
    Runs parallel 1D convolutions with different kernel sizes (3, 5, 11)
    to simultaneously detect sharp/narrow transits and broad/shallow transits.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Split channels across 3 parallel kernels (e.g. 16 -> 5 + 5 + 6 = 16)
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - (c1 + c2)

        self.conv3 = nn.Conv1d(in_channels, c1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, c2, kernel_size=5, padding=2)
        self.conv11 = nn.Conv1d(in_channels, c3, kernel_size=11, padding=5)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = torch.cat([self.conv3(x), self.conv5(x), self.conv11(x)], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ConvBranch(nn.Module):
    def __init__(self, in_channels=1, channels=(16, 32, 64)):
        super().__init__()
        layers = []
        prev_c = in_channels
        for c in channels:
            layers.append(MultiScaleBlock(prev_c, c))
            prev_c = c
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = prev_c

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x).squeeze(-1)


class DualViewCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_branch = ConvBranch(channels=(16, 32, 64))
        self.local_branch = ConvBranch(channels=(16, 32, 64))

        combined_dim = self.global_branch.out_dim + self.local_branch.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(inplace=True),
            # Increased dropout to 0.5 to prevent dense layer co-adaptation
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, global_view, local_view):
        g = self.global_branch(global_view.unsqueeze(1))
        l = self.local_branch(local_view.unsqueeze(1))
        combined = torch.cat([g, l], dim=1)
        return self.classifier(combined).squeeze(-1)