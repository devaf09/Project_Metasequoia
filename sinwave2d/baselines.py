"""比較相手: ロジスティック回帰、小さな CNN、小さな MLP（画像 (B, 28, 28) → logit）。"""
from __future__ import annotations

import torch.nn as nn


def logreg(n_classes: int = 10) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(784, n_classes))


def mlp(hidden: int, n_classes: int = 10) -> nn.Module:
    """hidden=3 で 2,395 個、hidden=10 で 7,960 個。"""
    return nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))


class CNN(nn.Module):
    """3 段の畳み込み + 全体平均 + 線形。ch=(4,8,16) で 1,674 個、(8,16,32) で 6,218 個。"""

    def __init__(self, ch=(4, 8, 16), n_classes: int = 10):
        super().__init__()
        a, b, c = ch
        self.f = nn.Sequential(
            nn.Conv2d(1, a, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(b, c, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, n_classes))

    def forward(self, x):
        return self.f(x[:, None])
