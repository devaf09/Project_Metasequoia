"""MNIST / Fashion-MNIST の読み込み（torchvision で取得済みの data/ から）。

訓練 60,000 枚のうち末尾 5,000 枚を検証用に分ける。値は 0〜1 の float、形は (N, 28, 28)。
"""
from __future__ import annotations

import torch
import torchvision as tv

_SETS = {"mnist": tv.datasets.MNIST, "fashion": tv.datasets.FashionMNIST}


def load(name: str, root: str = "data", device=None, n_val: int = 5000):
    """→ dict(train=(X, y), val=(X, y), test=(X, y))。ダウンロードはしない。"""
    C = _SETS[name]
    tr = C(root, train=True, download=False)
    te = C(root, train=False, download=False)
    Xtr = tr.data.float().div(255)
    ytr = tr.targets.clone()
    Xte = te.data.float().div(255)
    yte = te.targets.clone()
    out = {"train": (Xtr[:-n_val], ytr[:-n_val]), "val": (Xtr[-n_val:], ytr[-n_val:]), "test": (Xte, yte)}
    if device is not None:
        out = {k: (x.to(device), y.to(device)) for k, (x, y) in out.items()}
    return out
