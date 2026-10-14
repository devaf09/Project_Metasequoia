"""学習・評価・計算量の測定（SinWave2D と比較相手で共通）。"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F


def train(model, X, y, steps: int = 3000, lr: float = 0.01, batch_size: int = 256, seed: int = 0,
          weight_decay: float = 0.0, log_every: int = 0):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    t0 = time.perf_counter()
    for step in range(steps):
        i = torch.randint(len(y), (batch_size,), generator=gen).to(X.device)
        loss = F.cross_entropy(model(X[i]), y[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if log_every and step % log_every == 0:
            print(f"  step {step:5d} loss {loss.item():.4f}", flush=True)
    if X.is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


@torch.no_grad()
def accuracy(model, X, y, batch_size: int = 500) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(y), batch_size):
        correct += (model(X[i:i + batch_size]).argmax(1) == y[i:i + batch_size]).sum().item()
    return correct / len(y)


@torch.no_grad()
def features(model, X, batch_size: int = 500) -> torch.Tensor:
    """SinWave2D の出力面の明るさ全体 (N, N*N)。"""
    model.eval()
    return torch.cat([model.intensity(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])


@torch.no_grad()
def proto_accuracy(Ftr, ytr, Fte, yte, n_classes: int = 10) -> float:
    """典型の模様での判定: クラスごとの平均の模様と、中心化した相関（コサイン）が最大のクラス。"""
    mu = Ftr.mean(0, keepdim=True)
    P = torch.stack([(Ftr[ytr == c] - mu).mean(0) for c in range(n_classes)])
    P = P / P.norm(dim=1, keepdim=True)
    Q = Fte - mu
    Q = Q / (Q.norm(dim=1, keepdim=True) + 1e-12)
    return ((Q @ P.T).argmax(1) == yte).float().mean().item()


def measure(model, X, batch_size: int = 256, reps: int = 5):
    """推論 1,000 枚の時間 (s)、学習 1 step の時間 (s)、学習時のピーク GPU メモリ (MB)。"""
    dev = X.device
    xb = X[:batch_size]
    yb = torch.zeros(batch_size, dtype=torch.long, device=dev)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    model.eval()
    with torch.no_grad():
        model(X[:500]); sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(X[:1000] if len(X) >= 1000 else X)
        sync()
        t_inf = (time.perf_counter() - t0) / reps
    model.train()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    F.cross_entropy(model(xb), yb).backward(); sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(xb), yb).backward()
    sync()
    t_step = (time.perf_counter() - t0) / reps
    mem = (torch.cuda.max_memory_allocated() - base) / 2**20 if dev.type == "cuda" else float("nan")
    model.zero_grad(set_to_none=True)
    return t_inf, t_step, mem
