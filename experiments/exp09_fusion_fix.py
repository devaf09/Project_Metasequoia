"""実験09（旧 B1）: 2 技能の融合で「どちらの技能の画像か」の取り違えを直す工夫（計画は docs/plan_stage_B.md）。

着想: 実験08 で、各技能が見たことのない相手の画像にも自分の窓を明るくするため、融合が大きく崩れた。
      相手のデータを使わずに、見たことのない画像への過信を抑える方法と、技能どうしのスコアの物差しを
      そろえる方法、1 次元で効いた「共通の出発点」（Sin-CNN2 実験23）を試す。
比べるもの: 工夫なしの融合（実験08b と同じ）、10 クラスを一度に学習（実験03 の N=32）。

    python experiments/exp09_fusion_fix.py --data fashion --seeds 0 1 2 --steps 3000 --start shared --neg shuffle

設定は実験 A7 と同じ（技能 A は 0〜4、技能 B は 5〜9 の画像だけで学習。相手のデータは使わない）。
  --start indep  : A と B を別々の乱数の板から学習（A7 と同じ）
  --start shared : A と B を同じ乱数の板 θ0 から学習（共通の出発点）。差分の足し算 θ0 + ΔA + ΔB で 1 本の光路にもできる
  --neg none / shuffle / noise / mix : 自分のデータを壊した画像（画素の並べ替え / 雑音 / 2 枚の重ね合わせ）を混ぜ、
                   10 個の窓のどれも選ばない（一様な確率）ように学ぶ。見たことのない画像への過信を抑えるため
  融合は 3 通りを同時に測る: 平均（A7 と同じ）、較正つき平均（技能ごとに、自分の訓練データでのスコアの平均と
                   標準偏差で標準化してから平均）、差分の足し算（--start shared のときだけ）
結果は results/exp<tag>.txt に 1 行 1 結果（引数をすべて含む JSON）で追記する。
"""
import argparse
import copy
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sinwave2d import SinWave2D, data  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="mnist")
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--tag", default="09")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--start", default="indep", choices=["indep", "shared"])
ap.add_argument("--neg", default="none", choices=["none", "shuffle", "noise", "mix"])
ap.add_argument("--neg_frac", type=float, default=0.25, help="バッチに混ぜる壊した画像の割合")
ap.add_argument("--out", default=None, help="省略時は results/exp<tag>.txt")
ap.add_argument("--device", default="cuda")
ap.add_argument("--N", type=int, default=32)
ap.add_argument("--scale", type=int, default=1)
ap.add_argument("--layers", type=int, default=16)
ap.add_argument("--patches", type=int, default=8)
ap.add_argument("--n_train", type=int, default=0, help="0 なら全部（テスト用に小さくできる）")
a = ap.parse_args()
a.out = a.out or f"results/exp{a.tag}.txt"
ARGS = dict(vars(a), script="exp09_fusion_fix.py")

dev = a.device
D = data.load(a.data, device=dev)
(Xtr, ytr), (Xte, yte) = D["train"], D["test"]
if a.n_train:
    Xtr, ytr, Xte, yte = Xtr[:a.n_train], ytr[:a.n_train], Xte[:a.n_train], yte[:a.n_train]
cfg = dict(N=a.N, scale=a.scale, layers=a.layers, patches=a.patches)
own = {"A": list(range(0, 5)), "B": list(range(5, 10))}


def corrupt(x, g):
    """自分のデータを壊した画像（どのクラスでもないはずの画像）。"""
    n = x.shape[0]
    if a.neg == "shuffle":                                            # 画素の並べ替え（明るさの分布は同じ、形が無い）
        perm = torch.randperm(28 * 28, generator=g).to(x.device)
        return x.reshape(n, -1)[:, perm].reshape(n, 28, 28)
    if a.neg == "noise":                                              # 平均の明るさをそろえた一様な雑音
        r = torch.rand(x.shape, generator=g).to(x.device)
        return r * x.mean((1, 2), keepdim=True) * 2
    if a.neg == "mix":                                                # 違うクラスかもしれない 2 枚の重ね合わせ
        j = torch.randperm(n, generator=g).to(x.device)
        return (x + x[j]) / 2
    raise ValueError(a.neg)


def train(m, X, y, seed):
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    g = torch.Generator().manual_seed(seed)
    m.train()
    nb = int(a.batch * a.neg_frac) if a.neg != "none" else 0
    for _ in range(a.steps):
        i = torch.randint(len(y), (a.batch,), generator=g).to(X.device)
        loss = F.cross_entropy(m(X[i]), y[i])
        if nb:                                                        # 壊した画像は 10 個の窓に一様（どれも選ばない）
            out = m(corrupt(X[i[:nb]], g))
            loss = loss + a.neg_frac * -(F.log_softmax(out, 1).mean(1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()


@torch.no_grad()
def scores(m, X, bs=500):
    m.eval()
    return torch.cat([m(X[i:i + bs]) for i in range(0, len(X), bs)])


def acc(S, y, cls=None):
    if cls is None:
        return (S.argmax(1) == y).float().mean().item()
    sel = torch.isin(y, torch.tensor(cls, device=y.device))
    mask = torch.full((10,), float("-inf"), device=S.device)
    mask[cls] = 0
    return ((S[sel] + mask).argmax(1) == y[sel]).float().mean().item()


def add_deltas(base, nets):
    """差分の足し算: θ0 + Σ (θk − θ0)。背景光は、それぞれの担当の窓の値を使う。"""
    out = copy.deepcopy(base)
    with torch.no_grad():
        for name in ("A", "phi", "gain"):
            b = getattr(base, name)
            getattr(out, name).copy_(b + sum(getattr(n, name) - b for n in nets.values()))
        for k, n in nets.items():
            out.bias[own[k]] = n.bias[own[k]]
    return out


for s in a.seeds:
    base = SinWave2D(seed=s, **cfg).to(dev)
    nets = {}
    for k, sd in (("A", s), ("B", s + 100)):
        sel = torch.isin(ytr, torch.tensor(own[k], device=dev))
        torch.manual_seed(sd)
        m = copy.deepcopy(base) if a.start == "shared" else SinWave2D(seed=sd, **cfg).to(dev)
        train(m, Xtr[sel], ytr[sel], sd)
        nets[k] = m
    S = {k: scores(m, Xte) for k, m in nets.items()}
    # 較正: 自分の訓練データ（最大 5,000 枚）での、担当の窓のスコアの平均と標準偏差で標準化
    cal = {}
    for k, m in nets.items():
        sel = torch.isin(ytr, torch.tensor(own[k], device=dev))
        St = scores(m, Xtr[sel][:5000])[:, own[k]]
        cal[k] = (St.mean().item(), St.std().item() + 1e-6)
    S_avg = (S["A"] + S["B"]) / 2
    S_cal = sum((S[k] - cal[k][0]) / cal[k][1] for k in S) / 2
    row = dict(exp=a.tag, data=a.data, seed=s, start=a.start, neg=a.neg, neg_frac=a.neg_frac, steps=a.steps, cfg=cfg, lr=a.lr, args=ARGS,
               A_alone=acc(S["A"], yte, own["A"]), B_alone=acc(S["B"], yte, own["B"]),
               fused_avg=acc(S_avg, yte), fused_cal=acc(S_cal, yte),
               fused_avg_onA=acc(S_avg[yte < 5], yte[yte < 5]), fused_avg_onB=acc(S_avg[yte >= 5], yte[yte >= 5]))
    if a.start == "shared":
        S_sum = scores(add_deltas(base, nets), Xte)
        row.update(fused_sum=acc(S_sum, yte), fused_sum_onA=acc(S_sum[yte < 5], yte[yte < 5]),
                   fused_sum_onB=acc(S_sum[yte >= 5], yte[yte >= 5]))
    print(json.dumps(row, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
