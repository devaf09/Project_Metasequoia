"""実験 A7: 2 技能に分けた並列光路の融合（0〜4 と 5〜9）。

    python experiments/a7_fusion.py --data mnist --seeds 0 1 2 --N 32 --scale 1 --steps 10000

技能 A は 0〜4 の画像だけ、技能 B は 5〜9 の画像だけで別々に学習する（seed も別）。
どちらも窓は 10 個すべてで損失を取るので、担当外の窓は暗くなるように学ぶ（1 次元版 sinwave_minimal.py と同じ）。
融合は 2 つの光路のスコア（窓の明るさ）の平均を取るだけで、追加の学習も相手のデータも使わない。
取消は融合から外すだけで、残った技能は元と完全に同じ。
結果は results/a7.jsonl に追記する。
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sinwave2d import SinWave2D, data, train as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="mnist")
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--tag", default="A7")
ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--out", default="results/a7.jsonl")
ap.add_argument("--N", type=int, default=32)
ap.add_argument("--scale", type=int, default=1)
ap.add_argument("--layers", type=int, default=16)
ap.add_argument("--patches", type=int, default=8)
ap.add_argument("--smear", type=int, default=0)
ap.add_argument("--illum", default="none")
ap.add_argument("--blocks", type=int, default=2)
ap.add_argument("--chunk", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
D = data.load(a.data, device=dev)
(Xtr, ytr), (Xte, yte) = D["train"], D["test"]
cfg = dict(N=a.N, scale=a.scale, layers=a.layers, patches=a.patches, smear=a.smear, illum=a.illum, blocks=a.blocks)
own = {"A": list(range(0, 5)), "B": list(range(5, 10))}


@torch.no_grad()
def scores(models, X, bs=500):
    for m in models:
        m.eval()
    return torch.cat([sum(m(X[i:i + bs]) for m in models) / len(models) for i in range(0, len(X), bs)])


def acc_own(S, y, cls):
    """担当のクラスの画像だけを、担当の窓の中で判定した正解率。"""
    sel = torch.isin(y, torch.tensor(cls, device=y.device))
    mask = torch.full((10,), float("-inf"), device=S.device)
    mask[cls] = 0
    return ((S[sel] + mask).argmax(1) == y[sel]).float().mean().item()


for s in a.seeds:
    nets, t_train = {}, 0.0
    for k, sd in (("A", s), ("B", s + 100)):
        sel = torch.isin(ytr, torch.tensor(own[k], device=dev))
        torch.manual_seed(sd)
        m = SinWave2D(seed=sd, chunk=a.chunk, **cfg).to(dev)
        t_train += T.train(m, Xtr[sel], ytr[sel], steps=a.steps, lr=a.lr, seed=sd)
        nets[k] = m
    SA, SB, SF = scores([nets["A"]], Xte), scores([nets["B"]], Xte), scores([nets["A"], nets["B"]], Xte)
    row = dict(exp=a.tag, data=a.data, seed=s, params_each=nets["A"].n_params(), steps=a.steps, cfg=cfg,
               A_alone=acc_own(SA, yte, own["A"]),            # 技能 A 単独、0〜4 の中で
               B_alone=acc_own(SB, yte, own["B"]),            # 技能 B 単独、5〜9 の中で
               fused=(SF.argmax(1) == yte).float().mean().item(),      # 融合、10 クラス
               fused_A=acc_own(SF, yte, own["A"]),            # 融合したまま、0〜4 の中で
               fused_B=acc_own(SF, yte, own["B"]),
               fused_on_A=(SF.argmax(1)[yte < 5] == yte[yte < 5]).float().mean().item(),   # 0〜4 の画像、10 クラスで
               fused_on_B=(SF.argmax(1)[yte >= 5] == yte[yte >= 5]).float().mean().item(),
               train_s=t_train)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
