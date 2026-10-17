"""実験01 / 01b（旧 A0 / A0b）: 比較相手。

着想: SinWave の数字が良いか悪いかを判断するには、同じデータ・同じ分け方・同じ学習の長さで測った
      定番の手法が要る（段階 A の計画 docs/plan_stage_A.md）。
方法: 画素のクラス平均との照合（学習なし）、ロジスティック回帰、小さな MLP（隠れ 3 / 10）、
      小さな CNN（1,674 / 6,218 パラメータ）を Adam・余弦スケジュール・バッチ 256 で学習する。
      01 は 3000 steps、01b は 10000 steps（01 で CNN が学習不足だったため）。
比べるもの: SinWave2D（exp02_sinwave.py）。

    python experiments/exp01_baselines.py --data mnist --seeds 0 1 2 [--steps 10000 --tag 01b]
結果は results/exp<tag>.txt に 1 行 1 結果（引数をすべて含む JSON）で追記する。
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sinwave2d import baselines, data, train as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="mnist")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--tag", default="01")
ap.add_argument("--out", default=None, help="省略時は results/exp<tag>.txt")
a = ap.parse_args()
a.out = a.out or f"results/exp{a.tag}.txt"
ARGS = dict(vars(a), script="exp01_baselines.py")

dev = "cuda" if torch.cuda.is_available() else "cpu"
D = data.load(a.data, device=dev)
(Xtr, ytr), (Xva, yva), (Xte, yte) = D["train"], D["val"], D["test"]

# 典型との照合（学習なし、seed によらない）
acc = T.proto_accuracy(Xtr.flatten(1), ytr, Xte.flatten(1), yte)
rows = [dict(exp=a.tag, data=a.data, model="proto_pixels", params=0, seed=-1, test=acc, args=ARGS)]
print(rows[-1], flush=True)

models = {
    "logreg": (lambda: baselines.logreg(), 0.01),
    "mlp_h3": (lambda: baselines.mlp(3), 0.003),
    "mlp_h10": (lambda: baselines.mlp(10), 0.003),
    "cnn_4_8_16": (lambda: baselines.CNN((4, 8, 16)), 0.003),
    "cnn_8_16_32": (lambda: baselines.CNN((8, 16, 32)), 0.003),
}
for name, (make, lr) in models.items():
    for s in a.seeds:
        torch.manual_seed(s)
        m = make().to(dev)
        t_train = T.train(m, Xtr, ytr, steps=a.steps, lr=lr, seed=s)
        t_inf, t_step, mem = T.measure(m, Xte)
        rows.append(dict(exp=a.tag, data=a.data, model=name, params=sum(p.numel() for p in m.parameters()),
                         seed=s, val=T.accuracy(m, Xva, yva), test=T.accuracy(m, Xte, yte),
                         train_s=t_train, infer_1k_s=t_inf, step_s=t_step, mem_mb=mem, steps=a.steps, lr=lr, args=ARGS))
        print(rows[-1], flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
