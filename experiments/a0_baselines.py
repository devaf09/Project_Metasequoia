"""実験 A0: 比較相手（ロジスティック回帰、典型との照合、小さな CNN / MLP）。

    python experiments/a0_baselines.py --data mnist --seeds 0 1 2
結果は results/a0.jsonl に 1 行 1 結果で追記する。
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
ap.add_argument("--out", default="results/a0.jsonl")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
D = data.load(a.data, device=dev)
(Xtr, ytr), (Xva, yva), (Xte, yte) = D["train"], D["val"], D["test"]

# 典型との照合（学習なし、seed によらない）
acc = T.proto_accuracy(Xtr.flatten(1), ytr, Xte.flatten(1), yte)
rows = [dict(exp="A0", data=a.data, model="proto_pixels", params=0, seed=-1, test=acc)]
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
        rows.append(dict(exp="A0", data=a.data, model=name, params=sum(p.numel() for p in m.parameters()),
                         seed=s, val=T.accuracy(m, Xva, yva), test=T.accuracy(m, Xte, yte),
                         train_s=t_train, infer_1k_s=t_inf, step_s=t_step, mem_mb=mem, steps=a.steps))
        print(rows[-1], flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
