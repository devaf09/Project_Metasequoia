"""実験 A1〜A6: SinWave2D の学習（オプションで構成を切り替える）。

    python experiments/a1_sinwave.py --data mnist --seeds 0 --tag A1 [--illum tilt --blocks 4 ...]
窓での判定に加え、典型の模様での判定（出力面全体とクラス平均の相関）も出す。
--random を付けると板を学習せず、典型の模様だけで判定する（干渉と回折そのものの力）。
結果は results/a1.jsonl に追記する。
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
ap.add_argument("--tag", default="A1")
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--random", action="store_true")
ap.add_argument("--n_proto", type=int, default=10000, help="典型を作る訓練画像の枚数")
ap.add_argument("--out", default="results/a1.jsonl")
# 構成
ap.add_argument("--N", type=int, default=64)
ap.add_argument("--scale", type=int, default=2)
ap.add_argument("--layers", type=int, default=16)
ap.add_argument("--patches", type=int, default=8)
ap.add_argument("--z", type=float, default=32.0)
ap.add_argument("--input_mode", default="amp")
ap.add_argument("--illum", default="none")
ap.add_argument("--blocks", type=int, default=4)
ap.add_argument("--patterns", type=int, default=8)
ap.add_argument("--smear", type=int, default=0)
ap.add_argument("--learn_w", action="store_true")
ap.add_argument("--det_radius", type=float, default=None)
ap.add_argument("--det_size", type=int, default=3)
ap.add_argument("--no_bias", action="store_true")
ap.add_argument("--chunk", type=int, default=0)
ap.add_argument("--ref", type=float, default=0.0)
ap.add_argument("--ref_tilt", type=float, default=0.0)
a = ap.parse_args()

dev = "cuda"
D = data.load(a.data, device=dev)
(Xtr, ytr), (Xva, yva), (Xte, yte) = D["train"], D["val"], D["test"]
cfg = dict(N=a.N, scale=a.scale, layers=a.layers, patches=a.patches, z=a.z, input_mode=a.input_mode,
           illum=a.illum, blocks=a.blocks, patterns=a.patterns, smear=a.smear, freeze_w=not a.learn_w,
           det_radius=a.det_radius, det_size=a.det_size, bias=not a.no_bias, chunk=a.chunk, ref=a.ref, ref_tilt=a.ref_tilt)

for s in a.seeds:
    torch.manual_seed(s)
    m = SinWave2D(seed=s, **cfg).to(dev)
    t_train = 0.0 if a.random else T.train(m, Xtr, ytr, steps=a.steps, lr=a.lr, batch_size=a.batch, seed=s)
    t_inf, t_step, mem = T.measure(m, Xte)
    Ftr = T.features(m, Xtr[:a.n_proto])
    proto = T.proto_accuracy(Ftr, ytr[:a.n_proto], T.features(m, Xte), yte)
    row = dict(exp=a.tag, data=a.data, model="sinwave2d", random=a.random, params=m.n_params(), seed=s,
               val=None if a.random else T.accuracy(m, Xva, yva), test=None if a.random else T.accuracy(m, Xte, yte),
               proto=proto, train_s=t_train, infer_1k_s=t_inf, step_s=t_step, mem_mb=mem, steps=a.steps,
               lr=a.lr, cfg={k: v for k, v in cfg.items() if k != "chunk"}, chunk=a.chunk)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
