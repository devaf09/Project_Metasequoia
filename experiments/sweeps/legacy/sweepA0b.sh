#!/bin/sh
# 実験 A0b: 比較相手が学習不足でないかの確認（10000 steps）。SinWave も同じ steps で
for d in mnist fashion; do
  python experiments/a0_baselines.py --data $d --seeds 0 1 2 --steps 10000 --out results/a0b.jsonl
  python experiments/a1_sinwave.py --data $d --seeds 0 1 2 --tag A0b --steps 10000
done
