#!/bin/sh
# 実験01b（旧 A0b）: 比較相手が学習不足でないかの確認（10000 steps）。SinWave2D の基本構成も同じ steps で
for d in mnist fashion; do
  python experiments/exp01_baselines.py --data $d --seeds 0 1 2 --steps 10000 --tag 01b
  python experiments/exp02_sinwave.py --data $d --seeds 0 1 2 --steps 10000 --tag 01b
done
