#!/bin/sh
# 実験01（旧 A0）: 比較相手を MNIST / Fashion-MNIST、3 seed、3000 steps で
for d in mnist fashion; do
  python experiments/exp01_baselines.py --data $d --seeds 0 1 2 --tag 01
done
