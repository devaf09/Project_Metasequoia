#!/bin/sh
# 実験08（旧 A7）: 2 技能（0〜4 / 5〜9）を別々に学習し、明るさの平均で融合。N=32、拡大なし、区画 8×8、10000 steps、3 seed
for d in mnist fashion; do
  python experiments/exp08_fusion.py --data $d --seeds 0 1 2 --N 32 --scale 1 --steps 10000 --tag 08
done
