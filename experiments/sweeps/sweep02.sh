#!/bin/sh
# 実験02（旧 A1）: 基本構成（N=64、拡大 2、16 層、区画 8×8、3000 steps）と、学習しないランダムな板。3 seed
for d in mnist fashion; do
  python experiments/exp02_sinwave.py --data $d --seeds 0 1 2 --tag 02
  python experiments/exp02_sinwave.py --data $d --seeds 0 1 2 --tag 02 --random
done
