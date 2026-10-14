#!/bin/sh
# 実験 A0（比較相手）と A1（基本構成、ランダムな板）を MNIST / Fashion-MNIST、3 seed で
set -e
for d in mnist fashion; do
  python experiments/a0_baselines.py --data $d --seeds 0 1 2
  python experiments/a1_sinwave.py --data $d --seeds 0 1 2 --tag A1
  python experiments/a1_sinwave.py --data $d --seeds 0 1 2 --tag A1 --random
done
