#!/bin/sh
# 実験 A6 の続き（途中で処理が止まったため、残りだけ）
R="python experiments/a1_sinwave.py --steps 10000 --tag A6 --data fashion"
$R --seeds 1 2 --N 32 --scale 1
$R --seeds 0 1 2 --N 32 --scale 1 --patches 16
$R --seeds 0 1 2 --N 64 --scale 2 --patches 16
$R --seeds 0 1 2 --N 32 --scale 1 --smear 1
for c in 0 1 4; do python experiments/a1_sinwave.py --data fashion --seeds 0 --steps 300 --tag A6mem --illum tilt --blocks 4 --chunk $c; done
