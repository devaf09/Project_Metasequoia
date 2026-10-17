#!/bin/sh
# 実験07b（旧 A6t）: 07 の時間の測り直し（他の処理と重なった可能性があったため）。1 本ずつ、seed 0、1000 steps
R="python experiments/exp02_sinwave.py --seeds 0 --steps 1000 --tag 07b"
for d in mnist fashion; do
  $R --data $d --N 32 --scale 1
  $R --data $d --N 32 --scale 1 --patches 16
  $R --data $d --N 64 --scale 2 --patches 16
  $R --data $d --N 32 --scale 1 --smear 1
done
