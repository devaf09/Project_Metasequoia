#!/bin/sh
# 実験 A6: 良い候補を 3 seed・10000 steps で（比較相手の A0b と同じ steps）、省メモリ実行の測定
R="python experiments/a1_sinwave.py --seeds 0 1 2 --steps 10000 --tag A6"
for d in mnist fashion; do
  $R --data $d --N 32 --scale 1
  $R --data $d --N 32 --scale 1 --patches 16
  $R --data $d --N 64 --scale 2 --patches 16
  $R --data $d --N 32 --scale 1 --smear 1
done
# 省メモリ: tilt 16 通りを一度に / 1 つずつ（勾配は再計算）で、時間とメモリを比べる（精度は見ない）
for c in 0 1 4; do python experiments/a1_sinwave.py --data fashion --seeds 0 --steps 300 --tag A6mem --illum tilt --blocks 4 --chunk $c; done
