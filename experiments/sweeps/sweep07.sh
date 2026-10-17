#!/bin/sh
# 実験07（旧 A6）: 良い候補を 3 seed・10000 steps で（比較相手の 01b と同じ steps）
# 当時は途中で 2 回止まり、残りを分けて再実行した（legacy/sweepA6.sh, A6b.sh, A6c.sh）。条件はこの一覧と同じ
R="python experiments/exp02_sinwave.py --seeds 0 1 2 --steps 10000 --tag 07"
for d in mnist fashion; do
  $R --data $d --N 32 --scale 1
  $R --data $d --N 32 --scale 1 --patches 16
  $R --data $d --N 64 --scale 2 --patches 16
  $R --data $d --N 32 --scale 1 --smear 1
done
