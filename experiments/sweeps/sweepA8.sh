#!/bin/sh
# 実験 A8: ホログラムの考え方。(1) 拡大と距離のスケーリング則（拡大 s 倍 ⇔ 距離 1/s²）、(2) 参照波を重ねた入力
R="python experiments/a1_sinwave.py --seeds 0"
for d in fashion mnist; do
  $R --data $d --tag A8 --N 128 --scale 4 --patches 16 --z 8
  $R --data $d --tag A8 --N 128 --scale 2 --patches 16 --z 128
  for r in 0.1 0.3 1.0; do $R --data $d --tag A8 --ref $r; done
  for t in 8 16; do $R --data $d --tag A8 --ref 0.3 --ref_tilt $t; done
done
