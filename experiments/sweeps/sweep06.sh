#!/bin/sh
# 実験06（旧 A8）: ホログラムの考え方。(1) 拡大と距離のスケーリング則（拡大 s 倍 ⇔ 距離 1/s²）、(2) 参照波を重ねた入力。seed 0
R="python experiments/exp02_sinwave.py --seeds 0 --tag 06"
for d in fashion mnist; do
  $R --data $d --N 128 --scale 2 --patches 16 --z 8    # 拡大 4・距離 32 と同等のはず
  $R --data $d --N 128 --scale 4 --patches 16 --z 128  # 拡大 2・距離 32 と同等のはず
  for r in 0.1 0.3 1.0; do $R --data $d --ref $r; done
  for t in 8 16; do $R --data $d --ref 0.3 --ref_tilt $t; done
done
