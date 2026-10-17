#!/bin/sh
# 実験08b（旧 A7b）: 融合を良くする工夫を軽く（滲み、tilt 照明）。3000 steps、3 seed
R="python experiments/exp08_fusion.py --seeds 0 1 2 --N 32 --scale 1 --steps 3000 --tag 08b"
for d in fashion mnist; do
  $R --data $d
  $R --data $d --smear 1
  $R --data $d --illum tilt --blocks 2 --chunk 1
done
