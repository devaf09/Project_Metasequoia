#!/bin/sh
# 実験 A7b: 融合を良くする工夫を軽く試す。3000 steps、3 seed。1 次元では tilt 照明が融合を大きく改善した（86.9 → 93.1%）
R="python experiments/a7_fusion.py --seeds 0 1 2 --N 32 --scale 1 --steps 3000 --tag A7b"
for d in fashion mnist; do
  $R --data $d;                                   echo "exit $? $d none"
  $R --data $d --smear 1;                         echo "exit $? $d smear1"
  $R --data $d --illum tilt --blocks 2 --chunk 1; echo "exit $? $d tilt4"
done
