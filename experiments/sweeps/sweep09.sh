#!/bin/sh
# 実験09（旧 B1）: 融合の取り違えを直す工夫。3000 steps、3 seed
R="python experiments/exp09_fusion_fix.py --seeds 0 1 2 --steps 3000 --tag 09"
for d in fashion mnist; do
  $R --data $d --start indep  --neg none
  $R --data $d --start shared --neg none
  $R --data $d --start indep  --neg shuffle
  $R --data $d --start indep  --neg noise
  $R --data $d --start indep  --neg mix
  $R --data $d --start shared --neg shuffle
done
