#!/bin/sh
# 実験 B1: 融合の取り違えを直す工夫。3000 steps、3 seed、MNIST と Fashion（計画: docs/plan_stage_B.md）
R="python experiments/b1_fusion.py --seeds 0 1 2 --steps 3000"
for d in fashion mnist; do
  $R --data $d --start indep  --neg none;    echo "exit $? $d indep none"
  $R --data $d --start shared --neg none;    echo "exit $? $d shared none"
  $R --data $d --start indep  --neg shuffle; echo "exit $? $d indep shuffle"
  $R --data $d --start indep  --neg noise;   echo "exit $? $d indep noise"
  $R --data $d --start indep  --neg mix;     echo "exit $? $d indep mix"
  $R --data $d --start shared --neg shuffle; echo "exit $? $d shared shuffle"
done
