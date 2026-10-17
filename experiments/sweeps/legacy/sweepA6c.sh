#!/bin/sh
# 実験 A6 の残り（2 回目の再開。終了コードも記録する）
R="python experiments/a1_sinwave.py --steps 10000 --tag A6 --data fashion"
$R --seeds 1 2 --N 32 --scale 1;               echo "exit $? N32"
$R --seeds 0 1 2 --N 32 --scale 1 --patches 16; echo "exit $? N32 p16"
$R --seeds 0 1 2 --N 64 --scale 2 --patches 16; echo "exit $? N64 p16"
