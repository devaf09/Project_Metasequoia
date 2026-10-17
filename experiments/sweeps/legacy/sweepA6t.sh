#!/bin/sh
# 実験 A6t: A6 の時間の測り直し（A6b・A6c は GPU の使用が重なった可能性があり、時間が当てにならないため）
# 精度は見ない。1 本ずつ順番に、seed 0、1000 steps。結果は results/a6t.jsonl
R="python experiments/a1_sinwave.py --seeds 0 --steps 1000 --tag A6t --out results/a6t.jsonl"
for d in mnist fashion; do
  $R --data $d --N 32 --scale 1;               echo "exit $? $d N32"
  $R --data $d --N 32 --scale 1 --patches 16;  echo "exit $? $d N32 p16"
  $R --data $d --N 64 --scale 2 --patches 16;  echo "exit $? $d N64 p16"
  $R --data $d --N 32 --scale 1 --smear 1;     echo "exit $? $d N32 smear1"
done
