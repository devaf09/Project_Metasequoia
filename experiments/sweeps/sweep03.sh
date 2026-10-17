#!/bin/sh
# 実験03（旧 A2）: 入力の当て方と場の大きさ。基本構成から 1 か所だけ変える。seed 0
R="python experiments/exp02_sinwave.py --seeds 0 --tag 03"
for d in fashion mnist; do
  $R --data $d --input_mode phase
  $R --data $d --input_mode both
  $R --data $d --N 32 --scale 1
  $R --data $d --N 128 --scale 2 --patches 16
  $R --data $d --N 128 --scale 4 --patches 16
done
