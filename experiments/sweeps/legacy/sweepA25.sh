#!/bin/sh
# 実験 A2〜A5: 構成の比較（seed 0、MNIST / Fashion-MNIST）。典型の模様での判定 (A5) は毎回同時に出る
R="python experiments/a1_sinwave.py --seeds 0"
for d in fashion mnist; do
  # A2 入力の当て方と場の大きさ
  $R --data $d --tag A2 --input_mode phase
  $R --data $d --tag A2 --input_mode both
  $R --data $d --tag A2 --N 32 --scale 1
  $R --data $d --tag A2 --N 128 --scale 2 --patches 16
  $R --data $d --tag A2 --N 128 --scale 4 --patches 16
  # A3 干渉を減らす工夫の 2 次元版
  $R --data $d --tag A3 --smear 1
  $R --data $d --tag A3 --smear 2
  $R --data $d --tag A3 --smear 4
  $R --data $d --tag A3 --illum tilt --blocks 2
  $R --data $d --tag A3 --illum tiltx --blocks 4
  $R --data $d --tag A3 --illum rand --blocks 7 --patterns 8
  $R --data $d --tag A3 --illum rand --blocks 28 --patterns 8
  $R --data $d --tag A3 --illum tilt --blocks 4 --chunk 8
  # A4 容量・ω・背景光・距離・窓
  $R --data $d --tag A4 --layers 4
  $R --data $d --tag A4 --layers 8
  $R --data $d --tag A4 --layers 32
  $R --data $d --tag A4 --patches 4
  $R --data $d --tag A4 --patches 16
  $R --data $d --tag A4 --learn_w
  $R --data $d --tag A4 --no_bias
  $R --data $d --tag A4 --z 8
  $R --data $d --tag A4 --z 128
  $R --data $d --tag A4 --det_size 5
  $R --data $d --tag A4 --det_radius 8
done
