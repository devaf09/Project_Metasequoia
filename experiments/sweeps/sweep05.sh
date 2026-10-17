#!/bin/sh
# 実験05（旧 A4）: 容量（層数・区画数）、ω の学習、背景光、距離、窓の形。seed 0
R="python experiments/exp02_sinwave.py --seeds 0 --tag 05"
for d in fashion mnist; do
  $R --data $d --layers 4
  $R --data $d --layers 8
  $R --data $d --layers 32
  $R --data $d --patches 4
  $R --data $d --patches 16
  $R --data $d --learn_w
  $R --data $d --no_bias
  $R --data $d --z 8
  $R --data $d --z 128
  $R --data $d --det_size 5
  $R --data $d --det_radius 8
done
