#!/bin/sh
# 実験04（旧 A3）: 干渉を減らす工夫の 2 次元版（滲み層、tilt 照明、乱数位相）。seed 0
R="python experiments/exp02_sinwave.py --seeds 0 --tag 04"
for d in fashion mnist; do
  $R --data $d --smear 1
  $R --data $d --smear 2
  $R --data $d --smear 4
  $R --data $d --illum tilt --blocks 2
  $R --data $d --illum tiltx --blocks 4
  $R --data $d --illum rand --blocks 7 --patterns 8
  $R --data $d --illum rand --blocks 28 --patterns 8
  $R --data $d --illum tilt --blocks 4 --chunk 8
done
