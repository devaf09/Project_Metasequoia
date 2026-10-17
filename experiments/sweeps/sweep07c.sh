#!/bin/sh
# 実験07c（旧 A6mem）: 省メモリ実行。tilt 照明 16 通りを一度に / 1 つずつ / 4 つずつ（勾配は再計算）で時間とメモリを比べる
for c in 0 1 4; do
  python experiments/exp02_sinwave.py --data fashion --seeds 0 --steps 300 --tag 07c --illum tilt --blocks 4 --chunk $c
done
