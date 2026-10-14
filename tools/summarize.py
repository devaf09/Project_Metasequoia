"""results/*.jsonl を構成ごとにまとめて表示する（平均±標準誤差、%）。

    python tools/summarize.py results/a0.jsonl results/a1.jsonl
"""
import collections
import json
import statistics as st
import sys

BASE = dict(N=64, scale=2, layers=16, patches=8, z=32.0, input_mode="amp", illum="none", blocks=4, patterns=8,
            smear=0, freeze_w=True, det_radius=None, det_size=3, bias=True, ref=0.0, ref_tilt=0.0)


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return "-"
    m = st.mean(v) * 100
    return f"{m:.1f}±{st.stdev(v) / len(v) ** .5 * 100:.1f}" if len(v) > 1 else f"{m:.1f}"


rows = [json.loads(line) for f in sys.argv[1:] for line in open(f, encoding="utf-8")]
groups = collections.defaultdict(list)
for r in rows:
    cfg = r.get("cfg", {})
    diff = {k: v for k, v in cfg.items() if BASE.get(k) != v}
    key = (r["exp"], r["data"], r["model"] + (" random" if r.get("random") else ""), json.dumps(diff, ensure_ascii=False))
    groups[key].append(r)
for (exp, d, model, diff), v in groups.items():
    avg = lambda k: st.mean((r.get(k) or 0) for r in v)
    print(f"{exp:3s} {d:7s} {model:18s} {diff:45s} n={len(v)} params={v[0]['params']:5d} "
          f"test={ms([r.get('test') for r in v]):9s} proto={ms([r.get('proto') for r in v]):9s} "
          f"step={avg('step_s'):.4f}s inf1k={avg('infer_1k_s'):.4f}s mem={avg('mem_mb'):.0f}MB train={avg('train_s'):.0f}s")
