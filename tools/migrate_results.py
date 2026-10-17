"""旧形式の結果（results/legacy/*.jsonl、実験記号 A0〜B1）を、新しい形式 results/expNN.txt に移す。

    python tools/migrate_results.py

- 旧ファイルは消さない（results/legacy/ に残す）。何度実行しても同じ expNN.txt ができる（上書き）。
- 各行に "exp"（新しい番号）、"old_tag"（旧記号）、"old_file"、"args"（その実行の引数すべて）を入れる。
  旧形式に引数が残っていなかったものは、当時のスクリプトの既定値と sweep の条件で補い、補った引数名を "filled" に列挙する。
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LEG = os.path.join(ROOT, "results", "legacy")

# 旧記号 → 新しい番号
TAG = {"A0": "01", "A0b": "01b", "A1": "02", "A2": "03", "A3": "04", "A4": "05", "A8": "06",
       "A6": "07", "A6t": "07b", "A6mem": "07c", "A7": "08", "A7b": "08b", "B1": "09"}
LR_BASE = {"logreg": 0.01, "mlp_h3": 0.003, "mlp_h10": 0.003, "cnn_4_8_16": 0.003, "cnn_8_16_32": 0.003}


def sinwave_args(r):
    """exp02_sinwave.py の引数を、旧形式の cfg などから組み立てる。"""
    c = r["cfg"]
    known = dict(data=r["data"], steps=r["steps"], lr=r["lr"], random=r.get("random", False), chunk=r.get("chunk", 0),
                 N=c["N"], scale=c["scale"], layers=c["layers"], patches=c["patches"], z=c["z"],
                 input_mode=c["input_mode"], illum=c["illum"], blocks=c["blocks"], patterns=c["patterns"],
                 smear=c["smear"], learn_w=not c["freeze_w"], det_radius=c["det_radius"], det_size=c["det_size"],
                 no_bias=not c["bias"], ref=c.get("ref", 0.0), ref_tilt=c.get("ref_tilt", 0.0))
    filled = dict(seeds=[r["seed"]], batch=256, n_proto=10000)
    return dict(known, **filled, script="exp02_sinwave.py"), sorted(filled)


def fusion_args(r, script):
    c = r["cfg"]
    known = dict(data=r["data"], steps=r["steps"], **{k: c[k] for k in ("N", "scale", "layers", "patches")})
    filled = dict(seeds=[r["seed"]], lr=0.01)
    if script == "exp08_fusion.py":
        known.update(smear=c.get("smear", 0), illum=c.get("illum", "none"), blocks=c.get("blocks", 2))
        filled["chunk"] = 1 if c.get("illum") == "tilt" else 0
    else:
        known.update(start=r["start"], neg=r["neg"], neg_frac=r["neg_frac"])
        filled.update(batch=256, n_train=0, device="cuda")
    return dict(known, **filled, script=script), sorted(filled)


def convert(fname):
    out = []
    for line in open(os.path.join(LEG, fname), encoding="utf-8"):
        r = json.loads(line)
        old = r.pop("exp")
        if fname == "a0b.jsonl":
            old = "A0b"                                               # 比較相手の 10000 steps 版
        new = TAG[old]
        if r.get("model") in LR_BASE or r.get("model") == "proto_pixels":
            m = r["model"]
            args = dict(data=r["data"], steps=r.get("steps", 3000 if old == "A0" else 10000), script="exp01_baselines.py")
            filled = dict(seeds=[0, 1, 2], lr=LR_BASE.get(m), batch=256)
            r.setdefault("lr", LR_BASE.get(m))
            args, filled = dict(args, **filled), sorted(filled)
        elif fname in ("a1.jsonl", "a6t.jsonl"):
            args, filled = sinwave_args(r)
        elif fname == "a7.jsonl":
            args, filled = fusion_args(r, "exp08_fusion.py")
        elif fname == "b1.jsonl":
            args, filled = fusion_args(r, "exp09_fusion_fix.py")
        else:
            raise ValueError(fname)
        args["tag"] = new
        out.append(dict(exp=new, old_tag=old, old_file=f"legacy/{fname}", **r, args=args, filled=filled))
    return out


rows = [row for f in ("a0.jsonl", "a0b.jsonl", "a1.jsonl", "a6t.jsonl", "a7.jsonl", "b1.jsonl") for row in convert(f)]
by = {}
for row in rows:
    by.setdefault(row["exp"], []).append(row)
for exp, v in sorted(by.items()):
    with open(os.path.join(ROOT, "results", f"exp{exp}.txt"), "w", encoding="utf-8") as f:
        for row in v:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"results/exp{exp}.txt: {len(v)} 行")
