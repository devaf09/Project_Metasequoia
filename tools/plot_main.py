"""主要な結果の図を results/ から作り直す（docs/images/ に PNG を書き出す）。

    python tools/plot_main.py

  accuracy_vs_params.png : 正解率とパラメータ数（横軸は対数）。SinWave2D（実験07）と比較相手（実験01b）、10000 steps、3 seed
  fusion.png             : 2 技能の融合の正解率（実験08・08b・09）と、10 クラスを一度に学習した値
"""
import collections
import json
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "docs", "images")
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "Noto Sans CJK JP", "IPAexGothic", "sans-serif"]


def load(exp):
    return [json.loads(line) for line in open(os.path.join(ROOT, "results", f"exp{exp}.txt"), encoding="utf-8")]


def mse(v):
    return 100 * st.mean(v), (100 * st.stdev(v) / len(v) ** .5 if len(v) > 1 else 0.0)


def accuracy_vs_params():
    names = {"logreg": "ロジスティック回帰", "mlp_h3": "MLP", "mlp_h10": "MLP", "cnn_4_8_16": "CNN", "cnn_8_16_32": "CNN"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, d, title in zip(axes, ("mnist", "fashion"), ("MNIST（手書き数字）", "Fashion-MNIST（衣類）")):
        g = collections.defaultdict(list)
        for r in load("01b"):
            if r["data"] == d and r["model"] in names:
                g[(names[r["model"]], r["params"])].append(r["test"])
        for r in load("07"):
            if r["data"] == d and r["cfg"]["N"] == 32:                 # N=64 は N=32 と同じ精度なので省く
                c = r["cfg"]
                g[("SinWave2D" + (" + 滲み" if c["smear"] else "") + (" (N=64)" if c["N"] == 64 else ""), r["params"])].append(r["test"])
        styles = {"ロジスティック回帰": ("s", "tab:gray"), "MLP": ("^", "tab:olive"), "CNN": ("D", "tab:blue")}
        done = set()
        for (name, p), v in sorted(g.items(), key=lambda kv: kv[0][1]):
            m, e = mse(v)
            base = "SinWave2D" if name.startswith("SinWave2D") else name
            mk, col = styles.get(base, ("o", "tab:red"))
            ax.errorbar(p, m, yerr=e, fmt=mk, color=col, ms=7, capsize=3, label=base if base not in done else None)
            done.add(base)
            dy = {"MLP": 6, "ロジスティック回帰": -12}.get(base, -3)
            ax.annotate(f"{m:.1f}" + (" 滲み" if "滲み" in name else ""), (p, m), textcoords="offset points",
                        xytext=(-34 if base == "SinWave2D" and "滲み" in name else 6, dy), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("学習するパラメータの数（対数）")
        ax.set_ylabel("テスト正解率 (%)")
        ax.set_title(title)
        lo = 88 if d == "mnist" else 74
        ax.set_ylim(lo, 99 if d == "mnist" else 89)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("SinWave2D（場 32×32）と比較相手（10000 steps、3 seed の平均±標準誤差。MNIST の MLP 隠れ 3 は範囲外）", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "accuracy_vs_params.png"), dpi=130)


def fusion():
    def pick(exp, cond, key):
        return {d: mse([r[key] for r in load(exp) if r["data"] == d and cond(r)]) for d in ("mnist", "fashion")}

    bars = [
        ("足すだけ\n(10000 steps)", pick("08", lambda r: True, "fused")),
        ("足すだけ\n(3000 steps)", pick("08b", lambda r: r["cfg"].get("illum") == "none" and not r["cfg"]["smear"], "fused")),
        ("+ tilt 照明", pick("08b", lambda r: r["cfg"].get("illum") == "tilt", "fused")),
        ("+ 壊した画像で\n「どれでもない」", pick("09", lambda r: r["start"] == "indep" and r["neg"] == "shuffle", "fused_avg")),
        ("+ 較正\n（今の最良）", pick("09", lambda r: r["start"] == "shared" and r["neg"] == "shuffle", "fused_cal")),
    ]
    joint = {"mnist": 95.2, "fashion": 85.1}                          # 実験03 N=32、3000 steps、seed 0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, d, title in zip(axes, ("mnist", "fashion"), ("MNIST", "Fashion-MNIST")):
        xs = range(len(bars))
        ms = [b[1][d][0] for b in bars]
        es = [b[1][d][1] for b in bars]
        ax.bar(xs, ms, yerr=es, capsize=3, color=["tab:gray"] * 3 + ["tab:orange", "tab:red"])
        for x, m, e in zip(xs, ms, es):
            ax.text(x, m + e + 1.5, f"{m:.1f}", ha="center", fontsize=8)
        ax.axhline(joint[d], color="tab:blue", ls="--", lw=1)
        ax.text(-0.4, joint[d] + 1.5, f"比較: 10 クラスを一度に学習 {joint[d]}（seed 0）", ha="left", color="tab:blue", fontsize=8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([b[0] for b in bars], fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("融合した 10 クラスの正解率 (%)")
        ax.set_title(title)
    fig.suptitle("0〜4 と 5〜9 を別々に学習した 2 技能の融合（相手のデータは使わない、3 seed）", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fusion.png"), dpi=130)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    accuracy_vs_params()
    fusion()
    print("書き出し:", OUT)
