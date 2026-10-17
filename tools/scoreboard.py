"""Project Metasequoia（2 次元の SinWave）の到達度を「ポイント」で表す（合計 100 点）。

    python tools/scoreboard.py            # 表を表示
    python tools/scoreboard.py --md       # docs/scoreboard.md を書き出す

考え方（前のプロジェクト Sin-CNN2 の tools/scoreboard.py と同じ）
  各項目に「ゼロ点の基準（何もしない場合: 偶然・最頻の答えなど）」と「目標（最低限の指標: 定番の手法の値）」を置き、
  今の値がその間のどこにあるかで採点する（目標に届けば満点、越えても満点）。
    点 = 配点 × clip((今の値 − ゼロ点の基準) / (目標 − ゼロ点の基準), 0, 1)
  小さいほど良い項目（融合の差、時間など）も、同じ式でそのまま扱える（基準と目標の大小が逆になるだけ）。
  時間のように桁で変わるものは、対数（log10）で採点する。
  画像の目標は、同じ条件（10000 steps）で測った小さな CNN の値（2026-10-17 ユーザーと決定。進めるうちに見直すことがある）。
新しい結果が出たら、下の ITEMS の「今の値」と「出典」を書き換えて再実行する。未測定の項目は None（0 点）として残す。
"""
import math
import sys

# (軸, 項目, 配点, ゼロ点の基準, 目標, 今の値, 対数で採点, 目標の根拠, 今の値の出典)。今の値が None なら未測定（0 点）
ITEMS = [
    ("画像", "MNIST の正解率 (%)", 15, 10.0, 96.3, 96.7, False,
     "同程度の大きさの CNN 1,674 パラメータ（10000 steps、3 seed）", "実験07 N=32・区画 16×16（8,203 パラメータ）"),
    ("画像", "Fashion-MNIST の正解率 (%)", 15, 10.0, 86.4, 87.3, False,
     "CNN 6,218 パラメータ（10000 steps、3 seed）", "実験07 N=32・区画 16×16"),
    ("得意分野", "2 技能の融合 MNIST（一度に学習との差、点。小さいほど良い）", 10, 45.0, 0.0, 4.5, False,
     "差なし（ゼロ点は片方の技能の画像をすべて取り違える場合）", "実験09 共通・並べ替え・較正 90.7 vs 95.2"),
    ("得意分野", "2 技能の融合 Fashion（一度に学習との差、点）", 10, 40.0, 0.0, 4.8, False,
     "差なし（ゼロ点は同上）", "実験09 80.3 vs 85.1"),
    ("得意分野", "Split-MNIST のクラス増分（5 課題の後の 10 クラス、%）", 15, 19.9, 80.0, None, False,
     "少量の過去データを使う既存手法（リプレイ）", "未測定（段階 B）"),
    ("得意分野", "4 技能以上の融合（一度に学習との差、点）", 10, 70.0, 0.0, None, False,
     "差なし（ゼロ点は 1 つの技能のクラスしか当たらない場合）", "未測定（段階 B）"),
    ("得意分野", "取消の厳密さ（外して元に戻るか）", 5, 0.0, 1.0, 1.0, False,
     "完全に戻る", "tests/test_minimal2d.py"),
    ("効率", "1 技能の大きさ（KB、小さいほど良い）", 5, 1000.0, 50.0, 32.0, False,
     "50KB 以下", "8,203 パラメータ × 4 バイト"),
    ("効率", "推論の速さ（µs/枚、GPU、小さいほど良い）", 7.5, 1000.0, 1.0, 10.0, True,
     "1µs（CNN 1,674 パラメータは約 1µs）", "実験07b N=32 推論 1,000 枚 0.010 秒"),
    ("効率", "1 技能の学習時間（秒、小さいほど良い）", 7.5, 3600.0, 10.0, 210.0, True,
     "10 秒", "実験07b 0.021 秒 × 10000 steps"),
]


def score(lo, hi, cur, use_log):
    if cur is None:
        return 0.0
    if use_log:
        lo, hi, cur = math.log10(lo), math.log10(hi), math.log10(cur)
    r = (cur - lo) / (hi - lo)
    return max(0.0, min(1.0, r))


def table():
    rows, axes = [], {}
    for axis, name, w, lo, hi, cur, lg, why, src in ITEMS:
        r = score(lo, hi, cur, lg)
        rows.append((axis, name, w, lo, hi, cur, r * w, why, src))
        a = axes.setdefault(axis, [0.0, 0.0, 0.0])
        a[0] += r * w
        a[1] += w
        a[2] += w if cur is not None else 0
    return rows, axes


def markdown():
    rows, axes = table()
    total = sum(v[0] for v in axes.values())
    measured = sum(v[2] for v in axes.values())
    out = ["# Metasequoia（2 次元の SinWave）の到達度（ポイント）", "",
           "各項目に「ゼロ点の基準」（何もしない場合）と「目標」（最低限の指標＝定番の手法の値）を置き、今の値がその間のどこにあるかで採点しています。",
           "目標に届けば満点です。計算式と数値は `tools/scoreboard.py` にあり、新しい結果が出たら書き換えて `python tools/scoreboard.py --md` で更新します。",
           "数字の元は [NOTES.md](../NOTES.md)、やさしい説明は [experiments_guide.md](experiments_guide.md)。", "",
           f"## 合計: **{total:.1f} / 100 点**（測定済みの項目だけなら {total:.1f} / {measured:g} 点）", "",
           "| 軸 | 点 | 配点 | 達成率 |", "|---|---|---|---|"]
    for axis, (got, w, mw) in axes.items():
        out.append(f"| {axis} | {got:.1f} | {w:g} | {100 * got / w:.0f}% |")
    out += ["", "## 項目ごと", "",
            "| 軸 | 項目 | 配点 | ゼロ点の基準 | 目標 | 今の値 | 点 | あと | 目標の根拠 | 出典 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for axis, name, w, lo, hi, cur, got, why, src in rows:
        c = "未測定" if cur is None else f"{cur:g}"
        out.append(f"| {axis} | {name} | {w:g} | {lo:g} | {hi:g} | {c} | {got:.1f} | {w - got:.1f} | {why} | {src} |")
    rest = sorted(rows, key=lambda r: -(r[2] - r[6]))[:5]
    out += ["", "## 伸びしろが大きい項目（残りの点が多い順）", ""]
    for axis, name, w, lo, hi, cur, got, why, src in rest:
        out.append(f"- **{name}**（{axis}）: あと {w - got:.1f} 点")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    md = markdown()
    if "--md" in sys.argv:
        import pathlib
        p = pathlib.Path(__file__).resolve().parents[1] / "docs" / "scoreboard.md"
        p.write_text(md, encoding="utf-8")
        print("書き出し:", p)
    else:
        print(md)
