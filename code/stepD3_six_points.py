"""
StepD3: 6 个情绪变化点的"左右对应"定位与时间结构
- 用户手画 6 点 = 两组镜像：左组(2021顶后) ① 顶 ② 次高 ③ 底；右组(2025顶后) ④ 顶 ⑤ 次高 ⑥ 底
- 本脚本：把 6 个点吸附到真实极值（top 取窗口内最高，bottom 取最低），
  算左右两组的内部时间跨度与对应比值，画窗口图便于人工校正。
- 6 点日期可直接改 SIX_POINTS；改完重跑即可。
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stepD1_core_points import read_series, zigzag_idx, classify, setup_cjk_font

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "core_points"
WIN_START, WIN_END = pd.Timestamp("2021-06-01"), pd.Timestamp("2026-05-31")

# (名义日期, 类型 top/bottom, 组 L/R, 标签, 吸附半窗天)
SIX_POINTS = [
    ("2021-11-08", "top",    "L", "①顶 2021", 20),
    ("2022-03-29", "top",    "L", "②次高",     25),
    ("2022-11-21", "bottom", "L", "③底",       30),
    ("2025-10-05", "top",    "R", "④顶 2025", 20),
    ("2025-12-15", "top",    "R", "⑤次高",     35),
    ("2026-02-05", "bottom", "R", "⑥底",       30),
]


def snap(s, date, kind, win):
    d = pd.Timestamp(date)
    seg = s.loc[d - pd.Timedelta(days=win): d + pd.Timedelta(days=win)]
    if len(seg) == 0:
        return d, float("nan")
    dd = seg.idxmax() if kind == "top" else seg.idxmin()
    return dd, float(s.loc[dd])


def main():
    setup_cjk_font()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    s = read_series()

    pts = []
    for date, kind, grp, label, win in SIX_POINTS:
        dd, px = snap(s, date, kind, win)
        pts.append({"label": label, "grp": grp, "kind": kind, "date": dd, "price": px})
    df = pd.DataFrame(pts)

    print("===== 6 点定位（吸附真实极值）=====")
    print(df[["label", "grp", "kind", "date", "price"]].to_string(index=False))

    L = df[df.grp == "L"].reset_index(drop=True)
    R = df[df.grp == "R"].reset_index(drop=True)

    def spans(g):
        dts = list(g["date"])
        seg1 = (dts[1] - dts[0]).days   # 顶→次高
        seg2 = (dts[2] - dts[1]).days   # 次高→底
        tot = (dts[2] - dts[0]).days    # 顶→底
        return seg1, seg2, tot

    Ls, Rs = spans(L), spans(R)
    print("\n===== 左右两组时间跨度（天）=====")
    print(f"{'':10}{'顶→次高':>10}{'次高→底':>10}{'顶→底':>10}")
    print(f"{'左(2021)':10}{Ls[0]:>10}{Ls[1]:>10}{Ls[2]:>10}")
    print(f"{'右(2025)':10}{Rs[0]:>10}{Rs[1]:>10}{Rs[2]:>10}")
    print(f"{'右/左比':10}{Rs[0]/Ls[0]:>10.3f}{Rs[1]/Ls[1]:>10.3f}{Rs[2]/Ls[2]:>10.3f}")

    print("\n===== 价格回撤幅度（相对各自顶）=====")
    for g, name in [(L, "左(2021)"), (R, "右(2025)")]:
        top = g[g.kind == "top"]["price"].iloc[0]
        for _, r in g.iterrows():
            print(f"{name} {r['label']:8} {r['price']:>12.0f}  ({(r['price']/top-1)*100:+.1f}% vs 顶)")

    # 画窗口图：价格 + 参考 pivots(灰) + 6 个对应点(大号红/绿)
    win = s.loc[WIN_START:WIN_END]
    piv = classify(win, zigzag_idx(win.values, 0.22))
    plt.figure(figsize=(15, 6))
    plt.plot(win.index, win.values, lw=1.0, color="#bbb", zorder=1)
    plt.yscale("log")
    plt.scatter(piv["date"], piv["price"], s=18, c="#ddd", edgecolors="#999",
                linewidths=0.4, zorder=2, label="参考 pivots(θ22%)")
    for _, r in df.iterrows():
        c = "#e74c3c" if r["kind"] == "top" else "#1db954"
        plt.scatter([r["date"]], [r["price"]], s=160, c=c, edgecolors="k",
                    linewidths=1.0, zorder=4)
        plt.annotate(f"{r['label']}\n{r['date']:%y-%m-%d}", (r["date"], r["price"]),
                     textcoords="offset points",
                     xytext=(0, 14 if r["kind"] == "top" else -30),
                     ha="center", fontsize=9, fontweight="bold",
                     color="#c0392b" if r["kind"] == "top" else "#0a5")
    plt.title("6 个情绪变化点：左组(2021顶后) 与 右组(2025顶后) 对应")
    plt.ylabel("价格(对数)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out = OUTDIR / "six_points.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n图：{out}")


if __name__ == "__main__":
    main()
