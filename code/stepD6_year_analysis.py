"""
StepD6: 按年分析手标的情绪变化点（一次一年，便于逐年核对）
- 读 output/core_points/manual_points.csv（手标点，自动去重）
- 指定年份：python code/stepD6_year_analysis.py 2025
- 输出：该年的点 + 相邻间隔(天) + 相邻涨跌(%) 表格；图存 output/core_points/year_<年>.png
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stepD1_core_points import read_series, setup_cjk_font

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "output" / "core_points"
CSV = OUTDIR / "manual_points.csv"


def load_points():
    df = pd.read_csv(CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = (df.drop_duplicates(subset=["date"])
            .sort_values("date").reset_index(drop=True))
    return df


def main():
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    setup_cjk_font()
    s = read_series()
    pts = load_points()
    yp = pts[pts["date"].dt.year == year].reset_index(drop=True)
    if yp.empty:
        print(f"{year} 年没有标点。手标点覆盖：{pts['date'].min():%Y-%m-%d} ~ {pts['date'].max():%Y-%m-%d}")
        return

    yp["gap_days"] = yp["date"].diff().dt.days
    yp["move_%"] = (yp["price"] / yp["price"].shift(1) - 1) * 100

    print(f"===== {year} 年 手标情绪变化点（去重后 {len(yp)} 个）=====")
    show = yp.copy()
    show["date"] = show["date"].dt.strftime("%Y-%m-%d")
    show["price"] = show["price"].round(0)
    show["move_%"] = show["move_%"].round(1)
    print(show[["date", "price", "type", "gap_days", "move_%"]].to_string(index=False))

    g = yp["gap_days"].dropna()
    if len(g):
        print(f"\n间隔(天)：均值 {g.mean():.0f}，范围 {int(g.min())}~{int(g.max())}，"
              f"序列 {list(g.astype(int))}")

    # 画该年
    a, b = pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)
    seg = s.loc[a:b]
    plt.figure(figsize=(15, 6))
    plt.plot(seg.index, seg.values, lw=1.0, color="#888", zorder=1)
    plt.yscale("log")
    for _, r in yp.iterrows():
        c = "#e74c3c" if r["type"] == "H" else "#1db954"
        plt.scatter([r["date"]], [r["price"]], s=90, c=c, edgecolors="k",
                    linewidths=0.7, zorder=3)
        plt.annotate(f"{r['date']:%m-%d}", (r["date"], r["price"]),
                     textcoords="offset points",
                     xytext=(0, 9 if r["type"] == "H" else -16),
                     ha="center", fontsize=8,
                     color="#c0392b" if r["type"] == "H" else "#0a5")
    plt.title(f"{year} 年 手标情绪变化点（红=高 绿=低，标注=日期）：{len(yp)} 个")
    plt.ylabel("价格(对数)")
    plt.tight_layout()
    out = OUTDIR / f"year_{year}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n图：{out}")


if __name__ == "__main__":
    main()
