"""
StepD2: 按年/半年切片审阅核心点候选（人工逐窗勾选缺/多）
- 复用 stepD1 的检测：用更细的 ZigZag 阈值（默认 8%）宁可多检，按日历年切成单图，每点标号。
- 大圆点=周期级（粗阈值也认的大点），小点=细阈值新增候选；绿=高点 H，橙=低点 L。
- 输入：data/btc_merged_daily.csv（经 stepD1.read_series）
- 输出：output/core_points/by_year/<窗口>.png + output/core_points/review_by_year.csv
注意：单一幅度阈值抓不到"缓dip启动点""双顶第二顶"等形态，这些靠人工在逐年图上补。
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
OUTDIR = ROOT / "output" / "core_points" / "by_year"
REVIEW_TH = 0.08   # 细阈值：宁可多检、你来删
MAJOR_TH = 0.50    # 粗阈值：标出周期级大点（大圆点）
HALF_YEAR = False  # True=半年一张


def plot_window(s, d, start, end, title, outpath, major_set):
    win = s.loc[start:end]
    if len(win) == 0:
        return
    plt.figure(figsize=(13, 5.5))
    plt.plot(win.index, win.values, lw=1.0, color="#888", zorder=1)
    plt.yscale("log")
    for k, r in d.iterrows():
        is_major = r["date"].normalize() in major_set
        color = "#1db954" if r["type"] == "H" else "#e67e22"
        plt.scatter([r["date"]], [r["price"]], s=130 if is_major else 45,
                    c=color, edgecolors="k", linewidths=0.7, zorder=3)
        plt.annotate(f"{k + 1}", (r["date"], r["price"]),
                     textcoords="offset points",
                     xytext=(0, 11 if r["type"] == "H" else -16),
                     ha="center", fontsize=9, fontweight="bold", color="#002")
    plt.title(title)
    plt.ylabel("价格(对数)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    setup_cjk_font()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    s = read_series()
    prices = s.values
    df = classify(s, zigzag_idx(prices, REVIEW_TH))
    major_set = {s.index[i].normalize() for i in zigzag_idx(prices, MAJOR_TH)}

    rows = []
    years = range(s.index.min().year, s.index.max().year + 1)
    for y in years:
        if HALF_YEAR:
            spans = [(pd.Timestamp(y, 1, 1), pd.Timestamp(y, 6, 30), f"{y}H1"),
                     (pd.Timestamp(y, 7, 1), pd.Timestamp(y, 12, 31), f"{y}H2")]
        else:
            spans = [(pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31), f"{y}")]
        for start, end, label in spans:
            d = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
            if len(d) == 0:
                continue
            title = (f"{label}  候选核心点  细阈值{int(REVIEW_TH * 100)}%  "
                     f"(大圆点=周期级 | 绿=高 橙=低): {len(d)}个")
            plot_window(s, d, start, end, title, OUTDIR / f"{label}.png", major_set)
            for k, r in d.iterrows():
                rows.append({"window": label, "n": k + 1, "date": r["date"].date(),
                             "price": r["price"], "type": r["type"],
                             "major": r["date"].normalize() in major_set})
            print(f"{label}: {len(d)} 个候选 → {label}.png")

    pd.DataFrame(rows).to_csv(ROOT / "output" / "core_points" / "review_by_year.csv",
                             index=False, encoding="utf-8-sig")
    print(f"\n输出目录：{OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
