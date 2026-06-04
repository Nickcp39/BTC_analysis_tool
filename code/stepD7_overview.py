"""
StepD7: 手标点总览 —— 逐年统计 + 周期级宏观节奏（顶-顶 / 底-底 间隔）
- 读 manual_points.csv（去重）。
- 逐年：点数、相邻间隔均值/范围。
- 宏观：从标注里提取各周期真顶(窗口内最高)/真底(窗口内最低)，算 顶-顶、底-底 间隔。
不做复杂对齐，只看全貌与"间隔比例"在宏观层是否成立。
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "output" / "core_points" / "manual_points.csv"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TOP_WIN = [("2017-06-01", "2018-03-01"), ("2021-06-01", "2022-02-01"),
           ("2025-06-01", "2026-02-01")]
BOT_WIN = [("2015-01-01", "2016-01-01"), ("2018-09-01", "2019-04-01"),
           ("2022-06-01", "2023-03-01")]


def main():
    df = pd.read_csv(CSV)
    df["date"] = pd.to_datetime(df["date"])
    n0 = len(df)
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    print(f"手标点：{n0} 行 → 去重后 {len(df)} 个，覆盖 {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}\n")

    # 逐年
    print("===== 逐年概览 =====")
    print(f"{'年':>6}{'点数':>6}{'间隔均值':>9}{'间隔范围':>12}")
    for y, g in df.groupby(df["date"].dt.year):
        gaps = g["date"].diff().dt.days.dropna()
        rng = f"{int(gaps.min())}~{int(gaps.max())}" if len(gaps) else "-"
        mean = f"{gaps.mean():.0f}" if len(gaps) else "-"
        print(f"{y:>6}{len(g):>6}{mean:>9}{rng:>12}")

    def pick(win, kind):
        a, b = pd.Timestamp(win[0]), pd.Timestamp(win[1])
        sub = df[(df["date"] >= a) & (df["date"] <= b)]
        if sub.empty:
            return None
        idx = sub["price"].idxmax() if kind == "top" else sub["price"].idxmin()
        return df.loc[idx]

    tops = [pick(w, "top") for w in TOP_WIN]
    bots = [pick(w, "bottom") for w in BOT_WIN]

    print("\n===== 周期级宏观节奏（取自你的标注）=====")
    print("真顶：", "  ".join(f"{t['date']:%Y-%m-%d}(${t['price']:.0f})" for t in tops if t is not None))
    for a, b in zip(tops[:-1], tops[1:]):
        if a is not None and b is not None:
            print(f"  顶→顶 {a['date']:%Y-%m-%d} → {b['date']:%Y-%m-%d}: {(b['date']-a['date']).days} 天")
    print("真底：", "  ".join(f"{t['date']:%Y-%m-%d}(${t['price']:.0f})" for t in bots if t is not None))
    for a, b in zip(bots[:-1], bots[1:]):
        if a is not None and b is not None:
            print(f"  底→底 {a['date']:%Y-%m-%d} → {b['date']:%Y-%m-%d}: {(b['date']-a['date']).days} 天")
    # 底→顶 / 顶→底（每周期）
    print("  —— 半程 ——")
    for bt, tp in [(bots[1], tops[1]), (bots[2], tops[2])]:
        if bt is not None and tp is not None:
            print(f"  底→顶 {bt['date']:%Y-%m-%d} → {tp['date']:%Y-%m-%d}: {(tp['date']-bt['date']).days} 天")
    for tp, bt in [(tops[0], bots[1]), (tops[1], bots[2])]:
        if tp is not None and bt is not None:
            print(f"  顶→底 {tp['date']:%Y-%m-%d} → {bt['date']:%Y-%m-%d}: {(bt['date']-tp['date']).days} 天")


if __name__ == "__main__":
    main()
