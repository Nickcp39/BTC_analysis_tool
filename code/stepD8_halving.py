"""
StepD8: 把"减半"硬时钟纳入节奏分析
- 减半是外部确定事件(日期精确)，作锚最稳。看它相对手标真顶/真底的间隔。
- 读 manual_points.csv 提取各周期真顶/真底；减半日固定。
- 输出每周期：底→减半、减半→顶、顶→底、减半→下一个底，以及减半→减半。
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

HALVINGS = [pd.Timestamp("2016-07-09"), pd.Timestamp("2020-05-11"), pd.Timestamp("2024-04-20")]
TOP_WIN = [("2017-06-01", "2018-03-01"), ("2021-06-01", "2022-02-01"), ("2025-06-01", "2026-02-01")]
BOT_WIN = [("2015-01-01", "2016-01-01"), ("2018-09-01", "2019-04-01"), ("2022-06-01", "2023-03-01")]


def main():
    df = pd.read_csv(CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    def pick(win, kind):
        a, b = pd.Timestamp(win[0]), pd.Timestamp(win[1])
        sub = df[(df["date"] >= a) & (df["date"] <= b)]
        if sub.empty:
            return None
        idx = sub["price"].idxmax() if kind == "top" else sub["price"].idxmin()
        return df.loc[idx, "date"]

    tops = [pick(w, "top") for w in TOP_WIN]      # 2017/2021/2025 真顶
    bots = [pick(w, "bottom") for w in BOT_WIN]   # 2015/2018/2022 真底
    D = lambda a, b: (b - a).days

    print("===== 减半→减半 =====")
    for a, b in zip(HALVINGS[:-1], HALVINGS[1:]):
        print(f"  {a:%Y-%m-%d} → {b:%Y-%m-%d}: {D(a,b)} 天")

    print("\n===== 减半→顶（项目 step01 的核心量）=====")
    for h, t in zip(HALVINGS, tops):
        if t is not None:
            print(f"  减半 {h:%Y-%m-%d} → 顶 {t:%Y-%m-%d}: {D(h,t)} 天")

    print("\n===== 减半→下一个底（决定底的时点）=====")
    next_bots = [bots[1], bots[2], None]  # 2016减半后的底=2018-12；2020后=2022-11；2024后=未到
    for h, nb in zip(HALVINGS, next_bots):
        if nb is not None:
            print(f"  减半 {h:%Y-%m-%d} → 底 {nb:%Y-%m-%d}: {D(h,nb)} 天")
        else:
            print(f"  减半 {h:%Y-%m-%d} → 底 ?: 未到")

    print("\n===== 底→减半（减半前的底） =====")
    prev_bots = [bots[0], bots[1], bots[2]]   # 2015→2016减半(锚偏)；2018→2020；2022→2024
    for b_, h in zip(prev_bots, HALVINGS):
        if b_ is not None:
            note = "  (2015锚偏)" if h == HALVINGS[0] else ""
            print(f"  底 {b_:%Y-%m-%d} → 减半 {h:%Y-%m-%d}: {D(b_,h)} 天{note}")

    print("\n===== 顶→底（熊市，对照）=====")
    for t, nb in [(tops[0], bots[1]), (tops[1], bots[2])]:
        if t is not None and nb is not None:
            print(f"  顶 {t:%Y-%m-%d} → 底 {nb:%Y-%m-%d}: {D(t,nb)} 天")

    # 由减半推下一个底：用历史 减半→底 的均值
    hb = [D(HALVINGS[0], bots[1]), D(HALVINGS[1], bots[2])]
    avg = sum(hb) / len(hb)
    pred = HALVINGS[2] + pd.Timedelta(days=avg)
    print(f"\n===== 预测 2025 周期真底 =====")
    print(f"  历史 减半→底：{hb} 天，均值 {avg:.0f}")
    print(f"  → 2024-04-20 + {avg:.0f} ≈ {pred:%Y-%m-%d}")
    if tops[2] is not None:
        print(f"  对照(顶+365熊市)：{tops[2]:%Y-%m-%d} + 365 ≈ {tops[2]+pd.Timedelta(days=365):%Y-%m-%d}")


if __name__ == "__main__":
    main()
