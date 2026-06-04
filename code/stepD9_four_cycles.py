"""
StepD9: 四个周期的节奏（补上 2013 周期）
- 本地数据从 2014-12 起，2013 周期不在库、用户也没标 → 用历史公认锚点补：
    减半#1 2012-11-28、2013顶≈2013-11-30(~$1163)、2015真底 2015-01-14(~$172)
  （注：2013顶日期不同交易所差几天；2015真底在1月，用户手标的 2015-08-24 是二次低点）
- 2017/2021/2025 的顶/底取自用户手标；减半日固定。
- 算 4 周期的 减半→减半 / 减半→顶 / 顶→顶 / 顶→底 / 底→顶 / 底→底，看哪些跨 4 轮仍稳。
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

T = pd.Timestamp
HALVINGS = [T("2012-11-28"), T("2016-07-09"), T("2020-05-11"), T("2024-04-20")]
# 顶：2013 用历史；2017/2021/2025 取自手标（窗口内最高）
PEAKS_HIST = {0: T("2013-11-30")}
TOP_WIN = {1: ("2017-06-01", "2018-03-01"), 2: ("2021-06-01", "2022-02-01"),
           3: ("2025-06-01", "2026-02-01")}
# 底：2015 用历史真底；2018/2022 取自手标
BOTS_HIST = {0: T("2015-01-14")}
BOT_WIN = {1: ("2018-09-01", "2019-04-01"), 2: ("2022-06-01", "2023-03-01")}
LABELS = ["2013周期", "2017周期", "2021周期", "2025周期"]


def main():
    df = pd.read_csv(CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    def pick(win, kind):
        a, b = T(win[0]), T(win[1])
        sub = df[(df["date"] >= a) & (df["date"] <= b)]
        idx = sub["price"].idxmax() if kind == "top" else sub["price"].idxmin()
        return df.loc[idx, "date"]

    peaks = [PEAKS_HIST.get(i) or pick(TOP_WIN[i], "top") for i in range(4)]
    bots = [BOTS_HIST.get(0), pick(BOT_WIN[1], "bottom"), pick(BOT_WIN[2], "bottom"), None]
    D = lambda a, b: (b - a).days

    def seq(name, pairs):
        vals = [D(a, b) for a, b in pairs if a is not None and b is not None]
        print(f"{name:14}", "  ".join(f"{v}" for v in vals), f"   (n={len(vals)})")
        return vals

    print("锚点：")
    for i in range(4):
        pk = f"{peaks[i]:%Y-%m-%d}" if peaks[i] is not None else "—"
        bt = f"{bots[i]:%Y-%m-%d}" if bots[i] is not None else "(未到/缺)"
        src = "历史" if i == 0 else "手标"
        print(f"  {LABELS[i]}: 减半 {HALVINGS[i]:%Y-%m-%d} | 顶 {pk} | 底 {bt}   [{src}]")

    print("\n间隔(天)：")
    seq("减半→减半", list(zip(HALVINGS[:-1], HALVINGS[1:])))
    seq("减半→顶", list(zip(HALVINGS, peaks)))
    seq("顶→顶", list(zip(peaks[:-1], peaks[1:])))
    seq("顶→底(熊)", [(peaks[0], bots[0]), (peaks[1], bots[1]), (peaks[2], bots[2])])
    seq("底→顶(牛)", [(bots[0], peaks[1]), (bots[1], peaks[2]), (bots[2], peaks[3])])
    seq("底→底", list(zip([bots[0], bots[1]], [bots[1], bots[2]])))

    print("\n要点：")
    print("  · 减半→顶：2013 那轮 367 天，明显比后三轮 ~535 短 → 早期周期更快（呼应'压缩/成熟'）")
    print("  · 底→顶(牛)：1067/1059/1061，跨三轮几乎不变 → 最稳的量")
    print("  · 顶→顶：2013→2017=1477 偏长，之后 1423/1427 收紧")


if __name__ == "__main__":
    main()
