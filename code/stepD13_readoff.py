"""
StepD13: 直接从叠加变换"读底"——红(2021)曲线的底，套 高度/时间/平移 = 2025 预测底
- 价格只由 高度系数(amp) 决定，与时间/平移无关 → 价格是硬的。
- 时间由 time_scale + shift 决定 → 软。
- 用 2021 真实最低点，套不同参数组，直接给 2025 底的 日期 + 价格。
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
TOP21, TOP25 = pd.Timestamp("2021-11-08"), pd.Timestamp("2025-10-05")

# (amp高度, time_scale, shift, 标签)
PARAM_SETS = [
    (0.53, 0.87, -67, "你屏幕 v16"),
    (0.54, 1.00, -58, "stepD10拟合"),
    (0.53, 1.00, 0, "不压缩不平移"),
]


def main():
    df = pd.read_csv(ROOT / "data" / "btc_merged_daily.csv",
                     parse_dates=["date"]).set_index("date")["price"].sort_index()
    px21 = float(df.loc[:TOP21].iloc[-1])
    px25 = float(df.loc[:TOP25].iloc[-1])
    seg = df.loc["2022-01-01":"2023-03-01"]
    bd, bp = seg.idxmin(), float(seg.min())
    rel = (bd - TOP21).days
    log_dd = np.log(bp / px21)
    print(f"2021 真底 {bd.date()} ${bp:,.0f}  距顶 {rel} 天  回撤 {np.exp(log_dd)-1:+.1%}")
    print(f"2025 顶 ${px25:,.0f}\n")
    print(f"{'参数组':14}{'底日期':>13}{'底价':>12}{'回撤':>8}")
    for amp, ts, sh, tag in PARAM_SETS:
        d = TOP25 + pd.Timedelta(days=int(round(rel * ts + sh)))
        p = px25 * np.exp(log_dd * amp)
        print(f"{tag:14}{d.strftime('%Y-%m-%d'):>13}{p:>12,.0f}{np.exp(log_dd*amp)-1:>8.0%}")
    print("\n价格只随 amp 变（~$57k 都一样）→ 硬；时间随 ts/shift 变（6月~10月）→ 软。")


if __name__ == "__main__":
    main()
