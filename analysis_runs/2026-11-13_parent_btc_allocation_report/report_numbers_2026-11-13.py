# -*- coding: utf-8 -*-
"""11-13 版正文里不是由图表脚本直接输出的数字 → report_numbers_2026-11-13.txt。

价格到 2026-11-13 Yahoo 盘中点（csv 末行），估值到 11-12 收盘（CoinMetrics 只到 11-12）。
其余数字的来源：
- 拉伸对比：../2026-11-13_stretch_compare/stretch_numbers.txt
- 人工点位、用户配对：../2026-11-13_manual_point_compare/*_numbers.txt
- 五轮底后回撤：../2026-11-13_drawdown_context/pullbacks_numbers.txt
- 水位 / MVRV / 均线 / 恐惧贪婪：water_level_numbers_2026-11-13.txt
- AHR 压缩读法：../2026-11-13_ahr_compressed/ahr_compressed_numbers.txt（commit eb6a486）
- 宏观：macro_fred_2026-11-13.txt
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
T = pd.Timestamp
TOP, BOTTOM, HI = T("2025-08-12"), T("2026-06-30"), T("2026-10-04")


def main() -> None:
    s = pd.read_csv(ROOT / "data" / "btc_merged_daily.csv", parse_dates=["date"]).set_index("date")["price"].sort_index()
    m = pd.read_csv(HERE / "macro_fred_2026-11-13.csv", parse_dates=["date"], index_col="date")
    d, px = s.index.max(), float(s.iloc[-1])
    c12 = float(s[T("2026-11-12")])
    L = []
    o = L.append
    o(f"最新 {d.date()} 盘中 ${px:,.0f}；11-12 收盘 ${c12:,.0f}；11-06 收盘 ${s[T('2026-11-06')]:,.0f}")
    for lab, p in [("盘中", px), ("11-12 收盘", c12)]:
        o(f"{lab}：相对顶 {p / s[TOP] - 1:+.1%}；相对底 {p / s[BOTTOM] - 1:+.1%}；相对 10-04 高点 {p / s[HI] - 1:+.1%}；"
          f"到 $62,835 {62835 / p - 1:+.1%}；到底 {s[BOTTOM] / p - 1:+.1%}")
    o(f"11-06 → 11-13：{px / s[T('2026-11-06')] - 1:+.1%}；11-06 → 11-12 收盘：{c12 / s[T('2026-11-06')] - 1:+.1%}")
    o(f"顶后第 {(d - TOP).days} 天；底后第 {(d - BOTTOM).days} 天；10-04 起第 {(d - HI).days} 天")
    for lab, cost in [("Strategy 平均成本 $75,456（11-08）", 75456), ("ETF 平均成本约 $81.5k（Glassnode）", 81500),
                      ("短期持有者成本 $75.6k（Glassnode 11-11）", 75600), ("真实市场均值 $77.5k（Glassnode 11-11）", 77500)]:
        o(f"盘中价相对{lab}：{px / cost - 1:+.1%}；11-12 收盘相对：{c12 / cost - 1:+.1%}")
    nd = m["NASDAQ100"].dropna()
    b = s.loc["2026-11-05":"2026-11-12"]
    o(f"11-05 → 11-12：纳斯达克 100 {nd[T('2026-11-12')] / nd[T('2026-11-05')] - 1:+.1%}；BTC 收盘 {b.iloc[-1] / b.iloc[0] - 1:+.1%}")
    o(f"ETF 持币（Bitbo）09-30 1,292,181 → 11-05 1,277,921：{1277921 - 1292181:+,} 枚")
    o(f"11 月 ETF 净流量（TFTC，11-02～11-11 八个交易日）："
      f"{sum([-398.6, -115.6, 74.6, -430.2, -411.9, -201.1, -95.0, -299.9]):+.1f} 百万美元")
    (HERE / "report_numbers_2026-11-13.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
