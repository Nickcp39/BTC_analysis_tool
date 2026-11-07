# -*- coding: utf-8 -*-
"""11-06 版 §6 的宏观数字：FRED 10 年期名义/实际利率、美元指数、布伦特原油、纳指 100、标普 500，和 BTC 同期对比。
输出：macro_fred_2026-11-06.csv（2026-09-25 起的原始序列）、macro_fred_2026-11-06.txt（正文引用的数字）。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SERIES = {"DGS10": "10 年期国债收益率 %", "DFII10": "10 年期实际利率 %", "DTWEXBGS": "美元指数（广义）",
          "DCOILBRENTEU": "布伦特原油 $", "NASDAQ100": "纳斯达克 100", "SP500": "标普 500"}
AS_OF = pd.Timestamp("2026-11-06")


def main() -> None:
    frames = []
    for sid in SERIES:
        df = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2026-09-25")
        df.columns = ["date", sid]
        df = df[df[sid] != "."]
        df["date"] = pd.to_datetime(df["date"])
        df[sid] = df[sid].astype(float)
        frames.append(df.set_index("date"))
    m = pd.concat(frames, axis=1).sort_index().loc[:AS_OF]
    m.to_csv(HERE / "macro_fred_2026-11-06.csv", encoding="utf-8-sig")
    btc = pd.read_csv(ROOT / "data" / "btc_merged_daily.csv", parse_dates=["date"]).set_index("date")["price"].loc["2026-09-25":AS_OF]

    lines = []
    for sid, name in SERIES.items():
        x = m[sid].dropna()
        lines.append(f"{name}（{sid}）：10-01 {x.get(pd.Timestamp('2026-10-01'), float('nan')):,.2f}；最高 {x.max():,.2f}（{x.idxmax().date()}）；"
                     f"最低 {x.min():,.2f}（{x.idxmin().date()}）；最新 {x.iloc[-1]:,.2f}（{x.index[-1].date()}）")
    for sid in ["NASDAQ100", "SP500"]:
        x = m[sid].dropna()
        pk = x.loc["2026-10-01":].idxmax()
        lines.append(f"{SERIES[sid]}：{pk.date()} 高点 {x[pk]:,.0f} → {x.index[-1].date()} {x.iloc[-1]:,.0f}（{x.iloc[-1] / x[pk] - 1:+.1%}）；"
                     f"11-05 单日 {x[pd.Timestamp('2026-11-05')] / x[pd.Timestamp('2026-11-04')] - 1:+.1%}")
    lines.append(f"BTC：10-04 高点 ${btc['2026-10-04']:,.0f} → 11-06 ${btc.iloc[-1]:,.0f}（{btc.iloc[-1] / btc['2026-10-04'] - 1:+.1%}）；"
                 f"11-05 单日 {btc['2026-11-05'] / btc['2026-11-04'] - 1:+.1%}")
    r = pd.concat([btc.pct_change(), m["NASDAQ100"].pct_change()], axis=1, keys=["btc", "ndx"]).dropna().loc["2026-10-01":]
    lines.append(f"10-01 起 BTC 与纳指 100 日收益相关系数 {r.corr().iloc[0, 1]:.2f}（{len(r)} 个共同交易日）")
    (HERE / "macro_fred_2026-11-06.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
