# -*- coding: utf-8 -*-
"""2026-10-01 版新增：报告正文里「不是图表脚本直接输出」的数字，统一在这里算，结果写到 report_numbers_2026-10-01.txt。

- §0/§1：价格位置、8/9 月均价、09-22 之后的横盘区间
- §2.2：AHR999 关键日读数 + 按 10-01 的 200 日几何均线精确反算 0.30/0.45/1.2 对应价格
- §2.3：stepC1 两条退火路径在 10-04~10-18 时间窗里的位置
- §3/§4：历史基准率（2015 年以来所有交易日）、熊市后期反弹幅度、「涨 25% 用了几天」、9 天安静期统计
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from signals.indicators import compute_ahr999  # noqa: E402

PRICE_CSV = ROOT / "data" / "btc_merged_daily.csv"
STEPC1_CSV = ROOT / "visualization" / "2026-10-01" / "stepC1_postpeak_aligned.csv"
OUT = HERE / "report_numbers_2026-10-01.txt"
T = pd.Timestamp
TRUE_TOP, TRUE_TOP_PX = T("2025-10-05"), 124720.09
REPORT_ANCHOR = T("2025-08-12")
WINDOW = (T("2026-10-04"), T("2026-10-18"))
WINDOW_DAYS = 17  # 今天到 10-18


def touch_rate(v: np.ndarray, horizon: int, move: float) -> float:
    """未来 horizon 天内任意一天收盘触及 (1+move) 倍的比例；move<0 看下跌，>0 看上涨。"""
    hits = n = 0
    for i in range(len(v) - horizon):
        fut = v[i + 1:i + 1 + horizon]
        n += 1
        hits += fut.min() <= v[i] * (1 + move) if move < 0 else fut.max() >= v[i] * (1 + move)
    return hits / n


def main() -> None:
    lines: list[str] = []
    out = lines.append
    s = pd.read_csv(PRICE_CSV, parse_dates=["date"]).set_index("date")["price"].sort_index()
    latest, px = s.index.max(), float(s.iloc[-1])
    low_d = s.loc["2026-06-01":"2026-07-15"].idxmin()
    low_px = float(s[low_d])

    out("== §0/§1 价格位置")
    out(f"最新 {latest.date()} ${px:,.0f}；相对真顶 {px/TRUE_TOP_PX-1:+.1%}；相对 6 月低点 {px/low_px-1:+.1%}；跌破 6 月低点需再跌 {1-low_px/px:.1%}")
    out(f"顶后天数：10-05 锚 {(latest-TRUE_TOP).days}；08-12 锚 {(latest-REPORT_ANCHOR).days}；"
        f"距 10-04 {(WINDOW[0]-latest).days} 天，距 10-18 {(WINDOW[1]-latest).days} 天")
    out(f"09-21 ${s['2026-09-21']:,.0f}；09-22 ${s['2026-09-22']:,.0f}")
    post = s.loc["2026-09-23":]
    out(f"09-23 以来收盘区间 ${post.min():,.0f}（{post.idxmin().date()}）~ ${post.max():,.0f}（{post.idxmax().date()}）")
    since = s.loc[low_d:]
    out(f"6 月低点以来最高 ${since.max():,.0f}（{since.idxmax().date()}，{since.max()/low_px-1:+.1%}，08-12 锚第 {(since.idxmax()-REPORT_ANCHOR).days} 天）")
    for m in ["2026-08", "2026-09"]:
        x = s.loc[m]
        prev = float(s[T(m + "-01") - pd.Timedelta(days=1)])
        out(f"{m}：均价 ${x.mean():,.0f}；最低 ${x.min():,.0f}（{x.idxmin().date()}）；最高 ${x.max():,.0f}（{x.idxmax().date()}）；月涨跌 {x.iloc[-1]/prev-1:+.1%}")
    out(f"09-01~09-22 均价（09-22 版口径，修订后数据）${s.loc['2026-09-01':'2026-09-22'].mean():,.0f}")

    out("")
    out("== §2.2 AHR999")
    a = compute_ahr999(s.rename("price").reset_index()).set_index("date")
    for d in ["2026-02-05", "2026-06-30", "2026-08-01", "2026-08-12", "2026-08-16", "2026-09-21", "2026-09-22", str(latest.date())]:
        out(f"{d}：价格 ${a.loc[d, 'price']:,.0f}  AHR {a.loc[d, 'ahr999']:.3f}  200 日几何均线 ${a.loc[d, 'gma200']:,.0f}  估值线 ${a.loc[d, 'estimate_price']:,.0f}")
    sep = a.loc["2026-09", "ahr999"]
    out(f"9 月 AHR 区间 {sep.min():.3f}（{sep.idxmin().date()}）~ {sep.max():.3f}（{sep.idxmax().date()}）")
    last = a.iloc[-1]
    logs199 = float(np.log(a["price"].iloc[-200:-1]).sum())

    def ahr_at(x: float) -> float:  # 今天价格换成 x，前 199 天不变
        g = np.exp((logs199 + np.log(x)) / 200)
        return (x / g) * (x / last["estimate_price"])

    for tgt in [0.30, 0.45, 1.20]:
        out(f"AHR {tgt:.2f} ↔ 今天约 ${brentq(lambda x: ahr_at(x) - tgt, 1000, 1e6):,.0f}")

    out("")
    out("== §2.3 stepC1 退火路径（真顶 10-05 锚）")
    ann = pd.read_csv(STEPC1_CSV, encoding="utf-8-sig").set_index("post_day")
    for d in [(latest - TRUE_TOP).days, 364, 371, 378]:
        r = ann.loc[d]
        out(f"第 {d} 天（{(TRUE_TOP + pd.Timedelta(days=d)).date()}）：仿 2017 {r['dd_2017_scaled']:.1f}%（${TRUE_TOP_PX*(1+r['dd_2017_scaled']/100):,.0f}）"
            f"  仿 2021 {r['dd_2021_scaled']:.1f}%（${TRUE_TOP_PX*(1+r['dd_2021_scaled']/100):,.0f}）")

    out("")
    out("== §3/§4 历史基准率（2015-01-01 以来所有交易日，不分牛熊）")
    v = s.loc["2015-01-01":].to_numpy()
    for h in [60, 120]:
        out(f"{h} 天内跌 30% 以上：{touch_rate(v, h, -0.30):.1%}")
    levels = [
        ("09-21 高点", float(s["2026-09-21"])),
        ("仿 2017 退火 10-04", TRUE_TOP_PX * (1 + ann.loc[364, "dd_2017_scaled"] / 100)),
        ("Glassnode 真实市场均价（09-30 周报）", 77200.0),
        ("AHR 0.45", brentq(lambda x: ahr_at(x) - 0.45, 1000, 1e6)),
        ("仿 2021 退火 10-18", TRUE_TOP_PX * (1 + ann.loc[378, "dd_2021_scaled"] / 100)),
        ("6 月低点", low_px),
    ]
    for name, lvl in levels:
        move = lvl / px - 1
        rates = "  ".join(f"{h} 天 {touch_rate(v, h, move):.1%}" for h in [WINDOW_DAYS, 30, 60])
        out(f"{name} ${lvl:,.0f}（{move:+.1%}）触及率：{rates}")

    out("")
    out("== §3 熊市后期反弹 / 起飞速度")
    for name, top, bot in [("2017 轮", T("2017-12-16"), T("2018-12-15")), ("2021 轮", T("2021-11-08"), T("2022-11-21"))]:
        seg = s.loc[top + pd.Timedelta(days=250): bot]
        out(f"{name} 顶后 250 天到底部之间最大反弹 {(seg / seg.cummin() - 1).max():+.1%}")
    for name, d0 in [("2022-11 真底", T("2022-11-21")), ("2026-06-30", low_d)]:
        p0 = float(s[d0])
        hit = s.loc[d0:][s.loc[d0:] >= p0 * 1.25].index[0]
        out(f"{name} 涨 25% 用了 {(hit-d0).days} 天（{hit.date()}）")

    out("")
    out("== §4 9 天安静期")
    r = np.log(s).diff()
    vol9 = r.rolling(9).std() * np.sqrt(365)
    rng9 = s.rolling(9).max() / s.rolling(9).min() - 1
    for start in ["2015-01-01", "2023-01-01"]:
        out(f"{start} 起：最近 9 天年化波动 {vol9.iloc[-1]:.1%}（分位 {(vol9.loc[start:].dropna() < vol9.iloc[-1]).mean():.1%}）；"
            f"9 天收盘最高/最低差 {rng9.iloc[-1]:.2%}（分位 {(rng9.loc[start:].dropna() < rng9.iloc[-1]).mean():.1%}）")
    tight = rng9[rng9 <= 0.015].loc["2016-01-01":str((latest - pd.Timedelta(days=31)).date())]
    starts, prev = [], None
    for d in tight.index:  # 相隔 20 天以上才算新的一次
        if prev is None or (d - prev).days > 20:
            starts.append(d)
        prev = d
    for d in starts:
        p0, fut = float(s[d]), s.loc[d + pd.Timedelta(days=1): d + pd.Timedelta(days=30)]
        out(f"{d.date()} 9 天差 ≤1.5%：之后 30 天最高 {fut.max()/p0-1:+.1%}，最低 {fut.min()/p0-1:+.1%}，第 30 天 {fut.iloc[-1]/p0-1:+.1%}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
