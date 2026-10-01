# -*- coding: utf-8 -*-
"""2026-10-01 版新增：报告正文里「不是图表脚本直接输出」的数字，统一在这里算，结果写到 report_numbers_2026-10-01.txt。

口径：本轮顶 = 2025-08-12，底 = 2026-06-30（AGENTS.md：不再用 2025-10-05）。
- §0/§1：价格相对顶/底的位置、8/9 月均价、09-22 之后的横盘区间
- §2.2：AHR999 关键日读数 + 按 10-01 的 200 日几何均线精确反算 0.30/0.45/1.2 对应价格
- §3：历史基准率（2015 年以来所有交易日）、熊市后期反弹幅度、「涨 25% 用了几天」
顶→底的天数、跌幅和时间压缩比例见 make_cycle_top_bottom.py。
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
OUT = HERE / "report_numbers_2026-10-01.txt"
T = pd.Timestamp
TOP, BOTTOM = T("2025-08-12"), T("2026-06-30")
PRICE_0922_REPORTED = 85938.03  # 09-22 版当日用的盘中价（之后 FRED 修订为收盘价）


def drop_rate(v: np.ndarray, horizon: int, move: float) -> float:
    """未来 horizon 天内任意一天收盘跌到 (1+move) 倍以下的比例（move < 0）。"""
    hits = sum(v[i + 1:i + 1 + horizon].min() <= v[i] * (1 + move) for i in range(len(v) - horizon))
    return hits / (len(v) - horizon)


def main() -> None:
    lines: list[str] = []
    out = lines.append
    s = pd.read_csv(PRICE_CSV, parse_dates=["date"]).set_index("date")["price"].sort_index()
    latest, px = s.index.max(), float(s.iloc[-1])
    top_px, low_px = float(s[TOP]), float(s[BOTTOM])

    out("== §0/§1 价格位置（顶 2025-08-12，底 2026-06-30）")
    out(f"顶 ${top_px:,.0f}；底 ${low_px:,.0f}；顶→底 {(BOTTOM - TOP).days} 天，{low_px / top_px - 1:+.1%}")
    out(f"最新 {latest.date()} ${px:,.0f}；相对顶 {px / top_px - 1:+.1%}；相对底 {px / low_px - 1:+.1%}；跌破底需再跌 {1 - low_px / px:.1%}")
    out(f"顶后第 {(latest - TOP).days} 天；底后第 {(latest - BOTTOM).days} 天")
    out(f"09-22 版当日价 ${PRICE_0922_REPORTED:,.0f} 相对顶 {PRICE_0922_REPORTED / top_px - 1:+.1%}（顶后第 {(T('2026-09-22') - TOP).days} 天）")
    out(f"09-21 ${s['2026-09-21']:,.0f}；09-22 ${s['2026-09-22']:,.0f}")
    post = s.loc["2026-09-23":]
    out(f"09-23 以来收盘区间 ${post.min():,.0f}（{post.idxmin().date()}）~ ${post.max():,.0f}（{post.idxmax().date()}）")
    since = s.loc[BOTTOM:]
    out(f"底以来最高 ${since.max():,.0f}（{since.idxmax().date()}，{since.max() / low_px - 1:+.1%}）")
    for m in ["2026-08", "2026-09"]:
        x = s.loc[m]
        prev = float(s[T(m + "-01") - pd.Timedelta(days=1)])
        out(f"{m}：均价 ${x.mean():,.0f}；最低 ${x.min():,.0f}（{x.idxmin().date()}）；最高 ${x.max():,.0f}（{x.idxmax().date()}）；月涨跌 {x.iloc[-1] / prev - 1:+.1%}")
    out(f"09-01~09-22 均价（09-22 版口径，修订后数据）${s.loc['2026-09-01':'2026-09-22'].mean():,.0f}")

    out("")
    out("== §2.2 AHR999")
    a = compute_ahr999(s.rename("price").reset_index()).set_index("date")
    for d in ["2026-02-05", "2026-06-30", "2026-08-01", "2026-08-16", "2026-09-21", "2026-09-22", str(latest.date())]:
        out(f"{d}（顶后第 {(T(d) - TOP).days} 天）：价格 ${a.loc[d, 'price']:,.0f}  AHR {a.loc[d, 'ahr999']:.3f}  "
            f"200 日几何均线 ${a.loc[d, 'gma200']:,.0f}")
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
    out("== §3 历史基准率（2015-01-01 以来所有交易日，不分牛熊）")
    v = s.loc["2015-01-01":].to_numpy()
    for h in [60, 120]:
        out(f"{h} 天内跌 30% 以上：{drop_rate(v, h, -0.30):.1%}")
    out(f"跌破底（需 {1 - low_px / px:.1%}）：60 天内 {drop_rate(v, 60, low_px / px - 1):.1%}，120 天内 {drop_rate(v, 120, low_px / px - 1):.1%}")

    out("")
    out("== §3 熊市后期反弹 / 起飞速度")
    for name, top, bot in [("2017 轮", T("2017-12-16"), T("2018-12-15")), ("2021 轮", T("2021-11-08"), T("2022-11-21"))]:
        seg = s.loc[top + pd.Timedelta(days=250): bot]
        out(f"{name} 顶后 250 天到底之间最大反弹 {(seg / seg.cummin() - 1).max():+.1%}")
    seg = s.loc[TOP + pd.Timedelta(days=250): BOTTOM]
    out(f"2025 轮 顶后 250 天到底之间最大反弹 {(seg / seg.cummin() - 1).max():+.1%}")
    for name, d0 in [("2022-11 真底", T("2022-11-21")), ("2026-06-30 底", BOTTOM)]:
        p0 = float(s[d0])
        hit = s.loc[d0:][s.loc[d0:] >= p0 * 1.25].index[0]
        out(f"{name} 涨 25% 用了 {(hit - d0).days} 天（{hit.date()}，顶后第 {(hit - TOP).days if d0 == BOTTOM else '-'} 天）")

    out("")
    out("== §5.3 2022-11 真底之后一年内的回撤（从阶段高点算，超过 12% 的）")
    seg = s.loc["2022-11-21":"2023-11-21"]
    dd = seg / seg.cummax() - 1
    inside = False
    for d, x in dd.items():
        if x <= -0.12 and not inside:
            inside, pk, low_d = True, seg[:d].idxmax(), d
        elif inside:
            low_d = d if x < dd[low_d] else low_d
            if x > -0.02:
                out(f"{pk.date()} ${seg[pk]:,.0f} → {low_d.date()} ${seg[low_d]:,.0f}（{dd[low_d]:.1%}）")
                inside = False

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
