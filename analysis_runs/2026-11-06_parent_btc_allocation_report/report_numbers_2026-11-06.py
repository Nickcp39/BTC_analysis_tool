# -*- coding: utf-8 -*-
"""2026-11-06 版：报告正文里「不是图表脚本直接输出」的数字，统一在这里算，结果写到 report_numbers_2026-11-06.txt。

口径：本轮顶 = 2025-08-12，底 = 2026-06-30（AGENTS.md：不用 2025-10-05）。价格截至 11-06 收盘（csv 末行 11-07 是盘中点，不用）。
- §0/§1：价格相对顶/底/底后高点的位置、8–11 月均价、近 7/30/60 天
- §2.2：AHR999 关键日读数 + 按 11-06 的 200 日几何均线反算 0.30/0.45/0.80/1.2 对应价格
- §3：底后第 N 天三轮对照（原始 + 退火）、前两轮底后一年内的回撤、历史基准率
- §3.3：MVRV / 已实现价格（CoinMetrics，读 analysis_runs/2026-11-06_water_level/coinmetrics_btc_mvrv.csv）、均线、全历史分位
- §5.7 / §9：家庭计划四个月的月均价；§5.6 路标进度
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
FNG_CSV = ROOT / "data" / "fear_greed.csv"
CM_CSV = ROOT / "analysis_runs" / "2026-11-06_water_level" / "coinmetrics_btc_mvrv.csv"
OUT = HERE / "report_numbers_2026-11-06.txt"
T = pd.Timestamp
AS_OF = T("2026-11-06")
TOP, BOTTOM = T("2025-08-12"), T("2026-06-30")
CYCLES = [("2017 轮", T("2017-12-16"), T("2018-12-15")), ("2021 轮", T("2021-11-08"), T("2022-11-21"))]
ANNEAL = 0.687
PRICE_1001_REPORTED = 83788.08  # 10-01 版当日用的盘中价（之后 FRED 修订为收盘 $84,760）


def drop_rate(v: np.ndarray, horizon: int, move: float) -> float:
    hits = sum(v[i + 1:i + 1 + horizon].min() <= v[i] * (1 + move) for i in range(len(v) - horizon))
    return hits / (len(v) - horizon)


def pullbacks(seg: pd.Series, thresh: float) -> list[tuple]:
    dd = seg / seg.cummax() - 1
    res, inside = [], False
    for d, x in dd.items():
        if x <= thresh and not inside:
            inside, pk, low_d = True, seg[:d].idxmax(), d
        elif inside:
            low_d = d if x < dd[low_d] else low_d
            if x > -0.02:
                res.append((pk, seg[pk], low_d, seg[low_d], dd[low_d], ""))
                inside = False
    if inside:
        res.append((pk, seg[pk], low_d, seg[low_d], dd[low_d], "（进行中）"))
    return res


def pct_rank(series: pd.Series, value: float) -> float:
    return float((series.dropna() < value).mean())


def main() -> None:
    lines: list[str] = []
    out = lines.append
    s = pd.read_csv(PRICE_CSV, parse_dates=["date"]).set_index("date")["price"].sort_index().loc[:AS_OF]
    px = float(s.iloc[-1])
    top_px, low_px = float(s[TOP]), float(s[BOTTOM])
    since = s.loc[BOTTOM:]
    hi_d, hi = since.idxmax(), float(since.max())
    n_days = (AS_OF - BOTTOM).days

    out("== §0/§1 价格位置（顶 2025-08-12，底 2026-06-30）")
    out(f"最新 {AS_OF.date()} 收盘 ${px:,.0f}；相对顶 {px / top_px - 1:+.1%}；相对底 {px / low_px - 1:+.1%}；跌破底需再跌 {1 - low_px / px:.1%}")
    out(f"顶后第 {(AS_OF - TOP).days} 天；底后第 {n_days} 天")
    out(f"10-01 版当日价 ${PRICE_1001_REPORTED:,.0f}（盘中）→ FRED 收盘 ${s['2026-10-01']:,.0f}")
    out(f"底以来最高 ${hi:,.0f}（{hi_d.date()}，底后第 {(hi_d - BOTTOM).days} 天，{hi / low_px - 1:+.1%}）；09-21 ${s['2026-09-21']:,.0f}；现在距高点 {px / hi - 1:+.1%}")
    for m in ["2026-08", "2026-09", "2026-10", "2026-11"]:
        x = s.loc[m]
        prev = float(s[T(m + "-01") - pd.Timedelta(days=1)])
        out(f"{m}：均价 ${x.mean():,.0f}；最低 ${x.min():,.0f}（{x.idxmin().date()}）；最高 ${x.max():,.0f}（{x.idxmax().date()}）；"
            f"涨跌 {x.iloc[-1] / prev - 1:+.1%}{'（11-01~11-06）' if m == '2026-11' else ''}")
    for n in [7, 30, 60]:
        out(f"近 {n} 天 {px / float(s.iloc[-1 - n]) - 1:+.1%}")
    oct_ = s.loc["2026-10-01":"2026-10-31"]
    out(f"10-04 → 10-30：${s['2026-10-04']:,.0f} → ${s['2026-10-30']:,.0f}（{s['2026-10-30'] / s['2026-10-04'] - 1:+.1%}）；"
        f"10-30 → 11-04：{s['2026-11-04'] / s['2026-10-30'] - 1:+.1%}；11-04 → 11-06：{px / s['2026-11-04'] - 1:+.1%}")
    out(f"10 月最大单日跌幅 {oct_.pct_change().min():+.1%}（{oct_.pct_change().idxmin().date()}）；11-05 单日 {s['2026-11-05'] / s['2026-11-04'] - 1:+.1%}")
    m_avg = [float(s.loc[m].mean()) for m in ["2026-08", "2026-09", "2026-10"]]
    out(f"8/9/10 三个月均价的平均 ${np.mean(m_avg):,.0f}；加上 11 月至今均价 ${s.loc['2026-11'].mean():,.0f} 的四个月平均 ${np.mean(m_avg + [float(s.loc['2026-11'].mean())]):,.0f}")
    out(f"按月均价算，三笔（8–10 月）到今天 {px / np.mean(m_avg) - 1:+.1%}")

    out("")
    out("== §2.2 AHR999")
    a = compute_ahr999(s.rename("price").reset_index()).set_index("date")
    for d in ["2026-02-05", "2026-06-30", "2026-08-16", "2026-09-21", "2026-10-01", "2026-10-04", "2026-10-30", "2026-11-04", str(AS_OF.date())]:
        out(f"{d}（顶后第 {(T(d) - TOP).days} 天）：价格 ${a.loc[d, 'price']:,.0f}  AHR {a.loc[d, 'ahr999']:.3f}  200 日几何均线 ${a.loc[d, 'gma200']:,.0f}")
    octa = a.loc["2026-10", "ahr999"]
    out(f"10 月 AHR 区间 {octa.min():.3f}（{octa.idxmin().date()}）~ {octa.max():.3f}（{octa.idxmax().date()}）")
    below = a.loc["2026-09-01":, "ahr999"]
    first_below = below[below < 0.45]
    out("9 月以来 AHR < 0.45 的日子：" + ", ".join(f"{d.date()} {v:.3f}" for d, v in first_below.items()))
    last = a.iloc[-1]
    logs199 = float(np.log(a["price"].iloc[-200:-1]).sum())

    def ahr_at(x: float) -> float:
        g = np.exp((logs199 + np.log(x)) / 200)
        return (x / g) * (x / last["estimate_price"])

    for tgt in [0.30, 0.45, 0.80, 1.20]:
        out(f"AHR {tgt:.2f} ↔ 今天约 ${brentq(lambda x: ahr_at(x) - tgt, 1000, 1e6):,.0f}")
    out(f"AHR 全历史（2015 起）分位 {pct_rank(a['ahr999'].loc['2015-01-01':], last.ahr999):.0%}")
    p21 = T("2021-11-08")
    w = a.loc[p21 + pd.Timedelta(days=470): p21 + pd.Timedelta(days=500), "ahr999"]
    out(f"2021 轮顶后第 {(w.idxmin() - p21).days} 天（{w.idxmin().date()}）AHR {w.min():.3f}")

    out("")
    out("== §3 底后第 N 天三轮对照")
    lg = {}
    for name, top, bot in CYCLES + [("2025 轮", TOP, BOTTOM)]:
        p0 = float(s[bot])
        seg = s.loc[bot: bot + pd.Timedelta(days=n_days)]
        at = float(seg.iloc[-1])
        lg[name] = np.log(at / p0)
        out(f"{name}：底 {bot.date()} ${p0:,.0f}；第 {n_days} 天（{seg.index[-1].date()}）{at / p0 - 1:+.1%}；其间最高 {seg.max() / p0 - 1:+.1%}（{seg.idxmax().date()}）；"
            f"当时距其间高点 {at / seg.max() - 1:+.1%}")
    e17 = np.exp(lg["2017 轮"] * ANNEAL ** 2) - 1
    e21 = np.exp(lg["2021 轮"] * ANNEAL) - 1
    out(f"退火后同期：2017 轮 ×{ANNEAL}² → {e17:+.1%}（${low_px * (1 + e17):,.0f}）；2021 轮 ×{ANNEAL} → {e21:+.1%}（${low_px * (1 + e21):,.0f}）；本轮 {px / low_px - 1:+.1%}")
    for name, top, bot in CYCLES:
        seg = s.loc[bot: bot + pd.Timedelta(days=365)]
        out(f"{name} 底后一年内 >12% 回撤：" + "；".join(
            f"{p[0].date()} ${p[1]:,.0f} → {p[2].date()} ${p[3]:,.0f}（{p[4]:.1%}，{(p[2] - p[0]).days} 天）" for p in pullbacks(seg, -0.12)))
        out(f"   一年内是否跌破底：{'是' if float(s.loc[bot + pd.Timedelta(days=1): bot + pd.Timedelta(days=365)].min()) < float(s[bot]) else '否'}")
    out("本轮底以来 >10% 回撤：" + "；".join(
        f"{p[0].date()} ${p[1]:,.0f} → {p[2].date()} ${p[3]:,.0f}（{p[4]:.1%}，{(p[2] - p[0]).days} 天）{p[5]}" for p in pullbacks(since, -0.10)))
    out("本轮底以来的低点序列：06-30 $58,586；08-01 $62,835；08-16 $62,859；"
        f"09-15 ${s['2026-09-15']:,.0f}；10-30 ${s['2026-10-30']:,.0f}；11-06 ${px:,.0f}")

    out("")
    out("== §3 历史基准率")
    v = s.loc["2015-01-01":].to_numpy()
    for h in [60, 120]:
        out(f"全样本（2015 起）{h} 天内再跌 15% 以上 {drop_rate(v, h, -0.15):.1%}；跌破底（需 {1 - low_px / px:.1%}）{drop_rate(v, h, low_px / px - 1):.1%}")
    phase = pd.concat([s.loc[b + pd.Timedelta(days=90): b + pd.Timedelta(days=365)] for _, _, b in CYCLES])
    for h in [60, 120]:
        for mv in [-0.15, low_px / px - 1]:
            hits = tot = 0
            for d, p in phase.items():
                fut = s.loc[d + pd.Timedelta(days=1): d + pd.Timedelta(days=h)]
                if len(fut) == h:
                    tot += 1
                    hits += fut.min() <= p * (1 + mv)
            out(f"前两轮「底后 90–365 天」：{h} 天内再跌 {abs(mv):.1%} 以上 {hits / tot:.1%}（{tot} 个起点）")

    out("")
    out("== §3.3 水位：MVRV / 已实现价格 / 均线 / 情绪")
    cm = pd.read_csv(CM_CSV, parse_dates=["date"]).set_index("date")
    mv, rp = float(cm.loc[AS_OF, "mvrv"]), float(cm.loc[AS_OF, "realized_price"])
    out(f"{AS_OF.date()} MVRV {mv:.3f}；已实现价格 ${rp:,.0f}（价格高出 {px / rp - 1:+.1%}）；全历史（2013 起）分位 {pct_rank(cm['mvrv'].loc['2013-01-01':], mv):.0%}")
    for name, d in [("本轮顶", TOP), ("本轮底", BOTTOM), ("10-04 高点", hi_d), ("2018-12 底", T("2018-12-15")), ("2022-11 底", T("2022-11-21")),
                    ("2021 轮底后同日", T("2022-11-21") + pd.Timedelta(days=n_days))]:
        out(f"  {name} {d.date()} MVRV {cm.loc[d, 'mvrv']:.2f}")
    out(f"  MVRV 1.2 ↔ ${rp * 1.2:,.0f}")
    sma200, sma730, sma1400 = (s.rolling(n).mean().iloc[-1] for n in (200, 730, 1400))
    r200w = s / s.rolling(1400).mean()
    out(f"200 日均线 ${sma200:,.0f}（价格 ÷ 均线 {px / sma200:.3f}）；2 年均线 ${sma730:,.0f}（{px / sma730:.3f}）；200 周均线 ${sma1400:,.0f}（{px / sma1400:.3f}）")
    out(f"价格/200 周均线 全历史分位 {pct_rank(r200w.loc['2015-01-01':], px / sma1400):.0%}；价格/2 年均线分位 {pct_rank((s / s.rolling(730).mean()).loc['2015-01-01':], px / sma730):.0%}")
    fng = pd.read_csv(FNG_CSV, parse_dates=["date"]).set_index("date")["value"]
    out(f"恐惧贪婪：10-01 {fng['2026-10-01']}；10-04 {fng['2026-10-04']}；10-30 {fng['2026-10-30']}；11-05 {fng['2026-11-05']}；11-06 {fng['2026-11-06']}；"
        f"11-07 {fng.get(T('2026-11-07'), 'NA')}；10 月均值 {fng.loc['2026-10'].mean():.0f}；10 月最高 {fng.loc['2026-10'].max()}（{fng.loc['2026-10'].idxmax().date()}）")

    out("")
    out("== §5.6 路标进度")
    out(f"距 2025 年的顶 $123,366 还要涨 {top_px / px - 1:+.1%}；距底后高点 $86,424 {hi / px - 1:+.1%}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
