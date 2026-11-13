# -*- coding: utf-8 -*-
"""2026-11-13 当前水位重算（BTC 从 09-21 $86k 回落到 $76k 之后）。

口径：本轮顶 = 2025-08-12，底 = 2026-06-30（AGENTS.md / 10-01 版起不再用 2025-10-05）。
数据：data/btc_merged_daily.csv（FRED CBBTCUSD 到 11-06 收盘；最后一行 11-07 是 Yahoo 盘中点，只做参考）；
      CoinMetrics 社区 API（MVRV、已实现价格，缓存到本目录 coinmetrics_btc_mvrv.csv）；
      data/fear_greed.csv（alternative.me，11-06 由 signals/sources.fetch_fear_greed 刷新）。

输出（全部写在本目录）：
- water_level_numbers_2026-11-13.txt   所有正文数字
- water_level_table_2026-11-13.csv     各指标「现在 / 本轮底 / 上两轮底 / 本轮顶」对照 + 全历史分位
- png/water_level_dashboard.png        价格 + 已实现价格 + 200 周均线 + AHR999 + MVRV 三联图
- png/post_bottom_aligned_annealed.png 三轮「底后第 N 天」对齐（原始 + 按 0.687 退火）
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PNG = HERE / "png"
sys.path.insert(0, str(ROOT))
from signals.indicators import compute_ahr999  # noqa: E402

PRICE_CSV = ROOT / "data" / "btc_merged_daily.csv"
FNG_CSV = ROOT / "data" / "fear_greed.csv"
CM_CACHE = HERE / "coinmetrics_btc_mvrv.csv"
OUT_TXT = HERE / "water_level_numbers_2026-11-13.txt"
OUT_CSV = HERE / "water_level_table_2026-11-13.csv"
T = pd.Timestamp

AS_OF = T("2026-11-12")              # 最后一个完整日收盘（FRED）
TOP, BOTTOM = T("2025-08-12"), T("2026-06-30")
# 前两轮：顶（日收盘最高）、底（顶后最低收盘）——与 10-01 版 make_cycle_top_bottom 一致
CYCLES = [
    ("2017 轮", T("2017-12-16"), T("2018-12-15")),
    ("2021 轮", T("2021-11-08"), T("2022-11-21")),
    ("2025 轮", TOP, BOTTOM),
]
ANNEAL = 0.687                        # 四轮拟合：牛市对数涨幅每轮 ×0.687（analysis_runs/2026-10-01_cycle_time_amplitude_fit）


def fetch_coinmetrics() -> pd.DataFrame:
    url = ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc"
           "&metrics=CapMVRVCur,CapMrktCurUSD,SplyCur&frequency=1d&start_time=2010-07-18&page_size=10000")
    rows = []
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "btc-analysis-tool"})
        with urllib.request.urlopen(req, timeout=60) as r:
            j = json.load(r)
        rows += j["data"]
        url = j.get("next_page_url")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    for c in ["CapMVRVCur", "CapMrktCurUSD", "SplyCur"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["CapMVRVCur", "CapMrktCurUSD", "SplyCur"])
    df["mvrv"] = df["CapMVRVCur"]
    df["realized_price"] = df["CapMrktCurUSD"] / df["mvrv"] / df["SplyCur"]
    df = df[["date", "mvrv", "realized_price"]].sort_values("date").reset_index(drop=True)
    df.to_csv(CM_CACHE, index=False)
    return df


def pct_rank(series: pd.Series, value: float) -> float:
    s = series.dropna()
    return float((s < value).mean())


def pullbacks(seg: pd.Series, thresh: float = -0.10) -> list[tuple]:
    """seg 内从阶段高点回撤超过 thresh 的每一段：(高点日, 高点价, 低点日, 低点价, 回撤)。"""
    dd = seg / seg.cummax() - 1
    out, inside = [], False
    for d, x in dd.items():
        if x <= thresh and not inside:
            inside, pk, low_d = True, seg[:d].idxmax(), d
        elif inside:
            low_d = d if x < dd[low_d] else low_d
            if x > -0.02:
                out.append((pk, seg[pk], low_d, seg[low_d], dd[low_d]))
                inside = False
    if inside:
        out.append((pk, seg[pk], low_d, seg[low_d], dd[low_d], "进行中"))
    return out


def setup_mpl() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    setup_mpl()
    PNG.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    out = lines.append

    raw = pd.read_csv(PRICE_CSV, parse_dates=["date"]).set_index("date")["price"].sort_index()
    live_d, live_px = raw.index.max(), float(raw.iloc[-1])
    s = raw.loc[:AS_OF]
    px = float(s.iloc[-1])
    top_px, low_px = float(s[TOP]), float(s[BOTTOM])

    # ---------------- 指标面板 ----------------
    a = compute_ahr999(s.rename("price").reset_index()).set_index("date")
    sma200 = s.rolling(200).mean()
    sma200w = s.rolling(1400).mean()
    sma2y = s.rolling(730).mean()
    try:
        cm = fetch_coinmetrics()
    except Exception as e:  # 离线时用缓存
        print("CoinMetrics 抓取失败，用缓存：", e)
        cm = pd.read_csv(CM_CACHE, parse_dates=["date"])
    cm = cm.set_index("date")
    fng = pd.read_csv(FNG_CSV, parse_dates=["date"]).set_index("date")["value"]

    out("== 一、价格位置（顶 2025-08-12，底 2026-06-30）")
    out(f"最新完整收盘 {AS_OF.date()} ${px:,.0f}；盘中参考 {live_d.date()} ${live_px:,.0f}")
    out(f"顶 ${top_px:,.0f}；底 ${low_px:,.0f}；相对顶 {px / top_px - 1:+.1%}；相对底 {px / low_px - 1:+.1%}；跌破底需再跌 {1 - low_px / px:.1%}")
    out(f"顶后第 {(AS_OF - TOP).days} 天；底后第 {(AS_OF - BOTTOM).days} 天")
    since = s.loc[BOTTOM:]
    hi_d, hi = since.idxmax(), float(since.max())
    out(f"底以来最高 ${hi:,.0f}（{hi_d.date()}，底后第 {(hi_d - BOTTOM).days} 天，{hi / low_px - 1:+.1%}）；现在距这个高点 {px / hi - 1:+.1%}，已过 {(AS_OF - hi_d).days} 天")
    for m in ["2026-08", "2026-09", "2026-10", "2026-11"]:
        x = s.loc[m]
        prev = float(s[T(m + "-01") - pd.Timedelta(days=1)])
        out(f"{m}：均价 ${x.mean():,.0f}；最低 ${x.min():,.0f}（{x.idxmin().date()}）；最高 ${x.max():,.0f}（{x.idxmax().date()}）；"
            f"月涨跌 {x.iloc[-1] / prev - 1:+.1%}{'（月内至今）' if m == '2026-11' else ''}")
    for n in [7, 30, 60]:
        out(f"近 {n} 天：{px / float(s.iloc[-1 - n]) - 1:+.1%}")
    # 底以来的高低点结构
    lows = [("06-30 底", BOTTOM), ("08-01 回踩", T("2026-08-01")), ("08-16 回踩", T("2026-08-16")),
            ("09 月低点", s.loc["2026-09"].idxmin()), ("10-11 月低点", s.loc["2026-10-01":].idxmin())]
    out("底以来的低点序列：" + "；".join(f"{n} {d.date()} ${s[d]:,.0f}" for n, d in lows))

    out("")
    out("== 二、AHR999")
    last = a.iloc[-1]
    out(f"{AS_OF.date()}：AHR {last.ahr999:.3f}；200 日几何均线 ${last.gma200:,.0f}；拟合估值 ${last.estimate_price:,.0f}")
    for d in ["2026-02-05", "2026-06-30", "2026-08-16", "2026-09-21", "2026-10-01"]:
        out(f"  {d}：价格 ${a.loc[d, 'price']:,.0f}  AHR {a.loc[d, 'ahr999']:.3f}")
    hist = a["ahr999"].loc["2015-01-01":]
    out(f"AHR 全历史（2015 起）分位：{pct_rank(hist, last.ahr999):.0%}（越低越便宜）；本轮（顶后）分位 {pct_rank(a['ahr999'].loc[TOP:], last.ahr999):.0%}")
    logs199 = float(np.log(a["price"].iloc[-200:-1]).sum())

    def ahr_at(x: float) -> float:
        g = np.exp((logs199 + np.log(x)) / 200)
        return (x / g) * (x / last.estimate_price)

    for tgt in [0.30, 0.45, 0.80, 1.20]:
        out(f"  AHR {tgt:.2f} ↔ 今天约 ${brentq(lambda x: ahr_at(x) - tgt, 1000, 1e6):,.0f}")
    for name, top, bot in CYCLES[:2]:
        d = bot + (AS_OF - BOTTOM)
        out(f"  {name} 底后同一天（{d.date()}）AHR {a.loc[d, 'ahr999']:.3f}；底当天 {a.loc[bot, 'ahr999']:.3f}")

    out("")
    out("== 三、MVRV / 已实现价格（CoinMetrics）")
    cm_last_d = cm.index.max()
    mv, rp = float(cm.loc[cm_last_d, "mvrv"]), float(cm.loc[cm_last_d, "realized_price"])
    out(f"{cm_last_d.date()}：MVRV {mv:.3f}；已实现价格 ${rp:,.0f}；价格高出已实现价格 {px / rp - 1:+.1%}")
    out(f"MVRV 全历史（2013 起）分位：{pct_rank(cm['mvrv'].loc['2013-01-01':], mv):.0%}")
    for name, top, bot in CYCLES:
        out(f"  {name}：顶 MVRV {cm.loc[top, 'mvrv']:.2f}；底 MVRV {cm.loc[bot, 'mvrv']:.2f}（已实现价格 ${cm.loc[bot, 'realized_price']:,.0f}）"
            + ("" if bot == BOTTOM else f"；底后同一天 {cm.loc[bot + (AS_OF - BOTTOM), 'mvrv']:.2f}"))
    out(f"  MVRV 1.0 ↔ 约 ${rp:,.0f}；MVRV 1.2 ↔ 约 ${rp * 1.2:,.0f}；本轮底 MVRV 对应今天价格 ${rp * cm.loc[BOTTOM, 'mvrv']:,.0f}")

    out("")
    out("== 四、均线")
    for name, ma in [("200 日均线", sma200), ("2 年均线", sma2y), ("200 周均线", sma200w)]:
        out(f"{name} ${ma.iloc[-1]:,.0f}；价格/均线 {px / ma.iloc[-1]:.3f}")
    out(f"200 周均线全历史最低倍数 {float((s / sma200w).min()):.3f}（{(s / sma200w).idxmin().date()}）；本轮底 {low_px / sma200w[BOTTOM]:.3f}")

    out("")
    out("== 五、情绪：恐惧贪婪指数")
    out(f"最新 {fng.index.max().date()} = {int(fng.iloc[-1])}；{AS_OF:%m-%d} = {int(fng.get(AS_OF, -1))}；近 30 天均值 {fng.iloc[-30:].mean():.0f}；"
        f"06-30 底当天 {int(fng.get(BOTTOM, np.nan))}；09-21 高点 {int(fng.get(T('2026-09-21'), np.nan))}")
    out(f"近 30 天 ≤25（极度恐惧）的天数 {(fng.iloc[-30:] <= 25).sum()}")

    out("")
    out("== 六、三轮「底后第 N 天」对照")
    n_days = (AS_OF - BOTTOM).days
    rows_cmp = []
    for name, top, bot in CYCLES:
        p0 = float(s[bot])
        d = bot + pd.Timedelta(days=n_days)
        seg = s.loc[bot:d]
        at = float(seg.iloc[-1])
        hi_seg = float(seg.max())
        rows_cmp.append((name, bot, p0, at, hi_seg, seg.idxmax()))
        out(f"{name}：底 {bot.date()} ${p0:,.0f}；底后第 {n_days} 天（{seg.index[-1].date()}）${at:,.0f}（{at / p0 - 1:+.1%}，对数 {np.log(at / p0):.3f}）；"
            f"其间最高 {hi_seg / p0 - 1:+.1%}（{seg.idxmax().date()}）；当时距其间高点 {at / hi_seg - 1:+.1%}")
    lg = [np.log(r[3] / r[2]) for r in rows_cmp]
    exp_lg = lg[1] * ANNEAL
    out(f"对数涨幅比：2025/2021 = {lg[2] / lg[1]:.2f}；2021/2017 = {lg[1] / lg[0]:.2f}（四轮拟合每轮 ×{ANNEAL}）")
    out(f"若按 ×{ANNEAL} 退火 2021 轮同期：应在 {np.exp(exp_lg) - 1:+.1%} ≈ ${low_px * np.exp(exp_lg):,.0f}；实际 {px / low_px - 1:+.1%}，低于退火路径 {px / (low_px * np.exp(exp_lg)) - 1:+.1%}")
    for name, top, bot in CYCLES[:2]:
        seg = s.loc[bot: bot + pd.Timedelta(days=365)]
        pb = pullbacks(seg, -0.12)
        out(f"{name} 底后一年内 >12% 的回撤（从阶段高点）：" + ("；".join(f"{p[0].date()} ${p[1]:,.0f}→{p[2].date()} ${p[3]:,.0f}（{p[4]:.1%}）" for p in pb) or "无"))
        out(f"   其间是否跌破底：{'是' if float(s.loc[bot + pd.Timedelta(days=1): bot + pd.Timedelta(days=365)].min()) < float(s[bot]) else '否'}")
    pb = pullbacks(s.loc[BOTTOM:], -0.10)
    out("2025 轮 底以来 >10% 的回撤：" + "；".join(f"{p[0].date()} ${p[1]:,.0f}→{p[2].date()} ${p[3]:,.0f}（{p[4]:.1%}）{p[5] if len(p) > 5 else ''}" for p in pb))

    out("")
    out("== 七、历史基准率（2015 以来所有交易日）：从今天的价格出发")
    v = s.loc["2015-01-01":].to_numpy()
    for h in [60, 120]:
        for mv_ in [-0.15, low_px / px - 1]:
            hits = sum(v[i + 1:i + 1 + h].min() <= v[i] * (1 + mv_) for i in range(len(v) - h))
            out(f"{h} 天内再跌 {abs(mv_):.1%} 以上：{hits / (len(v) - h):.1%}")
    # 只看「底后 3–12 个月」这个阶段（2019-03~2019-12、2023-03~2023-11）
    phase = pd.concat([s.loc[b + pd.Timedelta(days=90): b + pd.Timedelta(days=365)] for _, _, b in CYCLES[:2]])
    for h in [60, 120]:
        for mv_ in [-0.15, low_px / px - 1]:
            hits = tot = 0
            for d, p in phase.items():
                fut = s.loc[d + pd.Timedelta(days=1): d + pd.Timedelta(days=h)]
                if len(fut) == h:
                    tot += 1
                    hits += fut.min() <= p * (1 + mv_)
            out(f"只看前两轮「底后 90–365 天」阶段：{h} 天内再跌 {abs(mv_):.1%} 以上 {hits / tot:.1%}（{tot} 个起点）")

    # ---------------- 对照表 ----------------
    def row(name, d):
        return {
            "时点": name, "日期": d.date().isoformat(), "价格": round(float(s[d])),
            "AHR999": round(float(a.loc[d, "ahr999"]), 3) if d in a.index and not np.isnan(a.loc[d, "ahr999"]) else None,
            "MVRV": round(float(cm.loc[d, "mvrv"]), 3) if d in cm.index else None,
            "价格/200周均线": round(float(s[d] / sma200w[d]), 3) if not np.isnan(sma200w[d]) else None,
            "恐惧贪婪": int(fng[d]) if d in fng.index else None,
        }
    tbl = [row("现在", AS_OF), row("本轮顶", TOP), row("本轮底", BOTTOM), row("底后高点", hi_d),
           row("2021 轮底", T("2022-11-21")), row("2021 轮底后同日", T("2022-11-21") + (AS_OF - BOTTOM)),
           row("2017 轮底", T("2018-12-15")), row("2017 轮底后同日", T("2018-12-15") + (AS_OF - BOTTOM))]
    tdf = pd.DataFrame(tbl)
    tdf.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    out("")
    out("== 八、对照表（water_level_table_2026-11-13.csv）")
    out(tdf.to_string(index=False))
    out("")
    out("== 九、全历史分位（越低越便宜；AHR/价格倍数 2015 起，MVRV 2013 起）")
    ranks = {
        "AHR999": pct_rank(hist, last.ahr999),
        "MVRV": pct_rank(cm["mvrv"].loc["2013-01-01":], mv),
        "价格/200周均线": pct_rank((s / sma200w).loc["2015-01-01":], px / sma200w.iloc[-1]),
        "价格/2年均线": pct_rank((s / sma2y).loc["2015-01-01":], px / sma2y.iloc[-1]),
    }
    for k, r in ranks.items():
        out(f"{k}：{r:.0%}")
    out(f"四项平均分位：{np.mean(list(ranks.values())):.0%}")

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---------------- 图 1：水位三联图 ----------------
    start = T("2018-01-01")
    fig, axs = plt.subplots(3, 1, figsize=(13.5, 11), dpi=160, sharex=True,
                            gridspec_kw={"height_ratios": [2.2, 1, 1]})
    ax = axs[0]
    ax.semilogy(s.loc[start:].index, s.loc[start:], color="#0f172a", lw=1.4, label="BTC 日收盘")
    ax.semilogy(cm.loc[start:].index, cm.loc[start:, "realized_price"], color="#2563eb", lw=1.4, label="已实现价格（全网平均持币成本）")
    ax.semilogy(sma200w.loc[start:].index, sma200w.loc[start:], color="#16a34a", lw=1.4, label="200 周均线")
    ax.semilogy(sma200.loc[start:].index, sma200.loc[start:], color="#f59e0b", lw=1.0, ls="--", label="200 日均线")
    for name, top, bot in CYCLES:
        ax.scatter([bot], [s[bot]], color="#16a34a", s=36, zorder=6)
        ax.scatter([top], [s[top]], color="#dc2626", s=36, zorder=6)
    ax.scatter([AS_OF], [px], color="#7c3aed", s=70, zorder=7, label=f"现在 {AS_OF:%m-%d} ${px:,.0f}")
    ax.set_title(f"BTC 水位（截至 {AS_OF:%Y-%m-%d}）：价格 vs 持币成本 vs 长均线", fontsize=13.5, weight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", color="#e2e8f0")
    ax = axs[1]
    ah = a["ahr999"].loc[start:]
    ax.plot(ah.index, ah, color="#0f766e", lw=1.2)
    ax.axhspan(0, 0.30, color="#bbf7d0", alpha=0.6)
    ax.axhspan(0.30, 0.45, color="#d9f99d", alpha=0.5)
    ax.axhline(1.2, color="#f97316", lw=0.9, ls="--")
    ax.set_ylim(0, 3)
    ax.set_ylabel("AHR999")
    ax.text(ah.index[5], 0.12, "<0.30 历史深底区", fontsize=8.5, color="#166534")
    ax.text(ah.index[5], 0.36, "0.30–0.45 抄底区", fontsize=8.5, color="#3f6212")
    ax.scatter([AS_OF], [last.ahr999], color="#7c3aed", s=50, zorder=7)
    ax.annotate(f"{last.ahr999:.2f}", (AS_OF, last.ahr999), xytext=(8, 8), textcoords="offset points", color="#7c3aed", weight="bold")
    ax.grid(True, color="#e2e8f0")
    ax = axs[2]
    m = cm["mvrv"].loc[start:]
    ax.plot(m.index, m, color="#2563eb", lw=1.2)
    ax.axhspan(0, 1.0, color="#bbf7d0", alpha=0.6)
    ax.axhline(3.0, color="#f97316", lw=0.9, ls="--")
    ax.set_ylabel("MVRV")
    ax.scatter([cm_last_d], [mv], color="#7c3aed", s=50, zorder=7)
    ax.annotate(f"{mv:.2f}", (cm_last_d, mv), xytext=(8, 8), textcoords="offset points", color="#7c3aed", weight="bold")
    ax.grid(True, color="#e2e8f0")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(0.99, 0.005, "红点＝各轮顶（2017-12-16 / 2021-11-08 / 2025-08-12）；绿点＝各轮底。数据：FRED CBBTCUSD、CoinMetrics 社区版。",
             ha="right", fontsize=8.3, color="#334155")
    fig.tight_layout()
    fig.savefig(PNG / "water_level_dashboard.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------- 图 2：底后对齐（原始 + 退火） ----------------
    fig, axs = plt.subplots(1, 2, figsize=(14.5, 6), dpi=160, sharey=False)
    colors = {"2017 轮": "#94a3b8", "2021 轮": "#0d9488", "2025 轮": "#0f172a"}
    horizon = 365
    for k, ax in enumerate(axs):
        for i, (name, top, bot) in enumerate(CYCLES):
            p0 = float(s[bot])
            seg = s.loc[bot: bot + pd.Timedelta(days=horizon)]
            x = (seg.index - bot).days
            lgv = np.log(seg.values / p0)
            if k == 1 and name != "2025 轮":
                lgv = lgv * ANNEAL ** (2 - i)   # 2017 退火两次，2021 退火一次
            ax.plot(x, (np.exp(lgv) - 1) * 100, color=colors[name], lw=3.2 if name == "2025 轮" else 1.7,
                    label=name + ("" if k == 0 or name == "2025 轮" else f"（×{ANNEAL}^{2 - i} 退火）"))
        ax.axvline(n_days, color="#7c3aed", ls=":", lw=1.2)
        ax.axhline(0, color="#475569", lw=0.9)
        ax.set_xlabel("距本轮底的天数")
        ax.grid(True, color="#e2e8f0")
        ax.legend(loc="upper left", fontsize=9)
    axs[0].set_ylabel("相对底的涨幅（%）")
    axs[0].set_title("原始涨幅：三轮底后第 0–365 天", fontsize=12.5, weight="bold")
    axs[1].set_title(f"按「每轮对数涨幅 ×{ANNEAL}」退火后", fontsize=12.5, weight="bold")
    axs[1].annotate(f"现在：底后第 {n_days} 天 {px / low_px - 1:+.0%}", (n_days, (px / low_px - 1) * 100),
                    xytext=(12, -28), textcoords="offset points", color="#7c3aed", weight="bold",
                    arrowprops=dict(arrowstyle="->", color="#7c3aed"))
    fig.suptitle(f"本轮底后的反弹，和前两轮同期比（截至 {AS_OF:%Y-%m-%d}）", fontsize=13.5, weight="bold")
    fig.tight_layout()
    fig.savefig(PNG / "post_bottom_aligned_annealed.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
