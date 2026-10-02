# -*- coding: utf-8 -*-
"""检验「每个阶段时间加速、幅度退火」，并据此拟合下一个顶（2026-10-01）。

口径：2025 轮顶 = 2025-08-12，底 = 2026-06-30（AGENTS.md，不用 2025-10-05）。
数据：2015-02 以前用 Bitstamp 日收盘（bitstamp_btcusd_daily_2011_2015.csv，FRED/Coinbase 2015-01 的 $120 是早期薄市场异常值）；
之后用 data/btc_merged_daily.csv（FRED CBBTCUSD）。

输出（本目录）：
- cycle_phases.csv          四轮的锚点、各阶段天数和对数幅度
- phase_ratios.csv          相邻两轮的比例（<1 = 时间加速 / 幅度退火）
- next_top_models.csv       下一个顶的时间模型、价格模型
- png/phase_time_amplitude.png、png/next_top_projection.png
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PNG = HERE / "png"
EARLY_CSV = HERE / "bitstamp_btcusd_daily_2011_2015.csv"
MERGED_CSV = ROOT / "data" / "btc_merged_daily.csv"
T = pd.Timestamp
SWITCH = T("2015-02-01")  # 之前用 Bitstamp，之后用 FRED 合并数据

# (轮次, 减半日, 上一轮底, 顶, 底)
CYCLES = [
    ("2013", T("2012-11-28"), T("2011-10-20"), T("2013-12-04"), T("2015-01-14")),
    ("2017", T("2016-07-09"), T("2015-01-14"), T("2017-12-16"), T("2018-12-15")),
    ("2021", T("2020-05-11"), T("2018-12-15"), T("2021-11-08"), T("2022-11-21")),
    ("2025", T("2024-04-20"), T("2022-11-21"), T("2025-08-12"), T("2026-06-30")),
]
HALVING_BLOCK_2024, NEXT_HALVING_BLOCK = 840_000, 1_050_000
HALVING_2024_UTC = datetime(2024, 4, 20, 0, 9, tzinfo=timezone.utc)
VOL_SQRT_ANNEAL = (1 / 3) ** 0.5  # 项目原设定：波动等级每轮 ÷3、alpha=0.5 → 0.577


def setup() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"


def load_price() -> pd.Series:
    early = pd.read_csv(EARLY_CSV, parse_dates=["date"]).set_index("date")["close"]
    late = pd.read_csv(MERGED_CSV, parse_dates=["date"]).set_index("date")["price"]
    return pd.concat([early[early.index < SWITCH], late[late.index >= SWITCH]]).sort_index()


def next_halving_estimate() -> tuple[pd.Timestamp, str]:
    """按 2024 减半以来的实际出块速度推算区块 1,050,000 的日期；联网失败就按 10 分钟/块。"""
    now = datetime.now(timezone.utc)
    try:
        req = Request("https://blockchain.info/q/getblockcount", headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=30) as r:
            height = int(r.read().decode().strip())
        per_day = (height - HALVING_BLOCK_2024) / ((now - HALVING_2024_UTC).total_seconds() / 86400)
        eta = now.timestamp() + (NEXT_HALVING_BLOCK - height) / per_day * 86400
        note = f"区块高度 {height:,}，2024 减半以来平均 {per_day:.1f} 块/天（{1440 / per_day:.2f} 分钟/块）"
    except Exception as e:  # noqa: BLE001
        eta = HALVING_2024_UTC.timestamp() + (NEXT_HALVING_BLOCK - HALVING_BLOCK_2024) * 600
        note = f"联网取区块高度失败（{e}），按 10 分钟/块"
    return T(datetime.fromtimestamp(eta, timezone.utc).date()), note


def phase_table(s: pd.Series) -> pd.DataFrame:
    rows = []
    prev_top = None
    for name, halving, pbot, top, bot in CYCLES:
        p = {k: float(s[d]) for k, d in [("halving", halving), ("pbot", pbot), ("top", top), ("bot", bot)]}
        rows.append({
            "轮次": name, "减半": halving.date(), "上一轮底": pbot.date(), "顶": top.date(), "底": bot.date(),
            "减半价": round(p["halving"], 2), "上一轮底价": round(p["pbot"], 2), "顶价": round(p["top"], 2), "底价": round(p["bot"], 2),
            "牛市_天": (top - pbot).days, "牛市_对数涨幅": math.log(p["top"] / p["pbot"]),
            "熊市_天": (bot - top).days, "熊市_对数跌幅": -math.log(p["bot"] / p["top"]),
            "减半→顶_天": (top - halving).days, "减半→顶_对数涨幅": math.log(p["top"] / p["halving"]),
            "底→底_天": (bot - pbot).days, "底→底_对数涨幅": math.log(p["bot"] / p["pbot"]),
            "顶→顶_天": (top - prev_top[0]).days if prev_top else np.nan,
            "顶→顶_对数涨幅": math.log(p["top"] / prev_top[1]) if prev_top else np.nan,
        })
        prev_top = (top, p["top"])
    return pd.DataFrame(rows)


def ratio_table(ph: pd.DataFrame) -> pd.DataFrame:
    phases = [("牛市（底→顶）", "牛市"), ("熊市（顶→底）", "熊市"), ("顶→顶", "顶→顶"), ("底→底", "底→底"), ("减半→顶", "减半→顶")]
    rows = []
    for label, key in phases:
        for kind, col in [("时间", f"{key}_天"), ("幅度", f"{key}_对数涨幅" if key != "熊市" else "熊市_对数跌幅")]:
            vals = ph[col].to_numpy(dtype=float)
            row = {"阶段": label, "量": kind}
            for i in range(1, len(vals)):
                if np.isfinite(vals[i]) and np.isfinite(vals[i - 1]):
                    row[f"{ph['轮次'][i - 1]}→{ph['轮次'][i]}"] = round(vals[i] / vals[i - 1], 3)
            rows.append(row)
    return pd.DataFrame(rows)


def models(ph: pd.DataFrame, next_halving: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cur = ph.iloc[-1]
    top25, bot26 = T(str(cur["顶"])), T(str(cur["底"]))
    tt = ph["顶→顶_天"].dropna().to_numpy()
    n = np.arange(len(tt))
    slope, icpt = np.polyfit(n, tt, 1)
    tt_next_lin = icpt + slope * len(tt)
    tt_next_geo = tt[-1] * float(np.exp(np.mean(np.log(tt[1:] / tt[:-1]))))
    bull = ph["牛市_天"].to_numpy(dtype=float)
    h2t = ph["减半→顶_天"].to_numpy(dtype=float)
    time_rows = [
        ("T1 顶→顶时钟（每轮约 -50 天，线性）", top25 + pd.Timedelta(days=round(tt_next_lin)), f"顶→顶 {tt.astype(int).tolist()} → {tt_next_lin:.0f} 天"),
        ("T2 顶→顶时钟（等比 ×{:.3f}）".format(tt_next_geo / tt[-1]), top25 + pd.Timedelta(days=round(tt_next_geo)), f"→ {tt_next_geo:.0f} 天"),
        ("T3 牛市天数按上一轮比例再缩短", bot26 + pd.Timedelta(days=round(bull[-1] * bull[-1] / bull[-2])), f"牛市 {bull[-2]:.0f}→{bull[-1]:.0f} 天，×{bull[-1] / bull[-2]:.3f}"),
        ("T4 牛市天数持平", bot26 + pd.Timedelta(days=int(bull[-1])), f"牛市 {bull[-1]:.0f} 天"),
        ("T5 减半→顶按上一轮比例再缩短", next_halving + pd.Timedelta(days=round(h2t[-1] * h2t[-1] / h2t[-2])), f"下次减半约 {next_halving.date()}；减半→顶 {h2t[-2]:.0f}→{h2t[-1]:.0f} 天"),
        ("T6 减半→顶持平", next_halving + pd.Timedelta(days=int(h2t[-1])), f"减半→顶 {h2t[-1]:.0f} 天"),
    ]
    tdf = pd.DataFrame(time_rows, columns=["模型", "日期", "依据"])

    g = ph["牛市_对数涨幅"].to_numpy(dtype=float)
    b, a = np.polyfit(np.arange(len(g)), np.log(g), 1)  # ln G = a + b n
    g_fit_next = float(np.exp(a + b * len(g)))
    bot_px, top_px = float(cur["底价"]), float(cur["顶价"])
    ttl = ph["顶→顶_对数涨幅"].dropna().to_numpy()
    ttl_ratio = float(np.exp(np.mean(np.log(ttl[1:] / ttl[:-1]))))
    price_rows = [
        (f"P1 牛市对数涨幅等比拟合（4 轮，每轮 ×{math.exp(b):.3f}）", bot_px * math.exp(g_fit_next), f"G 下一轮 = {g_fit_next:.3f}"),
        (f"P2 牛市对数涨幅按上一轮比例（×{g[-1] / g[-2]:.3f}）", bot_px * math.exp(g[-1] * g[-1] / g[-2]), f"G = {g[-1] * g[-1] / g[-2]:.3f}"),
        (f"P3 项目原退火系数（×{VOL_SQRT_ANNEAL:.3f}）", bot_px * math.exp(g[-1] * VOL_SQRT_ANNEAL), f"G = {g[-1] * VOL_SQRT_ANNEAL:.3f}"),
        (f"P4 顶→顶对数涨幅退火（×{ttl_ratio:.3f}）", top_px * math.exp(ttl[-1] * ttl_ratio), f"顶→顶对数涨幅 {np.round(ttl, 3).tolist()}"),
    ]
    pdf = pd.DataFrame(price_rows, columns=["模型", "价格", "依据"])
    fit = {"bull_a": a, "bull_b": b, "tt_slope": slope, "tt_icpt": icpt, "tt_next_lin": tt_next_lin}
    return tdf, pdf, fit


def chart_phases(ph: pd.DataFrame, fit: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.0), dpi=170)
    x = np.arange(len(ph))
    labels = [f"{r} 轮" for r in ph["轮次"]]
    ax = axes[0]
    for col, color, name in [("顶→顶_天", "#0f766e", "顶→顶"), ("牛市_天", "#2563eb", "牛市（底→顶）"),
                             ("减半→顶_天", "#a855f7", "减半→顶"), ("熊市_天", "#dc2626", "熊市（顶→底）")]:
        v = ph[col].to_numpy(dtype=float)
        ax.plot(x, v, "o-", color=color, lw=2.6 if name == "顶→顶" else 1.8, label=name)
        for xi, vi in zip(x, v):
            if np.isfinite(vi):
                below = name == "减半→顶"  # 2013 轮的减半→顶和熊市数值很近，放到点下方
                ax.text(xi, vi - 45 if below else vi + 25, f"{vi:.0f}", ha="center", fontsize=8.5, color=color)
    ax.plot([len(ph)], [fit["tt_next_lin"]], "*", color="#0f766e", ms=15)
    ax.text(len(ph), fit["tt_next_lin"] + 30, f"下一轮 ≈{fit['tt_next_lin']:.0f}", ha="center", fontsize=9, color="#0f766e", weight="bold")
    ax.set_xticks(list(x) + [len(ph)], labels + ["2029 轮（外推）"])
    ax.set_ylabel("天数")
    ax.set_title("时间：只有「顶→顶」每轮稳定缩短（约 -50 天）", fontsize=12.5, weight="bold")
    ax.grid(True, color="#e2e8f0")
    ax.legend(loc="center right", fontsize=9)

    ax = axes[1]
    for col, color, name in [("牛市_对数涨幅", "#2563eb", "牛市对数涨幅"), ("熊市_对数跌幅", "#dc2626", "熊市对数跌幅"),
                             ("顶→顶_对数涨幅", "#0f766e", "顶→顶对数涨幅")]:
        v = ph[col].to_numpy(dtype=float)
        ax.plot(x, v, "o-", color=color, lw=2.2, label=name)
        for xi, vi in zip(x, v):
            if np.isfinite(vi):
                ax.text(xi + 0.06, vi * 1.06, f"{vi:.2f}（×{math.exp(vi):,.0f}）" if "涨" in name else f"{vi:.2f}（{math.exp(-vi) - 1:.0%}）",
                        fontsize=8, color=color)
    xf = np.linspace(0, len(ph), 50)
    ax.plot(xf, np.exp(fit["bull_a"] + fit["bull_b"] * xf), ls="--", color="#2563eb", lw=1.2, alpha=0.7,
            label=f"牛市拟合：G = {math.exp(fit['bull_a']):.2f} × {math.exp(fit['bull_b']):.3f}^n")
    g_next = math.exp(fit["bull_a"] + fit["bull_b"] * len(ph))
    ax.plot([len(ph)], [g_next], "*", color="#2563eb", ms=15)
    ax.set_yscale("log")
    ax.set_xticks(list(x) + [len(ph)], labels + ["2029 轮（外推）"])
    ax.set_ylabel("对数幅度（对数刻度：直线 = 每轮按固定比例退火）")
    ax.set_title("幅度：每个阶段、每一轮都在缩小", fontsize=12.5, weight="bold")
    ax.grid(True, color="#e2e8f0", which="both")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.suptitle("「时间加速、幅度退火」检验：四轮周期（2013 / 2017 / 2021 / 2025）", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(PNG / "phase_time_amplitude.png", bbox_inches="tight")
    plt.close(fig)


def chart_projection(s: pd.Series, ph: pd.DataFrame, tdf: pd.DataFrame, pdf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.6), dpi=170)
    ax.plot(s.index, s.values, color="#0f172a", lw=1.3, label="BTC 日收盘（2015-02 前 Bitstamp，之后 FRED/Coinbase）")
    for _, r in ph.iterrows():
        for d, px, c, tag in [(r["顶"], r["顶价"], "#dc2626", "顶"), (r["底"], r["底价"], "#16a34a", "底")]:
            d = T(str(d))
            ax.scatter([d], [px], color=c, s=40, zorder=5)
            ax.annotate(f"{tag} {d:%Y-%m}\n${px:,.0f}", (d, px), xytext=(0, 12 if tag == "顶" else -26),
                        textcoords="offset points", ha="center", fontsize=8, color=c)
    core = tdf[tdf["模型"].str.startswith(("T1", "T2", "T3", "T4"))]["日期"]
    hal = tdf[tdf["模型"].str.startswith(("T5", "T6"))]["日期"]
    ax.axvspan(core.min(), core.max(), color="#0f766e", alpha=0.16, label=f"顶→顶 / 牛市时钟：{core.min():%Y-%m} – {core.max():%Y-%m}")
    ax.axvspan(hal.min(), hal.max(), color="#a855f7", alpha=0.10, label=f"减半时钟：{hal.min():%Y-%m} – {hal.max():%Y-%m}")
    ax.axhspan(pdf["价格"].min(), pdf["价格"].max(), color="#f59e0b", alpha=0.14,
               label=f"价格模型：\\${pdf['价格'].min() / 1000:,.0f}k – \\${pdf['价格'].max() / 1000:,.0f}k")
    star_d = T(np.median([d.value for d in core]))
    ax.scatter([star_d], [float(pdf["价格"].median())], marker="*", s=260, color="#b91c1c", zorder=7,
               label=f"中心：{star_d:%Y-%m} / ${pdf['价格'].median() / 1000:,.0f}k")
    ax.set_yscale("log")
    ax.set_xlim(T("2011-06-01"), T("2030-06-01"))
    ax.set_ylim(1, 600_000)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, color="#e2e8f0", which="both")
    ax.legend(loc="lower right", fontsize=8.6)
    ax.set_title("按「时间加速、幅度退火」外推下一个顶（2025 轮顶 = 2025-08-12，底 = 2026-06-30）", fontsize=13, weight="bold")
    fig.savefig(PNG / "next_top_projection.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    PNG.mkdir(exist_ok=True)
    s = load_price()
    ph = phase_table(s)
    rt = ratio_table(ph)
    nh, nh_note = next_halving_estimate()
    tdf, pdf, fit = models(ph, nh)
    ph.to_csv(HERE / "cycle_phases.csv", index=False, encoding="utf-8-sig")
    rt.to_csv(HERE / "phase_ratios.csv", index=False, encoding="utf-8-sig")
    out = pd.concat([tdf.assign(类=" 时间"), pdf.assign(类="价格")], ignore_index=True)
    out.to_csv(HERE / "next_top_models.csv", index=False, encoding="utf-8-sig")
    chart_phases(ph, fit)
    chart_projection(s, ph, tdf, pdf)
    pd.set_option("display.width", 260, "display.max_columns", 30, "display.max_colwidth", 70)
    print(ph.round(3).to_string(index=False))
    print(rt.to_string(index=False))
    print("下次减半：", nh.date(), "|", nh_note)
    print(tdf.to_string(index=False))
    print(pdf.round(0).to_string(index=False))
    print(f"牛市对数涨幅拟合 G(n) = {math.exp(fit['bull_a']):.3f} × {math.exp(fit['bull_b']):.4f}^n（n=0 为 2013 轮）")
    print(f"顶→顶线性拟合 D(n) = {fit['tt_icpt']:.1f} + {fit['tt_slope']:.2f}·n（n=0 为 2013→2017）")


if __name__ == "__main__":
    main()
