"""历史买点标注 + 结构/形态对比。

产出到 outputs/：
  buypoints_overview_*.png   价格/AHR999/MVRV 三面板，竖线标出历史三轮底 + 当前
  pattern_overlay_*.png      各轮"顶→底"跌势归一化叠加（峰顶对齐），看当前结构位置
  buypoint_comparison.csv    每个买点当时的四指标读数对照表
  report_buypoints_*.html    汇总（图 + 表，内嵌）

用法: python signals/buypoints.py   （用本地数据；如需最新先跑 charts.py --refresh）
"""
from __future__ import annotations
import base64
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import indicators as I

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
BLUE, AMBER, GREEN, RED, PURPLE, GREY = "#1e40af", "#b45309", "#15803d", "#b91c1c", "#7c3aed", "#6b7280"
CYCLE_COLORS = ["#9ca3af", "#0ea5e9", "#f59e0b", "#b91c1c"]

# 周期峰/底的检测窗口（在窗口内取价格极值，避免硬编码不准）
PEAK_WINDOWS = [
    ("2013周期顶", "2013-11-01", "2014-01-31"),
    ("2017周期顶", "2017-11-01", "2018-01-31"),
    ("2021周期顶", "2021-10-01", "2021-12-31"),
    ("2025周期顶", "2025-09-15", "2025-12-15"),
]
BOTTOM_WINDOWS = [
    ("2015底", "2014-10-01", "2015-09-30"),
    ("2018底", "2018-10-01", "2019-03-31"),
    ("2022底", "2022-09-01", "2023-02-28"),
]


def _extremum(price, start, end, kind):
    w = price[(price["date"] >= pd.Timestamp(start)) & (price["date"] <= pd.Timestamp(end))]
    if w.empty:
        return None, None
    row = w.loc[w["price"].idxmax()] if kind == "max" else w.loc[w["price"].idxmin()]
    return row["date"], float(row["price"])


def _at(df, col, date):
    """取离 date 最近一天的指标值。"""
    if df is None or col not in df.columns:
        return None
    s = df.dropna(subset=[col])
    if s.empty:
        return None
    i = (s["date"] - date).abs().idxmin()
    if abs((s.loc[i, "date"] - date).days) > 15:
        return None
    return float(s.loc[i, col])


def load():
    price = pd.read_csv(C.PRICE_CSV, parse_dates=["date"])
    fng = pd.read_csv(C.FNG_CSV, parse_dates=["date"])
    ahr = I.compute_ahr999(price)
    onc = pd.read_csv(C.ONCHAIN_CSV, parse_dates=["date"]) if C.ONCHAIN_CSV.exists() else None
    return price, fng, ahr, onc


def detect_peaks(price):
    peaks = []
    for name, s, e in PEAK_WINDOWS:
        d, p = _extremum(price, s, e, "max")
        if d is not None:
            peaks.append({"name": name, "date": d, "price": p})
    return peaks


def detect_bottoms(price):
    bots = []
    for name, s, e in BOTTOM_WINDOWS:
        d, p = _extremum(price, s, e, "min")
        if d is not None:
            bots.append({"name": name, "date": d, "price": p})
    return bots


def build_comparison(price, fng, ahr, onc):
    peaks = detect_peaks(price)
    bots = detect_bottoms(price)
    rows = []
    # 历史三轮底：配对前一个周期顶算跌幅
    for i, b in enumerate(bots):
        prior_peak = peaks[i]  # 第 i 个底对应第 i 个顶
        dd = b["price"] / prior_peak["price"] - 1
        days = (b["date"] - prior_peak["date"]).days
        rows.append({
            "事件": b["name"], "日期": b["date"].date().isoformat(), "价格": b["price"],
            "距前高跌幅": dd, "顶→底天数": days,
            "AHR999": _at(ahr, "ahr999", b["date"]),
            "MVRV": _at(onc, "mvrv", b["date"]),
            "恐慌": _at(fng, "value", b["date"]),
        })
    # 本轮：当前周期顶 + 本轮至今最低 + 当前
    cur_peak = peaks[-1]
    low_d, low_p = _extremum(price, "2026-01-01", price["date"].max().isoformat(), "min")
    last = price.iloc[-1]
    for tag, d, p in [("本轮至今最低", low_d, low_p), ("当前", last["date"], float(last["price"]))]:
        rows.append({
            "事件": tag, "日期": d.date().isoformat(), "价格": p,
            "距前高跌幅": p / cur_peak["price"] - 1,
            "顶→底天数": (d - cur_peak["date"]).days,
            "AHR999": _at(ahr, "ahr999", d),
            "MVRV": _at(onc, "mvrv", d),
            "恐慌": _at(fng, "value", d),
        })
    return pd.DataFrame(rows), peaks, bots


# ----------------------------------------------------------------- 图1：买点标注总览
def chart_overview(price, ahr, onc, bots, end):
    a = ahr[ahr["date"] >= pd.Timestamp("2013-01-01")]
    o = onc[onc["date"] >= pd.Timestamp("2013-01-01")] if onc is not None else None
    bot_dates = [b["date"] for b in bots]

    n = 3 if o is not None else 2
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.4 * n), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.5, 1.5][:n], "hspace": 0.12})

    def marks(ax):
        for b in bots:
            ax.axvline(b["date"], color=GREEN, ls="--", lw=1.0, alpha=0.7)
        ax.axvline(end, color=RED, ls="-", lw=1.0, alpha=0.6)

    ax = axes[0]
    ax.semilogy(a["date"], a["price"], color=BLUE, lw=1.2)
    for b in bots:
        ax.scatter([b["date"]], [b["price"]], color=GREEN, zorder=6, s=45)
        ax.annotate(f"{b['name']}\n${b['price']:,.0f}", (b["date"], b["price"]),
                    color=GREEN, fontsize=8.5, fontweight="bold",
                    xytext=(0, -34), textcoords="offset points", ha="center")
    marks(ax)
    ax.set_title("历史买点(绿)与当前(红) — 价格 / AHR999 / MVRV 对照", fontsize=15, fontweight="bold")
    ax.set_ylabel("价格(对数)")
    ax.grid(True, which="both", alpha=0.2)

    ax = axes[1]
    ax.plot(a["date"], a["ahr999"], color="#0f766e", lw=1.1)
    ax.axhspan(0, C.AHR_DEEP_VALUE, color=GREEN, alpha=0.12)
    ax.axhline(C.AHR_DEEP_VALUE, color=GREEN, ls=":", lw=0.9)
    for b in bots:
        v = _at(ahr, "ahr999", b["date"])
        if v is not None:
            ax.scatter([b["date"]], [v], color=GREEN, zorder=6, s=40)
            ax.annotate(f"{v:.2f}", (b["date"], v), fontsize=8.5, color=GREEN,
                        fontweight="bold", xytext=(4, 4), textcoords="offset points")
    marks(ax)
    ax.set_ylabel("AHR999"); ax.set_ylim(0, min(5, a["ahr999"].max() * 1.1)); ax.grid(True, alpha=0.2)

    if o is not None:
        ax = axes[2]
        ax.plot(o["date"], o["mvrv"], color=PURPLE, lw=1.1)
        ax.axhspan(0, 1.0, color=GREEN, alpha=0.12)
        ax.axhline(1.0, color=GREEN, ls=":", lw=0.9)
        for b in bots:
            v = _at(onc, "mvrv", b["date"])
            if v is not None:
                ax.scatter([b["date"]], [v], color=GREEN, zorder=6, s=40)
                ax.annotate(f"{v:.2f}", (b["date"], v), fontsize=8.5, color=GREEN,
                            fontweight="bold", xytext=(4, 4), textcoords="offset points")
        marks(ax)
        ax.set_ylabel("MVRV"); ax.grid(True, alpha=0.2)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes:
        ax.set_xlim(pd.Timestamp("2013-01-01"), end)
    out = C.CHART_DIR / f"buypoints_overview_{end.date()}.png"
    fig.savefig(out, dpi=135, bbox_inches="tight"); plt.close(fig)
    print(f"[chart] {out.name}")
    return out


# ----------------------------------------------------------------- 图2：跌势形态叠加（峰顶对齐）
def chart_pattern_overlay(price, peaks, end):
    """每轮顶→底跌势归一化（价格/峰顶, 对数），按距峰天数对齐叠加；当前轮高亮。"""
    fig, ax = plt.subplots(figsize=(14, 6.5))
    horizon = 500  # 看峰后 500 天
    for i, pk in enumerate(peaks):
        seg = price[(price["date"] >= pk["date"]) &
                    (price["date"] <= pk["date"] + pd.Timedelta(days=horizon))].copy()
        if seg.empty:
            continue
        seg["t"] = (seg["date"] - pk["date"]).dt.days
        seg["norm"] = seg["price"] / pk["price"]
        is_cur = (i == len(peaks) - 1)
        ax.plot(seg["t"], seg["norm"], color=CYCLE_COLORS[i % len(CYCLE_COLORS)],
                lw=2.6 if is_cur else 1.5, alpha=1.0 if is_cur else 0.75,
                label=f"{pk['name']} (峰 ${pk['price']:,.0f}){' ← 当前' if is_cur else ''}")
        # 标各轮最低点
        lo = seg.loc[seg["norm"].idxmin()]
        ax.scatter([lo["t"]], [lo["norm"]], color=CYCLE_COLORS[i % len(CYCLE_COLORS)], s=45, zorder=5)
        if not is_cur:
            ax.annotate(f"{(lo['norm']-1)*100:.0f}%\n第{int(lo['t'])}天",
                        (lo["t"], lo["norm"]), fontsize=8, ha="center",
                        color=CYCLE_COLORS[i % len(CYCLE_COLORS)],
                        xytext=(0, -28), textcoords="offset points")
    ax.axhline(1.0, color=GREY, ls=":", lw=0.8)
    for lv, txt in [(0.7, "-30%"), (0.5, "-50%"), (0.3, "-70%")]:
        ax.axhline(lv, color=GREY, ls=":", lw=0.5, alpha=0.6)
        ax.text(horizon, lv, f" {txt}", fontsize=8, color=GREY, va="center")
    ax.set_yscale("log")
    ax.set_title("跌势形态对比：各轮「顶→底」归一化叠加（峰顶=1，对数）— 当前轮看走到结构哪一步",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("距周期顶天数"); ax.set_ylabel("价格 / 峰顶价")
    ax.set_xlim(0, horizon); ax.legend(loc="lower left", fontsize=9); ax.grid(True, which="both", alpha=0.2)
    out = C.CHART_DIR / f"pattern_overlay_{end.date()}.png"
    fig.savefig(out, dpi=135, bbox_inches="tight"); plt.close(fig)
    print(f"[chart] {out.name}")
    return out


# ----------------------------------------------------------------- HTML
def _b64(p): return base64.b64encode(p.read_bytes()).decode()


def _fmt(df):
    d = df.copy()
    d["价格"] = d["价格"].map(lambda x: f"${x:,.0f}")
    d["距前高跌幅"] = d["距前高跌幅"].map(lambda x: f"{x*100:.0f}%")
    d["AHR999"] = d["AHR999"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    d["MVRV"] = d["MVRV"].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
    d["恐慌"] = d["恐慌"].map(lambda x: "—" if pd.isna(x) else f"{int(x)}")
    return d


def build_html(table_df, imgs, day):
    d = _fmt(table_df)
    th = "".join(f"<th>{c}</th>" for c in d.columns)
    trs = ""
    for _, r in d.iterrows():
        hl = ' style="background:#fef3c7;font-weight:bold;"' if r["事件"] in ("当前", "本轮至今最低") else ""
        trs += "<tr>" + "".join(f"<td{hl}>{r[c]}</td>" for c in d.columns) + "</tr>"
    figs = "".join(
        f'<figure><img src="data:image/png;base64,{_b64(p)}"/><figcaption>{cap}</figcaption></figure>'
        for p, cap in imgs)
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>BTC 历史买点与形态对比 {day}</title><style>
 body{{font-family:-apple-system,"Microsoft YaHei",sans-serif;background:#fafaf9;color:#1a1a1a;margin:0;}}
 .page{{max-width:1100px;margin:0 auto;padding:40px 28px 80px;}}
 h1{{font-size:23px;border-bottom:2px solid #1a1a1a;padding-bottom:12px;}}
 table{{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px;}}
 th,td{{border:1px solid #e5e5e3;padding:8px 11px;text-align:center;}} th{{background:#f3f4f6;}}
 figure{{margin:30px 0;}} img{{width:100%;border:1px solid #e5e5e3;border-radius:8px;}}
 figcaption{{color:#444;font-size:13.5px;margin-top:8px;line-height:1.6;}}
 .foot{{color:#9ca3af;font-size:12px;margin-top:36px;border-top:1px solid #e5e5e3;padding-top:14px;}}
</style></head><body><div class="page">
<h1>BTC 历史买点与结构/形态对比</h1>
<p style="color:#6b6b6b;font-size:14px;">生成于 {day}　·　历史三轮底(绿) vs 当前(黄)</p>
<table><tr>{th}</tr>{trs}</table>
{figs}
<div class="foot">买点=各周期价格最低日；指标取该日最近读数。形态图按距峰天数对齐、价格归一到峰顶。辅助分析，非买卖建议。</div>
</div></body></html>"""
    out = C.OUT_DIR / f"report_buypoints_{day}.html"
    out.write_text(html, encoding="utf-8")
    print(f"[html] {out}")
    return out


def main():
    price, fng, ahr, onc = load()
    end = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None).date())
    day = end.date().isoformat()

    table, peaks, bots = build_comparison(price, fng, ahr, onc)
    table.to_csv(C.OUT_DIR / "buypoint_comparison.csv", index=False, encoding="utf-8")
    print("\n=== 买点对照表 ===")
    print(_fmt(table).to_string(index=False))

    p1 = chart_overview(price, ahr, onc, bots, end)
    p2 = chart_pattern_overlay(price, peaks, end)
    build_html(table, [
        (p1, "历史三轮底(绿色竖线)在各指标上的位置 vs 当前(红线)。历史底都伴随 AHR999<0.45、MVRV<1。"),
        (p2, "各轮「顶→底」跌势按峰顶对齐、归一化叠加。粗红线=当前轮，看它的下跌深度与节奏走到了历史哪一步。"),
    ], day)
    print(f"\n完成，见 {C.OUT_DIR}")


if __name__ == "__main__":
    main()
