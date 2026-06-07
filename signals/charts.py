"""出图层：每个指标单独出一张大图，并汇总成一个自包含 HTML 页面。

产出到 outputs/charts/：
  price_*.png / ahr999_*.png / mvrv_*.png / fng_*.png / mcap_vs_realized_*.png
并生成 outputs/report_YYYY-MM-DD.html （图片以 base64 内嵌，单文件可分享）。

用法:
    python signals/charts.py             # 用本地数据出图 + HTML
    python signals/charts.py --refresh   # 先抓最新数据再出
"""
from __future__ import annotations
import argparse
import base64
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import sources as S
import indicators as I

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

BLUE, AMBER, GREEN, RED, PURPLE, GREY = "#1e40af", "#b45309", "#15803d", "#b91c1c", "#7c3aed", "#6b7280"
FIGSIZE = (14, 5.2)
DPI = 135


def _load(refresh: bool):
    if refresh:
        S.fetch_btc_daily(); S.fetch_fear_greed(); S.fetch_onchain()
    price = pd.read_csv(C.PRICE_CSV, parse_dates=["date"])
    fng = pd.read_csv(C.FNG_CSV, parse_dates=["date"])
    ahr = I.compute_ahr999(price)
    onc = None
    if C.ONCHAIN_CSV.exists():
        tmp = pd.read_csv(C.ONCHAIN_CSV, parse_dates=["date"])
        if not tmp.dropna(subset=["mvrv"]).empty:
            onc = tmp
    return price, fng, onc, ahr


def _clip(df, start):
    return df[df["date"] >= pd.Timestamp(start)].copy()


def _years(ax, start, end):
    ax.set_xlim(pd.Timestamp(start), end)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.22)


def _save(fig, name) -> Path:
    out = C.CHART_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] {out.name}")
    return out


# ----------------------------------------------------------------- 单图
def chart_price(ahr, onc, end, start="2015-01-01"):
    a = _clip(ahr, start)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogy(a["date"], a["price"], color=BLUE, lw=1.4, label="BTC 价格")
    if onc is not None:
        o = _clip(onc, start)
        ax.semilogy(o["date"], o["realized_price"], color=AMBER, lw=2.0, ls="--",
                    label="已实现价格 (全网成本线)")
    last = a.iloc[-1]
    ax.scatter([last["date"]], [last["price"]], color=RED, zorder=5)
    ax.set_title(f"BTC 价格 vs 全网成本线  (现价 \\${last['price']:,.0f}，{last['date'].date()})",
                 fontsize=15, fontweight="bold")
    ax.set_ylabel("价格 USD（对数）")
    ax.legend(loc="upper left")
    _years(ax, start, end)
    return _save(fig, f"price_{end.date()}.png")


def chart_ahr(ahr, end, start="2015-01-01"):
    a = _clip(ahr, start).dropna(subset=["ahr999"])
    last = a.iloc[-1]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(a["date"], a["ahr999"], color="#0f766e", lw=1.4)
    ax.axhspan(0, C.AHR_DEEP_VALUE, color=GREEN, alpha=0.13)
    ax.axhspan(C.AHR_DEEP_VALUE, C.AHR_ACC_TOP, color=AMBER, alpha=0.08)
    ax.axhline(C.AHR_DEEP_VALUE, color=GREEN, lw=1.0, ls=":")
    ax.axhline(C.AHR_ACC_TOP, color=AMBER, lw=1.0, ls=":")
    ax.text(a["date"].iloc[0], C.AHR_DEEP_VALUE, " 0.45 抄底线", color=GREEN, fontsize=9, va="bottom")
    ax.text(a["date"].iloc[0], C.AHR_ACC_TOP, " 1.2 定投上沿", color=AMBER, fontsize=9, va="bottom")
    ax.scatter([last["date"]], [last["ahr999"]], color=RED, zorder=5)
    ax.annotate(f"{last['ahr999']:.3f}", (last["date"], last["ahr999"]), color=RED,
                fontsize=11, fontweight="bold", xytext=(8, 0), textcoords="offset points")
    ax.set_title(f"AHR999 估值指标  (当前 {last['ahr999']:.3f} — 深度价值区)",
                 fontsize=15, fontweight="bold")
    ax.set_ylabel("AHR999")
    ax.set_ylim(0, min(5, a["ahr999"].max() * 1.1))
    _years(ax, start, end)
    return _save(fig, f"ahr999_{end.date()}.png")


def chart_mvrv(onc, end, start="2013-01-01"):
    o = _clip(onc, start).dropna(subset=["mvrv"])
    last = o.iloc[-1]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(o["date"], o["mvrv"], color=PURPLE, lw=1.4)
    ax.axhspan(0, 1.0, color=GREEN, alpha=0.13)
    ax.axhline(1.0, color=GREEN, lw=1.1, ls=":")
    ax.text(o["date"].iloc[0], 1.0, " MVRV=1 成本线（深熊投降底）", color=GREEN, fontsize=9, va="bottom")
    ax.scatter([last["date"]], [last["mvrv"]], color=RED, zorder=5)
    ax.annotate(f"{last['mvrv']:.2f}", (last["date"], last["mvrv"]), color=RED,
                fontsize=11, fontweight="bold", xytext=(8, 0), textcoords="offset points")
    ax.set_title(f"MVRV 比率  (当前 {last['mvrv']:.2f} — 仍在成本线之上，未到投降底)",
                 fontsize=15, fontweight="bold")
    ax.set_ylabel("MVRV = 账面市值 / 已实现市值")
    _years(ax, start, end)
    return _save(fig, f"mvrv_{end.date()}.png")


def chart_fng(fng, end, start="2018-02-01"):
    f = _clip(fng, start)
    last = f.iloc[-1]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(f["date"], f["value"], color=GREY, lw=1.0)
    ax.axhspan(0, 25, color=GREEN, alpha=0.13)
    ax.axhspan(75, 100, color=RED, alpha=0.11)
    ax.axhline(25, color=GREEN, lw=0.9, ls=":")
    ax.axhline(75, color=RED, lw=0.9, ls=":")
    ax.text(f["date"].iloc[0], 25, " 25 极度恐惧", color=GREEN, fontsize=9, va="bottom")
    ax.text(f["date"].iloc[0], 75, " 75 极度贪婪", color=RED, fontsize=9, va="bottom")
    ax.scatter([last["date"]], [last["value"]], color=RED, zorder=5)
    ax.annotate(f"{int(last['value'])}", (last["date"], last["value"]), color=RED,
                fontsize=11, fontweight="bold", xytext=(8, 0), textcoords="offset points")
    ax.set_title(f"恐慌贪婪指数  (当前 {int(last['value'])} — 极度恐惧)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Fear & Greed (0-100)")
    ax.set_ylim(0, 100)
    _years(ax, start, end)
    return _save(fig, f"fng_{end.date()}.png")


def chart_mcap_realized(onc, end, start="2013-01-01"):
    o = _clip(onc, start).copy()
    o["market_cap"] = o["mvrv"] * o["realized_cap"]
    last = o.dropna(subset=["realized_cap"]).iloc[-1]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogy(o["date"], o["market_cap"] / 1e9, color=BLUE, lw=1.5, label="账面市值（现价×流通量）")
    ax.semilogy(o["date"], o["realized_cap"] / 1e9, color=AMBER, lw=2.0, label="已实现市值（真实沉淀资本）")
    ax.fill_between(o["date"], o["realized_cap"] / 1e9, o["market_cap"] / 1e9,
                    where=(o["market_cap"] >= o["realized_cap"]), color=GREEN, alpha=0.09)
    ax.fill_between(o["date"], o["realized_cap"] / 1e9, o["market_cap"] / 1e9,
                    where=(o["market_cap"] < o["realized_cap"]), color=RED, alpha=0.16)
    ax.scatter([last["date"]], [last["market_cap"] / 1e9], color=BLUE, zorder=5)
    ax.scatter([last["date"]], [last["realized_cap"] / 1e9], color=AMBER, zorder=5)
    ax.set_title(f"账面市值 vs 已实现市值  (账面 \\${last['market_cap']/1e12:.2f}T | "
                 f"已实现 \\${last['realized_cap']/1e12:.2f}T | MVRV {last['mvrv']:.2f}) — 红色区=跌破成本线(投降底)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("十亿美元 (B，对数)")
    ax.legend(loc="upper left")
    _years(ax, start, end)
    return _save(fig, f"mcap_vs_realized_{end.date()}.png")


# ----------------------------------------------------------------- HTML
def _b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def build_html(items, header_html, day) -> Path:
    blocks = []
    for path, caption in items:
        blocks.append(
            f'<figure><img src="data:image/png;base64,{_b64(path)}" />'
            f'<figcaption>{caption}</figcaption></figure>'
        )
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BTC 底部信号报告 {day}</title>
<style>
 body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;background:#fafaf9;color:#1a1a1a;margin:0;}}
 .page{{max-width:1080px;margin:0 auto;padding:40px 28px 80px;}}
 h1{{font-size:24px;border-bottom:2px solid #1a1a1a;padding-bottom:12px;}}
 .sub{{color:#6b6b6b;font-size:14px;margin-top:-6px;}}
 table{{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px;}}
 th,td{{border:1px solid #e5e5e3;padding:8px 12px;text-align:left;}}
 th{{background:#f3f4f6;}}
 .verdict{{background:#dcfce7;border:1px solid #15803d;border-radius:8px;padding:14px 18px;font-size:16px;font-weight:bold;margin:18px 0;}}
 figure{{margin:34px 0;}}
 img{{width:100%;border:1px solid #e5e5e3;border-radius:8px;}}
 figcaption{{color:#444;font-size:13.5px;margin-top:8px;line-height:1.6;}}
 .foot{{color:#9ca3af;font-size:12px;margin-top:40px;border-top:1px solid #e5e5e3;padding-top:14px;}}
</style></head><body><div class="page">
<h1>BTC 底部信号报告</h1>
<div class="sub">生成于 {day}　·　数据源：CryptoCompare / alternative.me / bitcoin-data.com</div>
{header_html}
{''.join(blocks)}
<div class="foot">本页由 signals/charts.py 自动生成。指标为辅助判断，非买卖建议；底通常是一段区间，建议分批/区间执行。</div>
</div></body></html>"""
    out = C.OUT_DIR / f"report_{day}.html"
    out.write_text(html, encoding="utf-8")
    print(f"[html] {out}")
    return out


def _header(price, fng, onc, ahr):
    a = I.ahr999_signal(price)
    f = I.fng_signal(fng)
    rows = [
        ("AHR999", f"{a['value']}", a["zone"]),
        ("恐慌贪婪", f"{f['value']}", f["zone"]),
    ]
    verdict = "估值与情绪已到历史底部色带（适合分批吸筹）"
    if onc is not None:
        last = onc.dropna(subset=["mvrv"]).iloc[-1]
        rc = last["realized_cap"] / 1e12
        rows.append(("MVRV", f"{last['mvrv']:.2f}", "仍在成本线之上，未到投降底"))
        rows.append(("已实现市值", f"${rc:.2f}T", "真实沉淀资本，缓慢上升"))
        verdict += "；但 MVRV 仍 >1，真实价值维度显示绝对底可能更低"
    trs = "".join(f"<tr><td>{n}</td><td><b>{v}</b></td><td>{z}</td></tr>" for n, v, z in rows)
    return (f"<div class='verdict'>结论：{verdict}</div>"
            f"<table><tr><th>指标</th><th>当前值</th><th>区位</th></tr>{trs}</table>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    price, fng, onc, ahr = _load(args.refresh)
    end = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None).date())
    day = end.date().isoformat()

    items = []
    if onc is not None:
        items.append((chart_mcap_realized(onc, end), "橙线（已实现市值=真实沉淀资本）平滑缓慢上升；蓝线（账面市值）随币价剧烈波动。两线相交（MVRV=1）即历史投降底。当前蓝线仍高于橙线。"))
    items.append((chart_price(ahr, onc, end), "对数价格与全网成本线（已实现价格）。币价回落、正逼近成本线，但尚未跌破。"))
    items.append((chart_ahr(ahr, end), "AHR999 已跌回绿色深度价值区（<0.45），历史上对应抄底窗口。"))
    if onc is not None:
        items.append((chart_mvrv(onc, end), "MVRV 从牛市高位回落，但仍 >1（持币者整体浮盈）。跌破 1.0（绿区）才是 2022 那种资本投降底。"))
    items.append((chart_fng(fng, end), "恐慌贪婪指数贴近极度恐惧区，情绪冰点常出现在底部区域。"))

    header = _header(price, fng, onc, ahr)
    build_html(items, header, day)
    print(f"\n完成：{len(items)} 张图 + HTML 报告，见 {C.OUT_DIR}")


if __name__ == "__main__":
    main()
