# -*- coding: utf-8 -*-
"""AHR999 按「周期压缩」重读当前位置（用户 2026-11-09 / 11-13 的要求）。

思路：配对点位给时间，AHR 给「离底多远」，但 AHR 的数字每轮被压缩，不能用固定阈值（0.30 / 0.45 / 1.2）。
1. 每轮的 AHR 刻度：底 AHR 三轮几乎不动（0.272 / 0.260–0.267 / 0.280–0.281），顶 AHR 每轮被压缩。
   离底距离统一用 L = ln(AHR / 该轮底 AHR)。
2. 压缩怎么量：在时间配对的点上（user_pairs.csv：顶、用户三对、底后跳涨 / 小高峰）比 L(2021 轮) 和 L(2025 轮)，
   拟合 L25 = a · L21^p（高处压得多、低处压得少）；另给两个整体倍数做对照：×0.57（顶→底的对数幅度每轮比）和
   ×0.78（底后两个配对点的实测比）。
3. 把 2021 轮底后第一年的几个回调低点（2023-03-10 硅谷银行、2023-06-14、2023-09-11）按压缩换到本轮的 L，
   再按今天的 200 日几何均线和拟合估值换成今天的价格。
4. 底 AHR 不压缩：AHR 回到本轮底的水平（约 0.28）对应今天的价格 = 「06-30 是底」的硬下沿。

输出（本目录）：ahr_compressed_numbers.txt、ahr_compressed_ladder.csv、png/fig1–fig4（刻度、压缩量、配对时间 + 压缩水位、换成今天的价格）
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PNG = HERE / "png"
sys.path.insert(0, str(ROOT))
from signals.indicators import compute_ahr999  # noqa: E402

PRICE_CSV = ROOT / "data" / "btc_merged_daily.csv"
PAIRS_CSV = ROOT / "analysis_runs" / "2026-11-13_manual_point_compare" / "user_pairs.csv"
AS_OF = pd.Timestamp("2026-11-12")          # 最后一个 FRED 收盘；csv 末行 11-13 是盘中点
T = pd.Timestamp
BOT = {"2017": T("2018-12-15"), "2021": T("2022-11-21"), "2025": T("2026-06-30")}
TOP = {"2017": T("2017-12-16"), "2021": T("2021-11-08"), "2025": T("2025-08-12")}
LOWS_2021 = [("硅谷银行回调低点", T("2023-03-10")), ("第二次回调低点", T("2023-06-14")), ("第三次回调低点", T("2023-09-11"))]


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    PNG.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(PRICE_CSV, parse_dates=["date"]).set_index("date")["price"].sort_index()
    live_d, live_px = raw.index.max(), float(raw.iloc[-1])
    s = raw.loc[:AS_OF]
    full = compute_ahr999(s.rename("price").reset_index()).set_index("date")
    a = full["ahr999"]
    px = float(s.iloc[-1])
    out: list[str] = []
    P = out.append
    P(f"数据：{AS_OF.date()} 收盘 ${px:,.0f}（FRED）；{live_d.date()} 盘中 ${live_px:,.0f}（Yahoo，只作参考）")

    # ---------- 1. 每轮的刻度 ----------
    P("")
    P("== 1. 每轮的 AHR 刻度（底几乎不动，顶被压缩）")
    bA = {c: float(a[d]) for c, d in BOT.items()}
    for c in ["2017", "2021", "2025"]:
        P(f"{c} 轮：顶 {TOP[c].date()} AHR {a[TOP[c]]:.2f}；底 {BOT[c].date()} AHR {bA[c]:.3f}；顶→底 ln {np.log(a[TOP[c]] / bA[c]):.2f}")
    bull = {"2018-12 底 → 2021 顶": np.log(a[TOP["2021"]] / bA["2017"]), "2022-11 底 → 2025 顶": np.log(a[TOP["2025"]] / bA["2021"])}
    for k, v in bull.items():
        P(f"牛市段 {k}：ln(顶 AHR / 底 AHR) = {v:.2f}")
    r_bull = bull["2022-11 底 → 2025 顶"] / bull["2018-12 底 → 2021 顶"]
    nxt = bull["2022-11 底 → 2025 顶"] * r_bull
    P(f"牛市段每轮压缩 ×{r_bull:.2f} → 下一轮（2026-06 底 → 下一个顶）约 ln {nxt:.2f}，即顶 AHR 约 {bA['2025'] * np.exp(nxt):.2f}")

    # ---------- 2. 配对点上的压缩 ----------
    pr = pd.read_csv(PAIRS_CSV, parse_dates=["date_2021", "date_2025"])
    pr["L21"] = [np.log(a[d] / bA["2021"]) for d in pr.date_2021]
    pr["L25"] = [np.log(a[d] / bA["2025"]) for d in pr.date_2025]
    fit = pr[pr.L21 > 0.2]                      # 离底太近的点（对3、底）比值没意义，不进拟合
    p_, loga = np.polyfit(np.log(fit.L21), np.log(fit.L25), 1)
    a_ = float(np.exp(loga))
    comp = lambda x: a_ * np.power(np.maximum(x, 1e-9), p_)     # noqa: E731
    P("")
    P("== 2. 时间配对点上的压缩：L = ln(AHR / 该轮底 AHR)")
    for _, r in pr.iterrows():
        ratio = f"{r.L25 / r.L21:.2f}" if r.L21 > 0.2 else "—"
        P(f"{r.pair}：2021 {r.date_2021.date()} AHR {a[r.date_2021]:.3f}（L {r.L21:+.2f}） ↔ 本轮 {r.date_2025.date()} AHR {a[r.date_2025]:.3f}（L {r.L25:+.2f}）；比 {ratio}")
    P(f"拟合（{len(fit)} 个点，L21 > 0.2）：L25 = {a_:.2f} × L21^{p_:.2f}  → 高处压得多、低处压得少")
    for x in [0.25, 0.5, 1.0, 2.0]:
        P(f"   L21 = {x:.2f} → L25 = {comp(x):.2f}（相当于 ×{comp(x) / x:.2f}）")

    # ---------- 3. 今天在哪 ----------
    L_now = float(np.log(a.iloc[-1] / bA["2025"]))
    P("")
    P("== 3. 今天")
    P(f"{AS_OF.date()} AHR {a.iloc[-1]:.3f}，L {L_now:+.3f}；底后最高 L {np.log(a.loc[BOT['2025']:].max() / bA['2025']):+.3f}（{a.loc[BOT['2025']:].idxmax().date()} AHR {a.loc[BOT['2025']:].max():.3f}）")
    pos = L_now / nxt
    P(f"在本轮牛市段（底 → 外推的下一个顶，ln {nxt:.2f}）里的位置：{pos:.0%}")
    for name, top_c, bot_c in [("2021 轮（2022-11 底 → 2025 顶）", "2025", "2021"), ("2017 轮（2018-12 底 → 2021 顶）", "2021", "2017")]:
        band = np.log(a[TOP[top_c]] / bA[bot_c])
        b = BOT[bot_c]
        d_same = b + (AS_OF - BOT["2025"])
        L_same = np.log(a[d_same] / bA[bot_c])
        P(f"{name}：底后同一天 {d_same.date()} AHR {a[d_same]:.3f}，在它那一段里的位置 {L_same / band:.0%}")
    # 2021 轮底后第一年里（跳涨之后）AHR 的最低
    after_jump = a.loc[T("2023-01-14"):T("2023-11-21")]
    P(f"2021 轮跳涨之后的一年里 AHR 最低 {after_jump.min():.3f}（{after_jump.idxmin().date()}，L {np.log(after_jump.min() / bA['2021']):+.2f}）")
    inv = lambda y: float(np.exp((np.log(max(y, 1e-9)) - np.log(a_)) / p_))   # noqa: E731
    x_eq = inv(L_now)
    P(f"把今天的 L {L_now:+.2f} 反推回 2021 轮的刻度：L21 {x_eq:+.2f}（AHR {bA['2021'] * np.exp(x_eq):.3f}）；"
      f"按 ×0.57 反推：L21 {L_now / 0.57:+.2f}（AHR {bA['2021'] * np.exp(L_now / 0.57):.3f}）")

    # ---------- 4. 换成今天的价格 ----------
    logs199 = float(np.log(full["price"].iloc[-200:-1]).sum())
    est = float(full["estimate_price"].iloc[-1])

    def price_for(ahr: float) -> float:
        f = lambda x: (x / np.exp((logs199 + np.log(x)) / 200)) * (x / est) - ahr   # noqa: E731
        return brentq(f, 1000, 1e6)

    P("")
    P(f"== 4. 换成今天的价格（今天的 200 日几何均线 ${full['gma200'].iloc[-1]:,.0f}、拟合估值 ${est:,.0f}；价格 ≈ AHR 的平方根关系）")
    rows = []

    def add(name, L, note):
        ahr = bA["2025"] * np.exp(L)
        p = price_for(ahr)
        rows.append({"水位": name, "L": round(L, 3), "AHR": round(ahr, 3), "今天对应价格": round(p), "相对今天收盘": round(p / px - 1, 4), "说明": note})

    add("本轮底的 AHR（底不压缩）", 0.0, f"AHR {bA['2025']:.3f}；三轮底 0.272 / 0.267 / 0.281")
    for name, d in LOWS_2021:
        x = float(np.log(a[d] / bA["2021"]))
        add(f"2021 轮{name} {d:%Y-%m-%d}（按配对拟合压缩）", float(comp(x)), f"2021 轮 AHR {a[d]:.3f}，L21 {x:+.2f}")
        add(f"2021 轮{name} {d:%Y-%m-%d}（×0.57）", x * 0.57, "整体倍数：顶→底对数幅度每轮比")
        add(f"2021 轮{name} {d:%Y-%m-%d}（×0.78）", x * 0.78, "整体倍数：底后跳涨 / 小高峰两个配对点的实测比")
    add("今天", L_now, f"{AS_OF.date()} 收盘")
    lad = pd.DataFrame(rows)
    lad.to_csv(HERE / "ahr_compressed_ladder.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    P(lad.to_string(index=False))
    P(f"盘中 {live_d.date()} ${live_px:,.0f} 相对 11-12 收盘 {live_px / px - 1:+.1%}")

    # 前两轮：底之后 AHR 有没有再回到底的水平
    P("")
    P("== 5. 前两轮：真底之后，AHR 有没有再回到底的水平")
    for c, nxt_top in [("2017", "2021"), ("2021", "2025")]:
        b = BOT[c]
        seg = a.loc[b + pd.Timedelta(days=30): TOP[nxt_top]]
        m, md = float(seg.min()), seg.idxmin()
        P(f"{c} 轮底 {b.date()}（AHR {bA[c]:.3f}）→ 下一个顶之前，AHR 最低 {m:.3f}（{md.date()}，L {np.log(m / bA[c]):+.2f}，价格比底 {s[md] / s[b] - 1:+.0%}）")
    # ---------- 6. 按配对把 2021 轮的时间对到本轮，再把它的 AHR 压缩到本轮刻度 ----------
    prs = pr.sort_values("date_2021").reset_index(drop=True)
    x21 = np.array([(d - TOP["2021"]).days for d in prs.date_2021], float)
    x25 = np.array([(d - TOP["2025"]).days for d in prs.date_2025], float)

    def warp(days):
        v = np.interp(days, x21, x25)
        v = np.where(days > x21[-1], x25[-1] + (days - x21[-1]), v)
        return np.where(days < x21[0], x25[0] + (days - x21[0]), v)

    def to25(ahr21, how="fit"):
        L = np.log(np.asarray(ahr21, float) / bA["2021"])
        if how == "fit":
            Lc = np.where(L > 0, comp(np.maximum(L, 1e-9)), L)
        else:
            Lc = np.where(L > 0, L * how, L)
        return bA["2025"] * np.exp(Lc)

    seg21 = a.loc[TOP["2021"] - pd.Timedelta(days=60): T("2023-12-31")]
    d25 = TOP["2025"] + pd.to_timedelta(warp((seg21.index - TOP["2021"]).days.values.astype(float)), unit="D")
    map_fit = pd.Series(to25(seg21.values, "fit"), index=d25)
    map_057 = pd.Series(to25(seg21.values, 0.57), index=d25)
    i_now = int(np.argmin(np.abs((d25 - AS_OF).days)))
    today_map = float(map_fit.iloc[i_now])
    eq21 = seg21.index[i_now]
    P("")
    P("== 6. 配对时间 + 压缩水位")
    P(f"按配对（最后一对之后按日历天 1:1），本轮今天对应 2021 轮 {eq21.date()}：2021 轮那天 AHR {a[eq21]:.3f}，"
      f"压缩到本轮刻度是 {today_map:.3f}；本轮实际 {a.iloc[-1]:.3f}")
    Lj = float(np.log(a[T('2026-08-15')] / bA['2025']))
    Lp = float(np.log(a[T('2026-10-03')] / bA['2025']))
    P(f"本轮跳涨 L {Lj:+.2f}，小高峰 L {Lp:+.2f}，今天 L {L_now:+.2f}：跳涨→小高峰这段按 L 回吐了 {(Lp - L_now) / (Lp - Lj):.0%}")
    for name, b0, t1 in [("2018-12 底 → 2021 顶", BOT["2017"], TOP["2021"]), ("2022-11 底 → 2025 顶", BOT["2021"], TOP["2025"])]:
        seg = a.loc[b0:t1]
        P(f"牛市段 {name}：AHR 最高 {seg.max():.2f}（{seg.idxmax().date()}，往往早于价格顶）；价格顶那天 {a[t1]:.2f}")
    low_dates = {}
    for name, d in LOWS_2021:
        dd = TOP["2025"] + pd.Timedelta(days=float(warp(np.array([(d - TOP["2021"]).days], float))[0]))
        low_dates[name] = dd
        P(f"2021 轮{name} {d.date()} → 本轮日历约 {dd.date()}")
    (HERE / "ahr_compressed_numbers.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))

    draw(dict(ladder=lad, pr=pr, a_=a_, p_=p_, comp=comp, map_fit=map_fit, map_057=map_057, prs=prs, nxt=nxt,
              px=px, live_px=live_px, live_d=live_d, bA=bA, a=a, today_map=today_map, eq21=eq21, low_dates=low_dates))


def draw(f: dict) -> None:
    a, bA = f["a"], f["bA"]
    C21, C25 = "#0d9488", "#0f172a"
    SRC = {"user": "#dc2626", "marked_points": "#2563eb", "anchor": "#475569"}

    # 图 1：每轮的 AHR 刻度
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=170)
    xs = [0, 1, 2, 3]
    labs = ["2017 轮", "2021 轮", "2025 轮", "下一轮（外推）"]
    top_at = [float(a[TOP["2017"]]), float(a[TOP["2021"]]), float(a[TOP["2025"]]), float(bA["2025"] * np.exp(f["nxt"]))]
    bull_max = [float(a.loc[:TOP["2017"]].max()), float(a.loc[BOT["2017"]:TOP["2021"]].max()), float(a.loc[BOT["2021"]:TOP["2025"]].max())]
    bots = [bA["2017"], bA["2021"], bA["2025"]]
    ax.semilogy(xs[:3], top_at[:3], "o-", color="#dc2626", lw=2.2, ms=8, label="价格见顶那天的 AHR")
    ax.semilogy(xs[2:], top_at[2:], "--", color="#dc2626", lw=1.6)
    ax.semilogy([xs[3]], [top_at[3]], "o", color="#dc2626", ms=8, mfc="white")
    ax.semilogy(xs[:3], bull_max, "s-", color="#f59e0b", lw=1.6, ms=7, label="整段牛市里 AHR 的最高")
    ax.semilogy(xs[:3], bots, "o-", color="#16a34a", lw=2.2, ms=8, label="熊市底的 AHR")
    for x, v in zip(xs, top_at):
        ax.annotate(f"{v:.2f}", (x, v), xytext=(8, 4), textcoords="offset points", fontsize=10, color="#dc2626", weight="bold")
    for x, v, tv in zip(xs, bull_max, top_at):
        if abs(v / tv - 1) < 0.02:          # 2017 轮牛市最高就在价格顶那天，不重复标
            continue
        ax.annotate(f"{v:.2f}", (x, v), xytext=(8, 4), textcoords="offset points", fontsize=9, color="#b45309")
    for x, v in zip(xs, bots):
        ax.annotate(f"{v:.3f}", (x, v), xytext=(8, -14), textcoords="offset points", fontsize=10, color="#16a34a", weight="bold")
    for y, v in [(0.30, "0.30"), (0.45, "0.45"), (1.2, "1.2")]:
        ax.axhline(y, color="#94a3b8", lw=0.8, ls=":")
        ax.text(3.35, y, f"旧固定阈值 {v}", fontsize=8, color="#64748b", va="center")
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.set_xlim(-0.3, 3.95)
    ax.set_ylabel("AHR999（对数刻度）")
    ax.grid(True, which="both", color="#eef2f7")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("图 1　AHR999 的刻度每轮都在变：顶被压缩，底几乎不动", fontsize=13, weight="bold", loc="left")
    fig.text(0.01, -0.01, "「下一轮」的顶 = 本轮底的 AHR × exp(牛市段对数幅度 × 每轮压缩比 0.59)，只是外推。AHR 数据从 2015-06 起。",
             fontsize=8, color="#475569")
    fig.tight_layout()
    fig.savefig(PNG / "fig1_ahr_scale_by_cycle.png", bbox_inches="tight")
    plt.close(fig)

    # 图 2：压缩量（配对点）
    pr, comp = f["pr"], f["comp"]
    fig, ax = plt.subplots(figsize=(10, 7), dpi=170)
    xs2 = np.linspace(0.01, 2.9, 200)
    ax.plot(xs2, xs2, color="#cbd5e1", ls="--", lw=1.2, label="不压缩（1:1）")
    ax.plot(xs2, xs2 * 0.57, color="#f59e0b", ls=":", lw=1.6, label="整体 ×0.57")
    ax.plot(xs2, comp(xs2), color=C25, lw=2.4, label=f"配对拟合：本轮 L = {f['a_']:.2f} × (2021 轮 L)^{f['p_']:.2f}")
    for _, r in pr.iterrows():
        if r.L21 <= 0.05:
            continue
        col = SRC.get(r.source, "#475569")
        ax.scatter([r.L21], [r.L25], color=col, s=70, zorder=5, edgecolor="white", linewidth=1.2)
        near = r.L21 <= 0.2
        tag = "（离底太近，不进拟合）" if near else f"　比 {r.L25 / r.L21:.2f}"
        ax.annotate(f"{r.pair}{tag}\n{r.date_2021:%Y-%m-%d} 对 {r.date_2025:%Y-%m-%d}", (r.L21, r.L25),
                    xytext=(-6, 16) if near else (8, -20), textcoords="offset points", fontsize=8.5, color=col)
    ax.set_xlabel("2021 轮：L = ln(AHR ÷ 该轮底的 AHR)")
    ax.set_ylabel("本轮：L = ln(AHR ÷ 该轮底的 AHR)")
    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 2.0)
    ax.grid(True, color="#e2e8f0")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title("图 2　压缩量：同一个时间点上，两轮离底有多远", fontsize=13, weight="bold", loc="left")
    fig.text(0.01, -0.01, "红点 = 你给的配对；灰点 = 报告的顶；蓝点 = 底后标点（跳涨、第一个小高峰）。离底太近的点（对3、底）不进拟合。",
             fontsize=8.5, color="#475569")
    fig.tight_layout()
    fig.savefig(PNG / "fig2_compression_fit.png", bbox_inches="tight")
    plt.close(fig)

    # 图 3：配对时间 + 压缩水位（主图）
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=170)
    lo_d, hi_d = T("2025-06-15"), T("2027-05-15")
    a25 = a.loc[lo_d:]
    mf, m57 = f["map_fit"], f["map_057"]
    mf = mf[(mf.index >= lo_d) & (mf.index <= hi_d)]
    m57 = m57[(m57.index >= lo_d) & (m57.index <= hi_d)]
    ax.semilogy(m57.index, m57.values, color=C21, lw=1.0, ls="--", alpha=0.6, label="2021 轮（时间按配对对齐，AHR 整体 ×0.57 压缩）")
    ax.semilogy(mf.index, mf.values, color=C21, lw=1.9, label="2021 轮（时间按配对对齐，AHR 按配对拟合压缩）")
    ax.semilogy(a25.index, a25.values, color=C25, lw=2.8, label="本轮 AHR999")
    ax.axhline(bA["2025"], color="#16a34a", lw=1.2, ls="-.")
    ax.text(hi_d, bA["2025"] * 0.97, f"本轮底的 AHR {bA['2025']:.3f}（底不压缩）", ha="right", va="top", fontsize=9, color="#16a34a")
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for _, r in f["prs"].iterrows():
        col = SRC.get(r.source, "#475569")
        ax.axvline(r.date_2025, color=col, lw=1.0, ls="-" if r.source == "user" else "--", alpha=0.8)
        ax.text(r.date_2025, 0.985, f" {r.pair}", transform=tr, fontsize=8.5, color=col, va="top")
    for name, d in LOWS_2021:
        dd = f["low_dates"][name]
        v = float(mf.iloc[int(np.argmin(np.abs((mf.index - dd).days)))])
        ax.scatter([dd], [v], color=C21, s=40, zorder=6, edgecolor="white")
        off = (-150, -16) if name.startswith("硅谷") else (4, -28)
        ax.annotate(f"2021 轮{name}\n（{d:%Y-%m-%d}）", (dd, v), xytext=off, textcoords="offset points", fontsize=8, color=C21)
    ax.scatter([AS_OF], [a.iloc[-1]], color="#7c3aed", s=70, zorder=7, edgecolor="white")
    ax.annotate(f"今天 {AS_OF:%m-%d}\nAHR {a.iloc[-1]:.3f}", (AS_OF, a.iloc[-1]), xytext=(10, -34), textcoords="offset points",
                fontsize=9.5, color="#7c3aed", weight="bold")
    ax.annotate(f"同一时间的 2021 轮（{f['eq21']:%Y-%m-%d}）\n压缩后 {f['today_map']:.3f}", (AS_OF, f["today_map"]), xytext=(18, 12),
                textcoords="offset points", fontsize=9, color=C21, arrowprops={"arrowstyle": "->", "color": C21})
    ax.set_xlim(lo_d, hi_d)
    ax.set_ylim(0.22, 1.6)
    ax.yaxis.set_major_locator(mticker.FixedLocator([0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3]))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, which="both", color="#eef2f7")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylabel("AHR999（本轮刻度，对数）")
    ax.set_title("图 3　配对给时间、压缩后的 AHR 给水位：本轮 vs 2021 轮", fontsize=13, weight="bold", loc="left")
    fig.text(0.01, -0.01, "2021 轮的时间按配对分段对齐到本轮日历（最后一个配对之后按日历天 1:1）；2021 轮在今天之后的部分只是参照路径，不是预测。",
             fontsize=8.5, color="#475569")
    fig.tight_layout()
    fig.savefig(PNG / "fig3_pairs_time_compressed_level.png", bbox_inches="tight")
    plt.close(fig)

    # 图 4：压缩后的水位 → 今天的价格
    lad, px = f["ladder"], f["px"]
    fig, ax = plt.subplots(figsize=(12, 5.8), dpi=170)
    items = [("本轮底的 AHR（不压缩）", "本轮底的 AHR", "#16a34a")]
    for name, d in LOWS_2021:
        items.append((f"2021 轮{name}\n{d:%Y-%m-%d}", f"{name} {d:%Y-%m-%d}", C21))
    ys = np.arange(len(items))[::-1]
    for y, (lab, key, col) in zip(ys, items):
        sub = lad[lad["水位"].str.contains(key, regex=False)]
        main_row = sub[sub["水位"].str.contains("拟合|底的")].iloc[0]
        lo, hi = sub["今天对应价格"].min(), sub["今天对应价格"].max()
        if hi > lo:
            ax.plot([lo, hi], [y, y], color=col, lw=7, alpha=0.25, solid_capstyle="round")
        ax.scatter([main_row["今天对应价格"]], [y], color=col, s=90, zorder=5, edgecolor="white")
        txt = f"${main_row['今天对应价格']:,.0f}（AHR {main_row['AHR']:.2f}）"
        if hi > lo:
            txt += f"　三种压缩 ${lo:,.0f}–${hi:,.0f}"
        ax.text(max(hi, main_row["今天对应价格"]) + 700, y, txt, va="center", fontsize=9.5)
    ax.axvline(px, color="#7c3aed", lw=1.8)
    ax.text(px, ys[0] + 0.55, f"11-12 收盘 ${px:,.0f}", color="#7c3aed", fontsize=9.5, ha="center")
    ax.axvline(f["live_px"], color="#7c3aed", lw=1.0, ls=":")
    ax.text(f["live_px"], ys[-1] - 0.62, f"{f['live_d']:%m-%d} 盘中 ${f['live_px']:,.0f}", color="#7c3aed", fontsize=8.5, ha="center")
    ax.axvline(62835, color="#dc2626", lw=1.0, ls="--")
    ax.text(62835, ys[-1] - 0.62, "8 月更高的低点 $62,835", color="#dc2626", fontsize=8.5, ha="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([i[0] for i in items], fontsize=9.5)
    ax.set_ylim(ys[-1] - 0.95, ys[0] + 0.9)
    ax.set_xlim(56000, 96000)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v / 1000:.0f}k"))
    ax.grid(True, axis="x", color="#eef2f7")
    ax.set_xlabel("按 11-12 的 200 日几何均线和拟合估值，换成今天的价格")
    ax.set_title("图 4　压缩后的 AHR 水位，换成今天的价格", fontsize=13, weight="bold", loc="left")
    fig.text(0.01, -0.01, "实心点 = 按配对拟合压缩；浅色条 = 三种压缩（拟合、×0.57、×0.78）的范围。底的 AHR 三轮几乎不变，所以不压缩。",
             fontsize=8.5, color="#475569")
    fig.tight_layout()
    fig.savefig(PNG / "fig4_levels_to_price.png", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
