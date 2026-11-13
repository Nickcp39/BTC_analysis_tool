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

输出（本目录）：ahr_compressed_numbers.txt、ahr_compressed_ladder.csv、png/ahr_compressed.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
    (HERE / "ahr_compressed_numbers.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))

    # ---------- 图 ----------
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.4), dpi=150, gridspec_kw={"width_ratios": [1.0, 1.55, 1.15]})
    ax = axs[0]
    xs = np.linspace(0.01, 3.0, 200)
    ax.plot(xs, xs, color="#cbd5e1", ls="--", lw=1, label="不压缩")
    ax.plot(xs, xs * 0.57, color="#f59e0b", ls=":", lw=1.3, label="×0.57")
    ax.plot(xs, comp(xs), color="#0f172a", lw=2, label=f"配对拟合 L25 = {a_:.2f}·L21^{p_:.2f}")
    for _, r in pr.iterrows():
        if r.L21 > 0.05:
            ax.scatter([r.L21], [r.L25], color="#dc2626" if r.source == "user" else "#2563eb", s=40, zorder=5)
            ax.annotate(r.pair, (r.L21, r.L25), xytext=(4, -10), textcoords="offset points", fontsize=8)
    ax.set_xlabel("2021 轮：ln(AHR / 底 AHR)")
    ax.set_ylabel("本轮：ln(AHR / 底 AHR)")
    ax.set_title("① 压缩量：同一时间点上两轮离底多远", fontsize=11, weight="bold", loc="left")
    ax.grid(True, color="#e2e8f0")
    ax.legend(fontsize=8, loc="upper left")

    ax = axs[1]
    for c, col, lw in [("2017", "#94a3b8", 1.2), ("2021", "#0d9488", 1.6), ("2025", "#0f172a", 2.6)]:
        b = BOT[c]
        seg = a.loc[b - pd.Timedelta(days=30): min(b + pd.Timedelta(days=400), AS_OF)]
        d = (seg.index - b).days
        L = np.log(seg.values / bA[c])
        if c == "2021":
            ax.plot(d, comp(np.maximum(L, 0)) * np.sign(L) if False else np.where(L > 0, comp(np.maximum(L, 1e-9)), L * 0.78),
                    color=col, lw=lw, label="2021 轮（按配对拟合压缩）")
        elif c == "2017":
            ax.plot(d, np.where(L > 0, comp(comp(np.maximum(L, 1e-9))), L), color=col, lw=lw, label="2017 轮（拟合压缩两次）")
        else:
            ax.plot(d, L, color=col, lw=lw, label="本轮")
    for name, dlow in LOWS_2021:
        dd = (dlow - BOT["2021"]).days
        y = comp(np.log(a[dlow] / bA["2021"]))
        ax.scatter([dd], [y], color="#0d9488", s=30, zorder=6)
        ax.annotate(f"{dlow:%y-%m-%d}", (dd, y), xytext=(3, -12), textcoords="offset points", fontsize=7.5, color="#0d9488")
    ax.axhline(0, color="#475569", lw=0.9)
    ax.axhline(L_now, color="#7c3aed", lw=1, ls=":")
    ax.text(395, L_now, f"今天 L {L_now:+.2f}", color="#7c3aed", fontsize=8.5, va="bottom", ha="right")
    ax.set_xlabel("距各自的底的天数（日历天）")
    ax.set_ylabel("ln(AHR / 底 AHR)，前两轮已压缩")
    ax.set_title("② 底后 AHR 走法（压缩后）", fontsize=11, weight="bold", loc="left")
    ax.grid(True, color="#e2e8f0")
    ax.legend(fontsize=8, loc="upper left")

    ax = axs[2]
    show = lad[lad["水位"].str.contains("拟合|底的|今天")]
    ys = np.arange(len(show))[::-1]
    ax.barh(ys, show["今天对应价格"], color=["#16a34a" if "底的" in n else "#7c3aed" if n == "今天" else "#0d9488" for n in show["水位"]],
            alpha=0.85)
    for y, (_, r) in zip(ys, show.iterrows()):
        ax.text(r["今天对应价格"] + 500, y, f"${r['今天对应价格']:,.0f}（AHR {r['AHR']:.2f}）", va="center", fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([n.replace("（按配对拟合压缩）", "").replace("2021 轮", "2021 ") for n in show["水位"]], fontsize=8.5)
    ax.set_xlim(50000, px * 1.18)
    ax.set_xlabel("按今天的均线换成的价格（$）")
    ax.set_title("③ 压缩后的 AHR 水位 → 今天的价格", fontsize=11, weight="bold", loc="left")
    ax.grid(True, axis="x", color="#e2e8f0")
    fig.suptitle(f"AHR999 按周期压缩重读（数据截至 {AS_OF:%Y-%m-%d} 收盘 ${px:,.0f}）", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(PNG / "ahr_compressed.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
