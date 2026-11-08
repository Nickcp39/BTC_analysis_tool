# -*- coding: utf-8 -*-
"""按波段（高点/低点）一一对齐本轮和 2021 轮的大底之后走势（2026-11-07）。

为什么要这一版：post_bottom_similarity.py 的右图用的是 10-01 框架给的固定倍数（时间 ×0.94、幅度 ×0.70），
两条线的高低点错开了半个波段——本轮 10-04 见高点时，缩放后的 2021 线正在走「硅谷银行回调」；
本轮现在回调，2021 线却在 2023-04 的高点。数据自己最佳拟合的时间倍数是 0.71，不是 0.94。

做法：本轮底后的 4 个转折点（6% 折线法）
    ① 2026-09-03 高 $81.1k   ② 2026-09-15 低 $75.7k   ③ 2026-10-04 高 $86.4k   ④ 2026-11-06 今天 $75.9k（回调中，未确认是低点）
和 2021 轮的转折点按两种数法配对：
    数法 A（今天 = 第二次大回调）：① 2023-02-20  ② 2023-03-10  ③ 2023-04-14  ④ 2023-06-14
    数法 B（今天 = 硅谷银行回调）：① 2023-02-01  ② 2023-02-10  ③ 2023-02-20  ④ 2023-03-10
每种数法用 4 对点做过原点的最小二乘，得到一个时间倍数 kt（本轮天数 = kt × 2021 天数）和一个幅度倍数 ka
（本轮对数涨幅 = ka × 2021 对数涨幅）。再用这两个倍数，从今天往后接 2021 轮的走法，作为参照。

口径：本轮顶 2025-08-12 $123,366，底 2026-06-30 $58,586；2021 轮底 2022-11-21。数据 data/btc_merged_daily.csv。
输出（本目录）：swing_matched.txt、png/swing_matched_2021.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PNG = HERE / "png"
MERGED_CSV = ROOT / "data" / "btc_merged_daily.csv"
OUT_TXT = HERE / "swing_matched.txt"
T = pd.Timestamp

AS_OF = T("2026-11-06")
BOTTOM_NOW, TOP_NOW = T("2026-06-30"), T("2025-08-12")
BOTTOM_21, TOP_21 = T("2022-11-21"), T("2025-08-12")
NOW_PIV = [T("2026-09-03"), T("2026-09-15"), T("2026-10-04"), T("2026-11-06")]
NUM = ["①", "②", "③", "④"]
COUNTS = {
    "A": dict(
        title="数法 A：今天 ≈ 2023-06-14（2021 轮第二次大回调）",
        hist=[T("2023-02-20"), T("2023-03-10"), T("2023-04-14"), T("2023-06-14")],
        fwd=[T("2023-07-13"), T("2023-09-11"), T("2023-12-08"), T("2024-03-13")],
    ),
    "B": dict(
        title="数法 B：今天 ≈ 2023-03-10（2021 轮硅谷银行回调）",
        hist=[T("2023-02-01"), T("2023-02-10"), T("2023-02-20"), T("2023-03-10")],
        fwd=[T("2023-04-14"), T("2023-06-14"), T("2023-09-11"), T("2023-12-08")],
    ),
}
INK, TEAL, MUTED, TODAY = "#0f172a", "#0d9488", "#64748b", "#7c3aed"
XMAX = 430
LOWS = {T("2023-06-14"), T("2023-09-11")}  # 往后参照点里的低点，标签放下方


def log_path(s: pd.Series, bottom: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    seg = s.loc[bottom:end]
    return pd.Series(np.log(seg.values / float(s[bottom])), index=(seg.index - bottom).days)


def ls0(x: np.ndarray, y: np.ndarray) -> float:
    """过原点最小二乘斜率。"""
    return float(x @ y) / float(x @ x)


def pct(v: float) -> str:
    return f"{np.exp(v) - 1:+.1%}"


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    PNG.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    out = lines.append

    s = pd.read_csv(MERGED_CSV, parse_dates=["date"]).set_index("date")["price"].loc[:AS_OF]
    p0, p_now, p_top = float(s[BOTTOM_NOW]), float(s[AS_OF]), float(s[TOP_NOW])
    now = log_path(s, BOTTOM_NOW, AS_OF)
    h21 = log_path(s, BOTTOM_21, TOP_21)
    days = now.index.to_numpy().astype(float)
    y = now.to_numpy()
    n = int(days[-1])
    top_log = np.log(p_top / p0)

    nd = np.array([(d - BOTTOM_NOW).days for d in NOW_PIV], dtype=float)
    nv = np.array([now[int(d)] for d in nd])
    out(f"本轮底 {BOTTOM_NOW.date()} ${p0:,.0f}；今天 {AS_OF.date()} 第 {n} 天 ${p_now:,.0f}（{pct(y[-1])}）；前顶 ${p_top:,.0f} = 底 {pct(top_log)}")
    out("本轮转折点：" + "  ".join(f"{k} {d.date()} 第{int(x)}天 ${float(s[d]):,.0f}（{pct(v)}）"
                              for k, d, x, v in zip(NUM, NOW_PIV, nd, nv)))
    out("")

    res = {}
    for key, c in COUNTS.items():
        hd = np.array([(d - BOTTOM_21).days for d in c["hist"]], dtype=float)
        hv = np.array([h21[int(d)] for d in hd])
        kt, ka = ls0(hd, nd), ls0(hv, nv)
        yhat = ka * np.interp(days / kt, h21.index.to_numpy(), h21.to_numpy())
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        r = float(np.corrcoef(y, yhat)[0, 1])
        out(f"== {c['title']}")
        out(f"   拟合：时间倍数 kt = {kt:.3f}，幅度倍数 ka = {ka:.3f}；全程（第 0–{n} 天）对数 RMSE {rmse:.3f}，相关 r {r:.3f}")
        for k, d_now, x, v, d_h, xh, vh in zip(NUM, NOW_PIV, nd, nv, c["hist"], hd, hv):
            out(f"   {k} 本轮 {d_now:%m-%d} 第{int(x):3d}天 {pct(v):>7} ↔ 2021 轮 {d_h.date()} 第{int(xh):3d}天 {pct(vh):>7}"
                f"   时间比 {x / xh:.2f}  幅度比 {v / vh:.2f}")
        # 两次回调的深度
        dd_now = [np.exp(nv[1] - nv[0]) - 1, np.exp(nv[3] - nv[2]) - 1]
        dd_h = [np.exp(hv[1] - hv[0]) - 1, np.exp(hv[3] - hv[2]) - 1]
        out(f"   回调深度：①→② 本轮 {dd_now[0]:.1%} vs 2021 {dd_h[0]:.1%}；③→④ 本轮 {dd_now[1]:.1%}（进行中） vs 2021 {dd_h[1]:.1%}"
            f"（按幅度倍数折算 {np.exp(ka * np.log(1 + dd_h[1])) - 1:.1%} → ${float(s[NOW_PIV[2]]) * np.exp(ka * np.log(1 + dd_h[1])):,.0f}）")
        # 起涨那一跳：底后首次 +25%
        d25_now = next(int(d) for d, v in now.items() if v >= np.log(1.25))
        d25_h = next(int(d) for d, v in h21.items() if v >= np.log(1.25))
        out(f"   起涨（底后首次 +25%）：本轮第 {d25_now} 天；2021 轮第 {d25_h} 天 → 按 kt 折算到本轮第 {d25_h * kt:.0f} 天")
        out(f"   若整轮牛市都按 kt 缩：2021 轮牛市 {(TOP_21 - BOTTOM_21).days} 天 × {kt:.2f} = {(TOP_21 - BOTTOM_21).days * kt:.0f} 天 → 顶约 "
            f"{(BOTTOM_NOW + pd.Timedelta(days=int((TOP_21 - BOTTOM_21).days * kt))).date()}（10-01 框架：935–1,001 天，2029-01-20 / 2029-03-27）")

        # 从今天接 2021 轮的走法
        h_anchor = int(hd[-1])
        hh = h21.loc[h_anchor:]
        cont_d = n + kt * (hh.index.to_numpy() - h_anchor)
        cont_v = y[-1] + ka * (hh.to_numpy() - hh.iloc[0])
        out("   从今天接 2021 轮走法（参照，不是预测）：")
        for d in c["fwd"]:
            xh = (d - BOTTOM_21).days
            dd = n + kt * (xh - h_anchor)
            vv = y[-1] + ka * (h21[xh] - h21[h_anchor])
            out(f"     对应 {d.date()} → 本轮第 {dd:.0f} 天（{(BOTTOM_NOW + pd.Timedelta(days=round(dd))).date()}）≈ ${p0 * np.exp(vv):,.0f}")
        brk = np.argmax(cont_v >= top_log) if (cont_v >= top_log).any() else None
        if brk is not None:
            out(f"     首次站上前顶 ${p_top:,.0f}：本轮第 {cont_d[brk]:.0f} 天（{(BOTTOM_NOW + pd.Timedelta(days=round(cont_d[brk]))).date()}），"
                f"对应 2021 轮 {(BOTTOM_21 + pd.Timedelta(days=int(hh.index[brk]))).date()}")
        nxt = cont_v[cont_d <= n + 60]
        out(f"     未来 60 天参照路径：最低 ${p0 * np.exp(nxt.min()):,.0f}，最高 ${p0 * np.exp(nxt.max()):,.0f}")
        out("")
        res[key] = dict(hd=hd, hv=hv, kt=kt, ka=ka, rmse=rmse, r=r, cont_d=cont_d, cont_v=cont_v, h_anchor=h_anchor)

    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---------------- 图 ----------------
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.0), dpi=160, sharey=True)
    for ax, (key, c) in zip(axs, COUNTS.items()):
        R = res[key]
        kt, ka, ha = R["kt"], R["ka"], R["h_anchor"]
        # 2021 轮：配对段（实线）+ 从今天接着走（虚线）
        hpart = h21.loc[:ha]
        ax.plot(hpart.index * kt, (np.exp(ka * hpart.to_numpy()) - 1) * 100, color=TEAL, lw=2.0,
                label=f"2021 轮（底 2022-11-21），时间×{kt:.2f}、幅度×{ka:.2f}")
        m = R["cont_d"] <= XMAX
        ax.plot(R["cont_d"][m], (np.exp(R["cont_v"][m]) - 1) * 100, color=TEAL, lw=2.0, ls=(0, (5, 3)),
                label="2021 轮接下来的走法，从今天接上（参照）")
        ax.plot(days, (np.exp(y) - 1) * 100, color=INK, lw=3.2, label="本轮（底 2026-06-30）", zorder=6)
        # 转折点编号
        for i, k in enumerate(NUM):
            xb, vb = nd[i], (np.exp(nv[i]) - 1) * 100
            xt, vt = R["hd"][i] * kt, (np.exp(ka * R["hv"][i]) - 1) * 100
            hi = i % 2 == 0
            ax.scatter([xb], [vb], s=70, color=INK, edgecolor="white", linewidth=2, zorder=8)
            ax.scatter([xt], [vt], s=70, facecolor="white", edgecolor=TEAL, linewidth=2, zorder=8)
            ax.annotate(k, (xb, vb), xytext=(-7, 11 if hi else -17), textcoords="offset points", ha="center",
                        fontsize=12, weight="bold", color=INK, zorder=9)
            ax.annotate(k, (xt, vt), xytext=(9, 11 if hi else -17), textcoords="offset points", ha="center",
                        fontsize=11, color=MUTED, zorder=9)
        # 往后的几个参照点
        for d in c["fwd"]:
            xh = (d - BOTTOM_21).days
            dd = n + kt * (xh - ha)
            if dd > XMAX:
                continue
            vv = y[-1] + ka * (h21[xh] - h21[ha])
            dt = BOTTOM_NOW + pd.Timedelta(days=round(dd))
            ax.scatter([dd], [(np.exp(vv) - 1) * 100], s=26, color=TEAL, zorder=7)
            ax.annotate(f"{dt:%Y-%m} ≈${p0 * np.exp(vv) / 1000:,.0f}k\n（像 {d:%Y-%m-%d}）", (dd, (np.exp(vv) - 1) * 100),
                        xytext=(0, -30 if d in LOWS else 10), textcoords="offset points", ha="center",
                        fontsize=8.2, color="#334155")
        ax.axhline((p_top / p0 - 1) * 100, color=MUTED, lw=1.0, ls="--")
        ax.text(4, (p_top / p0 - 1) * 100 + 3, f"前顶 ${p_top:,.0f}", color=MUTED, fontsize=8.8)
        ax.axhline(0, color="#475569", lw=0.9)
        ax.axvline(n, color=TODAY, ls=":", lw=1.3)
        ax.annotate(f"今天 第 {n} 天\n${p_now:,.0f}（{np.exp(y[-1]) - 1:+.0%}）", (n, (np.exp(y[-1]) - 1) * 100),
                    xytext=(16, -62), textcoords="offset points", color=TODAY, weight="bold", fontsize=9.5,
                    arrowprops=dict(arrowstyle="->", color=TODAY))
        ax.set_xlim(0, XMAX)
        ax.set_ylim(-20, 200)
        ax.set_xlabel("距本轮底的天数")
        ax.grid(True, color="#e2e8f0")
        ax.legend(loc="upper left", fontsize=8.6, frameon=False)
        ax.set_title(c["title"], fontsize=12.5, weight="bold")
        pairs = "\n".join(f"{k} {NOW_PIV[i]:%m-%d} 对 {c['hist'][i]:%Y-%m-%d}" for i, k in enumerate(NUM))
        ax.text(0.985, 0.03, f"{pairs}\n时间×{kt:.2f}  幅度×{ka:.2f}\n全程误差 RMSE {R['rmse']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.8, color="#334155",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#cbd5e1"))
    axs[0].set_ylabel("相对本轮底的涨幅（%）")
    fig.suptitle(f"按高低点一一对齐：本轮 vs 2021 轮大底之后（截至 {AS_OF:%Y-%m-%d}）", fontsize=14, weight="bold")
    fig.text(0.99, 0.005, "黑点 = 本轮转折点，空心绿点 = 2021 轮对应的转折点，编号相同的配成一对。"
             "倍数由 4 对转折点最小二乘得出。④ 今天还在回调中，未确认是低点。虚线只是「如果照 2021 轮走」的参照。",
             ha="right", fontsize=8.3, color="#334155")
    fig.tight_layout()
    fig.savefig(PNG / "swing_matched_2021.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
