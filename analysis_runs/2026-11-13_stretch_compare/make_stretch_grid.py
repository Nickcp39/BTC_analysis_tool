# -*- coding: utf-8 -*-
"""多周期拉伸对比图（和 10-01 / 11-06 报告 §4 同一种图，按 11-13 的点位重新拟合）。

做法和以前的 peak_structure 图一样：历史那一轮的曲线做三件事再叠到本轮上——
  横轴：x = 历史天数 × time + shift（时间拉伸 + 平移）
  纵轴：y = ln(价格 / 锚点价) × amp（幅度拉伸）
以前 amp / time / shift 是用户在工作台手调的；这次用点位「近似拟合」：
  ① 点位拟合（实线，主图）：用窗口里的配对点（用户给的三对 + 报告的顶/底 + 底后标点）
     time、shift = 本轮天数对历史天数的最小二乘直线；amp = 两轮对数涨跌在配对点上过原点的最小二乘比
  ② 整条曲线拟合（细点线，对照）：time ∈ [0.6, 1.4]、shift ∈ [-90, 90] 网格搜索，amp 用最小二乘，取 RMSE 最小
历史曲线越过本轮最新一天的部分画成虚线，只是「照这条拉伸后的路走」的参照，不是预测。
不做 2017 → 2025 的直接对比（项目约定：2021 ↔ 2025 为主，2017 ↔ 2021 作桥）。

输入：data/btc_merged_daily.csv（到 2026-11-13，末行是 Yahoo 盘中点）；
      ../2026-11-13_manual_point_compare/user_pairs.csv、marked_points.csv
输出（本目录）：png/stretch_grid.png、png/stretch_<编号>.png（每个窗口一张大图）、stretch_params.csv、stretch_numbers.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PNG = HERE / "png"
PTS_DIR = ROOT / "analysis_runs" / "2026-11-13_manual_point_compare"
T = pd.Timestamp


def load_price() -> pd.Series:
    s = pd.read_csv(ROOT / "data" / "btc_merged_daily.csv", parse_dates=["date"]).set_index("date")["price"].sort_index()
    return s.asfreq("D").ffill()


def pairs_2021_2025() -> pd.DataFrame:
    p = pd.read_csv(PTS_DIR / "user_pairs.csv", parse_dates=["date_2021", "date_2025"])
    return p.rename(columns={"date_2021": "d_hist", "date_2025": "d_cur"})[["pair", "d_hist", "d_cur", "source"]]


def pairs_2017_2021() -> pd.DataFrame:
    m = pd.read_csv(PTS_DIR / "marked_points.csv", dtype={"cycle": str}, parse_dates=["date"])
    lab = {"drop": "垂直下跌线", "bottom": "底", "jump": "跳涨", "peak1": "小高峰", "pull1": "回调低点"}
    rows = [dict(pair="顶", d_hist=T("2017-12-16"), d_cur=T("2021-11-08"), source="anchor")]
    for k, l in lab.items():
        h = m[(m.cycle == "2017") & (m.key == k)].date.iloc[0]
        c = m[(m.cycle == "2021") & (m.key == k)].date.iloc[0]
        rows.append(dict(pair=l, d_hist=h, d_cur=c, source="candidate"))
    return pd.DataFrame(rows).sort_values("d_hist").reset_index(drop=True)


# (编号, 标题, 历史轮, 本轮, 锚点名, 窗口相对锚点的天数 [lo, hi]（本轮天数；hi=None 表示到最新）, 延伸天数)
WINDOWS = [
    ("A", "2021 → 2025 · 顶后至今（熊市 + 底后）", "2021", "2025", "顶", (0, None), 120),
    ("B", "2021 → 2025 · 底前一年 + 底后至今", "2021", "2025", "底", (-365, None), 120),
    ("C", "2021 → 2025 · 底后至今（放大）", "2021", "2025", "底", (0, None), 120),
    ("D", "2017 → 2021 · 顶后 500 天（桥接校验）", "2017", "2021", "顶", (0, 500), 0),
    ("E", "2017 → 2021 · 底前后各半年（桥接校验）", "2017", "2021", "底", (-183, 183), 0),
    ("F", "2017 → 2021 · 底后一年（桥接校验）", "2017", "2021", "底", (0, 365), 0),
]
COL = {"2017": "#7c3aed", "2021": "#2563eb", "2025": "#0f766e"}


def log_norm(s: pd.Series, anchor: pd.Timestamp, lo: int, hi: int) -> pd.Series:
    seg = s.loc[anchor + pd.Timedelta(days=lo): anchor + pd.Timedelta(days=hi)]
    return pd.Series(np.log(seg.values / s[anchor]), index=(seg.index - anchor).days)


def fit_points(pr: pd.DataFrame, s: pd.Series, ah: pd.Timestamp, ac: pd.Timestamp) -> tuple[float, float, float]:
    xh = np.array([(d - ah).days for d in pr.d_hist], float)
    xc = np.array([(d - ac).days for d in pr.d_cur], float)
    t, sh = np.polyfit(xh, xc, 1)
    yh = np.array([np.log(s[d] / s[ah]) for d in pr.d_hist])
    yc = np.array([np.log(s[d] / s[ac]) for d in pr.d_cur])
    keep = np.abs(yh) > 1e-9
    amp = float((yc[keep] * yh[keep]).sum() / (yh[keep] ** 2).sum()) if keep.any() else 1.0
    return float(t), float(sh), amp


def curve_rmse(cur: pd.Series, hist: pd.Series, t: float, sh: float, amp: float | None = None) -> tuple[float, float]:
    xs = (cur.index.values - sh) / t                       # 本轮第 x 天对应历史第几天
    ok = (xs >= hist.index.min()) & (xs <= hist.index.max())
    if ok.sum() < 20:
        return np.inf, np.nan
    h = np.interp(xs[ok], hist.index.values, hist.values)
    y = cur.values[ok]
    a = float((y * h).sum() / (h * h).sum()) if amp is None else amp
    a = float(np.clip(a, 0.15, 1.5))
    return float(np.sqrt(np.mean((y - a * h) ** 2))), a


def fit_curve(cur: pd.Series, hist: pd.Series) -> tuple[float, float, float, float]:
    best = (np.inf, 1.0, 0.0, 1.0)
    for t in np.arange(0.60, 1.401, 0.01):
        for sh in range(-90, 91, 2):
            r, a = curve_rmse(cur, hist, t, sh)
            if r < best[0]:
                best = (r, t, sh, a)
    return best[1], float(best[2]), best[3], best[0]


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    PNG.mkdir(parents=True, exist_ok=True)
    s = load_price()
    last = s.index.max()
    P = {("2021", "2025"): pairs_2021_2025(), ("2017", "2021"): pairs_2017_2021()}
    lines, rows, panels = [f"价格数据到 {last.date()}（${s[last]:,.0f}，末行为 Yahoo 盘中点）"], [], []

    for code, title, hc, cc, anc, (lo, hi), ext in WINDOWS:
        pr = P[(hc, cc)]
        ah = pr.loc[pr.pair == anc, "d_hist"].iloc[0]
        ac = pr.loc[pr.pair == anc, "d_cur"].iloc[0]
        hi_c = (last - ac).days if hi is None else hi
        cur = log_norm(s, ac, lo, hi_c)
        # 窗口里的配对点（本轮日期落在窗口内）
        inwin = pr[[(lo <= (d - ac).days <= hi_c) for d in pr.d_cur]]
        t1, s1, a1 = fit_points(inwin, s, ah, ac)
        hist = log_norm(s, ah, int((lo - 120 - s1) / max(t1, 0.3)) - 30, int((hi_c + ext + 120 - s1) / max(t1, 0.3)) + 30)
        r1, _ = curve_rmse(cur, hist, t1, s1, a1)
        t2, s2, a2, r2 = fit_curve(cur, hist)
        # 今天在历史哪一天、照拉伸后的历史路径往后走
        now_c = hi_c
        now_h = ah + pd.Timedelta(days=int(round((now_c - s1) / t1)))
        fut = None
        if ext:
            xs = np.arange(now_c, now_c + ext + 1)
            hx = (xs - s1) / t1
            hv = np.interp(hx, hist.index.values, hist.values) * a1
            px = s[ac] * np.exp(hv)
            fut = dict(lo=float(px.min()), lo_d=int(xs[px.argmin()] - now_c), hi=float(px.max()), hi_d=int(xs[px.argmax()] - now_c),
                       end=float(px[-1]), now_ref=float(px[0]), gap=float(s[last] / px[0] - 1))
        rows.append(dict(编号=code, 窗口=title, 锚点=f"{anc} {ah.date()} ↔ {ac.date()}", 配对点数=len(inwin),
                         点位拟合_time=round(t1, 3), 点位拟合_shift=round(s1, 1), 点位拟合_amp=round(a1, 3), 点位拟合_RMSE=round(r1, 4),
                         曲线拟合_time=round(t2, 3), 曲线拟合_shift=round(s2, 1), 曲线拟合_amp=round(a2, 3), 曲线拟合_RMSE=round(r2, 4),
                         本轮最新对应历史=now_h.date().isoformat() if hi is None else ""))
        lines.append("")
        lines.append(f"== {code}. {title}（锚点 {anc}：{ah.date()} ↔ {ac.date()}；配对点 {len(inwin)} 个：{'、'.join(inwin.pair)}）")
        lines.append(f"点位拟合：time {t1:.3f}，shift {s1:+.1f} 天，amp {a1:.3f}，整段 RMSE {r1:.4f}")
        lines.append(f"整条曲线拟合（对照）：time {t2:.2f}，shift {s2:+.0f} 天，amp {a2:.3f}，RMSE {r2:.4f}")
        for _, r in inwin.iterrows():
            xh = (r.d_hist - ah).days * t1 + s1
            xc = (r.d_cur - ac).days
            yh = np.log(s[r.d_hist] / s[ah]) * a1
            yc = np.log(s[r.d_cur] / s[ac])
            lines.append(f"  {r.pair}：{r.d_hist.date()} → 拉伸后第 {xh:.0f} 天（本轮实际第 {xc} 天，差 {xh - xc:+.0f}）；"
                         f"高度 {np.exp(yh) - 1:+.1%}（本轮实际 {np.exp(yc) - 1:+.1%}）")
        if hi is None:
            lines.append(f"本轮最新（{last.date()}，第 {now_c} 天）对应 {hc} 轮 {now_h.date()}")
        if fut:
            lines.append(f"拉伸后的 {hc} 轮在本轮最新这一天约 ${fut['now_ref']:,.0f}；本轮实际 ${s[last]:,.0f}，比它 {fut['gap']:+.1%}")
            lines.append(f"照拉伸后的 {hc} 轮往后 {ext} 天（参照，不是预测）：最低约 ${fut['lo']:,.0f}（+{fut['lo_d']} 天），"
                         f"最高约 ${fut['hi']:,.0f}（+{fut['hi_d']} 天），第 {ext} 天约 ${fut['end']:,.0f}")
        panels.append(dict(code=code, title=title, hc=hc, cc=cc, anc=anc, ah=ah, ac=ac, lo=lo, hi_c=hi_c, ext=ext, cur=cur,
                           hist=hist, fits=((t1, s1, a1, r1), (t2, s2, a2, r2)), inwin=inwin, now_h=now_h if hi is None else None))
    pd.DataFrame(rows).to_csv(HERE / "stretch_params.csv", index=False, encoding="utf-8-sig")
    (HERE / "stretch_numbers.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    def draw(ax, p, big=False):
        (t1, s1, a1, r1), (t2, s2, a2, r2) = p["fits"]
        cur, hist = p["cur"], p["hist"]
        hx1 = hist.index.values * t1 + s1
        end_x = p["hi_c"] + p["ext"]
        m_in = (hx1 >= p["lo"]) & (hx1 <= p["hi_c"])
        m_ext = (hx1 > p["hi_c"]) & (hx1 <= end_x)
        y_conv = lambda v: v      # 纵轴画 ln(价格/锚点价)，刻度标成百分比（和以前报告的图一样）
        ax.plot(hx1[m_in], y_conv(hist.values[m_in] * a1), color=COL[p["hc"]], lw=1.7 if not big else 2.0,
                label=f"{p['hc']} 轮（点位拟合：time {t1:.2f} / shift {s1:+.0f}d / amp {a1:.2f}）")
        if m_ext.any():
            ax.plot(hx1[m_ext], y_conv(hist.values[m_ext] * a1), color=COL[p["hc"]], lw=1.5, ls="--", alpha=0.85,
                    label=f"{p['hc']} 轮往后（参照，不是预测）")
        hx2 = hist.index.values * t2 + s2
        m2 = (hx2 >= p["lo"]) & (hx2 <= p["hi_c"])
        ax.plot(hx2[m2], y_conv(hist.values[m2] * a2), color=COL[p["hc"]], lw=0.9, ls=":", alpha=0.75,
                label=f"{p['hc']} 轮（整条曲线拟合：{t2:.2f} / {s2:+.0f}d / {a2:.2f}）")
        ax.plot(cur.index, y_conv(cur.values), color=COL[p["cc"]], lw=2.2 if not big else 2.6, label=f"{p['cc']} 轮")
        for _, r in p["inwin"].iterrows():
            xc = (r.d_cur - p["ac"]).days
            yc = y_conv(np.log(s[r.d_cur] / s[p["ac"]]))
            xh = (r.d_hist - p["ah"]).days * t1 + s1
            yh = y_conv(np.log(s[r.d_hist] / s[p["ah"]]) * a1)
            mk = "o" if r.source == "user" else ("s" if r.source == "anchor" else "^")
            ax.scatter([xc], [yc], s=46 if big else 30, color=COL[p["cc"]], marker=mk, edgecolor="white", zorder=6)
            ax.scatter([xh], [yh], s=46 if big else 30, color=COL[p["hc"]], marker=mk, edgecolor="white", zorder=6)
            ax.plot([xh, xc], [yh, yc], color="#64748b", lw=0.8, alpha=0.7)
            if big:
                ax.annotate(r.pair, (xc, yc), xytext=(4, 6), textcoords="offset points", fontsize=8.5, color="#0f172a")
        ax.axvline(0, color="#111827", lw=0.8, alpha=0.45)
        ax.axhline(0, color="#111827", lw=0.8, alpha=0.3)
        if p["now_h"] is not None:
            ax.axvline(p["hi_c"], color="#dc2626", lw=1.0, ls=":")
            ax.text(p["hi_c"], 0.97, f" 最新 {last:%m-%d}\n 对应 {p['hc']} 轮 {p['now_h']:%Y-%m-%d}", va="top",
                    transform=blended_transform_factory(ax.transData, ax.transAxes),
                    fontsize=7.8 if not big else 9, color="#dc2626")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{(np.exp(v) - 1) * 100:+.0f}%"))
        ax.grid(True, color="#cbd5e1", alpha=0.5, lw=0.6)
        ax.set_title(f"{p['code']}. {p['title']}", fontsize=10.5 if not big else 13, weight="bold", loc="left")
        ax.set_xlabel(f"距{p['anc']}的天数（{p['cc']} 轮：{p['ac']:%Y-%m-%d}）", fontsize=8.5 if not big else 10)
        ax.set_ylabel(f"相对{p['anc']}的涨跌（对数刻度）", fontsize=8.5 if not big else 10)
        ax.legend(loc="best", fontsize=7 if not big else 8.8, framealpha=0.85)
        ax.text(0.99, 0.02, f"点位拟合 RMSE {r1:.3f} ｜ 曲线拟合 RMSE {r2:.3f}", transform=ax.transAxes, ha="right",
                fontsize=7.5 if not big else 9, color="#334155")

    fig, axes = plt.subplots(3, 2, figsize=(15, 16.5), dpi=170, constrained_layout=True)
    for ax, p in zip(axes.ravel(), panels):
        ax.set_facecolor("#f0fdfa" if p["hc"] == "2021" else "#fff7ed")
        draw(ax, p)
    fig.suptitle("多周期拉伸对比图（2026-11-13）：历史一轮按点位拟合 time / shift / amp 后叠到下一轮\n"
                 "实线 = 点位拟合；细点线 = 整条曲线拟合（对照）；虚线 = 照拉伸后的历史往后走（参照，不是预测）；"
                 "圆点 = 用户配对，方块 = 报告锚点，三角 = 底后标点", fontsize=13, weight="bold")
    fig.savefig(PNG / "stretch_grid.png", bbox_inches="tight")
    plt.close(fig)
    for p in panels:
        fig, ax = plt.subplots(figsize=(13, 6.8), dpi=170)
        ax.set_facecolor("#f0fdfa" if p["hc"] == "2021" else "#fff7ed")
        draw(ax, p, big=True)
        fig.tight_layout()
        fig.savefig(PNG / f"stretch_{p['code']}.png", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
