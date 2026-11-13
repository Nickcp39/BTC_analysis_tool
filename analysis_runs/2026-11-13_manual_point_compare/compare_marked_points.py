# -*- coding: utf-8 -*-
"""人工点位比对：三轮「垂直下跌线 → 底 → 跳涨第一根线 → 第一个小高峰 → 回调低点」。

用户 2026-11-13 标了本轮三个点：垂直下跌线 2026-06-04、跳涨第一根线 2026-08-15、第一个小高峰 2026-10-03。
前两轮的对应点是按同样定义给的候选（marked_points.csv 的 source=candidate），用户可以直接改 csv 里的日期后重跑。

候选规则（只用于给前两轮出候选，用户的点不受规则约束）：
- 垂直下跌线：底前 90 天内最陡的 5 日跌幅窗口，取结束日前 1 天
- 跳涨第一根线：底后第一个 5 日涨幅 ≥ 15% 的窗口起点
- 第一个小高峰：跳涨后第一个「之后回撤 ≥ 10%」的高点
- 回调低点：小高峰之后、回升到小高峰之前的最低收盘

AHR999 按「周期压缩」读（见用户 2026-11-09 的说明）：用 ln(AHR / 本轮底 AHR) 表示离底多远；
跨轮比较时，前一轮乘以每轮压缩系数 0.57（ln(顶AHR/底AHR) 4.58 → 2.68 → 1.51，每轮约 ×0.57）。

输出（本目录）：
- marked_points_table.csv / marked_points_numbers.txt   每个点的日期、价格、离底天数、涨跌、AHR；各段天数和涨跌；三轮比例
- png/marked_points_by_cycle.png    三轮各自的价格 + AHR，五条竖线
- png/marked_points_aligned.png     以本轮为时间轴，前两轮按点位分段对齐（价格原始 / 幅度对齐 / AHR 压缩）
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PNG = HERE / "png"
sys.path.insert(0, str(ROOT))
from signals.indicators import compute_ahr999  # noqa: E402

PRICE_CSV = ROOT / "data" / "btc_merged_daily.csv"
POINTS_CSV = HERE / "marked_points.csv"
AS_OF = pd.Timestamp("2026-11-12")      # 最后一个 FRED 收盘（csv 末行 11-13 是 Yahoo 盘中点）
KEYS = ["drop", "bottom", "jump", "peak1", "pull1"]
LABEL = {"drop": "垂直下跌线", "bottom": "底", "jump": "跳涨第一根线", "peak1": "第一个小高峰", "pull1": "回调低点"}
COLOR = {"drop": "#dc2626", "bottom": "#16a34a", "jump": "#2563eb", "peak1": "#d97706", "pull1": "#7c3aed"}
CYC_COLOR = {"2017": "#94a3b8", "2021": "#0d9488", "2025": "#0f172a"}
AHR_COMP = 0.57
T = pd.Timestamp


def setup() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def rule_candidates(s: pd.Series, bottom: pd.Timestamp) -> dict:
    """按规则算前两轮的候选点，用来和 csv 里的日期互相核对。"""
    pre = s.loc[bottom - pd.Timedelta(days=90): bottom]
    r5 = (pre / pre.shift(5) - 1).dropna()
    drop = r5.idxmin() - pd.Timedelta(days=1)
    post = s.loc[bottom:]
    g5 = (post / post.shift(5) - 1).dropna()
    jump = g5[g5 >= 0.15].index[0] - pd.Timedelta(days=5)
    after = s.loc[jump:]
    peak1 = pull1 = None
    run_max_d = after.index[0]
    for d, p in after.items():
        if p > after[run_max_d]:
            run_max_d = d
        if p <= after[run_max_d] * 0.90:
            peak1 = run_max_d
            seg = s.loc[peak1:]
            back = seg[seg >= s[peak1]]
            end = back.index[1] if len(back) > 1 else seg.index[-1]
            pull1 = s.loc[peak1:end].idxmin()
            break
    return {"drop": drop, "jump": jump, "peak1": peak1, "pull1": pull1}


def main() -> None:
    setup()
    PNG.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(PRICE_CSV, parse_dates=["date"]).set_index("date")["price"].sort_index().loc[:AS_OF]
    a = compute_ahr999(s.rename("price").reset_index()).set_index("date")["ahr999"]
    pts = pd.read_csv(POINTS_CSV, dtype={"cycle": str}, parse_dates=["date"])

    # 本轮回调低点留空 → 第一个小高峰之后至今的最低收盘
    for i, r in pts.iterrows():
        if r.key == "pull1" and pd.isna(r.date):
            pk = pts[(pts.cycle == r.cycle) & (pts.key == "peak1")].date.iloc[0]
            pts.loc[i, "date"] = s.loc[pk:].idxmin()

    lines: list[str] = []
    out = lines.append
    out(f"数据截至 {AS_OF.date()} 收盘 ${s.iloc[-1]:,.0f}（FRED）")
    out("")
    out("== 规则候选 vs 账本日期（前两轮）")
    for cyc in ["2017", "2021"]:
        b = pts[(pts.cycle == cyc) & (pts.key == "bottom")].date.iloc[0]
        rc = rule_candidates(s, b)
        out(f"{cyc} 轮：" + "；".join(f"{LABEL[k]} 规则 {rc[k].date()} / 账本 {pts[(pts.cycle == cyc) & (pts.key == k)].date.iloc[0].date()}" for k in rc))
    b25 = pts[(pts.cycle == "2025") & (pts.key == "bottom")].date.iloc[0]
    rc = rule_candidates(s, b25)
    out("2025 轮（只核对，用户标的点优先）：" + "；".join(f"{LABEL[k]} 规则 {rc[k].date() if rc[k] is not None else '—'}" for k in rc))

    # ---------- 表 ----------
    rows = []
    P = {}
    for cyc in ["2017", "2021", "2025"]:
        sub = pts[pts.cycle == cyc].set_index("key")
        b = sub.loc["bottom", "date"]
        pb, ab = float(s[b]), float(a[b])
        P[cyc] = {k: sub.loc[k, "date"] for k in KEYS}
        for k in KEYS:
            d = sub.loc[k, "date"]
            rows.append({"周期": cyc, "点": LABEL[k], "日期": d.date().isoformat(), "来源": sub.loc[k, "source"],
                         "价格": round(float(s[d])), "离底天数": (d - b).days, "相对底": round(float(s[d]) / pb - 1, 4),
                         "AHR999": round(float(a[d]), 3), "ln(AHR/底AHR)": round(float(np.log(a[d] / ab)), 3)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(HERE / "marked_points_table.csv", index=False, encoding="utf-8-sig")
    out("")
    out("== 各点")
    out(tbl.to_string(index=False))

    out("")
    out("== 各段天数与涨跌")
    seg_names = [("drop", "bottom"), ("bottom", "jump"), ("jump", "peak1"), ("peak1", "pull1"), ("bottom", "peak1")]
    seg = {}
    for cyc in ["2017", "2021", "2025"]:
        parts = []
        for x, y in seg_names:
            dd = (P[cyc][y] - P[cyc][x]).days
            mv = float(s[P[cyc][y]]) / float(s[P[cyc][x]]) - 1
            seg[(cyc, x, y)] = (dd, mv)
            parts.append(f"{LABEL[x]}→{LABEL[y]} {dd} 天 {mv:+.1%}")
        out(f"{cyc} 轮：" + "；".join(parts))
    out("")
    out("== 时间比（本轮 ÷ 2021 轮 / 2021 轮 ÷ 2017 轮）与对数幅度比")
    for x, y in seg_names:
        t25 = seg[("2025", x, y)][0] / seg[("2021", x, y)][0]
        t21 = seg[("2021", x, y)][0] / seg[("2017", x, y)][0]
        m = lambda c: np.log(1 + seg[(c, x, y)][1])
        out(f"{LABEL[x]}→{LABEL[y]}：时间 {t25:.2f} / {t21:.2f}；对数幅度 {m('2025') / m('2021'):.2f} / {m('2021') / m('2017'):.2f}")
    pk25 = P["2025"]["peak1"]
    out("")
    out(f"本轮第一个小高峰之后已过 {(AS_OF - pk25).days} 天，回撤 {s.iloc[-1] / s[pk25] - 1:+.1%}（至今最低 {P['2025']['pull1'].date()} ${s[P['2025']['pull1']]:,.0f}）")
    for cyc in ["2017", "2021"]:
        pk, pl = P[cyc]["peak1"], P[cyc]["pull1"]
        nxt = s.loc[pl:pl + pd.Timedelta(days=60)]
        out(f"{cyc} 轮：小高峰 {pk.date()} → 回调低点 {pl.date()} 用了 {(pl - pk).days} 天、{s[pl] / s[pk] - 1:+.1%}；"
            f"低点后 60 天内最高 {nxt.max() / s[pl] - 1:+.1%}（{nxt.idxmax().date()}）；"
            f"小高峰后第 {(AS_OF - pk25).days} 天是 {(pk + (AS_OF - pk25)).date()}，当时相对小高峰 {s[pk + (AS_OF - pk25)] / s[pk] - 1:+.1%}")
    out("")
    out("== AHR999（压缩读法）")
    for k in KEYS:
        v = {c: float(np.log(a[P[c][k]] / a[P[c]["bottom"]])) for c in P}
        out(f"{LABEL[k]}：ln(AHR/底AHR) 2017 {v['2017']:+.2f} / 2021 {v['2021']:+.2f} / 2025 {v['2025']:+.2f}；"
            f"按 ×{AHR_COMP} 每轮压缩后，2021 轮折到本轮 {v['2021'] * AHR_COMP:+.2f}，2017 轮折到本轮 {v['2017'] * AHR_COMP ** 2:+.2f}")
    ab25 = float(a[P["2025"]["bottom"]])
    out(f"本轮底 AHR {ab25:.3f}；今天 AHR {a.iloc[-1]:.3f}（ln {np.log(a.iloc[-1] / ab25):+.2f}）")
    (HERE / "marked_points_numbers.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---------- 图 1：三轮各自 ----------
    fig = plt.figure(figsize=(14, 15), dpi=160)
    outer = fig.add_gridspec(3, 1, hspace=0.32)
    XL = (-95, 245)
    for i, cyc in enumerate(["2017", "2021", "2025"]):
        b = P[cyc]["bottom"]
        lo, hi = b + pd.Timedelta(days=XL[0]), min(b + pd.Timedelta(days=XL[1]), AS_OF)
        x = s.loc[lo:hi]
        inner = outer[i].subgridspec(2, 1, height_ratios=[3, 1.1], hspace=0.06)
        ax = fig.add_subplot(inner[0])
        ax2 = fig.add_subplot(inner[1], sharex=ax)
        ax.semilogy((x.index - b).days, x.values, color=CYC_COLOR[cyc], lw=2.0)
        ax.set_ylim(x.min() * 0.80, x.max() * 1.22)
        ah = a.loc[lo:hi]
        ax2.plot((ah.index - b).days, ah.values, color="#0f766e", lw=1.3)
        ax2.axhline(float(a[b]), color="#16a34a", lw=0.8, ls=":")
        for k in KEYS:
            d = P[cyc][k]
            dx = (d - b).days
            for axx in (ax, ax2):
                axx.axvline(dx, color=COLOR[k], lw=1.4, ls="-" if k != "pull1" or cyc != "2025" else "--", alpha=0.85)
            up = k in ("peak1", "jump", "bottom")
            ax.annotate(f"{LABEL[k]}\n{d:%Y-%m-%d}\n${s[d]:,.0f}  AHR {a[d]:.2f}", (dx, s[d]), xytext=(4, 8 if up else -40),
                        textcoords="offset points", fontsize=7.6, color=COLOR[k])
        ax.set_title(f"{cyc} 轮（底 {b:%Y-%m-%d}）" + ("　回调低点 = 小高峰之后至今最低，还在进行" if cyc == "2025" else ""),
                     fontsize=11.5, weight="bold", loc="left")
        ax.set_xlim(*XL)
        ax.grid(True, color="#e2e8f0", which="both")
        ax2.grid(True, color="#e2e8f0")
        ax2.set_ylabel("AHR999", fontsize=8.5)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax2.set_xlabel("距本轮底的天数", fontsize=8.5)
    fig.suptitle("人工点位比对：垂直下跌线 → 底 → 跳涨第一根线 → 第一个小高峰 → 回调低点（三轮）", fontsize=14, weight="bold", y=0.93)
    fig.text(0.99, 0.06, "2025 轮的垂直下跌线 / 跳涨 / 小高峰是用户手标；2017、2021 轮是同定义的候选，可在 marked_points.csv 里改。每轮下方小图是 AHR999，点线 = 该轮底的 AHR。",
             ha="right", fontsize=8, color="#334155")
    fig.savefig(PNG / "marked_points_by_cycle.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 图 2：按点位分段对齐到本轮时间轴 ----------
    tgt = {k: (P["2025"][k] - P["2025"]["bottom"]).days for k in ["drop", "bottom", "jump", "peak1"]}

    def warp(cyc: str, days: np.ndarray) -> np.ndarray:
        src = np.array([(P[cyc][k] - P[cyc]["bottom"]).days for k in ["drop", "bottom", "jump", "peak1"]], float)
        dst = np.array([tgt[k] for k in ["drop", "bottom", "jump", "peak1"]], float)
        outv = np.interp(days, src, dst)
        outv = np.where(days > src[-1], dst[-1] + (days - src[-1]), outv)     # 小高峰之后按日历天 1:1
        outv = np.where(days < src[0], dst[0] + (days - src[0]), outv)
        return outv

    fig, axs = plt.subplots(3, 1, figsize=(13.5, 13), dpi=160, sharex=True, gridspec_kw={"height_ratios": [1.2, 1.2, 1]})
    now_x = (AS_OF - P["2025"]["bottom"]).days
    amp = {}
    for cyc in ["2017", "2021", "2025"]:
        b = P[cyc]["bottom"]
        x = s.loc[P[cyc]["drop"] - pd.Timedelta(days=40): b + pd.Timedelta(days=260)].loc[:AS_OF]
        d = (x.index - b).days.values.astype(float)
        wx = d if cyc == "2025" else warp(cyc, d)
        lg = np.log(x.values / s[b])
        lw = 3.0 if cyc == "2025" else 1.7
        axs[0].plot(wx, np.exp(lg), color=CYC_COLOR[cyc], lw=lw, label=f"{cyc} 轮" + ("（本轮）" if cyc == "2025" else "（按点位分段对齐时间）"))
        amp[cyc] = np.log(s[P["2025"]["peak1"]] / s[P["2025"]["bottom"]]) / np.log(s[P[cyc]["peak1"]] / s[b])
        lg2 = lg if cyc == "2025" else lg * amp[cyc]
        axs[1].plot(wx, np.exp(lg2), color=CYC_COLOR[cyc], lw=lw,
                    label=f"{cyc} 轮" + ("" if cyc == "2025" else f"（对数幅度 ×{amp[cyc]:.2f}，让第一个小高峰等高）"))
        ah = a.loc[x.index]
        la = np.log(ah.values / a[b])
        k = {"2017": 2, "2021": 1, "2025": 0}[cyc]
        axs[2].plot(wx, la * AHR_COMP ** k, color=CYC_COLOR[cyc], lw=lw,
                    label=f"{cyc} 轮" + ("" if k == 0 else f"（×{AHR_COMP}^{k} 压缩）"))
        if cyc != "2025":
            # 本轮今天落在该轮的哪一天
            src_days = (s.loc[b - pd.Timedelta(days=200): b + pd.Timedelta(days=400)].index - b).days.values.astype(float)
            m = src_days[np.argmin(np.abs(warp(cyc, src_days) - now_x))]
            d_eq = b + pd.Timedelta(days=int(m))
            axs[0].scatter([now_x], [s[d_eq] / s[b]], color=CYC_COLOR[cyc], s=40, zorder=6, edgecolor="white")
            axs[0].annotate(f"本轮今天 对应 {cyc} 轮 {d_eq:%Y-%m-%d}", (now_x, s[d_eq] / s[b]), xytext=(8, 4),
                            textcoords="offset points", fontsize=8.2, color=CYC_COLOR[cyc])
    for kk, dx in tgt.items():
        for ax in axs:
            ax.axvline(dx, color=COLOR[kk], lw=1.2, alpha=0.8)
        axs[0].text(dx, 0.02, f" {LABEL[kk]}", color=COLOR[kk], fontsize=8.5, va="bottom",
                    transform=blended_transform_factory(axs[0].transData, axs[0].transAxes))
    for ax in axs:
        ax.axvline(now_x, color="#7c3aed", ls=":", lw=1.3)
        ax.grid(True, color="#e2e8f0")
        ax.legend(loc="upper left", fontsize=8.6)
    pct = mticker.FuncFormatter(lambda v, _: f"{(v - 1) * 100:+.0f}%")
    for ax in axs[:2]:
        ax.set_yscale("log")
        ax.axhline(1, color="#475569", lw=0.8)
        ax.yaxis.set_major_locator(mticker.FixedLocator([0.8, 1, 1.25, 1.5, 2, 2.5, 3, 4]))
        ax.yaxis.set_major_formatter(pct)
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    axs[2].axhline(0, color="#475569", lw=0.8)
    axs[0].set_ylabel("相对各自底的涨跌（对数刻度）")
    axs[1].set_ylabel("幅度对齐后（对数刻度）")
    axs[2].set_ylabel("ln(AHR / 底 AHR)")
    axs[0].set_title("① 时间按四个点位分段对齐（小高峰之后按日历天 1:1）；价格是原始涨跌", fontsize=11.5, weight="bold", loc="left")
    axs[1].set_title("② 再把幅度对齐：前两轮的对数涨幅缩放到第一个小高峰和本轮一样高", fontsize=11.5, weight="bold", loc="left")
    axs[2].set_title(f"③ AHR999 离底多远（压缩读法：每往前一轮乘 {AHR_COMP}）", fontsize=11.5, weight="bold", loc="left")
    axs[2].set_xlabel(f"本轮时间轴：距 2026-06-30 底的天数（紫色点线 = 今天 {AS_OF:%m-%d}）")
    fig.suptitle("按人工点位对齐三轮（以本轮为准）", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(PNG / "marked_points_aligned.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
