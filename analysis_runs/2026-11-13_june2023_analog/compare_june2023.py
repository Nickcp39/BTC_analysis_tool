# -*- coding: utf-8 -*-
"""这次下跌（2026-10-04 高点 → 今天）像不像 2023 年 6 月那次？（用户 2026-11-13 的问题）

2023 年 6 月那次 = 2021 轮底后第二次回调：2023-04-14 高点 → 2023-06-14 收盘低点（6-17 已开始反弹）。
同时对照 2021 轮另外两次（硅谷银行 2023-02-20 → 03-10、2023-07-13 → 09-11）和 2017 轮 2019-02 那次。

比四样：
1. 从高点跌了多少、跌了几天、最大单日跌幅；
2. 离本轮底还有多远（垫子有多厚）；
3. 回吐了「底后涨幅」的多少（对数口径；16 次历史回撤 + 本轮这次，来自 ../2026-11-13_drawdown_context/pullbacks_table.csv）；
4. AHR 按周期压缩后，那几次回调低点换成今天的价格（来自 ../2026-11-13_ahr_compressed/ahr_compressed_ladder.csv，按 11-12 收盘算）；
外加时间位置：三个拉伸拟合（../2026-11-13_stretch_compare/stretch_params.csv）把今天对到 2021 轮哪一天、把 2023-06-14 对到本轮哪一天。

输出：analog_numbers.txt、png/june2023_analog.png
用法：python compare_june2023.py [今天盘中最低价]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PNG = HERE / "png"
RUNS = ROOT / "analysis_runs"
T = pd.Timestamp

TOP21, TOP25 = T("2021-11-08"), T("2025-08-12")
# (标签, 高点, 低点, 本轮底, 颜色, 线宽)
CASES = [
    ("本轮 2026-10-04 → 今天", T("2026-10-04"), None, T("2026-06-30"), "#d9480f", 2.8),
    ("2023 年 6 月那次（04-14 → 06-14）", T("2023-04-14"), T("2023-06-14"), T("2022-11-21"), "#1c7ed6", 2.4),
    ("硅谷银行（2023-02-20 → 03-10）", T("2023-02-20"), T("2023-03-10"), T("2022-11-21"), "#74c0fc", 1.4),
    ("2023-07-13 → 09-11", T("2023-07-13"), T("2023-09-11"), T("2022-11-21"), "#4dabf7", 1.4),
    ("2017 轮 2018-12-20 → 2019-02-07", T("2018-12-20"), T("2019-02-07"), T("2018-12-15"), "#868e96", 1.4),
]
AFTER = 30  # 低点之后再画 30 天（虚线），只作参照


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    PNG.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(ROOT / "data" / "btc_merged_daily.csv", parse_dates=["date"]).set_index("date")["price"].sort_index()
    now_d, now_px = s.index.max(), float(s.iloc[-1])
    intraday_low = float(sys.argv[1]) if len(sys.argv) > 1 else None
    out: list[str] = []
    P = out.append
    P(f"数据：btc_merged_daily.csv 末行 {now_d.date()} ${now_px:,.0f}（Yahoo 盘中点）"
      + (f"；今天盘中最低 ${intraday_low:,.0f}（Yahoo 小时线）" if intraday_low else ""))

    # ---------- 1 & 2：深度、时长、单日、垫子 ----------
    P("")
    P("== 1. 深度、时长、最大单日跌幅、离底多远（收盘口径）")
    paths = {}
    for lab, hi, lo, bot, *_ in CASES:
        cur = lo is None
        lo_d = s.loc[hi:].idxmin() if cur else lo
        end = now_d if cur else lo + pd.Timedelta(days=AFTER)
        seg = s.loc[hi:end]
        dd = float(s[lo_d] / s[hi] - 1)
        r = s.loc[hi:lo_d].pct_change().dropna()
        worst = r.nsmallest(2)
        run_up = float(s[hi] / s[bot] - 1)
        cush = float(s[lo_d] / s[bot] - 1)
        give = float(-np.log(1 + dd) / np.log(1 + run_up))
        day40 = s.get(hi + pd.Timedelta(days=40))
        P(f"{lab}：高点 ${s[hi]:,.0f}（比底 {run_up:+.0%}）→ 低点 {lo_d.date()} ${s[lo_d]:,.0f}：{dd:+.1%}，{(lo_d - hi).days} 天；"
          f"低点比底 {cush:+.0%}；回吐底后涨幅 {give:.0%}；最大单日 "
          + "、".join(f"{d:%m-%d} {v:+.1%}" for d, v in worst.items())
          + (f"；第 40 天 {day40 / s[hi] - 1:+.1%}" if day40 is not None and not cur else ""))
        paths[lab] = dict(seg=seg, hi=hi, lo=lo_d, bot=bot, cur=cur)
    if intraday_low:
        hi, bot = T("2026-10-04"), T("2026-06-30")
        dd = intraday_low / s[hi] - 1
        P(f"本轮按今天盘中最低 ${intraday_low:,.0f}：{dd:+.1%}；比底 {intraday_low / s[bot] - 1:+.0%}；"
          f"回吐底后涨幅 {-np.log(1 + dd) / np.log(s[hi] / s[bot]):.0%}")

    # ---------- 3：16 次历史回撤的回吐比例 ----------
    tb = pd.read_csv(RUNS / "2026-11-13_drawdown_context" / "pullbacks_table.csv", encoding="utf-8-sig",
                     parse_dates=["高点", "低点"])
    tb["回吐"] = -np.log(1 + tb["深度"]) / np.log(1 + tb["高点相对底"])
    cur_row = tb["进行中"].astype(str).str.lower().eq("true")
    # 本轮那一行按今天收盘重算（table 是按 11-13 早些时候的盘中点算的）
    hi, bot = T("2026-10-04"), T("2026-06-30")
    tb.loc[cur_row, "深度"] = now_px / s[hi] - 1
    tb.loc[cur_row, "回吐"] = -np.log(now_px / s[hi]) / np.log(s[hi] / s[bot])
    P("")
    P("== 2. 回吐了底后涨幅的多少（对数口径；高点之前从底涨上来的那一段，回调吃掉了几成）")
    for _, r in tb.sort_values("回吐", ascending=False).iterrows():
        tag = "（本轮，进行中）" if r["进行中"] in (True, "True") else ""
        P(f"  {r['周期']} {r['高点']:%Y-%m-%d} → {r['低点']:%Y-%m-%d}：深度 {r['深度']:+.1%}，高点比底 {r['高点相对底']:+.0%}，"
          f"回吐 {r['回吐']:.0%}{tag}")
    hist = tb[~cur_row]
    n_more = int((hist["回吐"] >= tb.loc[cur_row, "回吐"].iloc[0]).sum())
    P(f"16 次历史回撤里回吐比例不低于本轮这次的：{n_more} 次；它们都没有跌破各自的底（一年内）")

    # ---------- 4：AHR 压缩水位换成今天的价格 ----------
    lad = pd.read_csv(RUNS / "2026-11-13_ahr_compressed" / "ahr_compressed_ladder.csv", encoding="utf-8-sig")
    P("")
    P("== 3. AHR 按周期压缩：2021 轮那几次回调低点的水位，换成今天的价格（ahr_compressed，按 11-12 收盘的均线算）")
    for _, r in lad[lad["水位"] != "今天"].iterrows():
        P(f"  {r['水位']}：AHR {r['AHR']:.3f} → ${r['今天对应价格']:,.0f}（相对今天盘中 {r['今天对应价格'] / now_px - 1:+.1%}）")
    P(f"  对照：11-12 收盘 $70,822 时 AHR 0.374；今天盘中 ${now_px:,.0f}")

    # ---------- 5：时间位置 ----------
    sp = pd.read_csv(RUNS / "2026-11-13_stretch_compare" / "stretch_params.csv", encoding="utf-8-sig")
    P("")
    P("== 4. 时间位置（三个点位拟合的拉伸窗口）")
    for _, r in sp[sp["编号"].isin(["A", "B", "C"])].iterrows():
        anchor21, anchor25 = (TOP21, TOP25) if r["编号"] == "A" else (T("2022-11-21"), T("2026-06-30"))
        x = (T("2023-06-14") - anchor21).days
        d = anchor25 + pd.Timedelta(days=round(r["点位拟合_time"] * x + r["点位拟合_shift"]))
        P(f"  {r['编号']} {r['窗口']}：今天 ≈ 2021 轮 {r['本轮最新对应历史']}；2023-06-14 低点 ≈ 本轮 {d.date()}")

    (HERE / "analog_numbers.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    draw(s, paths, tb, cur_row, lad, now_px, intraday_low)


def draw(s, paths, tb, cur_row, lad, now_px, intraday_low) -> None:
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.32, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    pct = mticker.FuncFormatter(lambda v, _: f"{(np.exp(v) - 1) * 100:+.0f}%")

    for lab, _, _, _, col, lw in CASES:
        p = paths[lab]
        seg, hi, lo = p["seg"], p["hi"], p["lo"]
        x = (seg.index - hi).days
        y1 = np.log(seg.values / s[hi])
        y2 = np.log(seg.values / s[p["bot"]])
        solid = seg.index <= lo if not p["cur"] else np.ones(len(seg), bool)
        for ax, y in [(ax1, y1), (ax2, y2)]:
            ax.plot(x[solid], y[solid], color=col, lw=lw, label=lab if ax is ax1 else None, zorder=5 if p["cur"] else 3)
            if not p["cur"]:
                ax.plot(x[~solid | (seg.index == lo)], y[~solid | (seg.index == lo)], color=col, lw=lw * 0.7, ls="--", alpha=0.8)
            xl, yl = (lo - hi).days, (y1 if ax is ax1 else y2)[list(seg.index).index(lo)]
            ax.scatter([xl], [yl], color=col, s=40 if lw > 2 else 22, zorder=6)
        if lw > 2:
            lo_px = s[lo]
            ax1.annotate(f"{lo:%m-%d} {lo_px / s[hi] - 1:+.1%}\n{(lo - hi).days} 天", ((lo - hi).days, np.log(lo_px / s[hi])),
                         xytext=(-58, -30) if p["cur"] else (6, -26), textcoords="offset points", color=col,
                         fontsize=10, fontweight="bold")
            ax2.annotate(f"比底 {lo_px / s[p['bot']] - 1:+.0%}", ((lo - hi).days, np.log(lo_px / s[p["bot"]])),
                         xytext=(6, -16), textcoords="offset points", color=col, fontsize=10, fontweight="bold")
    if intraday_low:
        hi = T("2026-10-04")
        xd = (s.index.max() - hi).days
        ax1.scatter([xd], [np.log(intraday_low / s[hi])], marker="v", color="#d9480f", s=45, zorder=7)
        ax1.annotate(f"今天盘中最低 \\${intraday_low:,.0f}（{intraday_low / s[hi] - 1:+.1%}）", (xd, np.log(intraday_low / s[hi])),
                     xytext=(8, -4), textcoords="offset points", color="#d9480f", fontsize=9)
    ax1.axhline(0, color="#adb5bd", lw=0.8)
    ax1.yaxis.set_major_formatter(pct)
    ax1.yaxis.set_major_locator(mticker.FixedLocator(np.log1p(np.arange(-0.25, 0.2, 0.05))))
    ax1.set_ylim(np.log(0.75), np.log(1.18))
    ax1.set_xlabel("离各自高点的天数")
    ax1.set_ylabel("相对高点")
    ax1.set_title("① 从高点跌了多少：实线到低点，虚线是低点之后 30 天（只作参照）", loc="left", fontsize=12)
    ax1.legend(loc="lower left", fontsize=10, frameon=False)
    ax1.grid(alpha=0.25)

    ax2.yaxis.set_major_formatter(pct)
    ax2.yaxis.set_major_locator(mticker.FixedLocator(np.log1p(np.array([0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]))))
    ax2.axhline(0, color="#495057", lw=1.0)
    ax2.text(1, 0.012, "各自的周期底", fontsize=9, color="#495057")
    ax2.set_xlabel("离各自高点的天数")
    ax2.set_ylabel("相对各自周期底")
    ax2.set_title("② 垫子有多厚：同一条路径，换成「离本轮底多远」", loc="left", fontsize=12)
    ax2.grid(alpha=0.25)

    # ③ 回吐比例
    t = tb.sort_values("回吐").reset_index(drop=True)
    cols = []
    for _, r in t.iterrows():
        if r["进行中"] in (True, "True"):
            cols.append("#d9480f")
        elif r["高点"] == T("2023-04-14"):
            cols.append("#1c7ed6")
        elif r["周期"] in ("2017 轮", "2021 轮"):
            cols.append("#a5d8ff")
        else:
            cols.append("#dee2e6")
    ax3.barh(range(len(t)), t["回吐"] * 100, color=cols)
    ax3.set_yticks(range(len(t)))
    ax3.set_yticklabels([f"{'本轮' if '本轮' in r['周期'] else r['周期'].replace(' 轮', '')} {r['高点']:%Y-%m}  {r['深度']:+.0%}" for _, r in t.iterrows()], fontsize=8.5)
    for i, v in enumerate(t["回吐"]):
        ax3.text(v * 100 + 1, i, f"{v:.0%}", va="center", fontsize=8)
    ax3.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax3.set_xlabel("回调吃掉了「底 → 高点」那段涨幅的几成（对数）")
    ax3.set_title("③ 回吐比例：16 次历史回撤 + 本轮（深蓝 = 2023-06，浅蓝 = 成熟两轮，灰 = 早期两轮）", loc="left", fontsize=10.5)
    ax3.grid(axis="x", alpha=0.25)

    # 右上角小框：AHR 压缩水位
    names = {"第二次": "2023-06 那次低点", "硅谷银行": "2023-03 硅谷银行低点", "第三次": "2023-09 低点", "本轮底": "本轮底的 AHR"}
    lines = ["AHR 按周期压缩，换成今天的价格（三种压缩口径）："]
    for k, nm in names.items():
        v = lad.loc[lad["水位"].str.contains(k), "今天对应价格"]
        rng = f"${v.min():,.0f}" if len(v) == 1 else f"${v.min():,.0f} – ${v.max():,.0f}"
        lines.append(f"  {nm}  ≈ {rng}")
    lines.append(f"  今天盘中  ${now_px:,.0f}")
    ax1.text(0.995, 0.97, "\n".join(lines).replace("$", r"\$"), transform=ax1.transAxes, ha="right", va="top", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#ced4da"))

    fig.suptitle(f"这次下跌 vs 2023 年 6 月那次（价格到 {s.index.max():%Y-%m-%d} 盘中）", fontsize=15, fontweight="bold", y=0.985)
    fig.savefig(PNG / "june2023_analog.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
