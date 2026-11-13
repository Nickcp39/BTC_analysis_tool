# -*- coding: utf-8 -*-
"""这次下跌放进五轮里看：每轮「底后第一年」出现过的所有回撤（从阶段高点算 ≥10%），深度、时长、发生在底后第几天。

五个底：2011-11-18、2015-01-14（这两轮是早期市场，用 Bitstamp 日收盘）、2018-12-15、2022-11-21、2026-06-30（FRED）。
本轮用到 2026-11-12 的 FRED 收盘，另加 11-13 的盘中价（Yahoo，命令行参数传入，默认 68,537）。
2015-01-14 之前后 FRED 有异常点（$120），所以 2015-06-30 以前一律用 Bitstamp。

输出（本目录）：
- pullbacks_table.csv、pullbacks_numbers.txt
- png/pullbacks_depth_vs_days.png      每次回撤一个点：横轴时长、纵轴深度；本轮这次是红星
- png/post_bottom_underwater.png      五轮底后 0–365 天：上图相对底的涨幅，下图距阶段高点的回撤
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
MERGED = ROOT / "data" / "btc_merged_daily.csv"
BITSTAMP = ROOT / "analysis_runs" / "2026-10-01_cycle_time_amplitude_fit" / "bitstamp_btcusd_daily_2011_2015.csv"
LAST_CLOSE = pd.Timestamp("2026-11-12")
LIVE = float(sys.argv[1]) if len(sys.argv) > 1 else 68537.0
LIVE_DATE = pd.Timestamp("2026-11-13")
T = pd.Timestamp
CYCLES = [  # (名字, 底, 颜色, 是否早期)
    ("2011 轮", T("2011-11-18"), "#d6d3d1", True),
    ("2013 轮", T("2015-01-14"), "#a8a29e", True),
    ("2017 轮", T("2018-12-15"), "#94a3b8", False),
    ("2021 轮", T("2022-11-21"), "#0d9488", False),
    ("2025 轮（本轮）", T("2026-06-30"), "#dc2626", False),
]


def load() -> pd.Series:
    bs = pd.read_csv(BITSTAMP, parse_dates=["date"]).set_index("date")["close"]
    m = pd.read_csv(MERGED, parse_dates=["date"]).set_index("date")["price"].loc[:LAST_CLOSE]
    s = pd.concat([bs.loc[:"2015-06-30"], m.loc["2015-07-01":]]).sort_index()
    s.loc[LIVE_DATE] = LIVE
    return s


def pullbacks(seg: pd.Series, bottom_px: float, is_current: bool, thresh: float = -0.10) -> list[dict]:
    dd = seg / seg.cummax() - 1
    res, inside = [], False
    for d, x in dd.items():
        if x <= thresh and not inside:
            inside, pk, lo = True, seg[:d].idxmax(), d
        elif inside:
            lo = d if x < dd[lo] else lo
            if x >= 0:          # 回到前高 = 这次回撤结束
                res.append(dict(高点=pk, 低点=lo, 收复日=d, 进行中=False))
                inside = False
    if inside:
        res.append(dict(高点=pk, 低点=lo, 收复日=None, 进行中=is_current))   # 别的轮：一年窗口结束时还没收复
    b0 = seg.index[0]
    for r in res:
        r.update(深度=seg[r["低点"]] / seg[r["高点"]] - 1, 下跌天数=(r["低点"] - r["高点"]).days,
                 高点在底后第几天=(r["高点"] - b0).days, 高点相对底=seg[r["高点"]] / bottom_px - 1,
                 低点相对底=seg[r["低点"]] / bottom_px - 1,
                 收复天数=(r["收复日"] - r["低点"]).days if r["收复日"] is not None else None)
    return res


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    PNG.mkdir(parents=True, exist_ok=True)
    s = load()
    rows, lines = [], []
    out = lines.append
    out(f"本轮：最后收盘 {LAST_CLOSE.date()} ${s[LAST_CLOSE]:,.0f}；{LIVE_DATE.date()} 盘中 ${LIVE:,.0f}")
    paths = {}
    for name, b0, color, early in CYCLES:
        w = s.loc[b0 - pd.Timedelta(days=10): b0 + pd.Timedelta(days=10)]
        b = w.idxmin()              # 早期数据源不同，底取 ±10 天内最低收盘
        bp = float(s[b])
        seg = s.loc[b: b + pd.Timedelta(days=365)]
        paths[name] = seg
        pbs = pullbacks(seg, bp, '本轮' in name)
        broke = float(seg.iloc[1:].min()) < bp
        out("")
        out(f"== {name}{'（早期）' if early else ''}：底 {b.date()} ${bp:,.2f}；底后一年内最高 {seg.max() / bp - 1:+.0%}；"
            f"一年内{'跌破过' if broke else '没有跌破'}底")
        for r in pbs:
            rows.append(dict(周期=name, 早期=early, **{k: (v.date().isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in r.items()}))
            out(f"  {r['高点'].date()}（底后第 {r['高点在底后第几天']} 天，比底 {r['高点相对底']:+.0%}）→ {r['低点'].date()}："
                f"{r['深度']:+.1%}，跌了 {r['下跌天数']} 天；低点比底 {r['低点相对底']:+.0%}；"
                + ("还在进行" if r["进行中"] else (f"{r['收复天数']} 天后收复前高" if r['收复天数'] is not None else "底后一年内没收复前高")))
    tb = pd.DataFrame(rows)
    tb.to_csv(HERE / "pullbacks_table.csv", index=False, encoding="utf-8-sig")

    mature = tb[(~tb["早期"]) & (~tb["进行中"])]
    early_t = tb[(tb["早期"]) & (~tb["进行中"])]
    cur = tb[tb["进行中"]].iloc[0]
    out("")
    out(f"== 本轮这次：{cur['高点']} → 至今 {cur['深度']:+.1%}，{cur['下跌天数']} 天")
    for lab, df in [("成熟两轮（2017、2021）", mature), ("早期两轮（2011、2013）", early_t), ("四轮合计", pd.concat([mature, early_t]))]:
        out(f"{lab}：底后一年内 ≥10% 的回撤共 {len(df)} 次；深度中位 {df['深度'].median():+.1%}、最深 {df['深度'].min():+.1%}；"
            f"时长中位 {df['下跌天数'].median():.0f} 天、最长 {df['下跌天数'].max()} 天；"
            f"比本轮这次更深的 {int((df['深度'] < cur['深度']).sum())} 次，更长的 {int((df['下跌天数'] > cur['下跌天数']).sum())} 次")
    # 本轮距底、距 8 月更高低点还有多远
    out("")
    for lab, lv in [("8 月更高的低点 $62,835", 62835), ("本轮底 $58,586", 58586)]:
        out(f"从盘中 ${LIVE:,.0f} 到{lab}：{lv / LIVE - 1:+.1%}；相当于从 {cur['高点']} 的高点回撤 {lv / s[T(cur['高点'])] - 1:+.1%}")
    (HERE / "pullbacks_numbers.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---------- 图 1：深度 vs 时长 ----------
    fig, ax = plt.subplots(figsize=(11.5, 7), dpi=170)
    for name, b, color, early in CYCLES:
        d = tb[(tb["周期"] == name) & (~tb["进行中"])]
        ax.scatter(d["下跌天数"], d["深度"] * 100, s=70, color=color, edgecolor="#334155" if not early else "#a8a29e",
                   marker="o" if not early else "s", label=name + ("（早期）" if early else ""), zorder=3)
        for _, r in d.iterrows():
            ax.annotate(f"{r['高点'][2:7]}", (r["下跌天数"], r["深度"] * 100), xytext=(5, 3), textcoords="offset points",
                        fontsize=7.4, color="#475569")
    ax.scatter([cur["下跌天数"]], [cur["深度"] * 100], s=320, marker="*", color="#dc2626", edgecolor="white", zorder=5,
               label=f"本轮这次（{cur['高点'][5:]} 起，还在进行）")
    ax.annotate(f"本轮这次：{cur['深度']:+.1%}，{cur['下跌天数']} 天，还在进行", (cur["下跌天数"], cur["深度"] * 100),
                xytext=(14, -6), textcoords="offset points", fontsize=10, color="#dc2626", weight="bold")
    ax.set_xlabel("从阶段高点跌到低点用了几天")
    ax.set_ylabel("回撤深度（%）")
    ax.grid(True, color="#e2e8f0")
    ax.legend(loc="lower right", fontsize=8.8)
    ax.set_title("五轮「见底后第一年」里的每一次回撤（≥10%）：深度 vs 时长", fontsize=13.5, weight="bold")
    ax.text(0.01, 0.02, "点旁边的数字 = 回撤开始的年月。早期两轮（2011、2013）用 Bitstamp 日收盘。", transform=ax.transAxes,
            fontsize=8.4, color="#334155")
    fig.savefig(PNG / "pullbacks_depth_vs_days.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 图 2：底后路径 + 水下图 ----------
    fig, axs = plt.subplots(2, 1, figsize=(12.5, 9.5), dpi=170, sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    for name, b, color, early in CYCLES:
        seg = paths[name]
        x = (seg.index - seg.index[0]).days
        lw = 3.0 if "本轮" in name else (1.2 if early else 1.8)
        ls = "--" if early else "-"
        axs[0].plot(x, seg.values / seg.iloc[0], color=color, lw=lw, ls=ls, label=name + ("（早期）" if early else ""))
        axs[1].plot(x, (seg / seg.cummax() - 1).values * 100, color=color, lw=lw, ls=ls)
    axs[0].set_yscale("log")
    axs[0].yaxis.set_major_locator(mticker.FixedLocator([0.8, 1, 1.5, 2, 3, 5, 10]))
    axs[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{(v - 1) * 100:+.0f}%"))
    axs[0].yaxis.set_minor_formatter(mticker.NullFormatter())
    axs[0].axhline(1, color="#475569", lw=0.9)
    axs[0].set_ylabel("相对各自底的涨跌（对数刻度）")
    axs[0].legend(loc="upper left", fontsize=8.8)
    axs[0].grid(True, color="#e2e8f0", which="both")
    axs[0].set_title("五轮见底后第一年：上图是离底多远，下图是离阶段高点跌了多少", fontsize=13, weight="bold")
    axs[1].set_ylabel("距阶段高点（%）")
    axs[1].set_xlabel("距各自底的天数")
    axs[1].grid(True, color="#e2e8f0")
    now_x = (LIVE_DATE - T("2026-06-30")).days
    for ax in axs:
        ax.axvline(now_x, color="#dc2626", ls=":", lw=1.2)
    fig.tight_layout()
    fig.savefig(PNG / "post_bottom_underwater.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
