# -*- coding: utf-8 -*-
"""减半和暴涨：四轮周期里减半、突破前高、最猛 90 天各在哪里，下一轮按不同「时钟」各落在哪里（2026-10-01）。

口径同 fit_cycles.py（2025 轮顶 = 2025-08-12，底 = 2026-06-30）。
2013 轮的「前高」= 2011-06-11 $33.80（blockchain.com 市场均价，Bitstamp 数据从 2011-08 才开始；两者重叠期中位差 0.7%）。

输出（本目录）：
- halving_surge_table.csv     每轮：减半在牛市里的位置、突破前高、最猛 90 天
- next_cycle_milestones.csv   下一轮各里程碑在三种时钟下的日期
- png/halving_surge_aligned.png   四轮按减半对齐：底、突破前高、最猛 90 天、顶
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_cycles import CYCLES, load_price, next_halving_estimate

HERE = Path(__file__).resolve().parent
PNG = HERE / "png"
T = pd.Timestamp
PREV_TOP_2011 = (T("2011-06-11"), 33.80)
SURGE_DAYS = 90
COLORS = {"2013": "#cbd5e1", "2017": "#94a3b8", "2021": "#475569", "2025": "#0f766e"}


def setup() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"


def strongest_window(s: pd.Series, start: pd.Timestamp, end: pd.Timestamp, days: int) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """[start, end] 里涨得最多的 days 天：返回 (起点, 终点, 倍数)。"""
    seg = s.loc[start:end].asfreq("D").ffill()
    ratio = seg / seg.shift(days)
    t_end = ratio.idxmax()
    return t_end - pd.Timedelta(days=days), t_end, float(ratio.max())


def table(s: pd.Series) -> pd.DataFrame:
    rows, prev_top = [], PREV_TOP_2011
    for name, halving, pbot, top, bot in CYCLES:
        bull = (top - pbot).days
        after = s.loc[pbot:top]
        bo = after[after > prev_top[1]].index[0]  # 第一次收盘站上上一轮的顶
        sw0, sw1, smult = strongest_window(s, pbot, top, SURGE_DAYS)
        rows.append({
            "轮次": name, "上一轮底": pbot.date(), "减半": halving.date(), "顶": top.date(),
            "牛市_天": bull,
            "底→减半_天": (halving - pbot).days, "减半在牛市的位置": round((halving - pbot).days / bull, 3),
            "减半→顶_天": (top - halving).days,
            "前高": round(prev_top[1], 2), "前高日期": prev_top[0].date(),
            "突破前高日": bo.date(), "突破_相对减半_天": (bo - halving).days,
            "突破在牛市的位置": round((bo - pbot).days / bull, 3),
            "突破→顶_天": (top - bo).days, "突破→顶_倍数": round(float(s[top]) / prev_top[1], 2),
            "最猛90天_起": sw0.date(), "最猛90天_止": sw1.date(), "最猛90天_倍数": round(smult, 2),
            "最猛90天止_相对减半_天": (sw1 - halving).days, "最猛90天止_距顶_天": (top - sw1).days,
            "最猛90天止在牛市的位置": round((sw1 - pbot).days / bull, 3),
        })
        prev_top = (top, float(s[top]))
    return pd.DataFrame(rows)


def milestones(tb: pd.DataFrame, next_halving: pd.Timestamp, top_px: float) -> pd.DataFrame:
    """下一轮（底 2026-06-30）各里程碑在三种时钟下的日期。"""
    bottom = T("2026-06-30")
    cur = tb.iloc[-1]
    # 三种时钟给出的「下一轮牛市天数」（底 → 顶）
    clocks = {
        "市场时钟（顶→顶每轮 -50 天）": (T("2029-03-27") - bottom).days,
        "减半→顶时钟（479 天）": (next_halving + pd.Timedelta(days=int(cur["减半→顶_天"])) - bottom).days,
        "减半居中时钟（减半在牛市 ~50% 处）": round((next_halving - bottom).days / float(tb["减半在牛市的位置"].mean())),
    }
    rows = []
    for clock, bull in clocks.items():
        top = bottom + pd.Timedelta(days=bull)
        rows.append({
            "时钟": clock, "牛市_天": bull, "顶": top.date(),
            "减半在牛市的位置": round((next_halving - bottom).days / bull, 3),
            "突破前高（按上一轮在牛市的位置 {:.2f}）".format(cur["突破在牛市的位置"]): (bottom + pd.Timedelta(days=round(bull * cur["突破在牛市的位置"]))).date(),
            "最猛90天止（按四轮平均位置 {:.2f}）".format(tb["最猛90天止在牛市的位置"].mean()): (bottom + pd.Timedelta(days=round(bull * tb["最猛90天止在牛市的位置"].mean()))).date(),
        })
    df = pd.DataFrame(rows)
    df.attrs["breakout_level"] = top_px
    return df


def chart(s: pd.Series, tb: pd.DataFrame, next_halving: pd.Timestamp) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.4), dpi=170)
    for (name, halving, pbot, top, bot), r in zip(CYCLES, tb.itertuples(index=False)):
        seg = s.loc[pbot:top]
        x = (seg.index - halving).days
        ax.plot(x, seg.values / float(s[halving]), color=COLORS[name], lw=3.0 if name == "2025" else 1.8, label=f"{name} 轮", zorder=4 if name == "2025" else 3)
        pts = [(pbot, "底", "o"), (T(str(r.突破前高日)), "突破前高", "^"), (T(str(r.最猛90天_止)), "最猛90天止", "s"), (top, "顶", "*")]
        for d, tag, mk in pts:
            ax.scatter([(d - halving).days], [float(s[d]) / float(s[halving])], marker=mk, s=110 if mk == "*" else 42,
                       color=COLORS[name], edgecolor="#0f172a", linewidth=0.6, zorder=6)
    xt = ax.get_xaxis_transform()  # x 用数据坐标，y 用坐标轴比例
    ax.axvline(0, color="#0f172a", lw=1.2, ls="--")
    ax.text(6, 0.03, "减半日", fontsize=9, color="#0f172a", transform=xt)
    nb = (T("2026-06-30") - next_halving).days
    ax.axvline(nb, color="#b91c1c", lw=1.2, ls=":")
    ax.text(nb + 6, 0.60, f"本轮的底 2026-06-30\n在下次减半（约 {next_halving:%Y-%m-%d}）前 {-nb} 天\n（前四轮：405 / 542 / 513 / 516 天）",
            fontsize=8.6, color="#b91c1c", transform=xt)
    from matplotlib.lines import Line2D
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([0], [0], marker=m, ls="", color="#64748b", markeredgecolor="#0f172a", label=l, markersize=8 if m != "*" else 12)
        for m, l in [("o", "上一轮底"), ("^", "突破上一轮的顶"), ("s", f"最猛 {SURGE_DAYS} 天的终点"), ("*", "顶")]]
    ax.legend(handles=handles, loc="upper left", fontsize=8.8, ncol=2)
    ax.set_yscale("log")
    ax.set_xlabel("距减半的天数")
    ax.set_ylabel("价格 ÷ 减半当天价格（对数刻度）")
    ax.set_xlim(-700, 600)
    ax.grid(True, color="#e2e8f0", which="both")
    ax.set_title("四轮按减半对齐：底、突破前高、最猛 90 天、顶各在哪里", fontsize=13.5, weight="bold")
    fig.savefig(PNG / "halving_surge_aligned.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    s = load_price()
    tb = table(s)
    nh, note = next_halving_estimate()
    ms = milestones(tb, nh, float(s[T("2025-08-12")]))
    tb.to_csv(HERE / "halving_surge_table.csv", index=False, encoding="utf-8-sig")
    ms.to_csv(HERE / "next_cycle_milestones.csv", index=False, encoding="utf-8-sig")
    chart(s, tb, nh)
    pd.set_option("display.width", 280, "display.max_columns", 40)
    print(tb.T.to_string())
    print("下次减半：", nh.date(), "|", note)
    print(ms.T.to_string())
    g = np.log(tb["最猛90天_倍数"].to_numpy())
    b, a = np.polyfit(np.arange(len(g)), np.log(g), 1)
    print(f"最猛 90 天对数涨幅：{np.round(g, 3).tolist()}，比例 {np.round(g[1:] / g[:-1], 3).tolist()}；"
          f"等比拟合 ×{math.exp(b):.3f}/轮 → 下一轮约 ×{math.exp(math.exp(a + b * len(g))):.2f}")


if __name__ == "__main__":
    main()
