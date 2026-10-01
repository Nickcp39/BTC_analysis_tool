# -*- coding: utf-8 -*-
"""2026-09-22 版新增：把「当年的预测」和「后来的实际」放在一起对照。
2026-10-01 版：口径统一为顶 = 2025-08-12、底 = 2026-06-30（AGENTS.md 规定不再用 2025-10-05）。
去掉 09-22 版按 10-05 画的退火路径和「10-05 + 364/378 天」时间窗；对照表按 08-12 顶、06-30 底重新判。

输出（全部写在本目录，不碰任何旧版本）：
- png/forecast_vs_actual.png         价格实际走势 + 06-01 五模型底部框 + 07-17 家庭计划 55-60k 区 + 「顶 08-12 + 364–378 天」时间窗
- png/bottom_aligned_rebound.png     把「真底」和「熊市中途低点」都对齐到 0 天，看本轮 06-30 之后的反弹更像哪一种
- forecast_scorecard.csv             对照表（报告 §1 的数字来源）
- rebound_comparison.csv             反弹对照表（报告 §3 的数字来源）
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PNG = HERE / "png"
PRICE_CSV = ROOT / "data" / "btc_merged_daily.csv"
T = pd.Timestamp

# ---- 当年的预测（原样抄自旧版产物，路径见 SOURCES）----
JUNE_MODELS = ROOT / "analysis_runs" / "2026-06-01_parent_report" / "tables" / "model_average_inputs.csv"
TOP = T("2025-08-12")                                      # 本轮的顶（AGENTS.md 统一锚点）
BOTTOM = T("2026-06-30")                                   # 本轮的底（顶后最低收盘）
HALVING = T("2024-04-20")
HIST_TOP_TO_BOTTOM = (364, 378)                            # 2017 / 2021 轮价格顶→价格底天数
FAMILY_BAND = (55000, 60000)                               # 07-17 版 §8「较合理的价格期望区间」
BUY_WINDOW = (T("2026-08-01"), T("2027-02-28"))            # 07-17 版 §8 按月分批窗口
CONSENSUS = dict(mean=54472, median=57626, lo=43611, hi=60385)  # stepD14 交叉验证（06 月模型共识）


def setup() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"


def load_price() -> pd.Series:
    df = pd.read_csv(PRICE_CSV, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index()


def low_between(s: pd.Series, a: str, b: str) -> tuple[pd.Timestamp, float]:
    sub = s.loc[a:b]
    return sub.idxmin(), float(sub.min())


def forecast_chart(s: pd.Series, june: pd.DataFrame) -> None:
    latest = s.index.max()
    fig, ax = plt.subplots(figsize=(13.6, 7.0), dpi=170)

    # 买入窗口（07-17 版家庭计划）
    ax.axvspan(*BUY_WINDOW, color="#e0f2fe", alpha=0.55, zorder=0)
    ax.text(BUY_WINDOW[0] + pd.Timedelta(days=4), 131000, "07-17 版家庭计划：2026-08 → 2027-02 按月分批",
            fontsize=8.8, color="#0369a1", va="top")

    # 55-60k 「合理价格区」
    ax.fill_between([BUY_WINDOW[0], BUY_WINDOW[1]], *FAMILY_BAND, color="#fde68a", alpha=0.8, zorder=1)
    ax.text(T("2026-11-06"), FAMILY_BAND[0] - 1800, "07-17 版「较合理价格区」\n$55k–60k",
            fontsize=8.8, color="#92400e", va="top")

    # 06-01 五模型底部框（价格 = 模型中枢价范围；日期 = 模型中枢日期范围）
    c_lo, c_hi = june["center_date"].min(), june["center_date"].max()
    p_lo, p_hi = june["center_price"].min(), june["center_price"].max()
    ax.add_patch(plt.Rectangle((mdates.date2num(c_lo), p_lo), (c_hi - c_lo).days, p_hi - p_lo,
                               fc="#fecaca", ec="#b91c1c", lw=1.2, alpha=0.75, zorder=2))
    ax.text(c_hi + pd.Timedelta(days=5), p_lo - 1500,
            f"06-01 五模型「底部」预测\n中枢价 \\${p_lo/1000:.1f}k–\\${p_hi/1000:.1f}k\n中枢日 {c_lo:%m-%d}–{c_hi:%m-%d}",
            fontsize=8.4, color="#b91c1c", va="top")

    # 「顶 08-12 + 前两轮顶→底天数」时间窗
    a, b = (TOP + pd.Timedelta(days=d) for d in HIST_TOP_TO_BOTTOM)
    ax.axvspan(a, b, color="#64748b", alpha=0.22, zorder=2)
    ax.text(b + pd.Timedelta(days=3), 118000, f"顶 08-12\n+{HIST_TOP_TO_BOTTOM[0]}–{HIST_TOP_TO_BOTTOM[1]} 天", fontsize=8.4,
            color="#334155", va="top")

    # 实际价格
    act = s.loc["2025-06-01":]
    ax.plot(act.index, act.values, color="#0f172a", lw=2.4, zorder=6, label="BTC 实际价格（FRED/Coinbase 日线）")

    retest = s.loc["2025-09-15":"2025-11-15"].idxmax()
    ax.annotate(f"{retest:%m-%d} 回测\n只比 8 月顶高 {s[retest] / s[TOP] - 1:.1%}", xy=(retest, float(s[retest])),
                xytext=(retest + pd.Timedelta(days=24), float(s[retest]) + 5000), fontsize=8, color="#64748b",
                arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.8})
    marks = [
        (TOP, float(s[TOP]), f"顶 2025-08-12\n${s[TOP]:,.0f}", (-70, 6000)),
        (*low_between(s, "2026-01-20", "2026-02-20"), "第一脚", (-40, -16000)),
        (BOTTOM, float(s[BOTTOM]), f"底 2026-06-30\n${s[BOTTOM]:,.0f}", (-100, -14000)),
        (*low_between(s, "2026-07-18", "2026-08-18"), "8 月更高的低点", (-35, -17000)),
        (latest, float(s.iloc[-1]), "今天", (12, 9000)),
    ]
    for d, px, txt, (dx, dy) in marks:
        ax.scatter([d], [px], s=46, color="#b91c1c" if txt == "今天" else "#0f172a", zorder=7,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(f"{txt}\n{d:%Y-%m-%d} ${px:,.0f}" if "\n" not in txt else txt,
                    xy=(d, px), xytext=(d + pd.Timedelta(days=dx), px + dy), fontsize=8.6,
                    color="#b91c1c" if txt == "今天" else "#0f172a", weight="bold" if txt == "今天" else None,
                    arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 0.9})

    ax.set_xlim(T("2025-06-01"), T("2027-03-01"))
    ax.set_ylim(30000, 135000)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, color="#e2e8f0")
    ax.legend(loc="lower left", fontsize=8.4, framealpha=0.92)
    ax.set_title(f"当年的预测 vs 实际走势（数据至 {latest:%Y-%m-%d}）", fontsize=14, weight="bold")
    fig.savefig(PNG / "forecast_vs_actual.png", bbox_inches="tight")
    plt.close(fig)


def rebound_chart(s: pd.Series) -> pd.DataFrame:
    """对齐到低点=0 天；看低点之后 150 天的走法。"""
    latest = s.index.max()
    cases = [
        ("2018-12 真底（2017 轮）", *low_between(s, "2018-11-01", "2019-01-31"), "#16a34a", "-", "true"),
        ("2022-11 真底（2021 轮）", *low_between(s, "2022-11-01", "2022-12-31"), "#0d9488", "-", "true"),
        ("2018-02 熊市中途低点", *low_between(s, "2018-01-25", "2018-02-28"), "#f97316", "--", "bear"),
        ("2022-06 熊市中途低点", *low_between(s, "2022-06-01", "2022-07-15"), "#dc2626", "--", "bear"),
        ("2026-06-30 本轮的底", *low_between(s, "2026-06-01", "2026-07-15"), "#0f172a", "-", "now"),
    ]
    rows = []
    fig, ax = plt.subplots(figsize=(12.8, 6.4), dpi=170)
    for name, d0, p0, color, ls, kind in cases:
        seg = s.loc[d0 - pd.Timedelta(days=30): d0 + pd.Timedelta(days=150)]
        x = (seg.index - d0).days
        ax.plot(x, (seg.values / p0 - 1) * 100, color=color, ls=ls, lw=3.2 if kind == "now" else 1.7,
                label=name, zorder=5 if kind == "now" else 3)
        after = s.loc[d0: d0 + pd.Timedelta(days=150)]
        n_now = min((latest - d0).days, 150)
        at_n = s.loc[: d0 + pd.Timedelta(days=(latest - s.loc["2026-06-01":"2026-07-15"].idxmin()).days)]
        future_min = s.loc[d0 + pd.Timedelta(days=1):]
        rows.append({
            "情形": name,
            "低点日期": d0.date().isoformat(),
            "低点价格": round(p0),
            "低点后150天内最大涨幅": round((after.max() / p0 - 1) * 100, 1),
            f"低点后{(latest - s.loc['2026-06-01':'2026-07-15'].idxmin()).days}天时涨幅": round((float(at_n.iloc[-1]) / p0 - 1) * 100, 1),
            "之后是否跌破这个低点": "是" if kind != "now" and float(future_min.min()) < p0 else ("否" if kind != "now" else "至今没有"),
            "之后最低价": round(float(future_min.min())) if kind != "now" else None,
        })
    ax.axhline(0, color="#475569", lw=0.9)
    ax.axvline(0, color="#475569", lw=0.9, ls=":")
    ax.set_xlabel("距低点的天数")
    ax.set_ylabel("相对低点的涨跌（%）")
    ax.grid(True, color="#e2e8f0")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("06-30 之后这波反弹，更像「真底之后」还是「熊市中途反弹」？", fontsize=13.5, weight="bold")
    ax.text(0.99, 0.02, "实线＝后来被证明是真底；虚线＝熊市中途低点（之后又跌破）。样本只有 4 个，只能当参照。",
            transform=ax.transAxes, ha="right", fontsize=8.6, color="#334155")
    fig.savefig(PNG / "bottom_aligned_rebound.png", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def scorecard(s: pd.Series, june: pd.DataFrame) -> pd.DataFrame:
    """按顶 2025-08-12、底 2026-06-30 重新判每一条当年的说法。"""
    latest, px = s.index.max(), float(s.iloc[-1])
    top_px, low_px = float(s[TOP]), float(s[BOTTOM])
    d_hl, p_hl = low_between(s, "2026-07-18", "2026-08-05")
    d_hl2, p_hl2 = low_between(s, "2026-08-06", "2026-08-18")
    win_a, win_b = (TOP + pd.Timedelta(days=d) for d in HIST_TOP_TO_BOTTOM)
    d_win, p_win = low_between(s, str(BUY_WINDOW[0].date()), str(latest.date()))
    h2t, t2b, dd = (TOP - HALVING).days, (BOTTOM - TOP).days, low_px / top_px - 1
    r = [
        ("减半→顶 536 天（2017/2021 两轮平均）", "本轮顶约 2025-10-08",
         f"顶在 2025-08-12（减半后 {h2t} 天）", f"错：早了 {536 - h2t} 天，本轮整体快约一成"),
        ("06-01 五模型：底部价格", f"中枢 ${june.center_price.min():,.0f}–${june.center_price.max():,.0f}；共识中位 ${CONSENSUS['median']:,}",
         f"底 ${low_px:,.0f}（2026-06-30），比共识中位只高 {low_px / CONSENSUS['median'] - 1:.1%}", "对"),
        ("06-01 五模型：底部时间", f"中枢 {june.center_date.min():%Y-%m-%d} ~ {june.center_date.max():%Y-%m-%d}",
         "底在 2026-06-30，早了 3–4 个月", "错"),
        ("AHR999 深底 0.26–0.28", "本轮也会探到这一带", "2026-02-05 0.280；2026-06-30 0.281", "对"),
        (f"顶 08-12 + 前两轮顶→底天数（{HIST_TOP_TO_BOTTOM[0]}–{HIST_TOP_TO_BOTTOM[1]} 天）",
         f"{win_a:%Y-%m-%d} ~ {win_b:%m-%d} 附近是时间底",
         f"这段时间只有更高的低点（{d_hl:%m-%d} ${p_hl:,.0f}、{d_hl2:%m-%d} ${p_hl2:,.0f}，08-17 起涨）；"
         f"真正的底在顶后第 {t2b} 天（06-30），早了 {HIST_TOP_TO_BOTTOM[0] - t2b}–{HIST_TOP_TO_BOTTOM[1] - t2b} 天",
         "半对：方向对，时间早了 6–8 周"),
        ("退火模型：顶后跌 28%–44%", "本轮跌幅会比前两轮浅很多",
         f"顶→底 {dd:.1%}，比模型最深的 -44% 还深约 {abs(dd) * 100 - 44.3:.0f} 个百分点", "半对：浅是对的，但浅得不够"),
        ("stepD12 价格区间", "$29k–$57k，中枢 $39k", f"底 ${low_px:,.0f}", "太悲观"),
        ("07-17 家庭计划「55–60k 较合理价格区」", "2026-08 → 2027-02 窗口里大概率能在这一带买",
         f"底在窗口开始前（06-30）；窗口内最低 ${p_win:,.0f}（{d_win:%m-%d}）；8 月均价 ${s.loc['2026-08'].mean():,.0f}，9 月均价 ${s.loc['2026-09'].mean():,.0f}",
         "错（09-22 版已作废这句话）"),
        ("「W 底」（2 月第一脚 → 6 月第二脚）", "第二脚后反弹",
         f"06-30 后 +{px / low_px - 1:.0%}（最高 +{s.loc[BOTTOM:].max() / low_px - 1:.0%}）", "对"),
        ("「真正时间底大概率在 8–10 月」", "8–10 月还会再跌回来",
         f"底在 6 月底；8、9 月都没再跌回去（9 月最低 ${s.loc['2026-09'].min():,.0f}）", "错"),
    ]
    return pd.DataFrame(r, columns=["当年的计算", "当时的结论", "实际（截至 " + f"{latest:%Y-%m-%d}）", "对/错"])


def main() -> None:
    setup()
    PNG.mkdir(parents=True, exist_ok=True)
    s = load_price()
    june = pd.read_csv(JUNE_MODELS, parse_dates=["center_date"])
    forecast_chart(s, june)
    rb = rebound_chart(s)
    rb.to_csv(HERE / "rebound_comparison.csv", index=False, encoding="utf-8-sig")
    sc = scorecard(s, june)
    sc.to_csv(HERE / "forecast_scorecard.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(rb.to_string(index=False))
    print(sc.to_string(index=False))


if __name__ == "__main__":
    main()
