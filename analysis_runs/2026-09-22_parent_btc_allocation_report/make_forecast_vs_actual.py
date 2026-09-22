# -*- coding: utf-8 -*-
"""2026-09-22 版新增：把「当年的预测」和「后来的实际」放在一起对照。

输出（全部写在本目录，不碰任何旧版本）：
- png/forecast_vs_actual.png         价格实际走势 + 06-01 五模型底部框 + 07-17 家庭计划 55-60k 区 + 退火路径 + 两个「365 天」时间窗
- png/bottom_aligned_rebound.png     把「真底」和「熊市中途低点」都对齐到 0 天，看本轮 06-30 之后的反弹更像哪一种
- forecast_scorecard.csv             对照表（报告 §2 的数字来源）
- rebound_comparison.csv             反弹对照表（报告 §6 的数字来源）
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
STEPC1_OLD = ROOT / "visualization" / "2026-07-17" / "stepC1_bottom_forecast.txt"
STEPC1_NEW_CSV = ROOT / "visualization" / "2026-09-22" / "stepC1_postpeak_aligned.csv"
TRUE_TOP, TRUE_TOP_PX = T("2025-10-05"), 124720.09       # stepC1 真顶
REPORT_ANCHOR = T("2025-08-12")                            # 家长报告统一锚点（AGENTS.md）
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
            f"06-01 五模型「底部」预测\n中枢价 ${p_lo/1000:.1f}k–${p_hi/1000:.1f}k\n中枢日 {c_lo:%m-%d}–{c_hi:%m-%d}",
            fontsize=8.4, color="#b91c1c", va="top")

    # 退火路径（从真顶 10-05 出发，换算成价格）
    ann = pd.read_csv(STEPC1_NEW_CSV, encoding="utf-8-sig")
    for col, color, name in [("dd_2017_scaled", "#16a34a", "仿 2017（退火）"),
                             ("dd_2021_scaled", "#ea580c", "仿 2021（退火）")]:
        d = TRUE_TOP + pd.to_timedelta(ann["post_day"], unit="D")
        px = TRUE_TOP_PX * (1 + ann[col] / 100.0)
        keep = d <= T("2027-03-01")
        ax.plot(d[keep], px[keep], color=color, lw=1.3, ls="--", alpha=0.85, zorder=3,
                label=f"stepC1 退火路径 {name}")

    # 两个「365 天」时间窗
    for a, b, txt, y in [
        (REPORT_ANCHOR + pd.Timedelta(days=364), REPORT_ANCHOR + pd.Timedelta(days=366), "报告锚点 08-12\n+364/366 天", 118000),
        (TRUE_TOP + pd.Timedelta(days=364), TRUE_TOP + pd.Timedelta(days=378), "真顶 10-05\n+364/378 天", 111000),
    ]:
        ax.axvspan(a, b, color="#64748b", alpha=0.22, zorder=2)
        ax.text(b + pd.Timedelta(days=3), y, txt, fontsize=8.4, color="#334155", va="top")

    # 实际价格
    act = s.loc["2025-06-01":]
    ax.plot(act.index, act.values, color="#0f172a", lw=2.4, zorder=6, label="BTC 实际价格（FRED/Coinbase 日线）")

    marks = [
        (TRUE_TOP, TRUE_TOP_PX, "真顶 2025-10-05\n$124,720", (-118, 4000)),
        (*low_between(s, "2026-01-20", "2026-02-20"), "第一脚", (-40, -16000)),
        (*low_between(s, "2026-06-01", "2026-07-15"), "第二脚（至今最低）", (-100, -14000)),
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
        ("2026-06-30 本轮至今最低", *low_between(s, "2026-06-01", "2026-07-15"), "#0f172a", "-", "now"),
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
            "之后是否跌破这个低点": "是" if kind != "now" and float(future_min.min()) < p0 else ("否" if kind != "now" else "未知（进行中）"),
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
    latest, px = s.index.max(), float(s.iloc[-1])
    d_low, p_low = low_between(s, "2025-10-05", str(latest.date()))
    d_hl, p_hl = low_between(s, "2026-07-18", "2026-08-05")
    d_hl2, p_hl2 = low_between(s, "2026-08-06", "2026-08-18")
    p_anchor365 = float(s.loc[REPORT_ANCHOR + pd.Timedelta(days=365)])
    d_win, p_win = low_between(s, str(BUY_WINDOW[0].date()), str(latest.date()))
    need = 1 - p_low / px
    r = [
        ("减半→顶 536 天（2017/2021 两轮）", "本轮顶约 2025-10-08", "实际顶 2025-10-05/06", "对（差 2–3 天）"),
        ("06-01 五模型：底部价格", f"中枢 ${june.center_price.min():,.0f}–${june.center_price.max():,.0f}；共识中位 ${CONSENSUS['median']:,}",
         f"至今最低 ${p_low:,.0f}（{d_low:%Y-%m-%d}）", "价格对（落在区间内、接近中位）"),
        ("06-01 五模型：底部时间", f"中枢 {june.center_date.min():%Y-%m-%d} ~ {june.center_date.max():%Y-%m-%d}",
         f"低点出现在 {d_low:%Y-%m-%d}，早了约 3–4 个月", "时间错（除非 10 月再跌破）"),
        ("AHR999 深底 0.26–0.28", "本轮也会探到这一带", "2026-02-05 0.280；2026-06-30 0.281", "对"),
        ("报告锚点 08-12 + 364/366 天", "2026-08-11 ~ 08-13 附近是时间底",
         f"第 365 天 ${p_anchor365:,.0f}；{d_hl:%m-%d} ${p_hl:,.0f} 与 {d_hl2:%m-%d} ${p_hl2:,.0f} 两次踩在同一位置，08-17 起飞",
         "时间点对，但它是「更高的低点」不是新低"),
        ("真顶 10-05 + 364/378 天（stepC1）", "2026-10-04 ~ 10-18 附近是时间底",
         f"还没到；要跌破 6 月低点需从今天再跌 {need:.0%}", "待验证"),
        ("stepC1 退火价格底 -28% ~ -44%", "$69.5k–$89.9k",
         f"6 月跌到 -53%（跌过头）；今天 {px/TRUE_TOP_PX-1:.0%} 又回到区间里", "半对：跌得比模型深，但现在回到模型路径"),
        ("stepD12 价格区间", "$29k–$57k，中枢 $39k", f"至今最低 ${p_low:,.0f}", "太悲观"),
        ("07-17 家庭计划「55–60k 较合理价格区」", "2026-08 → 2027-02 窗口里大概率能在这一带买",
         f"窗口内最低 ${p_win:,.0f}（{d_win:%m-%d}，离 60k 只差 {p_win/FAMILY_BAND[1]-1:.0%}）；8 月均价 ${s.loc['2026-08'].mean():,.0f}，9 月至今均价 ${s.loc['2026-09'].mean():,.0f}",
         "错：55–60k 在窗口开始前（6 月底）就到过，窗口里没再回来"),
        ("「W 底」（2 月第一脚 → 6 月第二脚）", "第二脚后反弹", f"06-30 后 +{px/p_low-1:.0%}", "对"),
        ("「真正时间底大概率在 8–10 月」", "8–10 月还会再跌回来", "8 月没破 6 月低点，反而起飞", "至今是错的（10 月未完）"),
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
