from __future__ import annotations

import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PNG = OUT / "png"
TABLES = OUT / "tables"
PNG.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

HALVING_2016 = pd.Timestamp("2016-07-09")
HALVING_2020 = pd.Timestamp("2020-05-11")
HALVING_2024 = pd.Timestamp("2024-04-20")
PEAK_2017 = pd.Timestamp("2017-12-19")
PEAK_2021 = pd.Timestamp("2021-11-10")
PEAK_2025 = pd.Timestamp("2025-10-05")
BOTTOM_2018 = pd.Timestamp("2018-12-15")
BOTTOM_2022 = pd.Timestamp("2022-11-21")

VOL_LEVEL = {"2021_peak_cycle": 3.0, "2025_peak_cycle": 1.0, "2029_peak_cycle": 1.0 / 3.0}
VOL_ALPHA = 0.5


def set_cn_font() -> None:
    for candidate in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\ARIALUNI.TTF"]:
        if Path(candidate).exists():
            font = fm.FontProperties(fname=candidate)
            plt.rcParams["font.family"] = font.get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def money(x: float) -> str:
    return f"${x:,.0f}"


def common(ax) -> None:
    ax.grid(True, which="both", ls=":", lw=0.7, alpha=0.45)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(PNG / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def load_price() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "btc_merged_daily.csv", parse_dates=["date"]).set_index("date").sort_index()


def bottom_anchor() -> dict[str, float]:
    coeff = pd.read_csv(ROOT / "analysis_runs" / "2026-06-01_trend_rework_v2" / "tables" / "model1_post_scale_coefficients.csv").iloc[0]
    ahr = pd.read_csv(ROOT / "analysis_runs" / "2026-06-01_trend_rework_v2" / "tables" / "ahr999_floor.csv").iloc[0]
    avg = pd.read_csv(ROOT / "analysis_runs" / "2026-06-01_trend_rework_v2" / "tables" / "average_trend_result.csv").iloc[0]
    center = float(np.average([coeff["mapped_center_price"], ahr["center_price"], avg["ahr_guarded_center_price"]], weights=[1.3, 1.1, 0.8]))
    return {
        "model1_coeff": float(coeff["mapped_center_price"]),
        "ahr_floor": float(ahr["center_price"]),
        "bottom_average_guarded": float(avg["ahr_guarded_center_price"]),
        "center": center,
        "lo": float(min(avg["price_lo"], coeff["mapped_2017_bottom_price"])),
        "hi": float(max(avg["price_hi"], coeff["mapped_2021_bottom_price"])),
    }


def forecast_halving() -> dict[str, object]:
    intervals = np.array([(HALVING_2020 - HALVING_2016).days, (HALVING_2024 - HALVING_2020).days], dtype=float)
    # Keep the user's original observation: halving intervals are getting longer.
    # So the next interval should not be shorter than 2020->2024.
    interval_center = float(intervals[-1] + 0.35 * (intervals[-1] - intervals[-2]))
    interval_lo = float(intervals[-1])
    interval_hi = float(intervals[-1] + 35)
    center = HALVING_2024 + pd.Timedelta(days=round(interval_center))
    lo = HALVING_2024 + pd.Timedelta(days=round(interval_lo))
    hi = HALVING_2024 + pd.Timedelta(days=round(interval_hi))
    return {"center": center, "lo": lo, "hi": hi, "interval_center": interval_center}


def time_models(next_halving: dict[str, object]) -> pd.DataFrame:
    h = pd.Timestamp(next_halving["center"])
    h_lo = pd.Timestamp(next_halving["lo"])
    h_hi = pd.Timestamp(next_halving["hi"])
    h2p = np.array([(PEAK_2017 - HALVING_2016).days, (PEAK_2021 - HALVING_2020).days, (PEAK_2025 - HALVING_2024).days], dtype=float)
    rows = []
    rows.append(
        {
            "model": "T1_halving_to_peak_recent_mean",
            "center_date": h + pd.Timedelta(days=round(float(np.mean(h2p[-3:])))),
            "lo": h_lo + pd.Timedelta(days=round(float(np.percentile(h2p[-3:], 20)))),
            "hi": h_hi + pd.Timedelta(days=round(float(np.percentile(h2p[-3:], 80)))),
            "weight": 1.15,
            "note": f"halving->peak days {h2p.astype(int).tolist()}",
        }
    )
    top_to_top = np.array([(PEAK_2021 - PEAK_2017).days, (PEAK_2025 - PEAK_2021).days], dtype=float)
    rows.append(
        {
            "model": "T2_top_to_top_clock",
            "center_date": PEAK_2025 + pd.Timedelta(days=round(float(np.mean(top_to_top)))),
            "lo": PEAK_2025 + pd.Timedelta(days=round(float(np.min(top_to_top) - 25))),
            "hi": PEAK_2025 + pd.Timedelta(days=round(float(np.max(top_to_top) + 35))),
            "weight": 1.05,
            "note": f"top->top days {top_to_top.astype(int).tolist()}",
        }
    )
    # Bottom->peak got shorter from 1061 to 1049. Keep the center near 1055 days after predicted bottom.
    bottom_center = pd.Timestamp("2026-10-22")
    rows.append(
        {
            "model": "T3_bottom_to_peak_clock",
            "center_date": bottom_center + pd.Timedelta(days=1055),
            "lo": bottom_center + pd.Timedelta(days=1015),
            "hi": bottom_center + pd.Timedelta(days=1100),
            "weight": 0.9,
            "note": "historical bottom->peak: 2018->2021=1061d, 2022->2025=1049d",
        }
    )
    return pd.DataFrame(rows)


def price_models(df: pd.DataFrame, bottom: dict[str, float]) -> pd.DataFrame:
    peak17 = float(df.loc[PEAK_2017, "price"])
    peak21 = float(df.loc[PEAK_2021, "price"])
    peak25 = float(df.loc[PEAK_2025, "price"])
    bot18 = float(df.loc[BOTTOM_2018, "price"])
    bot22 = float(df.loc[BOTTOM_2022, "price"])
    bot_next = float(bottom["center"])

    log_gain_2021 = math.log(peak21 / bot18)
    log_gain_2025 = math.log(peak25 / bot22)
    sqrt_next = (VOL_LEVEL["2029_peak_cycle"] / VOL_LEVEL["2025_peak_cycle"]) ** VOL_ALPHA
    log_gain_next = log_gain_2025 * sqrt_next
    p_sqrt = bot_next * math.exp(log_gain_next)

    # Decay in nominal peak multiples, softened because a pure extrapolation turns too flat.
    peak_ratio_21 = peak21 / peak17
    peak_ratio_25 = peak25 / peak21
    decay = peak_ratio_25 / peak_ratio_21
    p_ratio = peak25 * (1 + 0.55 * (peak_ratio_25 * decay - 1))

    # Bottom-to-bottom growth retained but attenuated by the same sqrt volatility scale.
    bottom_growth = bot22 / bot18
    next_bottom_growth = 1 + (bottom_growth - 1) * sqrt_next
    p_bottom_network = bot_next * next_bottom_growth * (peak25 / bot_next) ** 0.45

    # Prior-cycle actual multiple with a floor from cycle maturation.
    gain_ratio = log_gain_2025 / log_gain_2021
    p_log_trend = bot_next * math.exp(log_gain_2025 * max(0.55, gain_ratio))

    rows = [
        {
            "model": "P1_log_gain_sqrt_vol",
            "center_price": p_sqrt,
            "lo": p_sqrt * 0.84,
            "hi": p_sqrt * 1.18,
            "weight": 1.35,
            "note": f"log gain 2025={log_gain_2025:.3f}, next sqrt scale={sqrt_next:.3f}",
        },
        {
            "model": "P2_peak_ratio_decay_soft",
            "center_price": p_ratio,
            "lo": p_ratio * 0.82,
            "hi": p_ratio * 1.25,
            "weight": 0.85,
            "note": f"peak ratios 2021/2017={peak_ratio_21:.2f}, 2025/2021={peak_ratio_25:.2f}",
        },
        {
            "model": "P3_bottom_network_growth",
            "center_price": p_bottom_network,
            "lo": p_bottom_network * 0.80,
            "hi": p_bottom_network * 1.28,
            "weight": 0.75,
            "note": "bottom-to-bottom growth attenuated by next volatility scale",
        },
        {
            "model": "P4_log_gain_trend_floor",
            "center_price": p_log_trend,
            "lo": p_log_trend * 0.82,
            "hi": p_log_trend * 1.22,
            "weight": 1.05,
            "note": f"log gain decay ratio={gain_ratio:.3f}, floor=0.55",
        },
    ]
    return pd.DataFrame(rows)


def weighted_result(time_df: pd.DataFrame, price_df: pd.DataFrame) -> dict[str, object]:
    tw = time_df["weight"].to_numpy()
    center_ord = np.average([pd.Timestamp(d).toordinal() for d in time_df["center_date"]], weights=tw)
    lo_ord = np.average([pd.Timestamp(d).toordinal() for d in time_df["lo"]], weights=tw)
    hi_ord = np.average([pd.Timestamp(d).toordinal() for d in time_df["hi"]], weights=tw)

    pw = price_df["weight"].to_numpy()
    center_price = float(np.average(price_df["center_price"], weights=pw))
    lo_price = float(np.average(price_df["lo"], weights=pw))
    hi_price = float(np.average(price_df["hi"], weights=pw))

    return {
        "center_date": pd.Timestamp.fromordinal(int(round(center_ord))),
        "lo_date": pd.Timestamp.fromordinal(int(round(lo_ord))),
        "hi_date": pd.Timestamp.fromordinal(int(round(hi_ord))),
        "center_price": center_price,
        "lo_price": lo_price,
        "hi_price": hi_price,
    }


def plot_time(time_df: pd.DataFrame, avg: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ys = np.arange(len(time_df))
    for i, r in time_df.iterrows():
        c = pd.Timestamp(r["center_date"])
        ax.errorbar(c, i, xerr=[[c - pd.Timestamp(r["lo"])], [pd.Timestamp(r["hi"]) - c]], fmt="o", capsize=5, lw=2, label=r["model"])
        ax.text(c, i + 0.12, c.date().isoformat(), ha="center", fontsize=9)
    ax.axvspan(avg["lo_date"], avg["hi_date"], color="#d62728", alpha=0.12, label="average window")
    ax.axvline(avg["center_date"], color="#d62728", ls="--", lw=1.8, label=f"average {avg['center_date'].date()}")
    ax.set_yticks(ys)
    ax.set_yticklabels(time_df["model"])
    ax.set_title("下一轮 peak 时间模型：减半节奏 / 顶到顶 / 底到顶")
    ax.set_xlabel("日期")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="lower right", frameon=True)
    common(ax)
    save(fig, "model_time_next_peak.png")


def plot_price(price_df: pd.DataFrame, avg: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    xs = np.arange(len(price_df))
    ax.errorbar(xs, price_df["center_price"], yerr=[price_df["center_price"] - price_df["lo"], price_df["hi"] - price_df["center_price"]], fmt="o-", capsize=5, lw=2)
    ax.axhline(avg["center_price"], color="#d62728", ls="--", lw=1.8, label=f"average {money(avg['center_price'])}")
    ax.axhspan(avg["lo_price"], avg["hi_price"], color="#d62728", alpha=0.10, label=f"weighted band {money(avg['lo_price'])}~{money(avg['hi_price'])}")
    for i, r in price_df.iterrows():
        ax.text(i, r["center_price"] * 1.035, money(r["center_price"]), ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(price_df["model"], rotation=10, ha="right")
    ax.set_title("下一轮 peak 价格模型：log上涨开方退火为主")
    ax.set_ylabel("预测 peak 价格（美元）")
    ax.yaxis.set_major_formatter(lambda x, _: money(x))
    ax.legend(loc="upper left", frameon=True)
    common(ax)
    save(fig, "model_price_next_peak.png")


def plot_average(df: pd.DataFrame, bottom: dict[str, float], avg: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.0))
    view = df.loc["2024-01-01":].copy()
    ax.plot(view.index, view["price"], lw=2.2, label="BTC actual price")
    bottom_date = pd.Timestamp("2026-10-22")
    ax.scatter([bottom_date], [bottom["center"]], s=100, color="#2ca02c", zorder=5, label=f"bottom anchor {money(bottom['center'])}")
    ax.scatter([avg["center_date"]], [avg["center_price"]], s=130, color="#d62728", zorder=6, label=f"next peak avg {avg['center_date'].date()} / {money(avg['center_price'])}")
    ax.axvspan(avg["lo_date"], avg["hi_date"], color="#d62728", alpha=0.10, label="time window")
    ax.axhspan(avg["lo_price"], avg["hi_price"], color="#d62728", alpha=0.08, label="price band")
    ax.plot([bottom_date, avg["center_date"]], [bottom["center"], avg["center_price"]], color="#d62728", ls="--", lw=1.8)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda x, _: money(x))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_title("Average：本轮底部锚点 → 下一轮 peak")
    ax.set_xlabel("日期")
    ax.set_ylabel("BTC 价格（美元，对数轴）")
    ax.legend(loc="upper left", frameon=True)
    common(ax)
    save(fig, "average_next_peak_trend.png")


def write_summary(bottom, next_halving, time_df, price_df, avg) -> None:
    time_df.to_csv(TABLES / "time_models.csv", index=False, encoding="utf-8-sig")
    price_df.to_csv(TABLES / "price_models.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([bottom]).to_csv(TABLES / "bottom_anchor.csv", index=False)
    pd.DataFrame([next_halving]).to_csv(TABLES / "next_halving_estimate.csv", index=False)
    pd.DataFrame([avg]).to_csv(TABLES / "average_next_peak.csv", index=False)
    lines = [
        "# Next Peak Upside Forecast",
        "",
        f"- Bottom anchor center: {money(bottom['center'])}",
        f"- Next halving estimate: {pd.Timestamp(next_halving['center']).date()}",
        f"- Peak time average: {avg['lo_date'].date()} -> {avg['hi_date'].date()}, center {avg['center_date'].date()}",
        f"- Peak price average: {money(avg['center_price'])}, band {money(avg['lo_price'])} -> {money(avg['hi_price'])}",
        "",
        "The main price model is log upside compressed by sqrt volatility decay.",
    ]
    (OUT / "NEXT_PEAK_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    set_cn_font()
    df = load_price()
    bottom = bottom_anchor()
    next_halving = forecast_halving()
    tdf = time_models(next_halving)
    pdf = price_models(df, bottom)
    avg = weighted_result(tdf, pdf)
    plot_time(tdf, avg)
    plot_price(pdf, avg)
    plot_average(df, bottom, avg)
    write_summary(bottom, next_halving, tdf, pdf, avg)
    print(OUT)
    print("bottom", bottom)
    print("halving", next_halving)
    print("avg", avg)


if __name__ == "__main__":
    main()
