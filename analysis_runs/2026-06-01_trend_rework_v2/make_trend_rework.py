from __future__ import annotations

import math
from dataclasses import dataclass
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

BTC_BIRTH = pd.Timestamp("2009-01-03")
PEAK_2017 = pd.Timestamp("2017-12-19")
PEAK_2021 = pd.Timestamp("2021-11-10")
PEAK_2025_ACTUAL = pd.Timestamp("2025-10-05")
BOTTOM_2018 = pd.Timestamp("2018-12-15")
BOTTOM_2022 = pd.Timestamp("2022-11-21")

VOL_LEVEL = {"2017": 9.0, "2021": 3.0, "2025": 1.0}
VOL_ALPHA = 0.5
K_AHR = 5.84
B_AHR = -17.01


def set_cn_font() -> None:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\ARIALUNI.TTF",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            font = fm.FontProperties(fname=candidate)
            plt.rcParams["font.family"] = font.get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def money(x: float) -> str:
    return f"${x:,.0f}"


def ahr999_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_days"] = (out.index - BTC_BIRTH).days.astype(float)
    out["estimate_price"] = 10 ** (K_AHR * np.log10(out["age_days"]) + B_AHR)
    out["gma200"] = np.exp(np.log(out["price"]).rolling(200).mean())
    out["ahr999"] = (out["price"] / out["gma200"]) * (out["price"] / out["estimate_price"])
    return out


def projected_200d_tail(date: pd.Timestamp, price: float, hist: pd.DataFrame) -> list[float]:
    latest_date = hist.index.max()
    latest_price = float(hist.loc[latest_date, "price"])
    if date <= latest_date:
        tail = hist.loc[:date, "price"].tail(199).tolist() + [price]
        return tail[-200:]

    future_dates = pd.date_range(latest_date + pd.Timedelta(days=1), date, freq="D")
    future_prices = np.linspace(latest_price, price, len(future_dates) + 1)[1:]
    combined = pd.concat(
        [
            hist.loc[:latest_date, "price"],
            pd.Series(future_prices, index=future_dates),
        ]
    )
    tail = combined.tail(199).tolist() + [price]
    return tail[-200:]


def ahr_at_price(date: pd.Timestamp, price: float, hist: pd.DataFrame) -> float:
    age = (date - BTC_BIRTH).days
    estimate = 10 ** (K_AHR * math.log10(age) + B_AHR)
    tail = projected_200d_tail(date, price, hist)
    gma200 = math.exp(float(np.log(tail).mean()))
    return (price / gma200) * (price / estimate)


def solve_price_for_ahr(date: pd.Timestamp, target: float, hist: pd.DataFrame) -> float:
    lo, hi = 10000.0, 180000.0
    for _ in range(90):
        mid = (lo + hi) / 2
        if ahr_at_price(date, mid, hist) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cycle_drawdown(df: pd.DataFrame, peak_date: pd.Timestamp, end_days: int = 460, observed_only: bool = False) -> pd.DataFrame:
    peak_price = float(df.loc[peak_date, "price"])
    end_date = peak_date + pd.Timedelta(days=end_days)
    if observed_only:
        end_date = min(end_date, df.index.max())
    dates = pd.date_range(peak_date, end_date, freq="D")
    s = df.reindex(dates)["price"].ffill()
    rel_day = (s.index - peak_date).days
    return pd.DataFrame(
        {
            "date": s.index,
            "rel_day": rel_day,
            "price": s.values,
            "dd_pct": (s.values / peak_price - 1.0) * 100.0,
            "log_dd": np.log(s.values / peak_price),
        }
    ).dropna()


@dataclass
class Model:
    name: str
    center_date: pd.Timestamp
    date_lo: pd.Timestamp
    date_hi: pd.Timestamp
    center_price: float
    price_lo: float
    price_hi: float
    weight: float
    note: str


def load_models() -> list[Model]:
    source = ROOT / "analysis_runs" / "2026-06-01_bottom_conclusion_average" / "tables" / "bottom_model_average_inputs.csv"
    rows = pd.read_csv(source)
    return [
        Model(
            name=str(r["model"]),
            center_date=pd.Timestamp(r["center_date"]),
            date_lo=pd.Timestamp(r["date_lo"]),
            date_hi=pd.Timestamp(r["date_hi"]),
            center_price=float(r["center_price"]),
            price_lo=float(r["price_lo"]),
            price_hi=float(r["price_hi"]),
            weight=float(r["weight"]),
            note=str(r["note"]),
        )
        for _, r in rows.iterrows()
    ]


def common_style(ax) -> None:
    ax.grid(True, which="both", ls=":", lw=0.7, alpha=0.45)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(PNG / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_post_peak_clock(df: pd.DataFrame, models: list[Model]) -> None:
    c17 = cycle_drawdown(df, PEAK_2017)
    c21 = cycle_drawdown(df, PEAK_2021)
    c25 = cycle_drawdown(df, PEAK_2025_ACTUAL, end_days=420, observed_only=True)
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.plot(c25["rel_day"], c25["dd_pct"], lw=2.4, label="2025 当前峰后路径（实际）")
    ax.plot(c21["rel_day"], c21["dd_pct"], lw=1.9, alpha=0.82, label="2021 峰后真实路径")
    ax.plot(c17["rel_day"], c17["dd_pct"], lw=1.9, alpha=0.82, label="2017 峰后真实路径")
    ax.axvline(0, color="#1f77b4", ls="--", lw=1.8, alpha=0.8)
    ax.axvspan(363, 376, color="#d62728", alpha=0.12, label="历史峰→底窗口 363~376 天")
    avg_date = pd.Timestamp("2026-10-22")
    avg_rel = (avg_date - PEAK_2025_ACTUAL).days
    ax.axvline(avg_rel, color="#d62728", ls="--", lw=1.8, label=f"综合中心 {avg_date.date()}（+{avg_rel}天）")
    ax.scatter([(df.index.max() - PEAK_2025_ACTUAL).days], [float(c25.iloc[-1]["dd_pct"])], s=70, color="#1f77b4", zorder=5)
    ax.set_title("模型1：峰后时间钟 - 2025 已进入历史熊底前段")
    ax.set_xlabel("距本轮峰值天数（峰值=0）")
    ax.set_ylabel("相对峰值涨跌（%）")
    ax.set_xlim(-10, 430)
    ax.set_ylim(-88, 8)
    ax.legend(loc="lower left", frameon=True)
    common_style(ax)
    save(fig, "model_1_post_peak_time_clock.png")


def plot_post_peak_scaled_coefficient(df: pd.DataFrame) -> dict[str, float]:
    c17 = cycle_drawdown(df, PEAK_2017, end_days=600)
    c21 = cycle_drawdown(df, PEAK_2021, end_days=600)
    c25 = cycle_drawdown(df, PEAK_2025_ACTUAL, end_days=600, observed_only=True)

    std25 = float(c25["dd_pct"].std(ddof=0))
    std17 = float(c17["dd_pct"].std(ddof=0))
    std21 = float(c21["dd_pct"].std(ddof=0))
    post_scale_alpha = 1.0
    post_scale17 = (std25 / std17) ** post_scale_alpha
    post_scale21 = (std25 / std21) ** post_scale_alpha

    c17["dd_scaled"] = c17["dd_pct"] * post_scale17
    c21["dd_scaled"] = c21["dd_pct"] * post_scale21

    bottom17 = c17.loc[c17["date"] == BOTTOM_2018].iloc[0]
    bottom21 = c21.loc[c21["date"] == BOTTOM_2022].iloc[0]
    pred17_price = float(df.loc[PEAK_2025_ACTUAL, "price"]) * (1 + float(bottom17["dd_scaled"]) / 100)
    pred21_price = float(df.loc[PEAK_2025_ACTUAL, "price"]) * (1 + float(bottom21["dd_scaled"]) / 100)
    pred_center = float(np.average([pred17_price, pred21_price], weights=[0.85, 1.15]))

    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.plot(c25["rel_day"], c25["dd_pct"], lw=2.4, label=f"2025 实际峰后路径（std={std25:.2f}）")
    ax.plot(c21["rel_day"], c21["dd_scaled"], lw=2.0, label=f"2021 峰后×系数 {post_scale21:.2f}")
    ax.plot(c17["rel_day"], c17["dd_scaled"], lw=2.0, label=f"2017 峰后×系数 {post_scale17:.2f}")
    ax.axvline(0, color="#1f77b4", ls="--", lw=1.8, alpha=0.8)
    ax.axvspan(363, 376, color="#d62728", alpha=0.12, label="历史峰→底窗口 363~376 天")
    ax.axvline(382, color="#d62728", ls="--", lw=1.8, label="综合中心 2026-10-22（+382天）")
    ax.scatter([int(bottom21["rel_day"]), int(bottom17["rel_day"])], [float(bottom21["dd_scaled"]), float(bottom17["dd_scaled"])], s=70, zorder=5)
    ax.text(390, -58, f"系数版映射底部均值约 {money(pred_center)}", color="#d62728", fontsize=11)
    ax.set_title("模型1修正版：峰后路径 × A3 post_scale 系数（2025=1）")
    ax.set_xlabel("距本轮峰值天数（峰值=0）")
    ax.set_ylabel("相对峰值涨跌（%，历史已按峰后波动系数缩放）")
    ax.set_xlim(-10, 430)
    ax.set_ylim(-88, 8)
    ax.legend(loc="lower left", frameon=True)
    common_style(ax)
    save(fig, "model_1b_post_peak_scaled_coefficient.png")

    return {
        "std_2025_post_observed": std25,
        "std_2017_post_600d": std17,
        "std_2021_post_600d": std21,
        "post_scale17": post_scale17,
        "post_scale21": post_scale21,
        "mapped_2017_bottom_price": pred17_price,
        "mapped_2021_bottom_price": pred21_price,
        "mapped_center_price": pred_center,
    }


def plot_sqrt_log_replay(df: pd.DataFrame) -> dict[str, float]:
    c17 = cycle_drawdown(df, PEAK_2017)
    c21 = cycle_drawdown(df, PEAK_2021)
    c25 = cycle_drawdown(df, PEAK_2025_ACTUAL, end_days=420, observed_only=True)
    peak25 = float(df.loc[PEAK_2025_ACTUAL, "price"])
    scale17 = (VOL_LEVEL["2025"] / VOL_LEVEL["2017"]) ** VOL_ALPHA
    scale21 = (VOL_LEVEL["2025"] / VOL_LEVEL["2021"]) ** VOL_ALPHA
    c17["scaled_price"] = peak25 * np.exp(c17["log_dd"] * scale17)
    c21["scaled_price"] = peak25 * np.exp(c21["log_dd"] * scale21)
    b17 = float(c17.loc[c17["date"] == BOTTOM_2018, "scaled_price"].iloc[0])
    b21 = float(c21.loc[c21["date"] == BOTTOM_2022, "scaled_price"].iloc[0])
    center = np.average([b17, b21], weights=[0.85, 1.15])

    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.plot(c25["rel_day"], c25["price"], lw=2.4, label="2025 当前价格路径")
    ax.plot(c21["rel_day"], c21["scaled_price"], lw=2, label=f"2021 log回撤×开方缩放 {scale21:.3f}")
    ax.plot(c17["rel_day"], c17["scaled_price"], lw=2, label=f"2017 log回撤×开方缩放 {scale17:.3f}")
    ax.axvline(0, color="#1f77b4", ls="--", lw=1.8, alpha=0.8)
    ax.axhline(center, color="#d62728", ls="--", lw=1.7, label=f"开方波动回放均值 {money(center)}")
    ax.scatter([363, 376], [b17, b21], s=70, color=["#2ca02c", "#ff7f0e"], zorder=5)
    ax.set_title("模型2：log 回撤 × 波动率开方退火 - 把历史跌幅压到本轮尺度")
    ax.set_xlabel("距本轮峰值天数（峰值=0）")
    ax.set_ylabel("映射到 2025 峰值后的价格（美元，对数轴）")
    ax.set_xlim(-10, 430)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda x, _: money(x))
    ax.legend(loc="upper right", frameon=True)
    common_style(ax)
    save(fig, "model_2_sqrt_log_vol_replay.png")
    return {"scaled_2017_bottom": b17, "scaled_2021_bottom": b21, "center": center}


def plot_ratio_log_trend(df: pd.DataFrame) -> dict[str, float]:
    peak_prices = [float(df.loc[PEAK_2017, "price"]), float(df.loc[PEAK_2021, "price"]), float(df.loc[PEAK_2025_ACTUAL, "price"])]
    bottom_prices = [float(df.loc[BOTTOM_2018, "price"]), float(df.loc[BOTTOM_2022, "price"])]
    ratios = [bottom_prices[0] / peak_prices[0], bottom_prices[1] / peak_prices[1]]
    x_hist = np.array([0, 1], dtype=float)
    y = np.log(ratios)
    slope, intercept = np.polyfit(x_hist, y, 1)
    pred_ratio = float(np.exp(intercept + slope * 2))
    pred_price = pred_ratio * peak_prices[2]

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    xs = np.array([2017, 2021, 2025])
    vals = np.array(ratios + [pred_ratio]) * 100
    ax.plot(xs[:2], vals[:2], marker="o", lw=2.2, label="历史底/顶比例")
    ax.plot(xs[1:], vals[1:], marker="o", lw=2.2, ls="--", label="log 比例外推到本轮")
    for x, v, p in zip(xs, vals, bottom_prices + [pred_price]):
        ax.text(x, v + 1.0, f"{v:.1f}%\n{money(p)}", ha="center", va="bottom", fontsize=10)
    ax.set_title("模型3：底/顶比例的 log 趋势 - 波动收缩后跌幅比例上移")
    ax.set_xlabel("周期峰值年份")
    ax.set_ylabel("熊底价格 / 当轮峰值价格")
    ax.set_xticks(xs)
    ax.set_ylim(0, max(vals) * 1.45)
    ax.legend(loc="upper left", frameon=True)
    common_style(ax)
    save(fig, "model_3_bottom_peak_ratio_log_trend.png")
    return {"pred_ratio": pred_ratio, "pred_price": pred_price}


def plot_ahr_floor(df: pd.DataFrame) -> dict[str, float]:
    ahr = ahr999_frame(df)
    target = float(np.nanmean([ahr.loc[BOTTOM_2018, "ahr999"], ahr.loc[BOTTOM_2022, "ahr999"]]))
    dates = [pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-22"), pd.Timestamp("2026-11-05")]
    implied = [solve_price_for_ahr(d, target, df) for d in dates]

    fig, ax = plt.subplots(figsize=(13, 5.8))
    view = ahr.loc["2017-01-01":]
    ax.plot(view.index, view["ahr999"], lw=1.8, label="AHR999")
    ax.axhline(target, color="#d62728", ls="--", lw=1.8, label=f"历史底部均值 AHR≈{target:.3f}")
    ax.axhline(0.45, color="#ff7f0e", ls=":", lw=1.4, label="温和价值区 AHR=0.45")
    for d in [BOTTOM_2018, BOTTOM_2022]:
        ax.scatter([d], [float(ahr.loc[d, "ahr999"])], s=70, color="#d62728", zorder=5)
        ax.text(d, float(ahr.loc[d, "ahr999"]) + 0.05, f"{d.date()}", ha="center", fontsize=9)
    latest = ahr.dropna().iloc[-1]
    ax.scatter([latest.name], [latest["ahr999"]], s=75, color="#1f77b4", zorder=5, label=f"最新 {latest.name.date()} AHR={latest['ahr999']:.3f}")
    ax.set_title("模型4：AHR999 价值地板 - 46k 对应的 AHR 低于历史两次熊底")
    ax.set_xlabel("日期")
    ax.set_ylabel("AHR999")
    ax.set_ylim(0, 2.2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", frameon=True)
    common_style(ax)
    save(fig, "model_4_ahr999_value_floor.png")

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    ax.plot(dates, implied, marker="o", lw=2.3, label=f"AHR≈{target:.3f} 对应价格")
    ax.axhline(46000, color="#7f7f7f", ls="--", lw=1.5, label="$46,000 参考")
    for d, p in zip(dates, implied):
        ax.text(d, p + 900, f"{d.date()}\n{money(p)}", ha="center", fontsize=10)
    ax.set_title("模型4补图：若 AHR 回到历史熊底均值，价格应在 57k~58k")
    ax.set_xlabel("目标见底日期")
    ax.set_ylabel("AHR999 隐含价格（美元）")
    ax.yaxis.set_major_formatter(lambda x, _: money(x))
    ax.set_ylim(42000, 65000)
    ax.legend(loc="lower right", frameon=True)
    common_style(ax)
    save(fig, "model_4b_ahr999_implied_price_trend.png")
    return {"target_ahr": target, "center_price": implied[1], "lo": min(implied), "hi": max(implied)}


def plot_old_peak_accuracy(df: pd.DataFrame) -> None:
    actual_peak_price = float(df.loc[PEAK_2025_ACTUAL, "price"])
    time_rows = [
        ("MedianCenter", pd.Timestamp("2025-09-28"), 1.00),
        ("RegressionCenter", pd.Timestamp("2025-11-04"), 1.30),
        ("TopToTopCenter", pd.Timestamp("2025-11-01"), 1.10),
        ("FusionCenter", pd.Timestamp("2025-10-27"), 1.25),
    ]
    price_rows = [
        ("ModelA nominal 1%", 122062.0),
        ("Fusion price", 133754.0),
        ("ModelB nominal 3%", 133381.0),
        ("ModelC real base2025", 142615.0),
        ("Average peak", 141142.0),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    labels = [r[0] for r in time_rows]
    errs = [(r[1] - PEAK_2025_ACTUAL).days for r in time_rows]
    ax.axhline(0, color="#111111", lw=1)
    ax.plot(labels, errs, marker="o", lw=2.2)
    for label, err in zip(labels, errs):
        ax.text(label, err + (2 if err >= 0 else -4), f"{err:+d}天", ha="center", va="bottom" if err >= 0 else "top")
    ax.set_title("上次预测复盘：时间最准确的是 MedianCenter（比实际峰值早 7 天）")
    ax.set_ylabel("预测峰值日 - 实际峰值日（天）")
    common_style(ax)
    save(fig, "peak_time_accuracy_trend.png")

    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    labels = [r[0] for r in price_rows]
    vals = [r[1] for r in price_rows]
    err_pct = [(v / actual_peak_price - 1) * 100 for v in vals]
    ax.axhline(0, color="#111111", lw=1)
    ax.plot(labels, err_pct, marker="o", lw=2.2)
    for label, err, val in zip(labels, err_pct, vals):
        ax.text(label, err + (1 if err >= 0 else -1.6), f"{err:+.1f}%\n{money(val)}", ha="center", va="bottom" if err >= 0 else "top", fontsize=9)
    ax.set_title("上次预测复盘：价格最准确的是 ModelA nominal 1%（约 -2.1%）")
    ax.set_ylabel("预测峰值价格误差")
    common_style(ax)
    save(fig, "peak_price_accuracy_trend.png")


def plot_average_trend(df: pd.DataFrame, models: list[Model], ahr_center: float) -> dict[str, object]:
    weights = np.array([m.weight for m in models])
    center_prices = np.array([m.center_price for m in models])
    center_ord = np.array([m.center_date.toordinal() for m in models], dtype=float)
    raw_center_price = float(np.average(center_prices, weights=weights))
    raw_center_date = pd.Timestamp.fromordinal(int(round(float(np.average(center_ord, weights=weights)))))

    # AHR999 is used as a valuation guardrail instead of another equal-weight model.
    ahr_guard_weight = 1.25
    guarded_price = float(np.average([raw_center_price, ahr_center], weights=[weights.sum(), ahr_guard_weight]))
    price_lo = float(np.average([np.percentile([m.price_lo for m in models], 35), ahr_center * 0.94], weights=[1, 1]))
    price_hi = float(np.average([np.percentile([m.price_hi for m in models], 70), ahr_center * 1.13], weights=[1, 1]))

    fig, ax = plt.subplots(figsize=(13, 6.0))
    view = df.loc["2024-01-01":].copy()
    ax.plot(view.index, view["price"], lw=2.3, label="BTC 实际价格")
    for m in models:
        ax.errorbar(
            m.center_date,
            m.center_price,
            yerr=[[m.center_price - m.price_lo], [m.price_hi - m.center_price]],
            xerr=[[m.center_date - m.date_lo], [m.date_hi - m.center_date]],
            fmt="o",
            capsize=4,
            alpha=0.78,
            label=m.name.replace("_", " "),
        )
    ax.scatter([raw_center_date], [raw_center_price], s=110, color="#9467bd", zorder=6, label=f"原始多模型均值 {raw_center_date.date()} / {money(raw_center_price)}")
    ax.scatter([pd.Timestamp("2026-10-22")], [guarded_price], s=130, color="#d62728", zorder=7, label=f"AHR 修正后中心 {money(guarded_price)}")
    ax.axvspan(pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-05"), color="#d62728", alpha=0.10, label="综合时间窗口")
    ax.axhspan(price_lo, price_hi, color="#d62728", alpha=0.08, label=f"AHR修正价格区间 {money(price_lo)}~{money(price_hi)}")
    ax.set_title("Average：多模型趋势叠加 + AHR999 地板约束")
    ax.set_xlabel("日期")
    ax.set_ylabel("BTC 价格（美元，对数轴）")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(lambda x, _: money(x))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper right", fontsize=8, frameon=True, ncol=2)
    common_style(ax)
    save(fig, "average_multi_model_bottom_trend.png")

    return {
        "raw_center_date": raw_center_date.date().isoformat(),
        "raw_center_price": raw_center_price,
        "ahr_guarded_center_date": "2026-10-22",
        "ahr_guarded_center_price": guarded_price,
        "price_lo": price_lo,
        "price_hi": price_hi,
    }


def write_summary(results: dict[str, object]) -> None:
    lines = [
        "# Trend Rework Notes",
        "",
        "这版重做回旧趋势图语言：曲线、锚点、log轴、历史路径映射，不再使用卡片式模型图。",
        "",
        "## 核心结论",
        f"- 原始多模型均值：{results['average']['raw_center_date']} / {money(results['average']['raw_center_price'])}",
        f"- AHR999 地板修正后中心：{results['average']['ahr_guarded_center_date']} / {money(results['average']['ahr_guarded_center_price'])}",
        f"- AHR修正后核心价格区间：{money(results['average']['price_lo'])} ~ {money(results['average']['price_hi'])}",
        "- 时间窗口仍以 2026-10-01 → 2026-11-05 为主。",
        "",
        "## 上次 peak 预测复盘",
        "- 时间最准确：MedianCenter，2025-09-28，实际峰值 2025-10-05，误差 -7 天。",
        "- 价格最准确：ModelA nominal 1%，预测 $122,062，实际峰值 $124,720，误差约 -2.1%。",
        "",
        "## 输出图",
    ]
    for png in sorted(PNG.glob("*.png")):
        lines.append(f"- png/{png.name}")
    (OUT / "TREND_REWORK_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    set_cn_font()
    df = pd.read_csv(ROOT / "data" / "btc_merged_daily.csv", parse_dates=["date"]).set_index("date").sort_index()
    df = df[["price"]].dropna()
    models = load_models()

    results: dict[str, object] = {}
    plot_post_peak_clock(df, models)
    results["model1_coeff"] = plot_post_peak_scaled_coefficient(df)
    results["sqrt_log"] = plot_sqrt_log_replay(df)
    results["ratio"] = plot_ratio_log_trend(df)
    results["ahr"] = plot_ahr_floor(df)
    plot_old_peak_accuracy(df)
    results["average"] = plot_average_trend(df, models, float(results["ahr"]["center_price"]))

    pd.DataFrame([results["sqrt_log"]]).to_csv(TABLES / "sqrt_log_replay.csv", index=False)
    pd.DataFrame([results["model1_coeff"]]).to_csv(TABLES / "model1_post_scale_coefficients.csv", index=False)
    pd.DataFrame([results["ratio"]]).to_csv(TABLES / "ratio_log_trend.csv", index=False)
    pd.DataFrame([results["ahr"]]).to_csv(TABLES / "ahr999_floor.csv", index=False)
    pd.DataFrame([results["average"]]).to_csv(TABLES / "average_trend_result.csv", index=False)
    write_summary(results)
    print(OUT)
    print(results["average"])


if __name__ == "__main__":
    main()
