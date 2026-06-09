# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
PNG_DIR = OUT_DIR / "png"
PNG_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
fm.fontManager.addfont(FONT_PATH)
FONT_NAME = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["font.sans-serif"] = [FONT_NAME, "SimHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

PEAKS = [
    {"cycle": "2017", "peak": pd.Timestamp("2017-12-16"), "color": "#2563eb"},
    {"cycle": "2021", "peak": pd.Timestamp("2021-11-08"), "color": "#0f766e"},
    {"cycle": "2025", "peak": pd.Timestamp("2025-08-12"), "color": "#dc2626"},
]
WINDOW_BEFORE = 365
WINDOW_AFTER = 365


def fetch_blockchain_chart(slug: str) -> pd.DataFrame:
    url = f"https://api.blockchain.info/charts/{slug}?timespan=all&format=json&sampled=false"
    payload = requests.get(url, timeout=45).json()
    rows = payload["values"]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["x"], unit="s").dt.normalize()
    df = df.rename(columns={"y": "value"})[["date", "value"]]
    df = df.sort_values("date").reset_index(drop=True)
    df["ma30"] = df["value"].rolling(30, min_periods=7).mean()
    df.to_csv(OUT_DIR / f"blockchain_{slug}_all.csv", index=False)
    return df


def value_at_or_before(df: pd.DataFrame, target: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    hit = df.loc[df["date"] <= target].dropna(subset=["ma30"]).iloc[-1]
    return hit["date"], float(hit["ma30"])


def cycle_window(df: pd.DataFrame, peak: pd.Timestamp) -> pd.DataFrame:
    start = peak - pd.Timedelta(days=WINDOW_BEFORE)
    end = min(peak + pd.Timedelta(days=WINDOW_AFTER), df["date"].max())
    out = df.loc[(df["date"] >= start) & (df["date"] <= end)].copy()
    out["day"] = (out["date"] - peak).dt.days
    return out


def fmt_wan(v: float) -> str:
    return f"{v / 10000:.1f} 万"


def pct(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.1%}"


def make_cycle_summary(active: pd.DataFrame, txs: pd.DataFrame, compare_day: int) -> pd.DataFrame:
    rows: list[dict] = []
    for item in PEAKS:
        cycle = item["cycle"]
        peak = item["peak"]
        target = peak + pd.Timedelta(days=compare_day)
        active_peak_date, active_peak = value_at_or_before(active, peak)
        active_target_date, active_target = value_at_or_before(active, target)
        tx_peak_date, tx_peak = value_at_or_before(txs, peak)
        tx_target_date, tx_target = value_at_or_before(txs, target)
        rows.append(
            {
                "周期": cycle,
                "peak锚点": peak.date().isoformat(),
                "同相位天数": compare_day,
                "观察日": active_target_date.date().isoformat(),
                "活跃地址_peak_30DMA": active_peak,
                "活跃地址_同相位_30DMA": active_target,
                "活跃地址_变化": active_target / active_peak - 1.0,
                "确认交易_peak_30DMA": tx_peak,
                "确认交易_同相位_30DMA": tx_target,
                "确认交易_变化": tx_target / tx_peak - 1.0,
                "活跃地址_peak_日期": active_peak_date.date().isoformat(),
                "确认交易_peak_日期": tx_peak_date.date().isoformat(),
                "确认交易_观察日": tx_target_date.date().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def plot_cycle_lines(ax: plt.Axes, df: pd.DataFrame, metric_title: str, ylabel: str, compare_day: int) -> None:
    for item in PEAKS:
        cycle = item["cycle"]
        peak = item["peak"]
        color = item["color"]
        part = cycle_window(df, peak)
        ax.plot(part["day"], part["ma30"] / 10000, color=color, lw=2.1, label=f"{cycle} cycle")
    ax.axvline(0, color="#0f172a", lw=1.2, ls="--", alpha=0.65)
    ax.axvline(compare_day, color="#f97316", lw=1.1, ls=":", alpha=0.9)
    ax.set_xlim(-WINDOW_BEFORE, WINDOW_AFTER)
    ax.set_title(metric_title, fontsize=13.5, fontweight="bold")
    ax.set_xlabel("距离价格 peak 的天数")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", fontsize=9)


def plot_phase_bars(ax: plt.Axes, summary: pd.DataFrame, metric: str, title: str) -> None:
    colors = [item["color"] for item in PEAKS]
    y = summary[f"{metric}_变化"].tolist()
    x = range(len(summary))
    ax.bar(x, [v * 100 for v in y], color=colors, alpha=0.88)
    ax.axhline(0, color="#0f172a", lw=1.0)
    ax.set_xticks(list(x), summary["周期"].tolist())
    ax.set_ylabel("同相位变化")
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    for i, v in enumerate(y):
        va = "bottom" if v >= 0 else "top"
        offset = 1.2 if v >= 0 else -1.2
        ax.text(i, v * 100 + offset, pct(v), ha="center", va=va, fontsize=10, fontweight="bold")


def plot_lth_panel(ax: plt.Axes, lth: pd.DataFrame) -> tuple[float, float]:
    lth_start = lth.iloc[0]
    lth_latest = lth.iloc[-1]
    lth_supply_delta = float(lth_latest["lth_supply_btc"] - lth_start["lth_supply_btc"])
    lth_ratio_delta = float(lth_latest["lth_ratio"] - lth_start["lth_ratio"])

    ax.set_title("低频资本证据：LTH 公开基准日至最新仍增加", fontsize=12.5, fontweight="bold")
    ax.axis("off")
    green = "#0f766e"
    gray = "#64748b"
    y0, y1 = 0.68, 0.34
    x0, x1 = 0.16, 0.82

    ax.plot([x0, x1], [y0, y0 + 0.04], color=green, lw=4, solid_capstyle="round")
    ax.scatter([x0, x1], [y0, y0 + 0.04], s=78, color=green)
    ax.text(x0, y0 - 0.09, f"{lth_start['date'].date()}\n{lth_start['lth_supply_btc'] / 1e6:.2f}M BTC", ha="center", fontsize=9.6, color=gray)
    ax.text(x1, y0 + 0.08, f"{lth_latest['date'].date()}\n{lth_latest['lth_supply_btc'] / 1e6:.2f}M BTC", ha="center", fontsize=9.6, color=green, fontweight="bold")
    ax.text(0.50, y0 + 0.12, f"+{lth_supply_delta / 1e6:.2f}M BTC", ha="center", fontsize=11.5, color=green, fontweight="bold")

    ax.plot([x0, x1], [y1, y1 + 0.04], color=green, lw=4, solid_capstyle="round")
    ax.scatter([x0, x1], [y1, y1 + 0.04], s=78, color=green)
    ax.text(x0, y1 - 0.09, f"{lth_start['lth_ratio']:.1%}", ha="center", fontsize=9.6, color=gray)
    ax.text(x1, y1 + 0.08, f"{lth_latest['lth_ratio']:.1%}", ha="center", fontsize=9.6, color=green, fontweight="bold")
    ax.text(0.50, y1 + 0.12, f"+{lth_ratio_delta * 100:.1f} 个百分点", ha="center", fontsize=11.5, color=green, fontweight="bold")

    ax.text(
        0.5,
        0.06,
        "LTH 本地只有公开基准点，不能伪造成多年曲线；这里作为低频持有证据。",
        ha="center",
        fontsize=9.6,
        color="#334155",
    )
    return lth_supply_delta, lth_ratio_delta


def main() -> None:
    active = fetch_blockchain_chart("n-unique-addresses")
    txs = fetch_blockchain_chart("n-transactions")
    lth = pd.read_csv(ROOT / "data" / "lth_metrics.csv", parse_dates=["date"])

    current_date = min(active["date"].max(), txs["date"].max())
    compare_day = int((current_date - pd.Timestamp("2025-08-12")).days)
    compare_day = min(compare_day, WINDOW_AFTER)
    summary = make_cycle_summary(active, txs, compare_day)
    summary.to_csv(OUT_DIR / "real_user_cycle_phase_summary.csv", index=False)

    fig = plt.figure(figsize=(14.0, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.33, wspace=0.22)

    ax_active = fig.add_subplot(gs[0, 0])
    plot_cycle_lines(ax_active, active, "多周期对比：活跃地址 30日均线", "万地址", compare_day)
    ax_tx = fig.add_subplot(gs[0, 1])
    plot_cycle_lines(ax_tx, txs, "多周期对比：确认交易数 30日均线", "万笔交易", compare_day)

    ax_phase = fig.add_subplot(gs[1, 0])
    ax_phase.set_title(f"同相位 day +{compare_day}：峰后高频活动变化", fontsize=12.5, fontweight="bold")
    width = 0.36
    x = range(len(summary))
    active_changes = [v * 100 for v in summary["活跃地址_变化"]]
    tx_changes = [v * 100 for v in summary["确认交易_变化"]]
    ax_phase.bar([i - width / 2 for i in x], active_changes, width=width, color="#2563eb", label="活跃地址")
    ax_phase.bar([i + width / 2 for i in x], tx_changes, width=width, color="#dc2626", label="确认交易")
    ax_phase.axhline(0, color="#0f172a", lw=1.0)
    ax_phase.set_xticks(list(x), summary["周期"].tolist())
    ax_phase.set_ylabel("相对各自 peak 当日 30DMA")
    ax_phase.grid(axis="y", alpha=0.2)
    ax_phase.legend(fontsize=9)
    for i, row in summary.iterrows():
        for dx, key in [(-width / 2, "活跃地址_变化"), (width / 2, "确认交易_变化")]:
            v = float(row[key])
            va = "bottom" if v >= 0 else "top"
            offset = 1.1 if v >= 0 else -1.1
            ax_phase.text(i + dx, v * 100 + offset, pct(v), ha="center", va=va, fontsize=9.2, fontweight="bold")

    ax_lth = fig.add_subplot(gs[1, 1])
    lth_supply_delta, lth_ratio_delta = plot_lth_panel(ax_lth, lth)

    fig.suptitle(
        "真实用户结构：跨周期看高频链上活动，单独看低频长期持有资本",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        f"注：x=0 为每轮价格 peak；橙色虚线为 2025 peak 后当前同相位 day +{compare_day}。高频链上活动只代表转账/热钱，不等于 BTC 全部真实用户。",
        ha="center",
        fontsize=10.2,
        color="#334155",
    )
    fig.savefig(PNG_DIR / "real_user_structure_change.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    lth_start = lth.iloc[0]
    lth_latest = lth.iloc[-1]
    structure = pd.DataFrame(
        [
            {
                "口径": "long holder 供应",
                "最新日期": lth_latest["date"].date().isoformat(),
                "最新": float(lth_latest["lth_supply_btc"]),
                "基准日期": lth_start["date"].date().isoformat(),
                "基准": float(lth_start["lth_supply_btc"]),
                "变化": lth_supply_delta / float(lth_start["lth_supply_btc"]),
            },
            {
                "口径": "long holder 占流通供应比例",
                "最新日期": lth_latest["date"].date().isoformat(),
                "最新": float(lth_latest["lth_ratio"]),
                "基准日期": lth_start["date"].date().isoformat(),
                "基准": float(lth_start["lth_ratio"]),
                "变化": lth_ratio_delta,
            },
        ]
    )
    structure.to_csv(OUT_DIR / "real_user_structure_change_summary.csv", index=False)
    print("wrote real_user_cycle_phase_summary.csv")
    print("wrote real_user_structure_change_summary.csv")
    print(PNG_DIR / "real_user_structure_change.png")


if __name__ == "__main__":
    main()
