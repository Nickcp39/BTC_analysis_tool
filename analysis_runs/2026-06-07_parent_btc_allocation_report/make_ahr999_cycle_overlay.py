from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
PNG_DIR = REPORT_DIR / "png"
PRICE_CSV = ROOT / "data" / "btc_merged_daily.csv"
sys.path.insert(0, str(ROOT))

from signals.indicators import compute_ahr999

# (label, peak, color, linewidth, zorder, alpha, is_hero)
PEAKS = [
    ("2017（参照）", pd.Timestamp("2017-12-16"), "#b3bccb", 1.5, 2, 0.85, False),
    ("2021（参照）", pd.Timestamp("2021-11-08"), "#7c8595", 1.7, 2, 0.9, False),
    ("2025（本轮·重点看）", pd.Timestamp("2025-08-12"), "#0f766e", 3.8, 6, 1.0, True),
]
LATEST = pd.Timestamp("2026-06-05")


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def make_frame() -> pd.DataFrame:
    price = pd.read_csv(PRICE_CSV, parse_dates=["date"])[["date", "price"]]
    ahr = compute_ahr999(price)
    ahr.to_csv(REPORT_DIR / "ahr999_recalc_to_2026-06-05.csv", index=False)
    return ahr.dropna(subset=["ahr999"]).copy()


def cycle_slice(ahr: pd.DataFrame, peak: pd.Timestamp, latest_or_days: pd.Timestamp | int) -> pd.DataFrame:
    if isinstance(latest_or_days, int):
        end = peak + pd.Timedelta(days=latest_or_days)
    else:
        end = latest_or_days
    sub = ahr[(ahr["date"] >= peak) & (ahr["date"] <= end)].copy()
    sub["days_after_peak"] = (sub["date"] - peak).dt.days
    return sub


def main() -> None:
    setup_matplotlib()
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    ahr = make_frame()

    # Keep the historical windows long enough to compare with the current 2025 window.
    latest_days = int((LATEST - pd.Timestamp("2025-08-12")).days)

    fig, ax = plt.subplots(figsize=(13.4, 6.2), dpi=180)
    ax.axhspan(0, 0.30, color="#bbf7d0", alpha=0.55, label="历史深底区 AHR999 < 0.30", zorder=0)
    ax.axhspan(0.30, 0.45, color="#fde68a", alpha=0.42, label="深度价值区 0.30 - 0.45", zorder=0)
    ax.axhline(0.30, color="#16a34a", ls="--", lw=1.0, alpha=0.85, zorder=1)
    ax.axhline(0.45, color="#d97706", ls="--", lw=1.0, alpha=0.85, zorder=1)

    rows = []
    for label, peak, color, lw, zorder, alpha, is_hero in PEAKS:
        window = LATEST if is_hero else (latest_days + 160)
        sub = cycle_slice(ahr, peak, window)
        ax.plot(
            sub["days_after_peak"], sub["ahr999"],
            color=color, lw=lw, alpha=alpha, zorder=zorder,
            label=label, solid_capstyle="round",
        )

        current_end = LATEST if is_hero else sub["date"].max()
        usable = ahr[(ahr["date"] >= peak) & (ahr["date"] <= current_end)].copy()
        min_row = usable.loc[usable["ahr999"].idxmin()]
        latest_row = usable.iloc[-1]
        min_day = int((min_row["date"] - peak).days)
        latest_day = int((latest_row["date"] - peak).days)
        rows.append(
            {
                "cycle": label,
                "peak_date": peak.date().isoformat(),
                "latest_or_bottom_date": latest_row["date"].date().isoformat(),
                "days_after_peak": latest_day,
                "min_date": min_row["date"].date().isoformat(),
                "min_days_after_peak": min_day,
                "min_ahr999": float(min_row["ahr999"]),
                "last_ahr999": float(latest_row["ahr999"]),
                "last_price": float(latest_row["price"]),
            }
        )
        if is_hero:
            ax.scatter([min_day], [min_row["ahr999"]], s=80, color=color, zorder=8,
                       edgecolor="white", linewidth=1.6)
            ax.annotate(
                f"2025 最低 {min_row['ahr999']:.3f}\n（第 {min_day} 天就探到深底，\n比前两轮约第 365 天早得多）",
                xy=(min_day, min_row["ahr999"]),
                xytext=(min_day + 22, min_row["ahr999"] + 0.34),
                fontsize=9.6, color="#0f766e", weight="bold",
                arrowprops={"arrowstyle": "->", "color": "#0f766e", "lw": 1.3},
            )
        else:
            ax.scatter([min_day], [min_row["ahr999"]], s=20, color=color, zorder=3, alpha=0.8)
            ax.annotate(
                f"{label[:4]} 最低 {min_row['ahr999']:.3f}",
                xy=(min_day, min_row["ahr999"]),
                xytext=(min_day + 6, min_row["ahr999"] + 0.045),
                fontsize=7.5, color="#64748b",
            )

    current = rows[-1]
    ax.scatter([current["days_after_peak"]], [current["last_ahr999"]], s=95, color="#b91c1c", zorder=9,
               edgecolor="white", linewidth=1.6)
    ax.annotate(
        f"当前 2026-06-05\n第 {current['days_after_peak']} 天 / AHR {current['last_ahr999']:.3f}",
        xy=(current["days_after_peak"], current["last_ahr999"]),
        xytext=(current["days_after_peak"] + 18, current["last_ahr999"] + 0.22),
        fontsize=10,
        weight="bold",
        color="#b91c1c",
        arrowprops={"arrowstyle": "->", "color": "#b91c1c", "lw": 1.2},
    )

    ax.set_title("三轮顶部之后的「便宜程度」对比（AHR999 估值指标）", fontsize=14, weight="bold")
    ax.text(
        0.012,
        0.965,
        "注：2017 / 2021 / 2025 三轮都按各自价格顶部对齐；AHR999 越低越便宜，0.30 以下＝历史深底区。",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#334155",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1", "alpha": 0.85},
    )
    ax.set_xlabel("距价格顶部的天数")
    ax.set_ylabel("AHR999")
    ax.set_xlim(0, max(latest_days + 165, 460))
    ax.set_ylim(0, 1.6)
    ax.grid(True, color="#e2e8f0", alpha=0.7)
    ax.legend(loc="upper right", fontsize=8)

    out = PNG_DIR / "ahr999_cycle_overlay_peak_aligned.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(REPORT_DIR / "ahr999_cycle_overlay_summary.csv", index=False)
    print(f"chart={out.name}")
    print(f"summary=ahr999_cycle_overlay_summary.csv")


if __name__ == "__main__":
    main()
