from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
PNG_DIR = REPORT_DIR / "png"
SUMMARY_CSV = ROOT / "analysis_runs" / "2026-06-04_segment_experiments" / "segment_cycle_samples_v19_summary.csv"
PRICE_CSV = ROOT / "data" / "btc_merged_daily.csv"


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def load_price_series() -> pd.Series:
    df = pd.read_csv(PRICE_CSV, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV, parse_dates=["left_anchor", "right_anchor"])
    # Formal report anchor: 2025 peak is 2025-08-12. The old 2025-10-05 branch
    # was an exploratory hypothesis and must not enter the parent report.
    df = df[~((df["pair"] == "2021->2025") & (df["right_anchor"] != pd.Timestamp("2025-08-12")))].copy()
    df["window_label"] = df.apply(lambda r: f"-{int(r.pre_days)} / +{int(r.post_days)}", axis=1)
    df["anchor_label"] = df["right_anchor"].dt.strftime("%Y-%m-%d")
    order_key = []
    for _, row in df.iterrows():
        pair_rank = 0 if row["pair"] == "2021->2025" else 1
        anchor_rank = 0 if row["anchor_label"] == "2025-08-12" else 1
        window_rank = {
            (0, 500): 0,
            (183, 183): 1,
            (365, 0): 2,
            (720, 0): 3,
        }.get((int(row["pre_days"]), int(row["post_days"])), 9)
        order_key.append((pair_rank, anchor_rank, window_rank))
    df["_order"] = order_key
    df = df.sort_values("_order").reset_index(drop=True)
    df.drop(columns=["_order"]).to_csv(REPORT_DIR / "peak_structure_report_summary.csv", index=False)
    return df


def window_curve(series: pd.Series, anchor: pd.Timestamp, pre_days: int, post_days: int) -> pd.DataFrame:
    start = anchor - pd.Timedelta(days=int(pre_days))
    end = anchor + pd.Timedelta(days=int(post_days))
    sub = series.loc[(series.index >= start) & (series.index <= end)].dropna()
    if anchor not in series.index:
        raise ValueError(f"missing anchor {anchor.date()}")
    anchor_price = float(series.loc[anchor])
    rel_day = (sub.index - anchor).days.astype(float)
    return pd.DataFrame(
        {
            "date": sub.index,
            "rel_day": rel_day,
            "log_norm": np.log(sub.astype(float).to_numpy() / anchor_price),
        }
    )


def make_overlay_grid(summary: pd.DataFrame, series: pd.Series) -> Path:
    out = PNG_DIR / "peak_structure_all_windows_grid.png"
    fig, axes = plt.subplots(4, 2, figsize=(14, 15.5), dpi=180, constrained_layout=True)
    axes = axes.ravel()

    for i, row in summary.iterrows():
        ax = axes[i]
        pre = int(row["pre_days"])
        post = int(row["post_days"])
        left = window_curve(series, row["left_anchor"], pre, post)
        right = window_curve(series, row["right_anchor"], pre, post)

        left_x = left["rel_day"] * float(row["time_median"]) + float(row["shift_median"])
        left_y = left["log_norm"] * float(row["amp_median"])
        right_x = right["rel_day"]
        right_y = right["log_norm"]

        if row["pair"] == "2021->2025":
            left_color = "#2563eb"
            right_color = "#0f766e"
            face = "#f0fdfa"
        else:
            left_color = "#7c3aed"
            right_color = "#b45309"
            face = "#fff7ed"

        ax.set_facecolor(face)
        ax.plot(left_x, left_y, color=left_color, lw=1.65, label=f"{row['pair'].split('->')[0]} 缩放后")
        ax.plot(right_x, right_y, color=right_color, lw=1.9, label=f"{row['pair'].split('->')[1]} 目标窗口")
        ax.axvline(0, color="#111827", lw=0.8, alpha=0.45)
        ax.axhline(0, color="#111827", lw=0.8, alpha=0.25)
        ax.grid(True, color="#cbd5e1", alpha=0.45, lw=0.6)

        x_min = min(-pre, left_x.min(), right_x.min())
        x_max = max(post, left_x.max(), right_x.max())
        ax.set_xlim(x_min, x_max)
        y_min = min(left_y.min(), right_y.min())
        y_max = max(left_y.max(), right_y.max())
        pad = max(0.06, (y_max - y_min) * 0.14)
        ax.set_ylim(y_min - pad, y_max + pad)

        title = f"{i + 1}. {row['pair']}  peak {row['anchor_label']}  窗口 {row['window_label']}"
        subtitle = (
            f"amp {row['amp_median']:.3f}  time {row['time_median']:.3f}  "
            f"shift {row['shift_median']:+.0f}d  RMSE {row['rmse_median']:.4f}"
        )
        ax.set_title(title + "\n" + subtitle, fontsize=9.5, weight="bold", pad=8)
        ax.tick_params(labelsize=8)
        if i >= 6:
            ax.set_xlabel("距目标 peak 的天数", fontsize=8.5)
        ax.set_ylabel("log 价格相对 peak", fontsize=8.5)
        ax.legend(loc="best", fontsize=7.5, frameon=True, framealpha=0.82)

    for ax in axes[len(summary) :]:
        ax.axis("off")

    fig.suptitle(
        f"Peak 结构样本：{len(summary)} 个有效窗口组全部展开（2025 peak 统一锚定 2025-08-12）",
        fontsize=16,
        weight="bold",
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_metrics_map(summary: pd.DataFrame) -> Path:
    out = PNG_DIR / "peak_structure_window_metrics.png"
    labels = [
        f"{r.pair} | {r.anchor_label} | {r.window_label} | n={int(r.n)}"
        for r in summary.itertuples(index=False)
    ]
    y = np.arange(len(summary))
    colors = ["#0f766e" if p == "2021->2025" else "#7c3aed" for p in summary["pair"]]

    fig, axes = plt.subplots(1, 4, figsize=(14.5, 7.4), dpi=180, sharey=True, constrained_layout=True)
    fig.suptitle(
        f"Peak 窗口参数总览：{len(summary)} 个有效窗口组，旧分支已剔除",
        fontsize=16,
        weight="bold",
    )

    specs = [
        ("amp_median", "幅度比例 amp", 0.35, 0.82, [0.58, 0.60], "2021->2025 主收敛带 0.58-0.60"),
        ("time_median", "时间比例 time", 0.88, 1.04, [0.98, 1.01], "多数贴近 1.00"),
        ("shift_median", "平移 shift(days)", -85, 30, [-13, 2], "常见 -13d 到 +2d"),
        ("rmse_median", "RMSE 中位", 0, 0.55, None, "越低越贴合"),
    ]

    for ax, (col, title, xmin, xmax, band, note) in zip(axes, specs):
        ax.scatter(summary[col], y, s=80, c=colors, edgecolor="white", linewidth=1.2, zorder=3)
        for value, yy in zip(summary[col], y):
            if col == "shift_median":
                text = f"{value:+.0f}d"
            else:
                text = f"{value:.3f}" if col != "rmse_median" else f"{value:.4f}"
            ax.text(value, yy + 0.26, text, ha="center", va="bottom", fontsize=8.2, color="#111827")

        if col == "time_median":
            ax.axvline(1.0, color="#111827", lw=1.1, alpha=0.65)
        if col == "shift_median":
            ax.axvline(0, color="#111827", lw=1.1, alpha=0.65)
        if band is not None:
            ax.axvspan(band[0], band[1], color="#22c55e", alpha=0.13, zorder=0)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.7, len(summary) - 0.3)
        ax.invert_yaxis()
        ax.grid(True, axis="x", color="#cbd5e1", alpha=0.65)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel(note, fontsize=8.5)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8.7)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0f766e", label="2021->2025 当前分支", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7c3aed", label="2017->2021 历史校验", markersize=8),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncols=2, frameon=False, fontsize=9)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    setup_matplotlib()
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    series = load_price_series()
    grid = make_overlay_grid(summary, series)
    metrics = make_metrics_map(summary)
    print(f"grid={grid.name}")
    print(f"metrics={metrics.name}")


if __name__ == "__main__":
    main()
