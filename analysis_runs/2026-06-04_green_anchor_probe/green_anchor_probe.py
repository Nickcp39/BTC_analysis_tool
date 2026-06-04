from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "btc_merged_daily.csv"
OUT = Path(__file__).resolve().parent

HALVINGS = {
    "2017": pd.Timestamp("2016-07-09"),
    "2021": pd.Timestamp("2020-05-11"),
    "2025": pd.Timestamp("2024-04-20"),
}
NEXT_HALVINGS = {
    "2017": pd.Timestamp("2020-05-11"),
    "2021": pd.Timestamp("2024-04-20"),
    "2025": None,
}


@dataclass
class Anchor:
    date: pd.Timestamp
    kind: str
    price: float
    persistence: int
    score: float
    median_prominence: float


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    s = df.set_index("date")["price"].sort_index().asfreq("D").ffill()
    return s


def cluster_events(events: list[dict], tolerance_days: int = 12) -> list[Anchor]:
    if not events:
        return []

    events = sorted(events, key=lambda x: x["date"])
    clusters: list[list[dict]] = []
    for event in events:
        if not clusters:
            clusters.append([event])
            continue
        center = pd.Timestamp(np.median([e["date"].value for e in clusters[-1]]))
        if abs((event["date"] - center).days) <= tolerance_days:
            clusters[-1].append(event)
        else:
            clusters.append([event])

    anchors: list[Anchor] = []
    for cluster in clusters:
        dates = [e["date"] for e in cluster]
        kinds = [e["kind"] for e in cluster]
        prominences = [e["prominence"] for e in cluster]
        kind = max(set(kinds), key=kinds.count)
        chosen_date = pd.Timestamp(np.median([d.value for d in dates])).normalize()
        price = float(cluster[0]["series"].loc[chosen_date])
        persistence = len(cluster)
        median_prominence = float(np.median(prominences))
        score = persistence * (1.0 + median_prominence)
        anchors.append(
            Anchor(
                date=chosen_date,
                kind=kind,
                price=price,
                persistence=persistence,
                score=score,
                median_prominence=median_prominence,
            )
        )
    return anchors


def prune_close_anchors(anchors: list[Anchor], min_gap_days: int = 35) -> list[Anchor]:
    anchors = sorted(anchors, key=lambda a: a.score, reverse=True)
    kept: list[Anchor] = []
    for anchor in anchors:
        if all(abs((anchor.date - old.date).days) >= min_gap_days for old in kept):
            kept.append(anchor)
    return sorted(kept, key=lambda a: a.date)


def detect_anchors(series: pd.Series) -> list[Anchor]:
    logp = np.log(series)
    events: list[dict] = []
    smooth_windows = [7, 14, 30, 60, 90]

    for window in smooth_windows:
        smoothed = logp.rolling(window, center=True, min_periods=max(3, window // 3)).mean().dropna()
        if len(smoothed) < 100:
            continue
        min_distance = max(20, int(window * 1.25))
        min_prominence = 0.08 if window <= 30 else 0.10

        peak_idx, peak_props = find_peaks(
            smoothed.values,
            distance=min_distance,
            prominence=min_prominence,
        )
        trough_idx, trough_props = find_peaks(
            -smoothed.values,
            distance=min_distance,
            prominence=min_prominence,
        )

        for idx, prom in zip(peak_idx, peak_props["prominences"]):
            events.append(
                {
                    "date": smoothed.index[int(idx)].normalize(),
                    "kind": "peak",
                    "prominence": float(prom),
                    "series": series,
                }
            )
        for idx, prom in zip(trough_idx, trough_props["prominences"]):
            events.append(
                {
                    "date": smoothed.index[int(idx)].normalize(),
                    "kind": "trough",
                    "prominence": float(prom),
                    "series": series,
                }
            )

    anchors = cluster_events(events)
    anchors = [a for a in anchors if a.persistence >= 2]
    return prune_close_anchors(anchors)


def cycle_anchor_table(series: pd.Series, anchors: list[Anchor]) -> pd.DataFrame:
    rows = []
    latest = series.index.max()
    for cycle, halving in HALVINGS.items():
        end = NEXT_HALVINGS[cycle] or latest
        cycle_anchors = [a for a in anchors if halving <= a.date <= end]
        for i, anchor in enumerate(cycle_anchors):
            prev_date = cycle_anchors[i - 1].date if i else None
            rows.append(
                {
                    "cycle": cycle,
                    "anchor_index": i + 1,
                    "date": anchor.date.date().isoformat(),
                    "days_from_halving": (anchor.date - halving).days,
                    "kind": anchor.kind,
                    "price": round(anchor.price, 2),
                    "persistence": anchor.persistence,
                    "score": round(anchor.score, 4),
                    "median_prominence_log": round(anchor.median_prominence, 4),
                    "interval_from_prev_days": None if prev_date is None else (anchor.date - prev_date).days,
                }
            )
    return pd.DataFrame(rows)


def normalized_intervals(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle, g in table.groupby("cycle"):
        intervals = g["interval_from_prev_days"].dropna().astype(float).values
        if len(intervals) < 2:
            continue
        total = intervals.sum()
        median = float(np.median(intervals))
        for i, value in enumerate(intervals, start=2):
            rows.append(
                {
                    "cycle": cycle,
                    "to_anchor_index": i,
                    "interval_days": int(value),
                    "interval_div_total": round(float(value / total), 4),
                    "interval_div_median": round(float(value / median), 4),
                }
            )
    return pd.DataFrame(rows)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    dp = np.full((len(a) + 1, len(b) + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[-1, -1] / (len(a) + len(b)))


def similarity_report(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cycles = sorted(table["cycle"].unique())
    phase = {
        cycle: g["days_from_halving"].astype(float).values
        for cycle, g in table.groupby("cycle")
        if len(g) >= 3
    }
    for i, c1 in enumerate(cycles):
        for c2 in cycles[i + 1 :]:
            if c1 not in phase or c2 not in phase:
                continue
            a = phase[c1] / max(phase[c1].max(), 1.0)
            b = phase[c2] / max(phase[c2].max(), 1.0)
            rows.append(
                {
                    "cycle_a": c1,
                    "cycle_b": c2,
                    "anchors_a": len(a),
                    "anchors_b": len(b),
                    "dtw_phase_distance": round(dtw_distance(a, b), 4),
                }
            )
    return pd.DataFrame(rows)


def plot_anchors(series: pd.Series, table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    for ax, cycle in zip(axes, ["2017", "2021", "2025"]):
        halving = HALVINGS[cycle]
        end = NEXT_HALVINGS[cycle] or series.index.max()
        seg = series.loc[halving:end]
        ax.plot(seg.index, seg.values, color="#f06f61", linewidth=1.5)
        g = table[table["cycle"] == cycle]
        for _, row in g.iterrows():
            d = pd.Timestamp(row["date"])
            color = "#00b050" if row["kind"] == "trough" else "#00e676"
            ax.scatter(d, row["price"], s=42, color=color, edgecolor="black", linewidth=0.5, zorder=4)
            ax.text(d, row["price"], str(int(row["anchor_index"])), fontsize=8, color="black")
        ax.set_yscale("log")
        ax.set_title(f"{cycle} cycle auto anchors")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "green_anchor_probe.png", dpi=180)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    rows = []
    columns = list(df.columns)
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    anchors = detect_anchors(series)
    table = cycle_anchor_table(series, anchors)
    intervals = normalized_intervals(table)
    similarity = similarity_report(table)

    table.to_csv(OUT / "auto_anchors.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(OUT / "anchor_intervals.csv", index=False, encoding="utf-8-sig")
    similarity.to_csv(OUT / "cycle_similarity.csv", index=False, encoding="utf-8-sig")
    plot_anchors(series, table)

    lines = [
        "# Green Anchor Probe",
        "",
        "This is a first-pass, falsifiable translation of the hand-marked green dots:",
        "multi-scale persistent turning points on log BTC daily price.",
        "",
        "Lower `dtw_phase_distance` means the anchor-time layout is more similar after normalizing each cycle's anchor span.",
        "",
        "## Cycle Similarity",
        markdown_table(similarity) if not similarity.empty else "Not enough anchors.",
        "",
        "## Anchor Counts",
        markdown_table(table.groupby("cycle").size().rename("anchors").reset_index()),
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
