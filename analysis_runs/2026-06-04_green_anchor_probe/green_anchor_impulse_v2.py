from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


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
class ImpulseAnchor:
    date: pd.Timestamp
    price: float
    previous_peak_date: pd.Timestamp | None
    previous_peak_price: float | None
    next_peak_date: pd.Timestamp | None
    next_peak_price: float | None
    drop_into_anchor_pct: float | None
    rally_from_anchor_pct: float | None
    score: float


def load_price() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df.set_index("date")["price"].sort_index().asfreq("D").ffill()


def local_extrema(series: pd.Series, order: int = 18) -> pd.DataFrame:
    logp = np.log(series).rolling(7, center=True, min_periods=3).mean().dropna()
    highs = argrelextrema(logp.values, np.greater_equal, order=order)[0]
    lows = argrelextrema(logp.values, np.less_equal, order=order)[0]
    rows = []
    for idx in highs:
        d = logp.index[int(idx)].normalize()
        rows.append({"date": d, "kind": "peak", "price": float(series.loc[d])})
    for idx in lows:
        d = logp.index[int(idx)].normalize()
        rows.append({"date": d, "kind": "trough", "price": float(series.loc[d])})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def compress_zigzag(points: pd.DataFrame) -> pd.DataFrame:
    compressed = []
    for _, row in points.iterrows():
        item = row.to_dict()
        if not compressed:
            compressed.append(item)
            continue
        last = compressed[-1]
        if item["kind"] != last["kind"]:
            compressed.append(item)
            continue
        if item["kind"] == "peak" and item["price"] > last["price"]:
            compressed[-1] = item
        elif item["kind"] == "trough" and item["price"] < last["price"]:
            compressed[-1] = item
    return pd.DataFrame(compressed)


def impulse_anchors(series: pd.Series, cycle: str) -> list[ImpulseAnchor]:
    start = HALVINGS[cycle]
    end = NEXT_HALVINGS[cycle] or series.index.max()
    seg = series.loc[start:end]
    points = compress_zigzag(local_extrema(seg))

    anchors: list[ImpulseAnchor] = []
    # Dynamic thresholds: later cycles need lower raw pct thresholds because volatility anneals.
    min_drop = {"2017": 0.28, "2021": 0.22, "2025": 0.14}[cycle]
    min_rally = {"2017": 0.38, "2021": 0.30, "2025": 0.18}[cycle]

    for i, row in points.iterrows():
        if row["kind"] != "trough":
            continue
        prev_peaks = points[(points.index < i) & (points["kind"] == "peak")]
        next_peaks = points[(points.index > i) & (points["kind"] == "peak")]
        prev_peak = prev_peaks.iloc[-1] if not prev_peaks.empty else None
        next_peak = next_peaks.iloc[0] if not next_peaks.empty else None

        price = float(row["price"])
        drop = None
        rally = None
        if prev_peak is not None:
            drop = price / float(prev_peak["price"]) - 1.0
        if next_peak is not None:
            rally = float(next_peak["price"]) / price - 1.0

        is_drop_end = drop is not None and drop <= -min_drop
        is_rally_start = rally is not None and rally >= min_rally
        if not (is_drop_end or is_rally_start):
            continue

        drop_score = 0.0 if drop is None else abs(min(drop, 0.0))
        rally_score = 0.0 if rally is None else max(rally, 0.0)
        anchors.append(
            ImpulseAnchor(
                date=pd.Timestamp(row["date"]).normalize(),
                price=price,
                previous_peak_date=None if prev_peak is None else pd.Timestamp(prev_peak["date"]).normalize(),
                previous_peak_price=None if prev_peak is None else float(prev_peak["price"]),
                next_peak_date=None if next_peak is None else pd.Timestamp(next_peak["date"]).normalize(),
                next_peak_price=None if next_peak is None else float(next_peak["price"]),
                drop_into_anchor_pct=None if drop is None else drop * 100.0,
                rally_from_anchor_pct=None if rally is None else rally * 100.0,
                score=drop_score + rally_score,
            )
        )

    return merge_nearby_anchors(anchors)


def merge_nearby_anchors(anchors: list[ImpulseAnchor], min_gap_days: int = 35) -> list[ImpulseAnchor]:
    anchors = sorted(anchors, key=lambda x: x.score, reverse=True)
    kept: list[ImpulseAnchor] = []
    for anchor in anchors:
        if all(abs((anchor.date - old.date).days) >= min_gap_days for old in kept):
            kept.append(anchor)
    return sorted(kept, key=lambda x: x.date)


def make_tables(series: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cycle in ["2017", "2021", "2025"]:
        anchors = impulse_anchors(series, cycle)
        for i, anchor in enumerate(anchors):
            prev_anchor_date = anchors[i - 1].date if i else None
            rows.append(
                {
                    "cycle": cycle,
                    "anchor_index": i + 1,
                    "date": anchor.date.date().isoformat(),
                    "days_from_halving": (anchor.date - HALVINGS[cycle]).days,
                    "price": round(anchor.price, 2),
                    "drop_into_anchor_pct": None
                    if anchor.drop_into_anchor_pct is None
                    else round(anchor.drop_into_anchor_pct, 2),
                    "rally_from_anchor_pct": None
                    if anchor.rally_from_anchor_pct is None
                    else round(anchor.rally_from_anchor_pct, 2),
                    "prev_peak_date": None
                    if anchor.previous_peak_date is None
                    else anchor.previous_peak_date.date().isoformat(),
                    "next_peak_date": None
                    if anchor.next_peak_date is None
                    else anchor.next_peak_date.date().isoformat(),
                    "interval_from_prev_days": None
                    if prev_anchor_date is None
                    else (anchor.date - prev_anchor_date).days,
                    "score": round(anchor.score, 4),
                }
            )
    table = pd.DataFrame(rows)

    interval_rows = []
    for cycle, g in table.groupby("cycle"):
        intervals = g["interval_from_prev_days"].dropna().astype(float).values
        if len(intervals) == 0:
            continue
        for i, value in enumerate(intervals, start=2):
            interval_rows.append(
                {
                    "cycle": cycle,
                    "to_anchor_index": i,
                    "interval_days": int(value),
                    "interval_div_cycle_anchor_span": round(
                        float(value / max(g["days_from_halving"].max() - g["days_from_halving"].min(), 1)),
                        4,
                    ),
                }
            )
    return table, pd.DataFrame(interval_rows)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    dp = np.full((len(a) + 1, len(b) + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[-1, -1] / (len(a) + len(b)))


def similarity(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    phase = {}
    for cycle, g in table.groupby("cycle"):
        x = g["days_from_halving"].astype(float).values
        if len(x) < 2:
            continue
        phase[cycle] = (x - x.min()) / max(x.max() - x.min(), 1.0)
    cycles = sorted(phase)
    for i, a in enumerate(cycles):
        for b in cycles[i + 1 :]:
            rows.append(
                {
                    "cycle_a": a,
                    "cycle_b": b,
                    "anchors_a": len(phase[a]),
                    "anchors_b": len(phase[b]),
                    "dtw_phase_distance": round(dtw_distance(phase[a], phase[b]), 4),
                }
            )
    return pd.DataFrame(rows)


def plot(series: pd.Series, table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    for ax, cycle in zip(axes, ["2017", "2021", "2025"]):
        start = HALVINGS[cycle]
        end = NEXT_HALVINGS[cycle] or series.index.max()
        seg = series.loc[start:end]
        ax.plot(seg.index, seg.values, color="#ef6a5b", linewidth=1.4)
        g = table[table["cycle"] == cycle]
        for _, row in g.iterrows():
            d = pd.Timestamp(row["date"])
            ax.scatter(d, row["price"], s=58, color="#00c853", edgecolor="black", linewidth=0.6, zorder=4)
            ax.text(d, row["price"], str(int(row["anchor_index"])), fontsize=9, color="black")
        ax.set_yscale("log")
        ax.set_title(f"{cycle}: big-rally starts / big-drop ends")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "green_anchor_impulse_v2.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series = load_price()
    table, intervals = make_tables(series)
    sim = similarity(table)
    table.to_csv(OUT / "impulse_anchors_v2.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(OUT / "impulse_intervals_v2.csv", index=False, encoding="utf-8-sig")
    sim.to_csv(OUT / "impulse_similarity_v2.csv", index=False, encoding="utf-8-sig")
    plot(series, table)


if __name__ == "__main__":
    main()
